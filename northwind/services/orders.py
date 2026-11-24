import json
import secrets
import sqlite3
import time

from northwind.lib import jwt
from northwind.lib.broker_client import Publisher
from northwind.lib.http import HTTPError, UpstreamError, current_request_id
from northwind.lib.service import Service, require


def main():
    svc = Service("orders")
    svc.use_flags()
    db = svc.database("orders")
    auth = svc.upstream("auth")
    catalog = svc.upstream("catalog")
    inventory = svc.upstream("inventory")
    payments = svc.upstream("payments")
    publisher = Publisher(svc, svc.upstream("broker"))
    trusted = [
        {"kid": k["kid"], "file": svc.path(k["file"])} for k in svc.config["auth"]["trusted_keys"]
    ]
    keys = jwt.load_keys(trusted)
    svc.ready_checks.append(lambda: (bool(keys), "no trusted signing keys loaded"))
    max_items = int(svc.config.get("max_items_per_order", 10))

    def claims_from(req):
        h = req.headers.get("Authorization", "")
        require(h.startswith("Bearer "), 401, "missing bearer token")
        try:
            return jwt.verify(h[7:], keys)
        except jwt.TokenError as e:
            svc.metrics.inc("orders_auth_failures_total", reason=str(e).split(" kid=")[0])
            svc.log.warn("token verification failed", req=req.request_id, reason=str(e))
            raise HTTPError(401, "invalid token") from None

    def set_status(order_id, status, event=None, payload=None, reason=None, rid=None):
        now = time.time()
        with db.tx() as c:
            c.execute(
                "UPDATE orders SET status=?, updated_at=?, failure_reason=? WHERE id=?",
                (status, now, reason, order_id),
            )
            if event:
                c.execute(
                    "INSERT INTO outbox (topic, payload, request_id, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (event, json.dumps(payload), rid, now),
                )

    def fail(order_id, status, reason, rid, http_status):
        set_status(order_id, status, reason=reason)
        svc.metrics.inc("orders_checkouts_total", outcome=status)
        svc.log.error("checkout failed", req=rid, order=order_id, status=status, reason=reason)
        return http_status, {"error": reason, "order_id": order_id, "status": status}

    def call(up, method, path, body=None):
        try:
            return up.request(method, path, body=body)
        except UpstreamError as e:
            return e

    @svc.app.route("POST", "/checkout")
    def checkout(req):
        if svc.flag("checkout_mode", "maintenance") != "on":
            svc.log.warn(
                "checkout rejected: checkout_mode is not on",
                req=req.request_id,
                flags_source=svc.flags.source,
            )
            return 503, {"error": "checkout temporarily disabled"}
        claims = claims_from(req)
        rid = req.request_id
        items = req.json.get("items") or []
        require(0 < len(items) <= max_items, 400, f"between 1 and {max_items} items required")
        for it in items:
            require(
                isinstance(it.get("sku"), str) and int(it.get("qty", 0)) > 0,
                400,
                "each item needs sku and positive qty",
            )
        key = req.headers.get("Idempotency-Key")
        if key:
            prior = db.one("SELECT id, status FROM orders WHERE idempotency_key=?", (key,))
            if prior:
                return 200, {"order_id": prior["id"], "status": prior["status"], "duplicate": True}

        user = call(auth, "GET", f"/users/{claims['sub']}")
        if isinstance(user, UpstreamError) or not user.ok:
            svc.log.error(
                "user lookup failed",
                req=rid,
                user=claims["sub"],
                error=str(user) if isinstance(user, UpstreamError) else user.status,
            )
            return 503, {"error": "user service unavailable"}
        user = user.json()

        priced, total = [], 0
        for it in items:
            p = call(catalog, "GET", f"/products/{it['sku']}")
            if isinstance(p, UpstreamError):
                return 503, {"error": "catalog unavailable"}
            if p.status == 404:
                return 400, {"error": "unknown product", "sku": it["sku"]}
            if not p.ok:
                return 503, {"error": "catalog error"}
            price = int(p.json()["price_cents"])
            priced.append((it["sku"], int(it["qty"]), price))
            total += price * int(it["qty"])

        order_id = "ord_" + secrets.token_hex(6)
        now = time.time()
        try:
            with db.tx() as c:
                c.execute(
                    "INSERT INTO orders (id, user_id, email, status, total_cents, "
                    "idempotency_key, created_at, updated_at, shipping_address, channel) VALUES "
                    "(?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                    (
                        order_id,
                        user["id"],
                        user["email"],
                        total,
                        key,
                        now,
                        now,
                        user["address"],
                        req.headers.get("X-Channel", "web")[:16],
                    ),
                )
                c.executemany(
                    "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                    [(order_id, s, q, p) for s, q, p in priced],
                )
        except sqlite3.IntegrityError:
            prior = db.one("SELECT id, status FROM orders WHERE idempotency_key=?", (key,))
            return 200, {"order_id": prior["id"], "status": prior["status"], "duplicate": True}

        r = call(
            inventory,
            "POST",
            "/reservations",
            {"order_id": order_id, "items": [{"sku": s, "qty": q} for s, q, _ in priced]},
        )
        if isinstance(r, UpstreamError):
            return fail(order_id, "failed", f"inventory {r.kind}", rid, 503)
        if r.status == 409:
            return fail(order_id, "rejected", "insufficient stock", rid, 409)
        if not r.ok:
            return fail(order_id, "failed", f"inventory returned {r.status}", rid, 503)

        r = call(
            payments,
            "POST",
            "/charges",
            {"order_id": order_id, "amount_cents": total, "idempotency_key": order_id},
        )
        if isinstance(r, UpstreamError) or not r.ok:
            reason = (
                f"payments {r.kind}"
                if isinstance(r, UpstreamError)
                else f"payments returned {r.status}: {r.json().get('error', '')}"
            )
            rel = call(inventory, "POST", f"/reservations/{order_id}/release")
            if isinstance(rel, UpstreamError) or not rel.ok:
                svc.log.warn(
                    "could not release reservation, scheduler will expire it",
                    req=rid,
                    order=order_id,
                )
            return fail(
                order_id, "payment_failed", reason, rid, 504 if "timeout" in reason else 502
            )

        set_status(
            order_id,
            "paid",
            event="order.paid",
            rid=rid,
            payload={
                "order_id": order_id,
                "user_id": user["id"],
                "email": user["email"],
                "total_cents": total,
                "items": [{"sku": s, "qty": q} for s, q, _ in priced],
            },
        )
        svc.metrics.inc("orders_checkouts_total", outcome="paid")
        svc.log.info("checkout completed", req=rid, order=order_id, total_cents=total)
        return 201, {"order_id": order_id, "status": "paid", "total_cents": total}

    def order_view(order_id):
        o = db.one(
            "SELECT id, user_id, status, total_cents, created_at, updated_at, "
            "failure_reason FROM orders WHERE id=?",
            (order_id,),
        )
        require(o, 404, "order not found")
        o["items"] = db.query(
            "SELECT sku, qty, price_cents FROM order_items WHERE order_id=?", (order_id,)
        )
        s = db.one("SELECT tracking FROM shipments WHERE order_id=?", (order_id,))
        o["tracking"] = s["tracking"] if s else None
        return o

    @svc.app.route("GET", "/orders/{id}")
    def get_order(req):
        o = order_view(req.params["id"])
        if req.headers.get("X-Caller", "client") in ("edge", "client"):
            claims = claims_from(req)
            require(claims["sub"] == o["user_id"], 404, "order not found")
        return 200, o

    @svc.app.route("POST", "/orders/{id}/fulfill", internal=True)
    def fulfill(req):
        oid = req.params["id"]
        tracking = req.json.get("tracking")
        require(tracking, 400, "tracking required")
        o = db.one("SELECT status, email, user_id FROM orders WHERE id=?", (oid,))
        require(o, 404, "order not found")
        if o["status"] == "fulfilled":
            return 200, {"order_id": oid, "status": "fulfilled", "duplicate": True}
        require(o["status"] == "paid", 409, f"order is {o['status']}, not paid")
        now = time.time()
        with db.tx() as c:
            c.execute("INSERT OR IGNORE INTO shipments VALUES (?, ?, ?)", (oid, tracking, now))
            c.execute("UPDATE orders SET status='fulfilled', updated_at=? WHERE id=?", (now, oid))
            c.execute(
                "INSERT INTO outbox (topic, payload, request_id, created_at) VALUES (?, ?, ?, ?)",
                (
                    "order.shipped",
                    json.dumps(
                        {
                            "order_id": oid,
                            "email": o["email"],
                            "user_id": o["user_id"],
                            "tracking": tracking,
                        }
                    ),
                    req.request_id,
                    now,
                ),
            )
        return 200, {"order_id": oid, "status": "fulfilled"}

    @svc.app.route("POST", "/orders/{id}/refunded", internal=True)
    def mark_refunded(req):
        oid = req.params["id"]
        o = db.one("SELECT status FROM orders WHERE id=?", (oid,))
        require(o, 404, "order not found")
        set_status(oid, "refunded", reason=req.json.get("reason", "refunded"))
        return 200, {"order_id": oid, "status": "refunded"}

    @svc.app.route("POST", "/orders/{id}/abandon", internal=True)
    def abandon(req):
        oid = req.params["id"]
        o = db.one("SELECT status FROM orders WHERE id=?", (oid,))
        require(o, 404, "order not found")
        if o["status"] == "pending":
            set_status(oid, "failed", reason="abandoned: checkout did not complete")
            call(inventory, "POST", f"/reservations/{oid}/release")
        return 200, {"order_id": oid}

    @svc.app.route("GET", "/admin/migrations")
    def migrations(req):
        from northwind import schema

        with db.conn() as c:
            applied = [r[0] for r in c.execute("SELECT version FROM schema_migrations")]
            return 200, {"applied": sorted(applied), "pending": schema.pending(c, "orders")}

    @svc.app.route("POST", "/admin/migrations/apply")
    def apply_migrations(req):
        from northwind import schema
        from northwind.lib.db import connect

        c = connect(db.path)
        try:
            before = schema.pending(c, "orders")
            schema.migrate(c, "orders")
        finally:
            c.close()
        svc.log.info("schema migrations applied", versions=",".join(map(str, before)))
        return 200, {"applied": before}

    @svc.app.route("GET", "/orders", internal=True)
    def list_orders(req):
        status = req.arg("status")
        older = float(req.arg("older_than_s", 0))
        require(status, 400, "status required")
        rows = db.query(
            "SELECT id, status, total_cents, updated_at FROM orders WHERE status=? "
            "AND updated_at<=? ORDER BY updated_at LIMIT 200",
            (status, time.time() - older),
        )
        return 200, {"orders": rows}

    def relay_outbox():
        rows = db.query("SELECT * FROM outbox WHERE published_at IS NULL ORDER BY id LIMIT 50")
        for r in rows:
            try:
                tok = current_request_id.set(r["request_id"])
                try:
                    publisher.publish(r["topic"], json.loads(r["payload"]))
                finally:
                    current_request_id.reset(tok)
            except UpstreamError as e:
                svc.log.warn(
                    "outbox relay failed, will retry", error=str(e)[:120], pending=len(rows)
                )
                return
            with db.tx() as c:
                c.execute("UPDATE outbox SET published_at=? WHERE id=?", (time.time(), r["id"]))

    svc.metrics.gauge_fn(
        "orders_outbox_pending",
        lambda: db.one("SELECT COUNT(*) AS n FROM outbox WHERE published_at IS NULL")["n"],
    )
    svc.every(float(svc.config.get("outbox_interval_s", 0.3)), relay_outbox, "outbox_relay")
    svc.run()


if __name__ == "__main__":
    main()
