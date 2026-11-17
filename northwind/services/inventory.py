"""Stock levels and reservations with expiry.

Checkout reserves stock (reserved += qty) and the reservation is committed when the
order.paid event arrives (on_hand -= qty, reserved -= qty). Held reservations past their
expiry are released by the scheduler. available = on_hand - reserved never goes negative.
"""

import time

from northwind.lib.broker_client import Consumer
from northwind.lib.db import PoolTimeout
from northwind.lib.http import HTTPError
from northwind.lib.service import Service, require


def main():
    svc = Service("inventory")
    svc.use_flags()
    db = svc.database("inventory")
    default_ttl = float(svc.config.get("reservation_ttl_s", 120))

    @svc.app.route("GET", "/stock/{sku}")
    def stock(req):
        r = db.one(
            "SELECT sku, on_hand, reserved, on_hand - reserved AS available FROM stock WHERE sku=?",
            (req.params["sku"],),
        )
        require(r, 404, "unknown sku")
        return 200, r

    @svc.app.route("POST", "/reservations", internal=True)
    def reserve(req):
        b = req.json
        order_id, items = b.get("order_id"), b.get("items") or []
        require(order_id and items, 400, "order_id and items required")
        ttl = float(b.get("ttl_s", default_ttl))
        now = time.time()
        try:
            with db.tx() as c:
                existing = c.execute(
                    "SELECT id FROM reservations WHERE order_id=? AND status='held'", (order_id,)
                ).fetchall()
                if existing:
                    return 200, {
                        "order_id": order_id,
                        "reservations": [r[0] for r in existing],
                        "duplicate": True,
                    }
                ids = []
                for it in items:
                    sku, qty = it["sku"], int(it["qty"])
                    require(qty > 0, 400, "qty must be positive")
                    row = c.execute(
                        "SELECT on_hand - reserved FROM stock WHERE sku=?", (sku,)
                    ).fetchone()
                    require(row is not None, 404, "unknown sku", sku=sku)
                    if row[0] < qty:
                        svc.metrics.inc("inventory_reservations_total", outcome="insufficient")
                        raise HTTPError(409, "insufficient stock", sku=sku, available=row[0])
                    c.execute("UPDATE stock SET reserved = reserved + ? WHERE sku=?", (qty, sku))
                    cur = c.execute(
                        "INSERT INTO reservations (order_id, sku, qty, status, "
                        "expires_at, created_at, updated_at) VALUES "
                        "(?, ?, ?, 'held', ?, ?, ?)",
                        (order_id, sku, qty, now + ttl, now, now),
                    )
                    ids.append(cur.lastrowid)
        except PoolTimeout as e:
            svc.log.error("db pool exhausted", req=req.request_id, error=str(e))
            return 503, {"error": "database busy"}
        svc.metrics.inc("inventory_reservations_total", outcome="held")
        return 200, {"order_id": order_id, "reservations": ids, "expires_in": ttl}

    def finish(order_id: str, new_status: str) -> int:
        now = time.time()
        with db.tx() as c:
            rows = c.execute(
                "SELECT id, sku, qty FROM reservations WHERE order_id=? AND status='held'",
                (order_id,),
            ).fetchall()
            for r in rows:
                if new_status == "committed":
                    c.execute(
                        "UPDATE stock SET on_hand = on_hand - ?, reserved = reserved - ? "
                        "WHERE sku=?",
                        (r["qty"], r["qty"], r["sku"]),
                    )
                else:
                    c.execute(
                        "UPDATE stock SET reserved = reserved - ? WHERE sku=?", (r["qty"], r["sku"])
                    )
                c.execute(
                    "UPDATE reservations SET status=?, updated_at=? WHERE id=?",
                    (new_status, now, r["id"]),
                )
        return len(rows)

    @svc.app.route("POST", "/reservations/{order_id}/release", internal=True)
    def release(req):
        n = finish(req.params["order_id"], "released")
        if n:
            svc.metrics.inc("inventory_reservations_total", outcome="released")
        return 200, {"released": n}

    @svc.app.route("POST", "/reservations/{order_id}/commit", internal=True)
    def commit(req):
        return 200, {"committed": finish(req.params["order_id"], "committed")}

    def commit_expired(order_id: str) -> int:
        """The order was paid after its reservation expired (released stock). Ship anyway:
        take the stock now. Fails (CHECK on_hand >= 0) if it was sold out meanwhile."""
        now = time.time()
        with db.tx() as c:
            rows = c.execute(
                "SELECT id, sku, qty FROM reservations WHERE order_id=? AND status='expired'",
                (order_id,),
            ).fetchall()
            for r in rows:
                c.execute(
                    "UPDATE stock SET on_hand = on_hand - ? WHERE sku=?", (r["qty"], r["sku"])
                )
                c.execute(
                    "UPDATE reservations SET status='committed', updated_at=? WHERE id=?",
                    (now, r["id"]),
                )
        return len(rows)

    def on_order_paid(msg):
        order_id = msg["payload"]["order_id"]
        if finish(order_id, "committed"):
            svc.metrics.inc("inventory_reservations_total", outcome="committed")
            return
        if commit_expired(order_id):
            svc.metrics.inc("inventory_reservations_total", outcome="committed_after_expiry")
            svc.log.warn(
                "reservation had expired before payment; committed stock now", order=order_id
            )
            return
        r = db.one("SELECT status FROM reservations WHERE order_id=? LIMIT 1", (order_id,))
        if r is None or r["status"] != "committed":
            svc.log.warn("paid order has no reservation", order=order_id, status=r and r["status"])

    bcfg = svc.config["consumer"]
    consumer = Consumer(svc, svc.upstream("broker"), bcfg["queue"], bcfg["topics"], on_order_paid)
    svc.add_background(consumer.loop)
    svc.run()


if __name__ == "__main__":
    main()
