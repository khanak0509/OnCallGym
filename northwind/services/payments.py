"""Charges orders through the external provider: idempotency keys, retries with backoff and
a circuit breaker on the provider client (settings in etc/payments.yaml upstreams.provider)."""

import secrets
import sqlite3
import time

from northwind.lib.http import UpstreamError
from northwind.lib.service import Service, require


def main():
    svc = Service("payments")
    svc.use_flags()
    db = svc.database("payments")
    provider = svc.upstream("provider")
    send_key = bool(svc.config.get("send_idempotency_key", True))

    def update(cid, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        with db.tx() as c:
            c.execute(
                f"UPDATE charges SET {sets}, updated_at=? WHERE id=?",
                (*fields.values(), time.time(), cid),
            )

    @svc.app.route("POST", "/charges")
    def charge(req):
        b = req.json
        order_id, amount, key = (
            b.get("order_id"),
            int(b.get("amount_cents", 0)),
            b.get("idempotency_key"),
        )
        require(
            order_id and amount > 0 and key,
            400,
            "order_id, amount_cents and idempotency_key required",
        )
        row = db.one("SELECT * FROM charges WHERE idempotency_key=?", (key,))
        if row and row["status"] == "succeeded":
            svc.metrics.inc("payments_charges_total", outcome="duplicate_request")
            return 200, {"charge_id": row["id"], "status": "succeeded", "duplicate": True}
        if row is None:
            cid = "pay_" + secrets.token_hex(8)
            now = time.time()
            try:
                with db.tx() as c:
                    c.execute(
                        "INSERT INTO charges (id, order_id, idempotency_key, amount_cents, "
                        "status, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                        (cid, order_id, key, amount, now, now),
                    )
            except sqlite3.IntegrityError:
                return 409, {"error": "charge already in progress"}
        else:
            cid = row["id"]
        headers = {"Idempotency-Key": key} if send_key else {}
        t0 = time.monotonic()
        try:
            r = provider.post(
                "/v1/charges", {"merchant_ref": order_id, "amount_cents": amount}, headers=headers
            )
        except UpstreamError as e:
            update(cid, status="failed", error=e.kind)
            svc.metrics.inc("payments_charges_total", outcome=e.kind)
            svc.log.error(
                "charge failed",
                req=req.request_id,
                order=order_id,
                reason=e.kind,
                detail=e.detail,
                elapsed_ms=round((time.monotonic() - t0) * 1000),
            )
            status = 503 if e.kind == "breaker_open" else 504 if e.kind == "timeout" else 502
            return status, {"error": f"payment provider {e.kind}"}
        if not r.ok:
            update(cid, status="failed", error=f"provider_{r.status}")
            svc.metrics.inc("payments_charges_total", outcome=f"provider_{r.status}")
            svc.log.error(
                "charge failed",
                req=req.request_id,
                order=order_id,
                reason=f"provider returned {r.status}",
            )
            return 502, {"error": f"payment provider returned {r.status}"}
        pc = r.json()
        update(cid, status="succeeded", provider_charge_id=pc["id"], error=None)
        svc.metrics.inc("payments_charges_total", outcome="succeeded")
        svc.log.info(
            "charge succeeded", req=req.request_id, order=order_id, charge=cid, amount_cents=amount
        )
        return 200, {"charge_id": cid, "status": "succeeded"}

    @svc.app.route("POST", "/refunds")
    def refund(req):
        """Refund every successful provider charge for an order. The provider is the source
        of truth: a charge can have succeeded there even if our attempt timed out."""
        order_id = req.json.get("order_id")
        require(order_id, 400, "order_id required")
        try:
            r = provider.get(f"/v1/charges?merchant_ref={order_id}")
        except UpstreamError as e:
            return 502, {"error": f"payment provider {e.kind}"}
        if not r.ok:
            return 502, {"error": f"payment provider returned {r.status}"}
        remote = [c for c in r.json().get("charges", []) if c["status"] == "succeeded"]
        refunded = []
        for pc in remote:
            try:
                rr = provider.post("/v1/refunds", {"charge_id": pc["id"]})
            except UpstreamError as e:
                return 502, {"error": f"payment provider {e.kind}", "refunded": refunded}
            if not rr.ok:
                return 502, {"error": f"provider refund returned {rr.status}"}
            now = time.time()
            with db.tx() as c:
                if not c.execute(
                    "SELECT 1 FROM refunds WHERE provider_refund_id=?", (rr.json()["id"],)
                ).fetchone():
                    c.execute(
                        "INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            "rf_" + secrets.token_hex(8),
                            pc["id"],
                            order_id,
                            pc["amount_cents"],
                            rr.json()["id"],
                            now,
                        ),
                    )
                c.execute(
                    "UPDATE charges SET status='refunded', provider_charge_id=?, updated_at=? "
                    "WHERE order_id=?",
                    (pc["id"], now, order_id),
                )
            refunded.append(pc["id"])
        if refunded:
            svc.log.info(
                "refund issued",
                req=req.request_id,
                order=order_id,
                provider_charges=",".join(refunded),
            )
        return 200, {"order_id": order_id, "refunded": len(refunded), "charged": len(remote) > 0}

    @svc.app.route("GET", "/charges")
    def list_charges(req):
        order_id = req.arg("order_id")
        if order_id:
            return 200, {"charges": db.query("SELECT * FROM charges WHERE order_id=?", (order_id,))}
        since = float(req.arg("since", 0))
        return 200, {
            "charges": db.query(
                "SELECT id, order_id, status, amount_cents, updated_at FROM charges "
                "WHERE updated_at>=? ORDER BY updated_at LIMIT 500",
                (since,),
            )
        }

    svc.run()


if __name__ == "__main__":
    main()
