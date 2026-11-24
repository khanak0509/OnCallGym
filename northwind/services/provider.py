import random
import secrets
import sqlite3
import threading
import time

from northwind.lib.service import Service, require


def main():
    svc = Service("provider")
    cfg = svc.config
    db = svc.database("provider")
    lat = cfg.get("latency_ms", {"base": 40, "jitter": 30})
    error_rate = float(cfg.get("error_rate", 0.0))
    rng = random.Random()
    lock = threading.Lock()

    def delay():
        time.sleep(max(0.0, (lat.get("base", 40) + rng.uniform(0, lat.get("jitter", 30))) / 1000))

    @svc.app.route("POST", "/v1/charges", internal=True)
    def charge(req):
        b = req.json
        key = req.headers.get("Idempotency-Key")
        require(
            b.get("merchant_ref") and int(b.get("amount_cents", 0)) > 0,
            400,
            "merchant_ref and positive amount_cents required",
        )
        delay()
        if rng.random() < error_rate:
            svc.log.warn("upstream acquirer error", req=req.request_id)
            return 503, {"error": "acquirer unavailable, retry later"}
        with lock:
            if key:
                prior = db.one("SELECT * FROM charges WHERE idempotency_key=?", (key,))
                if prior:
                    svc.metrics.inc("provider_charges_total", outcome="idempotent_replay")
                    return 200, {**prior, "idempotent_replay": True}
            cid = "ch_" + secrets.token_hex(8)
            try:
                with db.tx() as c:
                    c.execute(
                        "INSERT INTO charges VALUES (?, ?, ?, ?, 'succeeded', ?)",
                        (cid, key, b["merchant_ref"], int(b["amount_cents"]), time.time()),
                    )
            except sqlite3.IntegrityError:
                prior = db.one("SELECT * FROM charges WHERE idempotency_key=?", (key,))
                return 200, {**prior, "idempotent_replay": True}
        svc.metrics.inc("provider_charges_total", outcome="succeeded")
        return 200, {
            "id": cid,
            "merchant_ref": b["merchant_ref"],
            "amount_cents": int(b["amount_cents"]),
            "status": "succeeded",
        }

    @svc.app.route("POST", "/v1/refunds", internal=True)
    def refund(req):
        b = req.json
        delay()
        ch = db.one("SELECT * FROM charges WHERE id=?", (b.get("charge_id"),))
        require(ch, 404, "no such charge")
        prior = db.one("SELECT * FROM refunds WHERE charge_id=?", (ch["id"],))
        if prior:
            return 200, prior
        rid = "re_" + secrets.token_hex(8)
        with db.tx() as c:
            c.execute(
                "INSERT INTO refunds VALUES (?, ?, ?, ?)",
                (rid, ch["id"], ch["amount_cents"], time.time()),
            )
            c.execute("UPDATE charges SET status='refunded' WHERE id=?", (ch["id"],))
        return 200, {"id": rid, "charge_id": ch["id"], "amount_cents": ch["amount_cents"]}

    @svc.app.route("GET", "/v1/charges", internal=True)
    def list_charges(req):
        ref = req.arg("merchant_ref")
        require(ref, 400, "merchant_ref required")
        return 200, {"charges": db.query("SELECT * FROM charges WHERE merchant_ref=?", (ref,))}

    @svc.app.route("GET", "/v1/status")
    def status(req):
        return 200, {
            "status": "operational"
            if error_rate < 0.2 and lat.get("base", 0) < 1000
            else "degraded",
            "latency_ms": lat,
            "error_rate": error_rate,
        }

    svc.run()


if __name__ == "__main__":
    main()
