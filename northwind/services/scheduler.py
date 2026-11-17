"""Periodic jobs: expire stale reservations, reconcile payments against orders, rotate logs.

Jobs and intervals come from etc/scheduler.yaml. Like many real cron runners, jobs touch
the databases directly (inventory.db, payments.db, orders.db) and call APIs where a side
effect must go through a service (refunds).
"""

import glob
import os
import threading
import time

from northwind.lib.db import connect
from northwind.lib.http import UpstreamError
from northwind.lib.service import Service


def main():
    svc = Service("scheduler")
    svc.use_flags()
    cfg = svc.config
    jobs_cfg = cfg["jobs"]
    payments = svc.upstream("payments")
    orders = svc.upstream("orders")
    status: dict[str, dict] = {}

    def expire_reservations(job):
        c = connect(svc.path(cfg["databases"]["inventory"]))
        try:
            now = time.time()
            c.execute("BEGIN IMMEDIATE")
            rows = c.execute(
                "SELECT id, sku, qty FROM reservations WHERE status='held' AND "
                "expires_at<? LIMIT ?",
                (now, int(job.get("batch", 200))),
            ).fetchall()
            for r in rows:
                c.execute(
                    "UPDATE stock SET reserved = reserved - ? WHERE sku=?", (r["qty"], r["sku"])
                )
                c.execute(
                    "UPDATE reservations SET status='expired', updated_at=? WHERE id=?",
                    (now, r["id"]),
                )
            hold = float(job.get("hold_lock_s", 0))
            if hold:
                time.sleep(hold)
            c.execute("COMMIT")
        finally:
            c.close()
        if rows:
            svc.log.info("expired stale reservations", count=len(rows))
        return {"expired": len(rows)}

    def reconcile_payments(job):
        grace = float(job.get("grace_s", 30))
        pc = connect(svc.path(cfg["databases"]["payments"]))
        oc = connect(svc.path(cfg["databases"]["orders"]))
        try:
            charged = pc.execute(
                "SELECT order_id FROM charges WHERE status='succeeded' AND updated_at<?",
                (time.time() - grace,),
            ).fetchall()
            ids = [r[0] for r in charged]
            bad = []
            for i in range(0, len(ids), 200):
                chunk = ids[i : i + 200]
                q = ",".join("?" * len(chunk))
                bad += [
                    dict(r)
                    for r in oc.execute(
                        f"SELECT id, status FROM orders WHERE id IN ({q}) AND status IN "
                        "('payment_failed','failed','rejected')",
                        chunk,
                    )
                ]
        finally:
            pc.close()
            oc.close()
        refunded = 0
        for o in bad:
            try:
                r = payments.post("/refunds", {"order_id": o["id"]})
                if r.ok:
                    orders.post(
                        f"/orders/{o['id']}/refunded",
                        {"reason": "charged but checkout failed; refunded by reconcile"},
                    )
                    refunded += 1
                    svc.log.warn(
                        "refunded charge for failed order", order=o["id"], order_status=o["status"]
                    )
            except UpstreamError as e:
                svc.log.warn("reconcile refund failed", order=o["id"], error=str(e)[:120])
        return {"checked": len(ids), "refunded": refunded}

    def rotate_logs(job):
        max_bytes = int(job.get("max_bytes", 2_000_000))
        keep = int(job.get("keep", 2))
        rotated = 0
        for path in glob.glob(svc.path("logs/*.log")):
            try:
                if os.path.getsize(path) < max_bytes:
                    continue
                for i in range(keep - 1, 0, -1):
                    if os.path.exists(f"{path}.{i}"):
                        os.replace(f"{path}.{i}", f"{path}.{i + 1}")
                os.replace(path, f"{path}.1")
                rotated += 1
            except OSError as e:
                svc.log.warn("log rotation failed", file=os.path.basename(path), error=str(e))
        if rotated:
            svc.log.info("rotated logs", count=rotated)
        return {"rotated": rotated}

    fns = {
        "expire_reservations": expire_reservations,
        "reconcile_payments": reconcile_payments,
        "rotate_logs": rotate_logs,
    }

    def run_job(name, job):
        interval = float(job["interval_s"])
        st = status.setdefault(
            name,
            {
                "runs": 0,
                "failures": 0,
                "last_run": None,
                "last_duration_ms": None,
                "last_error": None,
                "last_result": None,
                "interval_s": interval,
            },
        )
        while not svc.stopping.wait(interval):
            if not job.get("enabled", True):
                continue
            t0 = time.monotonic()
            try:
                st["last_result"] = fns[name](job)
                st["last_error"] = None
            except Exception as e:
                st["failures"] += 1
                st["last_error"] = f"{type(e).__name__}: {e}"[:200]
                svc.log.error("job failed", job=name, error=st["last_error"])
            st["runs"] += 1
            st["last_run"] = time.time()
            st["last_duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
            svc.metrics.observe("scheduler_job_duration_ms", st["last_duration_ms"], job=name)

    for name, job in jobs_cfg.items():
        if name not in fns:
            svc.log.warn("unknown job in config, ignoring", job=name)
            continue
        threading.Thread(target=run_job, args=(name, job), daemon=True).start()

    @svc.app.route("GET", "/jobs")
    def jobs(req):
        return 200, {"jobs": status}

    svc.run()


if __name__ == "__main__":
    main()
