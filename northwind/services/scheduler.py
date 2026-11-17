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

    checked: set[str] = set()

    def reconcile_payments(job):
        """Orders that failed (or were abandoned mid-checkout) may still have been charged at
        the provider, e.g. when our call timed out. Ask payments to refund those."""
        grace = float(job.get("grace_s", 30))
        now = time.time()
        oc = connect(svc.path(cfg["databases"]["orders"]))
        try:
            rows = [
                dict(r)
                for r in oc.execute(
                    "SELECT id, status FROM orders WHERE updated_at<? AND updated_at>? AND "
                    "status IN ('payment_failed','failed','rejected','pending') "
                    "ORDER BY updated_at LIMIT 200",
                    (now - grace, now - 3600),
                )
                if r["id"] not in checked
            ]
        finally:
            oc.close()
        refunded = 0
        for o in rows:
            try:
                r = payments.post("/refunds", {"order_id": o["id"]})
                if not r.ok:
                    continue
                if r.json().get("refunded"):
                    orders.post(
                        f"/orders/{o['id']}/refunded",
                        {"reason": "charged but checkout did not complete; refunded by reconcile"},
                    )
                    refunded += 1
                    svc.log.warn(
                        "refunded charge for failed order", order=o["id"], order_status=o["status"]
                    )
                elif o["status"] == "pending":
                    orders.post(f"/orders/{o['id']}/abandon", {})
                    svc.log.warn("abandoned stuck pending order", order=o["id"])
                checked.add(o["id"])
            except UpstreamError as e:
                svc.log.warn("reconcile failed", order=o["id"], error=str(e)[:120])
        return {"checked": len(rows), "refunded": refunded}

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
