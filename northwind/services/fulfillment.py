import hashlib

from northwind.lib.broker_client import Consumer
from northwind.lib.http import UpstreamError
from northwind.lib.service import Service


def main():
    svc = Service("fulfillment")
    svc.use_flags()
    orders = svc.upstream("orders")
    cfg = svc.config["consumer"]

    warehouse = svc.env("FULFILLMENT_WAREHOUSE_ID", "DRYRUN")
    if warehouse == "DRYRUN":
        svc.log.warn(
            "FULFILLMENT_WAREHOUSE_ID not set, running in dry-run mode "
            "(orders are acknowledged but not shipped)"
        )

    def ship(oid):
        tracking = "NW" + hashlib.sha1(oid.encode()).hexdigest()[:10].upper()
        try:
            r = orders.post(f"/orders/{oid}/fulfill", {"tracking": tracking})
        except UpstreamError as e:
            raise RuntimeError(f"orders api {e.kind}") from e
        return r, tracking

    @svc.app.route("POST", "/admin/redrive")
    def redrive(req):
        if warehouse == "DRYRUN":
            return 409, {"error": "dry-run mode: set FULFILLMENT_WAREHOUSE_ID first"}
        r = orders.get("/orders?status=paid&older_than_s=0")
        if not r.ok:
            return 502, {"error": f"orders api returned {r.status}"}
        shipped = 0
        for o in r.json().get("orders", []):
            rr, _ = ship(o["id"])
            shipped += int(rr.ok)
        svc.log.info("redrive finished", shipped=shipped)
        return 200, {"shipped": shipped}

    def handle(msg):
        p = msg["payload"]
        oid = p["order_id"]
        if warehouse == "DRYRUN":
            svc.log.info("dry run: would ship order", order=oid)
            return
        r, tracking = ship(oid)
        if r.status == 409:
            svc.log.warn("skipping shipment", order=oid, reason=r.json().get("error"))
            return
        if not r.ok:
            raise RuntimeError(f"orders api returned {r.status}")
        svc.metrics.inc("fulfillment_shipments_total")
        svc.log.info("shipment created", order=oid, tracking=tracking)

    consumer = Consumer(
        svc,
        svc.upstream("broker"),
        cfg["queue"],
        [cfg["topic"]],
        handle,
        batch=int(cfg.get("batch", 10)),
    )
    svc.add_background(consumer.loop)
    svc.run()


if __name__ == "__main__":
    main()
