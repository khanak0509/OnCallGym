"""Consumes order.paid, creates a shipment through the orders API."""

import hashlib

from northwind.lib.broker_client import Consumer
from northwind.lib.http import UpstreamError
from northwind.lib.service import Service


def main():
    svc = Service("fulfillment")
    svc.use_flags()
    orders = svc.upstream("orders")
    cfg = svc.config["consumer"]

    def handle(msg):
        p = msg["payload"]
        oid = p["order_id"]
        tracking = "NW" + hashlib.sha1(oid.encode()).hexdigest()[:10].upper()
        try:
            r = orders.post(f"/orders/{oid}/fulfill", {"tracking": tracking})
        except UpstreamError as e:
            raise RuntimeError(f"orders api {e.kind}") from e
        if r.status == 409:
            # refunded or failed orders are not shipped; ack and move on
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
