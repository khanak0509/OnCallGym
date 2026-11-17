"""Consumes order events and writes customer "emails" to data/outbox/emails.jsonl.

Delivery is at-least-once, so the consumer de-duplicates by (topic, order_id).
"""

import json
import os
import threading
import time

from northwind.lib.broker_client import Consumer
from northwind.lib.service import Service

TEMPLATES = {
    "order.paid": "Thanks for your order {order_id}. We'll let you know when it ships.",
    "order.shipped": "Your order {order_id} has shipped. Tracking number: {tracking}.",
}


def main():
    svc = Service("notify")
    svc.use_flags()
    cfg = svc.config
    path = svc.path(cfg.get("outbox_file", "data/outbox/emails.jsonl"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = threading.Lock()
    sent: set[tuple[str, str]] = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    sent.add((e["template"], e["order_id"]))
                except (ValueError, KeyError):
                    continue

    def handle(msg):
        p = msg["payload"]
        key = (msg["topic"], p["order_id"])
        with lock:
            if key in sent:
                return
            body = TEMPLATES[msg["topic"]].format(**{"tracking": "", **p})
            email = {
                "ts": time.time(),
                "to": p["email"],
                "template": msg["topic"],
                "order_id": p["order_id"],
                "body": body,
                "request_id": msg.get("request_id"),
            }
            with open(path, "a") as f:
                f.write(json.dumps(email) + "\n")
            sent.add(key)
        svc.metrics.inc("notify_emails_total", template=msg["topic"])

    c = cfg["consumer"]
    consumer = Consumer(svc, svc.upstream("broker"), c["queue"], c["topics"], handle)
    svc.add_background(consumer.loop)
    svc.run()


if __name__ == "__main__":
    main()
