"""Client for the Northwind broker: publish, and a polling consumer with ack/nack."""

import time

from .http import Upstream, UpstreamError, current_request_id


class Publisher:
    def __init__(self, svc, upstream: Upstream):
        self.svc = svc
        self.up = upstream

    def publish(self, topic: str, payload: dict) -> str:
        r = self.up.post(
            "/publish", {"topic": topic, "payload": payload, "request_id": current_request_id.get()}
        )
        if not r.ok:
            raise UpstreamError("broker", f"http_{r.status}", r.body[:200].decode(errors="replace"))
        return r.json()["id"]


class Consumer:
    """Pulls batches from `queue` (bound to `topics`) and calls handler(msg) for each.
    handler returning normally -> ack; raising -> nack (broker redelivers or dead-letters)."""

    def __init__(
        self,
        svc,
        upstream: Upstream,
        queue: str,
        topics: list[str],
        handler,
        batch: int = 10,
        poll_interval_s: float = 0.3,
        visibility_timeout_s: float = 10,
    ):
        self.svc = svc
        self.up = upstream
        self.queue = queue
        self.topics = topics
        self.handler = handler
        self.batch = batch
        self.poll_interval_s = poll_interval_s
        self.visibility_timeout_s = visibility_timeout_s
        self.subscribed = False
        self.processed = 0
        self.failed = 0
        svc.metrics.gauge_fn("consumer_processed_total", lambda: self.processed, queue=queue)
        svc.metrics.gauge_fn("consumer_failed_total", lambda: self.failed, queue=queue)
        svc.ready_checks.append(lambda: (self.subscribed, f"not subscribed to {queue}"))

    def subscribe(self):
        for t in self.topics:
            r = self.up.post("/subscribe", {"queue": self.queue, "topic": t})
            if not r.ok:
                raise UpstreamError("broker", f"http_{r.status}")
        if not self.subscribed:
            self.svc.log.info("subscribed", queue=self.queue, topics=",".join(self.topics))
        self.subscribed = True

    def loop(self):
        log = self.svc.log
        while not self.svc.stopping.is_set():
            try:
                if not self.subscribed:
                    self.subscribe()
                r = self.up.post(
                    "/consume",
                    {
                        "queue": self.queue,
                        "max": self.batch,
                        "visibility_timeout_s": self.visibility_timeout_s,
                    },
                )
                if not r.ok:
                    raise UpstreamError("broker", f"http_{r.status}")
                msgs = r.json().get("messages", [])
            except UpstreamError as e:
                log.warn("broker poll failed", queue=self.queue, error=str(e)[:160])
                self.svc.stopping.wait(1.0)
                continue
            for m in msgs:
                token = current_request_id.set(m.get("request_id") or None)
                t0 = time.monotonic()
                try:
                    self.handler(m)
                    self._settle("/ack", m)
                    self.processed += 1
                except Exception as e:
                    self.failed += 1
                    log.error(
                        "message handling failed",
                        req=m.get("request_id"),
                        queue=self.queue,
                        msg_id=m["id"],
                        delivery=m.get("deliveries"),
                        error=f"{type(e).__name__}: {e}"[:200],
                    )
                    self._settle("/nack", m, error=f"{type(e).__name__}: {e}"[:200])
                finally:
                    self.svc.metrics.observe(
                        "consumer_handle_ms", (time.monotonic() - t0) * 1000, queue=self.queue
                    )
                    current_request_id.reset(token)
            if not msgs:
                self.svc.stopping.wait(self.poll_interval_s)

    def _settle(self, path, m, error=None):
        body = {"queue": self.queue, "id": m["id"], "receipt": m["receipt"]}
        if error:
            body["error"] = error
        try:
            r = self.up.post(path, body)
            if not r.ok:
                # e.g. 409 stale receipt: visibility timeout expired while handling, so the
                # message will be delivered again (handlers must be idempotent)
                self.svc.log.warn(
                    "ack/nack rejected",
                    queue=self.queue,
                    msg_id=m["id"],
                    status=r.status,
                    error=r.json().get("error"),
                )
        except UpstreamError as e:
            self.svc.log.warn(
                "ack/nack failed", queue=self.queue, msg_id=m["id"], error=str(e)[:120]
            )
