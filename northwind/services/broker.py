"""Message broker: topics fan out to bound queues; consumers pull with a visibility timeout,
then ack or nack. Failed deliveries are retried up to max_redeliveries, then moved to the
queue's dead-letter list. Everything is persisted in data/broker.db.

etc/broker.yaml:
  defaults: {max_redeliveries: 5, redelivery_delay_ms: 500, ordered: false}
  queues: {fulfillment.order-paid: {max_redeliveries: 5}}
  bindings: [{queue: notify.events, topic: order.paid}]   # static bindings (optional)
"""

import contextlib
import json
import secrets
import threading
import time

from northwind.lib.db import connect
from northwind.lib.service import Service, require

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload TEXT NOT NULL,
  request_id TEXT, published_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS bindings (queue TEXT NOT NULL, topic TEXT NOT NULL,
  PRIMARY KEY (queue, topic));
CREATE TABLE IF NOT EXISTS entries (
  queue TEXT NOT NULL, msg_id INTEGER NOT NULL, state TEXT NOT NULL,
  deliveries INTEGER NOT NULL DEFAULT 0, visible_at REAL NOT NULL DEFAULT 0,
  receipt TEXT, last_error TEXT, updated_at REAL,
  PRIMARY KEY (queue, msg_id));
CREATE INDEX IF NOT EXISTS entries_ready ON entries (queue, state, visible_at, msg_id);
CREATE TABLE IF NOT EXISTS unrouted (msg_id INTEGER PRIMARY KEY, topic TEXT);
"""


def main():
    svc = Service("broker")
    cfg = svc.config
    defaults = {
        "max_redeliveries": 5,
        "redelivery_delay_ms": 500,
        "ordered": False,
        **(cfg.get("defaults") or {}),
    }
    qcfg = cfg.get("queues") or {}
    db = connect(svc.path(cfg.get("db", "data/broker.db")), threads=True)
    db.executescript(SCHEMA)
    for b in cfg.get("bindings") or []:
        db.execute("INSERT OR IGNORE INTO bindings VALUES (?, ?)", (b["queue"], b["topic"]))
    lock = threading.Lock()

    @contextlib.contextmanager
    def tx():
        """Write transaction on the shared connection; rolled back on any error so a failed
        write (e.g. disk full) never leaves the connection stuck inside a transaction."""
        with lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def qsetting(queue, key):
        return (qcfg.get(queue) or {}).get(key, defaults[key])

    def max_deliveries(queue):
        """None means unlimited (max_redeliveries < 0)."""
        m = int(qsetting(queue, "max_redeliveries"))
        return None if m < 0 else m + 1

    def dead_letter(queue, msg_id, err, now):
        db.execute(
            "UPDATE entries SET state='dead', last_error=?, updated_at=? "
            "WHERE queue=? AND msg_id=?",
            (err, now, queue, msg_id),
        )
        svc.log.warn(
            "message moved to dead-letter queue",
            queue=queue,
            msg_id=msg_id,
            reason=(err or "")[:120],
        )

    @svc.app.route("POST", "/publish")
    def publish(req):
        b = req.json
        require(b.get("topic"), 400, "missing topic")
        now = time.time()
        with tx():
            cur = db.execute(
                "INSERT INTO messages (topic, payload, request_id, published_at) "
                "VALUES (?, ?, ?, ?)",
                (b["topic"], json.dumps(b.get("payload", {})), b.get("request_id"), now),
            )
            mid = cur.lastrowid
            queues = [
                r[0] for r in db.execute("SELECT queue FROM bindings WHERE topic=?", (b["topic"],))
            ]
            for q in queues:
                db.execute(
                    "INSERT INTO entries (queue, msg_id, state, visible_at, updated_at) "
                    "VALUES (?, ?, 'ready', 0, ?)",
                    (q, mid, now),
                )
            if not queues:
                db.execute("INSERT INTO unrouted VALUES (?, ?)", (mid, b["topic"]))
        svc.metrics.inc("broker_published_total", topic=b["topic"])
        if not queues:
            svc.log.warn(
                "message published to topic with no subscribers",
                topic=b["topic"],
                msg_id=mid,
                req=b.get("request_id"),
            )
        return 200, {"id": mid, "queues": queues}

    @svc.app.route("POST", "/subscribe")
    def subscribe(req):
        b = req.json
        require(b.get("queue") and b.get("topic"), 400, "queue and topic required")
        with lock:
            cur = db.execute(
                "INSERT OR IGNORE INTO bindings VALUES (?, ?)", (b["queue"], b["topic"])
            )
        if cur.rowcount:
            svc.log.info("binding created", queue=b["queue"], topic=b["topic"])
        return 200, {"queue": b["queue"], "topic": b["topic"]}

    @svc.app.route("POST", "/consume")
    def consume(req):
        b = req.json
        queue = b.get("queue")
        require(queue, 400, "missing queue")
        n = max(1, min(int(b.get("max", 10)), 100))
        vis = float(b.get("visibility_timeout_s", 10))
        limit = max_deliveries(queue)
        ordered = bool(qsetting(queue, "ordered"))
        now = time.time()
        out = []
        select = (
            "SELECT e.msg_id, e.deliveries, e.state, e.visible_at, m.topic, m.payload, "
            "m.request_id FROM entries e JOIN messages m ON m.id=e.msg_id "
            "WHERE e.queue=? AND e.state IN ('ready','inflight') "
        )
        with tx():
            if ordered:
                # strict FIFO: only the head of the queue may be delivered, and only when
                # it is visible (not in flight, not waiting for its redelivery delay)
                rows = db.execute(select + "ORDER BY e.msg_id LIMIT 1", (queue,)).fetchall()
                rows = [r for r in rows if r["visible_at"] <= now]
            else:
                rows = db.execute(
                    select + "AND e.visible_at<=? ORDER BY e.msg_id LIMIT ?", (queue, now, n)
                ).fetchall()
            for r in rows:
                if r["state"] == "inflight" and limit is not None and r["deliveries"] >= limit:
                    dead_letter(queue, r["msg_id"], "visibility timeout expired", now)
                    continue
                receipt = secrets.token_hex(6)
                db.execute(
                    "UPDATE entries SET state='inflight', deliveries=deliveries+1, "
                    "visible_at=?, receipt=?, updated_at=? WHERE queue=? AND msg_id=?",
                    (now + vis, receipt, now, queue, r["msg_id"]),
                )
                out.append(
                    {
                        "id": r["msg_id"],
                        "topic": r["topic"],
                        "payload": json.loads(r["payload"]),
                        "request_id": r["request_id"],
                        "deliveries": r["deliveries"] + 1,
                        "receipt": receipt,
                    }
                )
        return 200, {"messages": out}

    def settle(req, ack: bool):
        b = req.json
        queue, mid, receipt = b.get("queue"), b.get("id"), b.get("receipt")
        now = time.time()
        with lock:
            row = db.execute(
                "SELECT state, deliveries, receipt FROM entries WHERE queue=? AND msg_id=?",
                (queue, mid),
            ).fetchone()
            if row is None:
                return 404, {"error": "unknown message"}
            if row["state"] != "inflight" or row["receipt"] != receipt:
                return 409, {"error": "stale receipt", "state": row["state"]}
            if ack:
                db.execute(
                    "UPDATE entries SET state='acked', updated_at=? WHERE queue=? AND msg_id=?",
                    (now, queue, mid),
                )
                return 200, {"acked": True}
            err = b.get("error", "")
            limit = max_deliveries(queue)
            if limit is not None and row["deliveries"] >= limit:
                dead_letter(queue, mid, err, now)
                return 200, {"dead_lettered": True}
            delay = (
                float(qsetting(queue, "redelivery_delay_ms")) / 1000 * min(row["deliveries"], 10)
            )
            db.execute(
                "UPDATE entries SET state='ready', visible_at=?, last_error=?, "
                "updated_at=? WHERE queue=? AND msg_id=?",
                (now + delay, err, now, queue, mid),
            )
        svc.metrics.inc("broker_redeliveries_total", queue=queue)
        return 200, {"requeued": True}

    svc.app.route("POST", "/ack")(lambda req: settle(req, True))
    svc.app.route("POST", "/nack")(lambda req: settle(req, False))

    def queue_stats():
        with lock:
            rows = db.execute(
                "SELECT queue, state, COUNT(*) AS n FROM entries GROUP BY queue, state"
            ).fetchall()
            bindings = db.execute("SELECT queue, topic FROM bindings").fetchall()
            unrouted = db.execute("SELECT COUNT(*) FROM unrouted").fetchone()[0]
        out = {}
        for b in bindings:
            out.setdefault(
                b["queue"], {"topics": [], "ready": 0, "inflight": 0, "acked": 0, "dead": 0}
            )["topics"].append(b["topic"])
        for r in rows:
            out.setdefault(
                r["queue"], {"topics": [], "ready": 0, "inflight": 0, "acked": 0, "dead": 0}
            )[r["state"]] = r["n"]
        for q, s in out.items():
            s["max_redeliveries"] = qsetting(q, "max_redeliveries")
        return out, unrouted

    cache = {"t": 0.0, "v": ({}, 0)}

    def cached_stats():
        if time.time() - cache["t"] > 1.0:
            cache["v"] = queue_stats()
            cache["t"] = time.time()
        return cache["v"]

    for state, metric in (
        ("ready", "broker_queue_depth"),
        ("inflight", "broker_inflight"),
        ("dead", "broker_dlq_size"),
    ):
        for q in set(list(qcfg) + [b["queue"] for b in cfg.get("bindings") or []]):
            svc.metrics.gauge_fn(
                metric, lambda q=q, s=state: cached_stats()[0].get(q, {}).get(s, 0), queue=q
            )
    svc.metrics.gauge_fn("broker_unrouted_total", lambda: cached_stats()[1])

    @svc.app.route("GET", "/queues")
    def queues(req):
        stats, unrouted = queue_stats()
        return 200, {"queues": stats, "unrouted": unrouted}

    @svc.app.route("GET", "/queues/{queue}/dlq")
    def dlq(req):
        with lock:
            rows = db.execute(
                "SELECT e.msg_id, e.deliveries, e.last_error, m.topic, m.payload FROM entries e "
                "JOIN messages m ON m.id=e.msg_id WHERE e.queue=? AND e.state='dead' "
                "ORDER BY e.msg_id LIMIT 50",
                (req.params["queue"],),
            ).fetchall()
        return 200, {"queue": req.params["queue"], "messages": [dict(r) for r in rows]}

    @svc.app.route("POST", "/queues/{queue}/dlq/replay")
    def replay(req):
        with lock:
            cur = db.execute(
                "UPDATE entries SET state='ready', deliveries=0, visible_at=0, "
                "updated_at=? WHERE queue=? AND state='dead'",
                (time.time(), req.params["queue"]),
            )
        svc.log.info("dead-letter queue replayed", queue=req.params["queue"], count=cur.rowcount)
        return 200, {"replayed": cur.rowcount}

    @svc.app.route("GET", "/queues/{queue}/messages")
    def head(req):
        """Messages waiting in a queue (ready or in flight), oldest first."""
        with lock:
            rows = db.execute(
                "SELECT e.msg_id AS id, e.state, e.deliveries, e.last_error, m.topic, m.payload "
                "FROM entries e JOIN messages m ON m.id=e.msg_id WHERE e.queue=? AND "
                "e.state IN ('ready','inflight') ORDER BY e.msg_id LIMIT ?",
                (req.params["queue"], max(1, min(int(req.arg("limit", 10)), 100))),
            ).fetchall()
        return 200, {"queue": req.params["queue"], "messages": [dict(r) for r in rows]}

    @svc.app.route("POST", "/queues/{queue}/messages/{id}/dead-letter")
    def move_to_dlq(req):
        q, mid = req.params["queue"], int(req.params["id"])
        with lock:
            row = db.execute(
                "SELECT state FROM entries WHERE queue=? AND msg_id=?", (q, mid)
            ).fetchone()
            require(row is not None, 404, "unknown message")
            require(row["state"] in ("ready", "inflight"), 409, f"message is {row['state']}")
            dead_letter(q, mid, (req.json.get("reason") or "moved by operator")[:200], time.time())
        return 200, {"dead_lettered": mid}

    svc.ready_checks.append(lambda: (db.execute("SELECT 1").fetchone() is not None, "db"))
    svc.run()


if __name__ == "__main__":
    main()
