import http.client
import json
import os
import random
import sqlite3
import time
import uuid

from northwind.provision import table_hash

from .stack import http_json

EXPECTED_QUEUES = {
    "order.paid": ["inventory.order-paid", "fulfillment.order-paid", "notify.events"],
    "order.shipped": ["notify.events"],
}

LIVENESS_INVARIANTS = {
    "paid_orders_fulfilled_or_refunded",
    "charged_failed_orders_refunded",
    "events_delivered",
    "customers_notified",
    "shipped_orders_stock_committed",
}

THRESHOLDS = {
    "error_rate": 0.02,
    "p95_ms": 1500.0,
    "journey_timeout_s": 15.0,
    "checkout_timeout_s": 10.0,
}


def _ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("BEGIN")
    c.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    return c


class Probe:
    def __init__(self, stack):
        self.stack = stack
        with open(os.path.join(stack.private, "verifier.json")) as f:
            self.secret = json.load(f)
        self.home = stack.home
        self.edge = stack.ports["edge"]
        self.rng = random.Random(f"probe-{self.secret['seed']}")

    def _req(self, method, path, body=None, token=None, timeout=10.0):
        headers = {
            "Content-Type": "application/json",
            "X-Client-Id": "probe",
            "X-Request-Id": "probe-" + uuid.uuid4().hex[:10],
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if path == "/api/checkout":
            headers["Idempotency-Key"] = "probe-" + uuid.uuid4().hex
        t0 = time.monotonic()
        c = http.client.HTTPConnection("127.0.0.1", self.edge, timeout=timeout)
        try:
            c.request(
                method,
                path,
                body=json.dumps(body).encode() if body is not None else None,
                headers=headers,
            )
            r = c.getresponse()
            raw = r.read()
            status = r.status
            try:
                data = json.loads(raw) if raw else {}
            except ValueError:
                data = {"raw": raw[:200].decode(errors="replace")}
        except (OSError, http.client.HTTPException) as e:
            status, data = 0, {"error": f"{type(e).__name__}: {e}"}
        finally:
            c.close()
        return status, data, (time.monotonic() - t0) * 1000

    def _emails_for(self, order_id):
        path = os.path.join(self.home, "data/outbox/emails.jsonl")
        out = set()
        try:
            with open(path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("order_id") == order_id:
                        out.add(e.get("template"))
        except OSError:
            pass
        return out

    def journey(self) -> dict:
        user = self.rng.choice(self.secret["probe_users"])
        sku = self.rng.choice(self.secret["skus"])
        steps = []

        def step(name, ok, ms, detail=None):
            steps.append(
                {
                    "step": name,
                    "ok": ok,
                    "ms": round(ms, 1),
                    **({"detail": detail} if detail and not ok else {}),
                }
            )
            return ok

        res = {"ok": False, "steps": steps}
        st, d, ms = self._req(
            "POST", "/api/auth/login", {"email": user["email"], "password": user["password"]}
        )
        if not step("login", st == 200 and "token" in d, ms, f"{st} {d.get('error')}"):
            return res
        token = d["token"]
        st, d, ms = self._req("GET", "/api/catalog/products?page=1")
        if not step("browse", st == 200 and len(d.get("products", [])) > 0, ms, st):
            return res
        st, d, ms = self._req("GET", f"/api/catalog/products/{sku}")
        if not step("product", st == 200 and d.get("sku") == sku, ms, st):
            return res
        st, d, ms = self._req("GET", "/api/catalog/search?q=steel")
        if not step("search", st == 200 and len(d.get("results", [])) > 0, ms, st):
            return res
        st, d, ms = self._req(
            "POST", "/api/checkout", {"items": [{"sku": sku, "qty": 1}]}, token=token
        )
        if not step(
            "checkout", st == 201 and d.get("status") == "paid", ms, f"{st} {d.get('error')}"
        ):
            return res
        order_id = d["order_id"]
        res["order_id"] = order_id
        t0 = time.monotonic()
        deadline = t0 + THRESHOLDS["journey_timeout_s"]
        status = None
        while time.monotonic() < deadline:
            st, d, _ = self._req("GET", f"/api/orders/{order_id}", token=token)
            status = d.get("status") if st == 200 else f"http {st}"
            if status == "fulfilled":
                break
            time.sleep(0.3)
        if not step(
            "fulfilled",
            status == "fulfilled",
            (time.monotonic() - t0) * 1000,
            f"order status {status}",
        ):
            return res
        t1 = time.monotonic()
        emails = set()
        while time.monotonic() < deadline:
            emails = self._emails_for(order_id)
            if {"order.paid", "order.shipped"} <= emails:
                break
            time.sleep(0.3)
        if not step(
            "notified",
            {"order.paid", "order.shipped"} <= emails,
            (time.monotonic() - t1) * 1000,
            f"emails {sorted(emails)}",
        ):
            return res
        res["ok"] = True
        return res

    def recovery_window(self, seconds: float = 10.0, interval: float = 0.5) -> dict:
        lat, errors, total, not_ready = [], 0, 0, {}
        checkouts = {"ok": 0, "failed": 0, "max_ms": 0.0}
        user = self.rng.choice(self.secret["probe_users"])
        end = time.monotonic() + seconds
        tick = 0
        while time.monotonic() < end:
            try:
                for s in self.stack.services():
                    if not s["ready"]:
                        not_ready[s["name"]] = not_ready.get(s["name"], 0) + 1
            except Exception:
                not_ready["supervisor"] = not_ready.get("supervisor", 0) + 1
            sku = self.rng.choice(self.secret["skus"])
            for path in (
                f"/api/catalog/products?page={1 + tick % 3}",
                f"/api/catalog/products/{sku}",
                "/api/catalog/search?q=" + self.rng.choice(["steel", "lamp", "cotton"]),
            ):
                st, _, ms = self._req("GET", path, timeout=5)
                total += 1
                lat.append(ms)
                if st != 200:
                    errors += 1
            if tick % 4 == 0:
                t0 = time.monotonic()
                st, d, _ = self._req(
                    "POST",
                    "/api/auth/login",
                    {"email": user["email"], "password": user["password"]},
                )
                ok = False
                if st == 200:
                    st, d, _ = self._req(
                        "POST",
                        "/api/checkout",
                        {"items": [{"sku": sku, "qty": 1}]},
                        token=d["token"],
                        timeout=THRESHOLDS["checkout_timeout_s"],
                    )
                    ok = st == 201
                ms = (time.monotonic() - t0) * 1000
                if not ok and "first_failure" not in checkouts:
                    checkouts["first_failure"] = f"{st} {str(d)[:160]}"
                checkouts["ok" if ok else "failed"] += 1
                checkouts["max_ms"] = round(max(checkouts["max_ms"], ms), 1)
            tick += 1
            time.sleep(interval)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None
        err = errors / total if total else 1.0
        ok = (
            not not_ready
            and err <= THRESHOLDS["error_rate"]
            and p95 is not None
            and p95 <= THRESHOLDS["p95_ms"]
            and checkouts["failed"] == 0
            and checkouts["ok"] > 0
        )
        return {
            "ok": ok,
            "seconds": seconds,
            "reads": total,
            "error_rate": round(err, 3),
            "p95_ms": round(p95, 1) if p95 else None,
            "checkouts": checkouts,
            "not_ready": not_ready,
        }

    def invariants(self, grace_s: float = 45.0) -> dict:
        h = self.home
        now = time.time()
        out = {}

        def res(name, ok, detail=None):
            out[name] = {"ok": bool(ok), **({"detail": detail} if detail else {})}

        t_orders = time.time()
        orders = _ro(f"{h}/data/orders.db")
        inv = _ro(f"{h}/data/inventory.db")
        broker = _ro(f"{h}/data/broker.db")
        prov = _ro(f"{h}/data/provider.db")
        users = _ro(f"{h}/data/users.db")
        catalog = _ro(f"{h}/data/catalog.db")
        try:
            dup = prov.execute(
                "SELECT merchant_ref, COUNT(*) n FROM charges GROUP BY "
                "merchant_ref HAVING n > 1 LIMIT 5"
            ).fetchall()
            res("no_duplicate_charges", not dup, [dict(r) for r in dup])

            neg = inv.execute(
                "SELECT sku, on_hand, reserved FROM stock WHERE on_hand < 0 OR "
                "reserved < 0 OR on_hand - reserved < 0 LIMIT 5"
            ).fetchall()
            n_sku = inv.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
            res(
                "stock_never_negative",
                not neg and n_sku == len(self.secret["skus"]),
                [dict(r) for r in neg]
                or (None if n_sku == len(self.secret["skus"]) else f"{n_sku} skus in stock table"),
            )

            committed = {
                r[0]: r[1]
                for r in inv.execute(
                    "SELECT sku, SUM(qty) FROM reservations WHERE status='committed' GROUP BY sku"
                )
            }
            held = {
                r[0]: r[1]
                for r in inv.execute(
                    "SELECT sku, SUM(qty) FROM reservations WHERE status='held' GROUP BY sku"
                )
            }
            bad = []
            for r in inv.execute("SELECT sku, on_hand, reserved FROM stock"):
                exp_on_hand = self.secret["stock_on_hand"] - committed.get(r["sku"], 0)
                if r["on_hand"] != exp_on_hand or r["reserved"] != held.get(r["sku"], 0):
                    bad.append(
                        {
                            "sku": r["sku"],
                            "on_hand": r["on_hand"],
                            "expected": exp_on_hand,
                            "reserved": r["reserved"],
                            "held": held.get(r["sku"], 0),
                        }
                    )
            res("stock_conserved", not bad, bad[:5])

            charges = {}
            created = {}
            for r in prov.execute(
                "SELECT merchant_ref, amount_cents, status, created_at FROM charges"
            ):
                charges.setdefault(r[0], []).append((r[1], r[2]))
                created[r[0]] = min(created.get(r[0], r[3]), r[3])
            bad, unrefunded = [], []
            for o in orders.execute("SELECT id, status, total_cents, updated_at FROM orders"):
                ch = charges.get(o["id"], [])
                if o["status"] in ("paid", "fulfilled"):
                    if len(ch) != 1 or ch[0][0] != o["total_cents"] or ch[0][1] != "succeeded":
                        bad.append({"order": o["id"], "status": o["status"], "charges": ch})
                elif o["status"] == "refunded":
                    if not ch or any(s != "refunded" for _, s in ch):
                        bad.append({"order": o["id"], "status": o["status"], "charges": ch})
                elif (
                    ch and any(s == "succeeded" for _, s in ch) and now - o["updated_at"] > grace_s
                ):
                    unrefunded.append({"order": o["id"], "status": o["status"], "charges": ch})
            known = {r[0] for r in orders.execute("SELECT id FROM orders")}
            orphan = [ref for ref in charges if ref not in known and created[ref] < t_orders - 1.0]
            res(
                "charges_match_orders",
                not bad and not orphan,
                (bad[:5] + [{"orphan_charge_for": o} for o in orphan[:5]]) or None,
            )
            orphan_items = orders.execute(
                "SELECT COUNT(*) FROM order_items i LEFT JOIN orders o ON o.id=i.order_id "
                "WHERE o.id IS NULL"
            ).fetchone()[0]
            old_res = {
                r[0]
                for r in inv.execute(
                    "SELECT DISTINCT order_id FROM reservations WHERE created_at < ?",
                    (t_orders - 1.0,),
                )
            }
            orphan_res = sorted(old_res - known)
            res(
                "order_records_intact",
                not orphan_items and not orphan_res,
                None
                if not orphan_items and not orphan_res
                else {
                    "items_without_order": orphan_items,
                    "reservations_without_order": orphan_res[:5],
                    "count": len(orphan_res),
                },
            )
            res("charged_failed_orders_refunded", not unrefunded, unrefunded[:5] or None)

            stuck = orders.execute(
                "SELECT id, status, updated_at FROM orders WHERE "
                "status='paid' AND updated_at < ? LIMIT 5",
                (now - grace_s,),
            ).fetchall()
            res("paid_orders_fulfilled_or_refunded", not stuck, [dict(r) for r in stuck])

            committed_by_order = {}
            for r in inv.execute(
                "SELECT order_id, sku, SUM(qty) FROM reservations WHERE status='committed' "
                "GROUP BY order_id, sku"
            ):
                committed_by_order[(r[0], r[1])] = r[2]
            uncommitted = []
            for r in orders.execute(
                "SELECT i.order_id, i.sku, i.qty FROM order_items i JOIN orders o ON "
                "o.id=i.order_id WHERE o.status='fulfilled' AND o.updated_at < ?",
                (now - grace_s,),
            ):
                if committed_by_order.get((r[0], r[1]), 0) != r[2]:
                    uncommitted.append({"order": r[0], "sku": r[1], "qty": r[2]})
            res("shipped_orders_stock_committed", not uncommitted, uncommitted[:5] or None)

            pending = orders.execute(
                "SELECT COUNT(*) FROM outbox WHERE published_at IS NULL AND created_at < ?",
                (now - grace_s,),
            ).fetchone()[0]
            n_outbox = {
                r[0]: r[1]
                for r in orders.execute(
                    "SELECT topic, COUNT(*) FROM outbox WHERE published_at IS NOT NULL "
                    "GROUP BY topic"
                )
            }
            n_msgs = {
                r[0]: r[1]
                for r in broker.execute("SELECT topic, COUNT(*) FROM messages GROUP BY topic")
            }
            order_ids = {r[0] for r in orders.execute("SELECT id FROM orders")}
            changes = {}
            for r in broker.execute("SELECT queue, topic, action, ts FROM binding_log ORDER BY id"):
                changes.setdefault(r["topic"], []).append((r["ts"], r["queue"], r["action"]))

            def bound_at(topic, ts):
                qs = set()
                for t, q, action in changes.get(topic, []):
                    if t > ts:
                        break
                    (qs.add if action == "bind" else qs.discard)(q)
                return qs

            valid = {}
            for r in broker.execute("SELECT id, topic, payload, published_at FROM messages"):
                try:
                    oid = json.loads(r["payload"]).get("order_id")
                except (ValueError, AttributeError):
                    oid = None
                if oid in order_ids and r["topic"] in EXPECTED_QUEUES:
                    valid[r["id"]] = bound_at(r["topic"], r["published_at"])
            entries = {}
            for r in broker.execute("SELECT queue, msg_id, state, updated_at FROM entries"):
                entries[(r["queue"], r["msg_id"])] = (r["state"], r["updated_at"])
            missing, dead, stale = [], [], []
            for (q, mid), (state, updated) in entries.items():
                if mid not in valid:
                    continue
                if state == "dead":
                    dead.append({"queue": q, "msg_id": mid})
                elif state != "acked" and now - updated > grace_s:
                    stale.append({"queue": q, "msg_id": mid, "state": state})
            for mid, queues in valid.items():
                for q in queues:
                    if (q, mid) not in entries:
                        missing.append({"queue": q, "msg_id": mid})
            short = [t for t, n in n_outbox.items() if n_msgs.get(t, 0) < n]
            res(
                "no_lost_messages",
                not missing and not short,
                None
                if not missing and not short
                else {
                    "missing_queue_entries": missing[:5],
                    "missing_count": len(missing),
                    "outbox_published_but_not_in_broker": short,
                },
            )
            undelivered = pending or dead or stale
            res(
                "events_delivered",
                not undelivered,
                None
                if not undelivered
                else {
                    "outbox_unpublished": pending,
                    "dead_lettered_valid": dead[:5],
                    "dead_count": len(dead),
                    "stale_unacked": stale[:5],
                    "stale_count": len(stale),
                },
            )

            emails = {}
            path = f"{h}/data/outbox/emails.jsonl"
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                        except ValueError:
                            continue
                        emails.setdefault(e.get("order_id"), set()).add(e.get("template"))
            acked_notify = set()
            for r in broker.execute(
                "SELECT m.payload, m.topic FROM entries e JOIN messages m ON m.id=e.msg_id "
                "WHERE e.queue='notify.events' AND e.state='acked'"
            ):
                try:
                    oid = json.loads(r[0]).get("order_id")
                except (ValueError, AttributeError):
                    oid = None
                if oid in order_ids:
                    acked_notify.add((oid, r[1]))
            missing_emails = sorted(
                f"{o}:{t}" for o, t in acked_notify if t not in emails.get(o, set())
            )
            res("acked_events_have_emails", not missing_emails, missing_emails[:5] or None)
            unnotified = [
                r[0]
                for r in orders.execute(
                    "SELECT id FROM orders WHERE status='fulfilled' AND updated_at < ?",
                    (now - grace_s,),
                )
                if "order.shipped" not in emails.get(r[0], set())
            ]
            res("customers_notified", not unnotified, unnotified[:5] or None)

            hashes = {
                "users.users": table_hash(users, "users"),
                "catalog.products": table_hash(catalog, "products"),
            }
            changed = [k for k, v in hashes.items() if v != self.secret["seed_hashes"][k]]
            res("seed_data_intact", not changed, changed or None)
        finally:
            for c in (prov, orders, inv, broker, users, catalog):
                c.close()
        return out

    def run(self, journeys: int = 3, window_s: float = 10.0) -> dict:
        t0 = time.time()
        js = [self.journey() for _ in range(journeys)]
        window = self.recovery_window(window_s)
        inv = self.invariants()
        passed = all(j["ok"] for j in js) and window["ok"] and all(v["ok"] for v in inv.values())
        return {
            "passed": passed,
            "journeys": js,
            "recovery_window": window,
            "invariants": inv,
            "probe_seconds": round(time.time() - t0, 1),
        }


def ready_all(stack) -> bool:
    try:
        st, body = http_json(stack.ports["supervisor"], "GET", "/services")
        return st == 200 and all(s["ready"] for s in body["services"])
    except Exception:
        return False
