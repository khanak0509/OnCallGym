"""Fault families. Spec numbers refer to ONCALLGYM_SPEC.md section 7.1."""

import json
import os
import time

from ..stack import http_json
from .base import (
    Fault,
    config_path,
    edit_service_config,
    edit_yaml,
    effective_config,
    effective_flags,
    restart_and_wait,
)


def _port_of(url: str) -> int:
    return int(url.rsplit(":", 1)[1].split("/")[0])


def _with_port(url: str, port: int) -> str:
    head, _ = url.rsplit(":", 1)
    return f"{head}:{port}"


class EdgeWrongUpstreamPort(Fault):
    """Spec 2. A routing change left one edge upstream pointing at the wrong port."""

    family = "edge_wrong_upstream_port"
    spec_number = 2
    category = "config"
    tier = "easy"

    def choose_params(self, rng):
        upstream = rng.choice(["catalog", "orders", "auth"])
        # a port inside this stack's own block: either unused (connection refused) or
        # another of our services (wrong service answers, 404s)
        wrong_offset = rng.choice([16, 17, 18, 19, "neighbour"])
        return {"upstream": upstream, "wrong_offset": wrong_offset}

    @property
    def component(self):
        return "edge"

    def describe(self):
        return (
            f"etc/edge.yaml upstreams.{self.params['upstream']}.url points at the wrong "
            f"port, so every {self.params['upstream']} route through the edge fails"
        )

    def wrong_port(self, stack):
        base = stack.ports["supervisor"]
        if self.params["wrong_offset"] == "neighbour":
            neighbours = {"catalog": "inventory", "orders": "payments", "auth": "catalog"}
            return stack.ports[neighbours[self.params["upstream"]]]
        return base + self.params["wrong_offset"]

    def alert(self):
        what = {
            "catalog": "Product pages and search",
            "orders": "Checkout and order pages",
            "auth": "Login",
        }[self.params["upstream"]]
        return f"{what} failing for most customers (5xx/404 at the edge) since a few minutes ago."

    def configure(self, stack):
        up = self.params["upstream"]
        port = self.wrong_port(stack)

        def change(doc):
            doc["upstreams"][up]["url"] = _with_port(doc["upstreams"][up]["url"], port)

        edit_service_config(stack, "edge", change)

    def root_cause_fixed(self, stack):
        up = self.params["upstream"]
        url = effective_config(stack, "edge")["upstreams"][up]["url"]
        return _port_of(url) == stack.ports[up]

    def solution_steps(self, stack):
        up, right = self.params["upstream"], stack.ports[self.params["upstream"]]
        wrong = self.wrong_port(stack)
        return [
            (
                "logs",
                {"service": "edge", "tail": 5, "grep": "upstream request failed"},
                "Customers see errors at the edge. Check what the edge logs.",
            ),
            (
                "topology",
                {},
                f"The edge cannot reach {up}. Compare with the port {up} really listens on.",
            ),
            (
                "exec",
                {"command": "grep -n 'url:' etc/edge.yaml"},
                "Look at the upstream URLs in the edge config.",
            ),
            (
                "exec",
                {
                    "command": f"sed -i '/^  {up}:/,/url:/ s#127.0.0.1:{wrong}#127.0.0.1:"
                    f"{right}#' etc/edge.yaml && grep -n 'url:' etc/edge.yaml"
                },
                f"The {up} upstream points at port {wrong} but {up} listens on {right}. Fix it.",
            ),
            ("restart", {"service": "edge"}, "Restart the edge so it reloads its config."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": "edge",
                    "summary": f"etc/edge.yaml upstream {up} pointed at port "
                    f"{wrong} instead of {right}; fixed and restarted "
                    "edge.",
                },
                "Requests work again. Submit.",
            ),
        ]

    def oracle(self, stack):
        up = self.params["upstream"]

        def fix(doc):
            doc["upstreams"][up]["url"] = _with_port(doc["upstreams"][up]["url"], stack.ports[up])

        edit_service_config(stack, "edge", fix)
        restart_and_wait(stack, "edge")


class OrdersPaymentsTimeoutInversion(Fault):
    """Spec 7. The payment provider is degraded (slow but working). payments->provider
    has a generous timeout, but orders->payments was tuned much lower, so orders gives up
    while payments is still waiting on the provider: checkouts fail with 504 at the edge
    even though many charges succeed (the scheduler later refunds those)."""

    family = "orders_payments_timeout_inversion"
    spec_number = 7
    category = "resilience_config"
    tier = "medium"
    settle_s = 4.0

    def choose_params(self, rng):
        base = rng.choice([700, 900, 1100])
        return {
            "provider_base_ms": base,
            "provider_jitter_ms": 300,
            "payments_timeout_ms": 2500,
            # always below the provider's minimum latency, so every checkout times out
            "orders_timeout_ms": base - rng.choice([200, 300, 400]),
        }

    @property
    def component(self):
        return "orders"

    def describe(self):
        p = self.params
        return (
            f"provider latency degraded to ~{p['provider_base_ms']}ms; orders->payments "
            f"timeout {p['orders_timeout_ms']}ms is below payments->provider timeout "
            f"{p['payments_timeout_ms']}ms, so checkouts time out in orders"
        )

    def alert(self):
        return (
            "Checkout failure rate above 50% (504 at the edge). Some customers report being "
            "charged for orders that show as failed."
        )

    def configure(self, stack):
        p = self.params

        def provider(doc):
            doc["latency_ms"] = {"base": p["provider_base_ms"], "jitter": p["provider_jitter_ms"]}

        def payments(doc):
            doc["upstreams"]["provider"]["timeout_ms"] = p["payments_timeout_ms"]

        def orders(doc):
            doc["upstreams"]["payments"]["timeout_ms"] = p["orders_timeout_ms"]

        edit_service_config(stack, "provider", provider)
        edit_service_config(stack, "payments", payments)
        edit_service_config(stack, "orders", orders)

    def root_cause_fixed(self, stack):
        orders = effective_config(stack, "orders")["upstreams"]["payments"]
        pay = effective_config(stack, "payments")["upstreams"]["provider"]
        # the outer call must wait longer than one inner attempt
        return int(orders["timeout_ms"]) > int(pay["timeout_ms"])

    def solution_steps(self, stack):
        p = self.params
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "checkout failed"},
                "Checkouts fail with 504. See why orders fails them.",
            ),
            (
                "logs",
                {"service": "payments", "tail": 5, "grep": "charge"},
                "Orders times out calling payments. Does payments itself succeed?",
            ),
            (
                "exec",
                {
                    "command": "grep -n -A3 '^  payments:' etc/orders.yaml; "
                    "grep -n -A3 '^  provider:' etc/payments.yaml"
                },
                "Payments succeeds slowly. Compare the two timeouts.",
            ),
            (
                "exec",
                {
                    "command": "sed -i '/^  payments:/,/timeout_ms/ s/timeout_ms: [0-9]*/"
                    "timeout_ms: 6000/' etc/orders.yaml && "
                    "grep -n -A3 '^  payments:' etc/orders.yaml"
                },
                f"orders gives up after {p['orders_timeout_ms']} ms while payments waits up to "
                f"{p['payments_timeout_ms']} ms for the slow provider. Raise orders' timeout.",
            ),
            ("restart", {"service": "orders"}, "Restart orders to load the new timeout."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "resilience_config",
                    "component": "orders",
                    "summary": "orders->payments timeout was below payments->"
                    "provider timeout while the provider is slow; "
                    "raised it to 6000 ms.",
                },
                "Checkouts complete again. Submit.",
            ),
        ]

    def oracle(self, stack):
        pay = effective_config(stack, "payments")["upstreams"]["provider"]
        budget = (int(pay.get("retries", 0)) + 1) * int(pay["timeout_ms"]) + 1000

        def fix(doc):
            doc["upstreams"]["payments"]["timeout_ms"] = budget

        edit_service_config(stack, "orders", fix)
        restart_and_wait(stack, "orders")


class CatalogLeakFlag(Fault):
    """Spec 12. A feature flag turns on catalog's new recommendations code, which leaks
    memory on every request; catalog hits its memory limit, is OOM-killed and restarted in
    a loop, so browsing fails intermittently. Restarting does not help (the flag stays on)."""

    family = "catalog_leak_flag"
    spec_number = 12
    category = "resource_exhaustion"
    tier = "medium"
    breaks = ("window",)  # intermittent: a single journey may slip through between OOM kills
    settle_s = 25.0

    # the leak is in catalog, the switch that enables it is a flag served by flags
    also_accepted = ("flags",)

    def choose_params(self, rng):
        return {"flag": "recommendations_v2", "leak_kb": rng.choice([3072, 4096, 6144])}

    @property
    def component(self):
        return "catalog"

    def describe(self):
        return (
            f"feature flag {self.params['flag']} enables catalog code that leaks memory per "
            "request; catalog is OOM-killed and restarted repeatedly"
        )

    def alert(self):
        return "Intermittent errors and timeouts when browsing products; error rate is flapping."

    def configure(self, stack):
        p = self.params

        def flags(doc):
            svc = doc.setdefault("services", {}).setdefault("catalog", {})
            svc[p["flag"]] = True
            svc["recommendations_cache_kb"] = p["leak_kb"]

        edit_yaml(config_path(stack, "feature-flags"), flags)

    def root_cause_fixed(self, stack):
        flags = effective_flags(stack)
        merged = {
            **(flags.get("global") or {}),
            **((flags.get("services") or {}).get("catalog") or {}),
        }
        return not merged.get(self.params["flag"], False)

    def solution_steps(self, stack):
        flag = self.params["flag"]
        return [
            ("list_services", {}, "Browsing flaps. Check service health and restarts."),
            (
                "logs",
                {"service": "supervisor", "tail": 5, "grep": "Out of memory"},
                "catalog keeps restarting. Was it killed for memory?",
            ),
            (
                "exec",
                {"command": "cat etc/feature-flags.yaml"},
                "catalog is OOM-killed repeatedly. Check what changed in its feature flags.",
            ),
            (
                "exec",
                {
                    "command": f"sed -i 's/{flag}: true/{flag}: false/' "
                    "etc/feature-flags.yaml && cat etc/feature-flags.yaml"
                },
                f"{flag} is on; it enables new catalog code. Turn it off.",
            ),
            ("restart", {"service": "catalog"}, "Restart catalog to free the leaked memory."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "resource_exhaustion",
                    "component": "catalog",
                    "summary": f"feature flag {flag} enables leaking catalog "
                    "code; catalog was OOM-killed in a loop. "
                    "Disabled the flag.",
                },
                "catalog is stable. Submit.",
            ),
        ]

    def oracle(self, stack):
        def flags(doc):
            doc["services"]["catalog"][self.params["flag"]] = False

        edit_yaml(config_path(stack, "feature-flags"), flags)
        restart_and_wait(stack, "catalog")


class JwtKeyRotationMismatch(Fault):
    """Spec 14. Auth's signing key was rotated (new kid) but orders still only trusts the
    old key, so every checkout is rejected with 401 although login works."""

    family = "jwt_key_rotation_mismatch"
    spec_number = 14
    category = "auth"
    tier = "easy"

    # rolling auth back to the old key is also a valid fix
    also_accepted = ("auth",)

    def choose_params(self, rng):
        return {
            "new_kid": rng.choice(["k-2026-11", "k-2026-12", "k-2027-01"]),
            "key": rng.randbytes(32).hex(),
        }

    @property
    def component(self):
        return "orders"

    def describe(self):
        return (
            f"auth signs tokens with new key {self.params['new_kid']}; orders "
            "auth.trusted_keys was not updated, so token verification fails at checkout"
        )

    def key_file(self):
        return f"etc/keys/jwt-{self.params['new_kid']}.key"

    def alert(self):
        return "Customers can log in but every checkout is rejected (401)."

    def configure(self, stack):
        path = os.path.join(stack.home, self.key_file())
        with open(path, "w") as f:
            f.write(self.params["key"] + "\n")

        def auth(doc):
            doc["signing_key"] = {"kid": self.params["new_kid"], "file": self.key_file()}

        edit_service_config(stack, "auth", auth)

    def root_cause_fixed(self, stack):
        auth_kid = effective_config(stack, "auth")["signing_key"]["kid"]
        trusted = effective_config(stack, "orders")["auth"]["trusted_keys"]
        return any(k.get("kid") == auth_kid for k in trusted)

    def solution_steps(self, stack):
        kid = self.params["new_kid"]
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "token verification failed"},
                "Checkouts get 401. See why orders rejects tokens.",
            ),
            (
                "exec",
                {
                    "command": "grep -n -A2 signing_key etc/auth.yaml; "
                    "grep -n -A4 trusted_keys etc/orders.yaml; ls etc/keys"
                },
                "Unknown signing key. Compare auth's signing key with what orders trusts.",
            ),
            (
                "exec",
                {
                    "command": "sed -i '/file: etc\\/keys\\/jwt-k-2026-10.key/a\\  - kid: "
                    f"{kid}\\n    file: etc/keys/jwt-{kid}.key' etc/orders.yaml && "
                    "grep -n -A6 trusted_keys etc/orders.yaml"
                },
                f"auth now signs with {kid}; orders only trusts the old key. Add the new one.",
            ),
            ("restart", {"service": "orders"}, "Restart orders to load the trusted keys."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "auth",
                    "component": "orders",
                    "summary": f"auth rotated to {kid} but orders did not trust "
                    "it; added the new key to orders.",
                },
                "Checkouts work. Submit.",
            ),
        ]

    def oracle(self, stack):
        def orders(doc):
            doc["auth"]["trusted_keys"].append(
                {"kid": self.params["new_kid"], "file": self.key_file()}
            )

        edit_service_config(stack, "orders", orders)
        restart_and_wait(stack, "orders")


class PoisonMessage(Fault):
    """Spec 18. fulfillment's queue is strictly ordered (shipments go out in payment order)
    and retries failed messages forever. A malformed order.paid event (old producer schema)
    sits at the head of the queue and fails on every delivery, so no order is shipped."""

    family = "poison_message"
    spec_number = 18
    category = "data"
    tier = "medium"
    settle_s = 4.0

    # the message sits in the broker queue that fulfillment consumes
    also_accepted = ("broker",)

    def choose_params(self, rng):
        shape = rng.choice(["camel_case", "missing_order_id", "null_order_id"])
        ref = "ord_" + rng.randbytes(6).hex()
        payloads = {
            "camel_case": {"orderId": ref, "userId": 7, "totalCents": 4599, "schema": 1},
            "missing_order_id": {"user_id": 7, "total_cents": 4599, "items": []},
            "null_order_id": {"order_id": None, "user_id": 7, "total_cents": 4599},
        }
        return {"shape": shape, "payload": payloads[shape]}

    @property
    def component(self):
        return "fulfillment"

    def describe(self):
        return (
            "malformed order.paid message at the head of the ordered queue "
            "fulfillment.order-paid fails on every delivery and blocks all shipments"
        )

    def alert(self):
        return "No orders have shipped for several minutes; paid orders are piling up."

    def inject(self, stack):
        st, body = stack.call(
            "broker",
            "POST",
            "/publish",
            {
                "topic": "order.paid",
                "payload": self.params["payload"],
                "request_id": "legacy-import",
            },
        )
        assert st == 200, body
        self.poison_id = body["id"]

    def root_cause_fixed(self, stack):
        import sqlite3

        c = sqlite3.connect(f"file:{stack.home}/data/broker.db?mode=ro", uri=True, timeout=5)
        try:
            row = c.execute(
                "SELECT e.state FROM entries e JOIN messages m ON m.id=e.msg_id WHERE "
                "e.queue='fulfillment.order-paid' AND m.request_id='legacy-import'"
            ).fetchone()
        finally:
            c.close()
        return row is None or row[0] not in ("ready", "inflight")

    def solution_steps(self, stack):
        b = stack.ports["broker"]
        q = "fulfillment.order-paid"
        return [
            (
                "logs",
                {"service": "fulfillment", "tail": 5, "grep": "failed"},
                "Nothing ships. See what fulfillment logs.",
            ),
            (
                "exec",
                {"command": f"curl -s 'http://127.0.0.1:{b}/queues/{q}/messages?limit=3'"},
                "One message fails on every delivery. Look at the head of its queue.",
            ),
            (
                "exec",
                {
                    "command": f"curl -s -X POST http://127.0.0.1:{b}/queues/{q}/messages/"
                    f'{self.poison_id}/dead-letter -d \'{{"reason": "poison '
                    "message, malformed order.paid\"}'"
                },
                "The head message is malformed and blocks the ordered queue. Dead-letter it.",
            ),
            (
                "logs",
                {"service": "fulfillment", "tail": 3, "grep": "shipment created"},
                "Check that shipments resume.",
            ),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "data",
                    "component": "fulfillment",
                    "summary": "a malformed order.paid message blocked the "
                    "ordered fulfillment queue; moved it to the DLQ.",
                },
                "Shipments flow again. Submit.",
            ),
        ]

    def oracle(self, stack):
        q = "fulfillment.order-paid"
        st, body = http_json(stack.ports["broker"], "GET", f"/queues/{q}/messages?limit=5")
        head = body["messages"][0]
        st, body = http_json(
            stack.ports["broker"],
            "POST",
            f"/queues/{q}/messages/{head['id']}/dead-letter",
            {"reason": "poison message, moved by on-call"},
        )
        assert st == 200, body


class WrongDatabasePath(Fault):
    """Spec 1. A config change points one service at a database file that does not exist;
    SQLite silently creates an empty one, so every query fails with "no such table"."""

    family = "wrong_database_path"
    spec_number = 1
    category = "config"
    tier = "easy"

    def choose_params(self, rng):
        service = rng.choice(["inventory", "payments", "catalog"])
        db = {"inventory": "inventory", "payments": "payments", "catalog": "catalog"}[service]
        wrong = rng.choice([f"data/{db}-new.db", f"data/{db}s.db", f"/var/lib/northwind/{db}.db"])
        return {"service": service, "db": db, "wrong": wrong}

    @property
    def component(self):
        return self.params["service"]

    def describe(self):
        p = self.params
        return f"etc/{p['service']}.yaml db.path is {p['wrong']} instead of data/{p['db']}.db"

    def alert(self):
        what = {"inventory": "Checkout", "payments": "Checkout", "catalog": "Product browsing"}
        return f"{what[self.params['service']]} failing with 5xx errors for all customers."

    def configure(self, stack):
        wrong = self.params["wrong"]
        if wrong.startswith("/var/lib"):
            # absolute path into a directory that does not exist inside the episode
            wrong = "data/missing-volume/" + os.path.basename(wrong)
            self.params["wrong"] = wrong
        edit_service_config(stack, self.params["service"], lambda d: d["db"].update(path=wrong))

    def root_cause_fixed(self, stack):
        path = effective_config(stack, self.params["service"])["db"]["path"]
        full = path if os.path.isabs(path) else os.path.join(stack.home, path)
        return os.path.realpath(full) == os.path.realpath(
            os.path.join(stack.home, f"data/{self.params['db']}.db")
        )

    def solution_steps(self, stack):
        p = self.params
        svc, db = p["service"], p["db"]
        return [
            ("list_services", {}, "Customers get 5xx. Check which services are healthy."),
            (
                "logs",
                {"service": svc, "tail": 5, "grep": "error"},
                f"{svc} is not ready. See its errors.",
            ),
            (
                "exec",
                {"command": f"grep -n -A3 '^db:' etc/{svc}.yaml; ls data/"},
                f"{svc} finds no tables. Check which database file it opens.",
            ),
            (
                "exec",
                {
                    "command": f"sed -i 's#path: {p['wrong']}#path: data/{db}.db#' "
                    f"etc/{svc}.yaml && grep -n -A3 '^db:' etc/{svc}.yaml"
                },
                f"It points at {p['wrong']}; the real database is data/{db}.db. Fix the path.",
            ),
            ("restart", {"service": svc}, f"Restart {svc}."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": svc,
                    "summary": f"etc/{svc}.yaml db.path pointed at {p['wrong']}; "
                    f"set it back to data/{db}.db.",
                },
                f"{svc} is ready. Submit.",
            ),
        ]

    def oracle(self, stack):
        db = self.params["db"]
        edit_service_config(
            stack, self.params["service"], lambda d: d["db"].update(path=f"data/{db}.db")
        )
        restart_and_wait(stack, self.params["service"])


class EdgeRateLimitTooLow(Fault):
    """Spec 3. A rate-limit change at the edge set the per-client limit to almost zero;
    most requests get 429."""

    family = "edge_rate_limit_too_low"
    spec_number = 3
    category = "config"
    tier = "easy"

    def choose_params(self, rng):
        return {"rps": rng.choice([0.05, 0.1, 0.2]), "burst": 1}

    @property
    def component(self):
        return "edge"

    def describe(self):
        return f"etc/edge.yaml rate_limit.rps={self.params['rps']} burst=1 (normally 20/40)"

    def alert(self):
        return "Customers getting 'Too Many Requests' (429) on most pages."

    def configure(self, stack):
        edit_service_config(
            stack,
            "edge",
            lambda d: d.update(
                rate_limit={"rps": self.params["rps"], "burst": self.params["burst"]}
            ),
        )

    def root_cause_fixed(self, stack):
        rl = effective_config(stack, "edge").get("rate_limit") or {}
        return float(rl.get("rps", 40)) >= 5 and float(rl.get("burst", 80)) >= 5

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "edge", "tail": 5, "grep": "rate limit"},
                "Customers get 429. Check the edge.",
            ),
            (
                "exec",
                {"command": "grep -n -A2 rate_limit etc/edge.yaml"},
                "The edge rate-limits nearly everything. Check the limit.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/^  rps: .*/  rps: 20/; s/^  burst: .*/  burst: 40/' "
                    "etc/edge.yaml && grep -n -A2 rate_limit etc/edge.yaml"
                },
                f"The limit is {self.params['rps']} requests/s per client. Restore 20/s, burst 40.",
            ),
            ("restart", {"service": "edge"}, "Restart the edge."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": "edge",
                    "summary": "edge rate_limit was set near zero; restored rps 20 burst 40.",
                },
                "Requests pass again. Submit.",
            ),
        ]

    def oracle(self, stack):
        edit_service_config(stack, "edge", lambda d: d.update(rate_limit={"rps": 20, "burst": 40}))
        restart_and_wait(stack, "edge")


class FulfillmentWrongTopic(Fault):
    """Spec 4. fulfillment was redeployed subscribed to a misspelled topic, so its queue no
    longer receives order.paid events and nothing ships. Fixing the topic is not enough: the
    events published meanwhile must be backfilled into the queue."""

    family = "fulfillment_wrong_topic"
    spec_number = 4
    category = "config"
    tier = "medium"
    settle_s = 5.0

    def choose_params(self, rng):
        return {"topic": rng.choice(["order.payed", "orders.paid", "order_paid"])}

    @property
    def component(self):
        return "fulfillment"

    def describe(self):
        return (
            f"etc/fulfillment.yaml consumer.topic is {self.params['topic']} instead of "
            "order.paid; events published meanwhile are missing from the queue"
        )

    def alert(self):
        return "Orders are paid but none have shipped for several minutes."

    def configure(self, stack):
        edit_service_config(
            stack, "fulfillment", lambda d: d["consumer"].update(topic=self.params["topic"])
        )

    def root_cause_fixed(self, stack):
        consumer = effective_config(stack, "fulfillment")["consumer"]
        return consumer.get("topic") == "order.paid"

    def solution_steps(self, stack):
        b = stack.ports["broker"]
        return [
            (
                "logs",
                {"service": "fulfillment", "tail": 5, "grep": "subscribed"},
                "Paid orders are not shipped. What does fulfillment consume?",
            ),
            (
                "exec",
                {"command": f"curl -s http://127.0.0.1:{b}/queues"},
                "Check which topics each queue is bound to.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/topic: .*/topic: order.paid/' "
                    "etc/fulfillment.yaml && grep -n -A3 consumer "
                    "etc/fulfillment.yaml"
                },
                f"fulfillment subscribes to {self.params['topic']}, but orders publishes "
                "order.paid. Fix the topic.",
            ),
            ("restart", {"service": "fulfillment"}, "Restart fulfillment to resubscribe."),
            (
                "exec",
                {
                    "command": f"curl -s -X POST http://127.0.0.1:{b}/queues/"
                    "fulfillment.order-paid/backfill -d "
                    '\'{"topic": "order.paid", "since_s": 3600}\''
                },
                "Events published meanwhile never reached the queue. Backfill them.",
            ),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": "fulfillment",
                    "summary": "fulfillment consumer topic was misspelled; set "
                    "order.paid and backfilled missed events.",
                },
                "Orders ship again. Submit.",
            ),
        ]

    def oracle(self, stack):
        edit_service_config(
            stack, "fulfillment", lambda d: d["consumer"].update(topic="order.paid")
        )
        restart_and_wait(stack, "fulfillment")
        st, body = http_json(
            stack.ports["broker"],
            "POST",
            "/queues/fulfillment.order-paid/backfill",
            {"topic": "order.paid", "since_s": 3600},
        )
        assert st == 200, body


class PaymentsBreakerStuckOpen(Fault):
    """Spec 9. The provider has elevated (but retryable) errors. payments' circuit breaker
    was set to open on a single failure and stay open for minutes, so it trips almost
    immediately and checkout fails fast with 503 although the provider mostly works."""

    family = "payments_breaker_stuck_open"
    spec_number = 9
    category = "resilience_config"
    tier = "medium"
    breaks = ("window",)  # checkouts fail once the first provider error trips the breaker
    settle_s = 15.0

    def choose_params(self, rng):
        return {
            "error_rate": rng.choice([0.08, 0.1, 0.12]),
            "reset_timeout_s": rng.choice([300, 600, 900]),
        }

    @property
    def component(self):
        return "payments"

    def describe(self):
        return (
            f"etc/payments.yaml breaker failure_threshold=1, reset_timeout_s="
            f"{self.params['reset_timeout_s']} with provider error rate "
            f"{self.params['error_rate']}: breaker stays open"
        )

    def alert(self):
        return "Checkout failing for most customers (502/503); payment provider status is green."

    def configure(self, stack):
        p = self.params
        edit_service_config(stack, "provider", lambda d: d.update(error_rate=p["error_rate"]))

        def payments(doc):
            doc["upstreams"]["provider"]["breaker"] = {
                "failure_threshold": 1,
                "window_s": 60,
                "reset_timeout_s": p["reset_timeout_s"],
            }

        edit_service_config(stack, "payments", payments)

    def inject(self, stack):
        """A short error blip trips the breaker: run customer checkouts until one fails fast
        with breaker_open (with threshold 1 the first provider error does it). Without this
        the fault only shows if background traffic happens to hit an error in time."""
        from ..probes import Probe

        p = Probe(stack)
        for _ in range(80):
            user = p.rng.choice(p.secret["probe_users"])
            st, d, _ = p._req(
                "POST", "/api/auth/login", {"email": user["email"], "password": user["password"]}
            )
            if st != 200:
                time.sleep(0.5)
                continue
            st, d, _ = p._req(
                "POST",
                "/api/checkout",
                {"items": [{"sku": p.rng.choice(p.secret["skus"]), "qty": 1}]},
                token=d["token"],
                timeout=15,
            )
            if "breaker_open" in json.dumps(d):
                return
        raise RuntimeError("payments breaker did not open")

    def root_cause_fixed(self, stack):
        b = effective_config(stack, "payments")["upstreams"]["provider"].get("breaker")
        if not b:
            return True  # no breaker at all is a (blunt) valid choice
        return int(b.get("failure_threshold", 5)) >= 3 and float(b.get("reset_timeout_s", 5)) <= 60

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "checkout failed"},
                "Checkouts fail. See why.",
            ),
            (
                "logs",
                {"service": "payments", "tail": 5, "grep": "breaker"},
                "payments reports the provider breaker as open. Check when it opens.",
            ),
            (
                "exec",
                {"command": "grep -n -A4 breaker etc/payments.yaml"},
                "The provider mostly works, yet the breaker stays open. Check its settings.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/failure_threshold: 1$/failure_threshold: 8/; "
                    "s/reset_timeout_s: [0-9]*/reset_timeout_s: 5/; "
                    "s/window_s: 60$/window_s: 10/' etc/payments.yaml && "
                    "grep -n -A4 breaker etc/payments.yaml"
                },
                "It opens on a single failure and stays open for minutes. Restore sane values.",
            ),
            ("restart", {"service": "payments"}, "Restart payments."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "resilience_config",
                    "component": "payments",
                    "summary": "payments circuit breaker opened on 1 failure "
                    "with a long reset; restored threshold 8, "
                    "reset 5 s.",
                },
                "Charges go through again. Submit.",
            ),
        ]

    def oracle(self, stack):
        def payments(doc):
            doc["upstreams"]["provider"]["breaker"] = {
                "failure_threshold": 8,
                "window_s": 10,
                "reset_timeout_s": 5,
            }

        edit_service_config(stack, "payments", payments)
        restart_and_wait(stack, "payments")


class NotifyDeadLetters(Fault):
    """Spec 19. The email relay is flaky by nature and notify relies on redelivery. The
    notify.events queue was changed to max_redeliveries 0, so every relay hiccup sends the
    event straight to the dead-letter queue and those customers never get their email."""

    family = "notify_dead_letters"
    spec_number = 19
    category = "data"
    tier = "medium"
    breaks = ("liveness",)  # dead-lettered emails for real orders
    settle_s = 15.0

    # max_redeliveries is a retry setting as much as a queue setting
    also_accepted_categories = ("resilience_config",)

    def choose_params(self, rng):
        return {"max_redeliveries": 0}

    @property
    def component(self):
        return "broker"

    def describe(self):
        return (
            "etc/broker.yaml queues.notify.events.max_redeliveries=0: transient relay "
            "errors dead-letter emails"
        )

    def alert(self):
        return "Customers report missing order confirmation and shipping emails."

    def configure(self, stack):
        def broker(doc):
            doc["queues"]["notify.events"] = {"max_redeliveries": 0}

        edit_service_config(stack, "broker", broker)

    def root_cause_fixed(self, stack):
        q = (effective_config(stack, "broker").get("queues") or {}).get("notify.events")
        return int((q or {}).get("max_redeliveries", 5)) != 0

    def solution_steps(self, stack):
        b = stack.ports["broker"]
        return [
            (
                "logs",
                {"service": "broker", "tail": 5, "grep": "dead-letter"},
                "Emails are missing. Check whether notify's events are being dropped.",
            ),
            (
                "exec",
                {
                    "command": f"curl -s http://127.0.0.1:{b}/queues; grep -n -A8 '^queues:' "
                    "etc/broker.yaml"
                },
                "Events go straight to the DLQ. Check the queue's redelivery setting.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/max_redeliveries: 0$/max_redeliveries: 5/' "
                    "etc/broker.yaml && grep -n -A8 '^queues:' etc/broker.yaml"
                },
                "notify.events allows 0 redeliveries, so every relay hiccup dead-letters. "
                "Allow retries again.",
            ),
            ("restart", {"service": "broker"}, "Restart the broker."),
            (
                "exec",
                {
                    "command": f"curl -s -X POST http://127.0.0.1:{b}/queues/"
                    "notify.events/dlq/replay"
                },
                "Replay the dead-lettered emails so customers get them.",
            ),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "data",
                    "component": "broker",
                    "summary": "notify.events max_redeliveries was 0; restored "
                    "5 and replayed the DLQ.",
                },
                "Emails are delivered. Submit.",
            ),
        ]

    def oracle(self, stack):
        def broker(doc):
            doc["queues"]["notify.events"] = {}

        edit_service_config(stack, "broker", broker)
        restart_and_wait(stack, "broker")
        st, body = http_json(stack.ports["broker"], "POST", "/queues/notify.events/dlq/replay")
        assert st == 200, body


class SchedulerLockHold(Fault):
    """Spec 20. The reservation-expiry job was changed to hold its write transaction on
    inventory.db for seconds every few seconds (a "batch window" setting); inventory's own
    writes then fail with "database is locked" and checkouts fail intermittently."""

    family = "scheduler_lock_hold"
    spec_number = 20
    category = "performance"
    tier = "hard"
    breaks = ("window",)
    settle_s = 10.0

    def choose_params(self, rng):
        return {"hold_s": rng.choice([2.5, 3.0, 4.0]), "interval_s": rng.choice([3, 4])}

    @property
    def component(self):
        return "scheduler"

    def describe(self):
        return (
            f"etc/scheduler.yaml jobs.expire_reservations.hold_lock_s="
            f"{self.params['hold_s']} every {self.params['interval_s']}s locks inventory.db"
        )

    def alert(self):
        return "Checkout failing intermittently (about half the attempts) with 5xx errors."

    def configure(self, stack):
        p = self.params

        def fn(doc):
            doc["jobs"]["expire_reservations"].update(
                hold_lock_s=p["hold_s"], interval_s=p["interval_s"]
            )

        edit_service_config(stack, "scheduler", fn)

    def root_cause_fixed(self, stack):
        job = effective_config(stack, "scheduler")["jobs"]["expire_reservations"]
        return float(job.get("hold_lock_s", 0)) <= 0.2

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "checkout failed"},
                "Checkouts fail now and then. See why.",
            ),
            (
                "logs",
                {"service": "inventory", "tail": 5, "grep": "locked"},
                "inventory errors on reservations. Is its database locked?",
            ),
            (
                "exec",
                {"command": "grep -n -A4 expire_reservations etc/scheduler.yaml"},
                "Something holds a lock on inventory.db. The scheduler writes to it directly.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/hold_lock_s: .*/hold_lock_s: 0/' etc/scheduler.yaml"
                    " && grep -n -A4 expire_reservations etc/scheduler.yaml"
                },
                f"expire_reservations holds its lock {self.params['hold_s']} s every "
                f"{self.params['interval_s']} s. Remove the hold.",
            ),
            ("restart", {"service": "scheduler"}, "Restart the scheduler."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "performance",
                    "component": "scheduler",
                    "summary": "scheduler expire_reservations held a long write "
                    "lock on inventory.db; removed hold_lock_s.",
                },
                "Checkouts succeed. Submit.",
            ),
        ]

    def oracle(self, stack):
        edit_service_config(
            stack, "scheduler", lambda d: d["jobs"]["expire_reservations"].update(hold_lock_s=0)
        )
        restart_and_wait(stack, "scheduler")


class CatalogBadBuildFlag(Fault):
    """Spec 15. catalog was deployed with build flag search_v2 on; the v2 search query needs
    a table that only exists from catalog 2.8, so every search returns 500."""

    family = "catalog_bad_build_flag"
    spec_number = 15
    category = "bad_deploy"
    tier = "easy"

    def choose_params(self, rng):
        return {"version": rng.choice(["2.7.2", "2.7.3"])}

    @property
    def component(self):
        return "catalog"

    def describe(self):
        return "etc/catalog.yaml build.flags.search_v2=true needs product_tags (not deployed)"

    def alert(self):
        return "Product search returns errors for every query since the last catalog deploy."

    def configure(self, stack):
        def fn(doc):
            doc["build"] = {"version": self.params["version"], "flags": {"search_v2": True}}

        edit_service_config(stack, "catalog", fn)

    def root_cause_fixed(self, stack):
        flags = effective_config(stack, "catalog").get("build", {}).get("flags") or {}
        return not flags.get("search_v2")

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "catalog", "tail": 5, "grep": "error"},
                "Search fails. Check catalog's errors.",
            ),
            (
                "exec",
                {"command": "grep -n -A4 '^build:' etc/catalog.yaml"},
                "The search query hits a missing table. Check catalog's build flags.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/search_v2: true/search_v2: false/' etc/catalog.yaml"
                    " && grep -n -A4 '^build:' etc/catalog.yaml"
                },
                "search_v2 is on but its table does not exist yet. Roll the flag back.",
            ),
            ("restart", {"service": "catalog"}, "Restart catalog."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "bad_deploy",
                    "component": "catalog",
                    "summary": "catalog deployed with build flag search_v2, which "
                    "needs a missing table; rolled the flag back.",
                },
                "Search works. Submit.",
            ),
        ]

    def oracle(self, stack):
        edit_service_config(stack, "catalog", lambda d: d["build"]["flags"].update(search_v2=False))
        restart_and_wait(stack, "catalog")


class FlagsServiceDown(Fault):
    """Spec 21. A bad edit left etc/feature-flags.yaml unparsable; the flags service crashes
    on start and keeps restarting. orders, which never got its flags, falls back to its local
    default checkout_mode=maintenance and rejects every checkout (symptom far from cause)."""

    family = "flags_service_down"
    spec_number = 21
    category = "dependency_down"
    tier = "hard"
    breaks = ("journeys",)
    settle_s = 6.0

    def choose_params(self, rng):
        return {"breakage": rng.choice(["tab", "unclosed_list", "bad_indent"])}

    @property
    def component(self):
        return "flags"

    def describe(self):
        return "etc/feature-flags.yaml is not valid YAML; flags crash-loops; orders fails closed"

    def alert(self):
        return "All checkouts rejected with 503 'checkout temporarily disabled'."

    def configure(self, stack):
        path = config_path(stack, "feature-flags")
        with open(path) as f:
            text = f.read()
        b = self.params["breakage"]
        if b == "tab":
            text = text.replace("  catalog:", "\tcatalog:", 1)
        elif b == "unclosed_list":
            text = text.replace("- kitchen", "[kitchen", 1)
        else:
            text = text.replace("  orders:", "     orders:", 1)
        with open(path, "w") as f:
            f.write(text)

    def root_cause_fixed(self, stack):
        import yaml

        try:
            doc = effective_flags(stack)
        except yaml.YAMLError:
            return False
        orders = (doc.get("services") or {}).get("orders") or {}
        return orders.get("checkout_mode") == "on"

    def solution_steps(self, stack):
        b = self.params["breakage"]
        fix = {
            "tab": "sed -i 's/^\\tcatalog:/  catalog:/' etc/feature-flags.yaml",
            "unclosed_list": "sed -i 's/\\[kitchen/- kitchen/' etc/feature-flags.yaml",
            "bad_indent": "sed -i 's/^     orders:/  orders:/' etc/feature-flags.yaml",
        }[b]
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "checkout"},
                "Checkouts are rejected. See what orders says.",
            ),
            ("list_services", {}, "orders runs in maintenance mode. Is the flags service up?"),
            (
                "exec",
                {"command": "tail -5 logs/flags.stdout; cat -A etc/feature-flags.yaml | head -20"},
                "flags keeps crashing. Check its error and its file.",
            ),
            (
                "exec",
                {"command": f"{fix} && cat etc/feature-flags.yaml"},
                "The flags file is not valid YAML. Repair it.",
            ),
            ("restart", {"service": "flags"}, "Restart flags; orders will pick up its flags."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "dependency_down",
                    "component": "flags",
                    "summary": "feature-flags.yaml was invalid so flags crash-"
                    "looped and orders fell back to maintenance; "
                    "repaired the file.",
                },
                "Checkouts work again. Submit.",
            ),
        ]

    def oracle(self, stack):
        import yaml

        from northwind.provision import FLAGS

        with open(config_path(stack, "feature-flags"), "w") as f:
            yaml.safe_dump(FLAGS, f, sort_keys=False)
        restart_and_wait(stack, "flags")
        time.sleep(3)  # orders polls flags every 2 s


class MissingMigration(Fault):
    """Spec 16. orders was deployed with code that writes the new column orders.channel, but
    migration 3 was not applied (rolled back by a failed deploy); every checkout fails on
    insert although orders reports ready."""

    family = "missing_migration"
    spec_number = 16
    category = "schema"
    tier = "medium"

    def choose_params(self, rng):
        return {}

    @property
    def component(self):
        return "orders"

    def describe(self):
        return "orders.db is missing migration 3 (column channel) that orders code writes"

    def alert(self):
        return "Every checkout fails with 500; browsing works."

    def configure(self, stack):
        import sqlite3

        c = sqlite3.connect(os.path.join(stack.home, "data/orders.db"), isolation_level=None)
        try:
            c.execute("ALTER TABLE orders DROP COLUMN channel")
            c.execute("DELETE FROM schema_migrations WHERE version=3")
        finally:
            c.close()

    def root_cause_fixed(self, stack):
        import sqlite3

        c = sqlite3.connect(f"file:{stack.home}/data/orders.db?mode=ro", uri=True)
        try:
            cols = {r[1] for r in c.execute("PRAGMA table_info(orders)")}
        finally:
            c.close()
        return "channel" in cols

    def solution_steps(self, stack):
        o = stack.ports["orders"]
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "error"},
                "Checkouts fail with 500. Check orders' errors.",
            ),
            (
                "exec",
                {"command": f"curl -s http://127.0.0.1:{o}/admin/migrations"},
                "The insert references a column that does not exist. Check schema migrations.",
            ),
            (
                "exec",
                {"command": f"curl -s -X POST http://127.0.0.1:{o}/admin/migrations/apply"},
                "Migration 3 is pending. Apply it.",
            ),
            (
                "logs",
                {"service": "orders", "tail": 3, "grep": "checkout completed"},
                "Check that checkouts complete.",
            ),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "schema",
                    "component": "orders",
                    "summary": "orders.db was missing migration 3 (channel "
                    "column) the code writes; applied it.",
                },
                "Checkouts work. Submit.",
            ),
        ]

    def oracle(self, stack):
        st, body = http_json(stack.ports["orders"], "POST", "/admin/migrations/apply")
        assert st == 200, body


def _env_path(stack, service):
    return os.path.join(stack.home, "etc", "env", f"{service}.env")


class FulfillmentEnvTypo(Fault):
    """Spec 5. A typo in fulfillment's env file (warehouse id variable misspelled) makes it
    fall back to its unsafe default, dry-run mode: order.paid events are acknowledged but
    nothing ships. After fixing the variable, the orders acknowledged in dry-run mode must
    be re-driven."""

    family = "fulfillment_env_typo"
    spec_number = 5
    category = "config"
    tier = "medium"
    settle_s = 6.0

    def choose_params(self, rng):
        return {
            "typo": rng.choice(
                ["FULFILMENT_WAREHOUSE_ID", "FULFILLMENT_WAREHOUSE", "FULFILLMENT_WAREHOUSEID"]
            )
        }

    @property
    def component(self):
        return "fulfillment"

    def describe(self):
        return (
            f"etc/env/fulfillment.env sets {self.params['typo']} instead of "
            "FULFILLMENT_WAREHOUSE_ID; fulfillment runs in dry-run mode"
        )

    def alert(self):
        return "Orders are paid but none have shipped for several minutes."

    def configure(self, stack):
        path = _env_path(stack, "fulfillment")
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace("FULFILLMENT_WAREHOUSE_ID=", self.params["typo"] + "="))

    def root_cause_fixed(self, stack):
        from northwind.supervisor import read_env_file

        wh = read_env_file(_env_path(stack, "fulfillment")).get("FULFILLMENT_WAREHOUSE_ID")
        return bool(wh) and wh != "DRYRUN"

    def solution_steps(self, stack):
        f = stack.ports["fulfillment"]
        typo = self.params["typo"]
        return [
            (
                "logs",
                {"service": "fulfillment", "tail": 5, "grep": "dry"},
                "Nothing ships. Check fulfillment.",
            ),
            (
                "exec",
                {"command": "cat etc/env/fulfillment.env"},
                "It runs in dry-run mode because the warehouse id is not set. Check its env.",
            ),
            (
                "exec",
                {
                    "command": f"sed -i 's/^{typo}=/FULFILLMENT_WAREHOUSE_ID=/' "
                    "etc/env/fulfillment.env && cat etc/env/fulfillment.env"
                },
                f"The variable is misspelled as {typo}. Fix it.",
            ),
            ("restart", {"service": "fulfillment"}, "Restart fulfillment to read its env."),
            (
                "exec",
                {"command": f"curl -s -X POST http://127.0.0.1:{f}/admin/redrive"},
                "Orders acknowledged in dry-run mode were never shipped. Re-drive them.",
            ),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": "fulfillment",
                    "summary": "typo in fulfillment env var left it in dry-run "
                    "mode; fixed the name and re-drove paid orders.",
                },
                "Orders ship. Submit.",
            ),
        ]

    def oracle(self, stack):
        path = _env_path(stack, "fulfillment")
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace(self.params["typo"] + "=", "FULFILLMENT_WAREHOUSE_ID="))
        restart_and_wait(stack, "fulfillment")
        st, body = http_json(stack.ports["fulfillment"], "POST", "/admin/redrive")
        assert st == 200, body


class OrdersConfigDrift(Fault):
    """Spec 6. orders is running last release's config: its env file still points
    NW_CONFIG_FILE at the archived config, which calls the decommissioned payments-v1
    endpoint. Checkout fails; etc/orders.yaml itself looks correct."""

    family = "orders_config_drift"
    spec_number = 6
    category = "config"
    tier = "hard"

    def choose_params(self, rng):
        return {}

    @property
    def component(self):
        return "orders"

    def describe(self):
        return (
            "etc/env/orders.env NW_CONFIG_FILE=etc/archive/orders-2026-09.yaml (old payments "
            "endpoint)"
        )

    def alert(self):
        return "Every checkout fails with 502 since this morning's deploy."

    def configure(self, stack):
        with open(_env_path(stack, "orders"), "a") as f:
            f.write("NW_CONFIG_FILE=etc/archive/orders-2026-09.yaml\n")

    def root_cause_fixed(self, stack):
        from northwind.supervisor import read_env_file

        cf = read_env_file(_env_path(stack, "orders")).get("NW_CONFIG_FILE", "etc/orders.yaml")
        return os.path.normpath(cf) == "etc/orders.yaml"

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "orders", "tail": 5, "grep": "checkout failed"},
                "Checkouts fail with 502. See why.",
            ),
            (
                "logs",
                {"service": "orders", "tail": 2, "grep": "service started"},
                "orders cannot reach payments. Which config did it start with?",
            ),
            (
                "exec",
                {
                    "command": "cat etc/env/orders.env; grep -n -A2 '^  payments:' "
                    "etc/orders.yaml etc/archive/orders-2026-09.yaml"
                },
                "It loaded an archived config with the old payments endpoint.",
            ),
            (
                "exec",
                {
                    "command": "sed -i '/^NW_CONFIG_FILE=/d' etc/env/orders.env && "
                    "cat etc/env/orders.env"
                },
                "Remove the override so orders uses etc/orders.yaml.",
            ),
            ("restart", {"service": "orders"}, "Restart orders."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "config",
                    "component": "orders",
                    "summary": "orders ran the archived 2026-09 config via "
                    "NW_CONFIG_FILE (old payments endpoint); removed it.",
                },
                "Checkouts work. Submit.",
            ),
        ]

    def oracle(self, stack):
        path = _env_path(stack, "orders")
        with open(path) as f:
            lines = [ln for ln in f if not ln.startswith("NW_CONFIG_FILE=")]
        with open(path, "w") as f:
            f.writelines(lines)
        restart_and_wait(stack, "orders")


class CacheFullNoEviction(Fault):
    """Spec 13. The cache was given a small memory limit and the noeviction policy. It fills
    up with catalog entries within seconds; afterwards every write is refused, and auth,
    which stores login sessions there, rejects every login."""

    family = "cache_full_noeviction"
    spec_number = 13
    category = "resource_exhaustion"
    tier = "medium"
    settle_s = 12.0

    def choose_params(self, rng):
        return {"max_bytes": rng.choice([8_000, 12_000, 16_000])}

    @property
    def component(self):
        return "cache"

    def describe(self):
        return (
            f"etc/cache.yaml max_memory_bytes={self.params['max_bytes']} with "
            "eviction_policy noeviction: cache full, session writes refused"
        )

    def alert(self):
        return "Customers cannot log in (503); browsing works."

    def configure(self, stack):
        edit_service_config(
            stack,
            "cache",
            lambda d: d.update(
                max_memory_bytes=self.params["max_bytes"], eviction_policy="noeviction"
            ),
        )

    def root_cause_fixed(self, stack):
        c = effective_config(stack, "cache")
        return (
            c.get("eviction_policy") != "noeviction"
            or int(c.get("max_memory_bytes", 0)) >= 1_000_000
        )

    def solution_steps(self, stack):
        return [
            (
                "logs",
                {"service": "auth", "tail": 5, "grep": "session"},
                "Logins fail. See why auth rejects them.",
            ),
            ("metrics", {"service": "cache"}, "auth cannot store sessions. Check the cache."),
            (
                "exec",
                {"command": "cat etc/cache.yaml"},
                "The cache is full and refuses writes. Check its limits and policy.",
            ),
            (
                "exec",
                {
                    "command": "sed -i 's/eviction_policy: noeviction/eviction_policy: "
                    "allkeys-lru/; s/max_memory_bytes: .*/max_memory_bytes: "
                    "4000000/' etc/cache.yaml && cat etc/cache.yaml"
                },
                "Tiny limit with noeviction: once full nothing can be written. Evict LRU and "
                "restore the limit.",
            ),
            ("restart", {"service": "cache"}, "Restart the cache."),
            (
                "submit_diagnosis",
                {
                    "root_cause_category": "resource_exhaustion",
                    "component": "cache",
                    "summary": "cache had a tiny memory limit and noeviction, so "
                    "it filled and refused session writes; set "
                    "allkeys-lru and 4 MB.",
                },
                "Logins work. Submit.",
            ),
        ]

    def oracle(self, stack):
        edit_service_config(
            stack,
            "cache",
            lambda d: d.update(max_memory_bytes=4_000_000, eviction_policy="allkeys-lru"),
        )
        restart_and_wait(stack, "cache")


FAMILIES = {
    cls.family: cls
    for cls in (
        FulfillmentEnvTypo,
        OrdersConfigDrift,
        CacheFullNoEviction,
        SchedulerLockHold,
        CatalogBadBuildFlag,
        FlagsServiceDown,
        MissingMigration,
        WrongDatabasePath,
        EdgeRateLimitTooLow,
        FulfillmentWrongTopic,
        PaymentsBreakerStuckOpen,
        NotifyDeadLetters,
        EdgeWrongUpstreamPort,
        OrdersPaymentsTimeoutInversion,
        CatalogLeakFlag,
        JwtKeyRotationMismatch,
        PoisonMessage,
    )
}


def make_fault(family: str, variant: int = 0) -> Fault:
    return FAMILIES[family](variant)


def describe_all() -> str:
    return json.dumps(
        {
            k: {"spec": v.spec_number, "category": v.category, "tier": v.tier}
            for k, v in FAMILIES.items()
        },
        indent=1,
    )
