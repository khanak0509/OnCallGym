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
    read_yaml,
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

    def inject(self, stack):
        up = self.params["upstream"]
        port = self.wrong_port(stack)

        def change(doc):
            doc["upstreams"][up]["url"] = _with_port(doc["upstreams"][up]["url"], port)

        edit_service_config(stack, "edge", change)
        restart_and_wait(stack, "edge")

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

    def inject(self, stack):
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
        for s in ("provider", "payments", "orders"):
            restart_and_wait(stack, s)

    def oracle(self, stack):
        pay = read_yaml(config_path(stack, "payments"))["upstreams"]["provider"]
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

    def inject(self, stack):
        p = self.params

        def flags(doc):
            svc = doc.setdefault("services", {}).setdefault("catalog", {})
            svc[p["flag"]] = True
            svc["recommendations_cache_kb"] = p["leak_kb"]

        edit_yaml(config_path(stack, "feature-flags"), flags)

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

    def inject(self, stack):
        with open(os.path.join(stack.home, self.key_file()), "w") as f:
            f.write(self.params["key"] + "\n")

        def auth(doc):
            doc["signing_key"] = {"kid": self.params["new_kid"], "file": self.key_file()}

        edit_service_config(stack, "auth", auth)
        restart_and_wait(stack, "auth")

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
        st, body = http_json(
            stack.ports["broker"],
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

    def inject(self, stack):
        wrong = self.params["wrong"]
        if wrong.startswith("/var/lib"):
            # absolute path into a directory that does not exist inside the episode
            wrong = "data/missing-volume/" + os.path.basename(wrong)
            self.params["wrong"] = wrong
        edit_service_config(stack, self.params["service"], lambda d: d["db"].update(path=wrong))
        restart_and_wait(stack, self.params["service"], timeout_s=8)

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

    def inject(self, stack):
        edit_service_config(
            stack,
            "edge",
            lambda d: d.update(
                rate_limit={"rps": self.params["rps"], "burst": self.params["burst"]}
            ),
        )
        restart_and_wait(stack, "edge")

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

    def inject(self, stack):
        self.injected_at = time.time()
        edit_service_config(
            stack, "fulfillment", lambda d: d["consumer"].update(topic=self.params["topic"])
        )
        restart_and_wait(stack, "fulfillment")

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
            "error_rate": rng.choice([0.12, 0.15, 0.2]),
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

    def inject(self, stack):
        p = self.params
        edit_service_config(stack, "provider", lambda d: d.update(error_rate=p["error_rate"]))

        def payments(doc):
            doc["upstreams"]["provider"]["breaker"] = {
                "failure_threshold": 1,
                "window_s": 60,
                "reset_timeout_s": p["reset_timeout_s"],
            }

        edit_service_config(stack, "payments", payments)
        restart_and_wait(stack, "provider")
        restart_and_wait(stack, "payments")

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

    def inject(self, stack):
        def broker(doc):
            doc["queues"]["notify.events"] = {"max_redeliveries": 0}

        edit_service_config(stack, "broker", broker)
        restart_and_wait(stack, "broker")

    def oracle(self, stack):
        def broker(doc):
            doc["queues"]["notify.events"] = {}

        edit_service_config(stack, "broker", broker)
        restart_and_wait(stack, "broker")
        st, body = http_json(stack.ports["broker"], "POST", "/queues/notify.events/dlq/replay")
        assert st == 200, body


FAMILIES = {
    cls.family: cls
    for cls in (
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
