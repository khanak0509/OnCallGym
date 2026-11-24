BLOCK_SIZE = 20

SERVICES = {
    "broker": (
        10,
        0,
        [],
        "message broker: topics, queues, acks, redelivery, dead-letter queues. "
        "Admin: GET /queues, GET /queues/<q>/messages, GET /queues/<q>/dlq, "
        "POST /queues/<q>/dlq/replay, POST /queues/<q>/backfill, "
        "POST /queues/<q>/messages/<id>/dead-letter",
    ),
    "cache": (11, 0, [], "key-value cache with memory budget and eviction policy"),
    "flags": (12, 0, [], "central config and feature flags; services poll it"),
    "provider": (7, 0, [], "external payment provider (third party, not ours to change)"),
    "auth": (
        2,
        1,
        ["cache", "flags"],
        "users, login (sessions kept in cache), JWT issue and verify (HTTPS)",
    ),
    "catalog": (3, 1, ["cache", "flags"], "products, search, pricing; read-through cache"),
    "inventory": (4, 1, ["broker", "flags"], "stock levels and reservations with expiry"),
    "payments": (
        6,
        1,
        ["provider", "flags"],
        "charges via provider: idempotency, retries, circuit breaker",
    ),
    "orders": (
        5,
        2,
        ["auth", "catalog", "inventory", "payments", "broker", "flags"],
        "checkout orchestration; transactional outbox to broker. Admin: GET /admin/migrations, "
        "POST /admin/migrations/apply",
    ),
    "fulfillment": (8, 3, ["broker", "orders", "flags"], "consumes order.paid, creates shipments"),
    "notify": (9, 3, ["broker", "flags"], "consumes order events, writes customer emails"),
    "scheduler": (
        13,
        3,
        ["inventory", "payments", "orders", "flags"],
        "periodic jobs: expire reservations, reconcile payments, rotate logs",
    ),
    "edge": (
        1,
        3,
        ["auth", "catalog", "orders"],
        "reverse proxy: routing, rate limits, request ids, timeouts",
    ),
    "telemetry": (14, 3, [], "scrapes metrics, ships logs, stores request traces"),
    "trafficgen": (15, 4, ["edge"], "synthetic user traffic"),
}
SUPERVISOR_OFFSET = 0

DATABASES = {
    "auth": "users",
    "catalog": "catalog",
    "inventory": "inventory",
    "orders": "orders",
    "payments": "payments",
    "provider": "provider",
}

MEMORY_LIMIT_MB = {name: 120 for name in SERVICES}


def ports(base: int) -> dict[str, int]:
    out = {name: base + spec[0] for name, spec in SERVICES.items()}
    out["supervisor"] = base + SUPERVISOR_OFFSET
    return out


def layers() -> list[list[str]]:
    n = max(s[1] for s in SERVICES.values()) + 1
    return [[name for name, s in SERVICES.items() if s[1] == i] for i in range(n)]
