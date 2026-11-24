import random

COMMON = [
    ("warn", "TLS certificate for internal CA expires in {days} days", {"days": (40, 90)}),
    ("info", "gc pause {ms}ms", {"ms": (8, 60)}),
    ("warn", "config key 'legacy_mode' is deprecated and ignored", {}),
    ("warn", "connection reset by peer, retrying", {"attempt": (1, 1)}),
]
PER_SERVICE = {
    "catalog": [("info", "slow query", {"dur_ms": (120, 280), "table": ("products",)})],
    "auth": [("warn", "client sent deprecated header X-Legacy-Client", {"client": ("ios-4.2",)})],
    "payments": [("warn", "provider latency above p50", {"latency_ms": (110, 190)})],
    "inventory": [("info", "stock below reorder point", {"sku": ("KIT-104", "OFF-107")})],
    "orders": [("warn", "order total recalculated after price change", {"delta_cents": (1, 9)})],
    "broker": [("info", "queue depth sample", {"queue": ("notify.events",), "depth": (0, 3)})],
    "edge": [("warn", "client closed connection before response", {"path": ("/api/catalog/",)})],
    "scheduler": [
        ("info", "job took longer than usual", {"job": ("rotate_logs",), "ms": (40, 90)})
    ],
}


def emitter(svc, seed: int = 0):
    rng = random.Random(f"noise:{svc.name}:{seed}")
    templates = COMMON + PER_SERVICE.get(svc.name, [])

    def emit():
        if rng.random() > 0.5:
            return
        level, msg, fields = rng.choice(templates)
        vals = {}
        for k, v in fields.items():
            vals[k] = rng.randint(*v) if isinstance(v[0], int) else rng.choice(v)
        svc.log.log(
            level, msg.format(**vals), **{k: v for k, v in vals.items() if "{" + k + "}" not in msg}
        )

    return emit
