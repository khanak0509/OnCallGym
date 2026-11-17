"""Create one Northwind deployment in a directory: configs (etc/), env files, keys, TLS
certs, seeded databases (data/), empty logs/. Deterministic for a given seed.

Layout of <home>:
  etc/<service>.yaml  etc/env/<service>.env  etc/flags.yaml  etc/keys/  etc/tls/
  data/*.db  data/outbox/  data/seed/manifest.json  logs/  run/
Verifier-only material (probe user credentials, seed hashes) goes to <private>, which is
outside <home>.
"""

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile

import yaml

from . import schema
from .services.auth import hash_password
from .topology import DATABASES, MEMORY_LIMIT_MB, SERVICES, layers, ports

VERSION = {
    "edge": "1.14.2",
    "auth": "3.2.0",
    "catalog": "2.7.1",
    "inventory": "2.3.4",
    "orders": "4.1.0",
    "payments": "3.0.3",
    "provider": "2025.10",
    "fulfillment": "1.9.0",
    "notify": "1.4.2",
    "broker": "0.9.7",
    "cache": "1.2.0",
    "flags": "1.1.5",
    "scheduler": "1.6.0",
    "telemetry": "2.0.1",
    "trafficgen": "0.4.0",
}

CATEGORIES = {
    "kitchen": [
        "Steel Mug",
        "Pour-over Kit",
        "Chef Knife",
        "Cutting Board",
        "Tea Kettle",
        "Spice Rack",
        "Cast Iron Pan",
        "Salad Bowl",
        "Coffee Grinder",
        "Oven Mitt",
        "Water Bottle",
        "Mini Blender",
    ],
    "office": [
        "Desk Lamp",
        "Standing Desk",
        "Ergo Chair",
        "Notebook Pro",
        "Monitor Arm",
        "Desk Mat",
        "Cable Kit",
        "Pen Set",
        "Laptop Stand",
        "Whiteboard",
        "Mini Shelf",
        "Paper Tray",
    ],
    "outdoor": [
        "Camp Chair",
        "Trail Lamp",
        "Steel Flask",
        "Tent Mini",
        "Rain Jacket",
        "Hiking Poles",
        "Cooler Box",
        "Hammock",
        "Fire Kit",
        "Map Case",
        "Dry Bag",
        "Picnic Mat",
    ],
    "apparel": [
        "Cotton Tee",
        "Wool Socks",
        "Denim Jacket",
        "Rain Hat",
        "Linen Shirt",
        "Cotton Hoodie",
        "Trail Shorts",
        "Knit Beanie",
        "Leather Belt",
        "Scarf",
        "Sneakers Pro",
        "Gloves",
    ],
}
FIRST = [
    "Ada",
    "Ben",
    "Chen",
    "Dara",
    "Eli",
    "Fay",
    "Gus",
    "Hana",
    "Ivo",
    "Jun",
    "Kai",
    "Lia",
    "Mo",
    "Nia",
    "Omar",
    "Pia",
    "Quin",
    "Rui",
    "Sam",
    "Tara",
]
LAST = ["Okafor", "Silva", "Kim", "Novak", "Haddad", "Larsen", "Mehta", "Rossi", "Tanaka", "Brown"]
STREETS = ["Elm St", "Harbor Rd", "Mill Ln", "Oak Ave", "Station Rd", "Park Pl"]

N_USERS = 60
N_PROBE_USERS = 10
STOCK_ON_HAND = 5000


def tls_material(cache_dir: str | None = None) -> str:
    """Self-signed CA + auth server cert, generated once per machine with openssl.
    Extensions satisfy Python 3.13's VERIFY_X509_STRICT."""
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "oncallgym", "tls-v2")
    if os.path.exists(os.path.join(cache_dir, "auth.pem")):
        return cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=cache_dir)

    def run(*args):
        subprocess.run(["openssl", *args], cwd=tmp, check=True, capture_output=True, timeout=60)

    with open(os.path.join(tmp, "ca.ext"), "w") as f:
        f.write(
            "[req]\ndistinguished_name=dn\n[dn]\n[v3_ca]\n"
            "basicConstraints=critical,CA:TRUE\n"
            "keyUsage=critical,keyCertSign,cRLSign\n"
            "subjectKeyIdentifier=hash\n"
        )
    with open(os.path.join(tmp, "leaf.ext"), "w") as f:
        f.write(
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            "subjectKeyIdentifier=hash\n"
            "authorityKeyIdentifier=keyid,issuer\n"
            "subjectAltName=IP:127.0.0.1,DNS:localhost\n"
        )
    run(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "ca.key",
        "-out",
        "ca.pem",
        "-days",
        "3650",
        "-subj",
        "/CN=northwind-internal-ca",
        "-config",
        "ca.ext",
        "-extensions",
        "v3_ca",
    )
    run(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "auth.key",
        "-out",
        "auth.csr",
        "-subj",
        "/CN=auth.northwind.internal",
    )
    run(
        "x509",
        "-req",
        "-in",
        "auth.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "auth.pem",
        "-days",
        "3650",
        "-extfile",
        "leaf.ext",
    )
    for f in ("ca.pem", "auth.key", "auth.pem"):
        os.replace(os.path.join(tmp, f), os.path.join(cache_dir, f))
    shutil.rmtree(tmp)
    return cache_dir


def service_configs(p: dict, seed: int) -> dict[str, dict]:
    url = lambda name, scheme="http": f"{scheme}://127.0.0.1:{p[name]}"  # noqa: E731
    flags_url = url("flags")

    def base(name, flag_defaults=None, **extra):
        c = {
            "listen": {"host": "127.0.0.1", "port": p[name]},
            "log": {"level": "info"},
            "build": {"version": VERSION[name], "flags": {}},
        }
        if flag_defaults is not None:
            c["flags"] = {"url": flags_url, "poll_interval_s": 2, "defaults": flag_defaults}
        c.update(extra)
        return c

    def db(name, pool=4, timeout=1000):
        return {"path": f"data/{name}.db", "pool_size": pool, "acquire_timeout_ms": timeout}

    broker_up = {
        "url": url("broker"),
        "timeout_ms": 2000,
        "retries": 2,
        "backoff_ms": 100,
        "pool_size": 4,
    }
    auth_up = {
        "url": url("auth", "https"),
        "ca_file": "etc/tls/ca.pem",
        "timeout_ms": 2000,
        "retries": 1,
        "backoff_ms": 50,
        "pool_size": 8,
    }
    key = {"kid": "k-2026-10", "file": "etc/keys/jwt-k-2026-10.key"}
    cfgs = {
        "broker": base(
            "broker",
            db="data/broker.db",
            defaults={"max_redeliveries": 5, "redelivery_delay_ms": 500, "ordered": False},
            queues={
                "inventory.order-paid": {},
                # shipments go out in payment order and an order is never dropped
                "fulfillment.order-paid": {"ordered": True, "max_redeliveries": -1},
                "notify.events": {},
            },
        ),
        "cache": base(
            "cache", max_memory_bytes=4_000_000, eviction_policy="allkeys-lru", default_ttl_s=60
        ),
        "flags": base("flags", flags_file="etc/feature-flags.yaml"),
        "provider": base(
            "provider", db=db("provider"), latency_ms={"base": 40, "jitter": 40}, error_rate=0.01
        ),
        "auth": base("auth", {"token_ttl_s": 900}, db=db("users"), signing_key=key, verify_keys=[]),
        "catalog": base(
            "catalog",
            {"promo_discount_pct": 0, "promo_categories": []},
            db=db("catalog"),
            cache={"enabled": True, "ttl_s": 30},
            upstreams={"cache": {"url": url("cache"), "timeout_ms": 300, "pool_size": 8}},
        ),
        "inventory": base(
            "inventory",
            {},
            db=db("inventory"),
            reservation_ttl_s=120,
            upstreams={"broker": broker_up},
            consumer={"queue": "inventory.order-paid", "topics": ["order.paid"]},
        ),
        "payments": base(
            "payments",
            {},
            db=db("payments"),
            send_idempotency_key=True,
            upstreams={
                "provider": {
                    "url": url("provider"),
                    "timeout_ms": 1500,
                    "retries": 2,
                    "backoff_ms": 100,
                    "backoff_max_ms": 800,
                    "pool_size": 8,
                    "retry_on_status": [502, 503, 504],
                    "breaker": {"failure_threshold": 8, "window_s": 10, "reset_timeout_s": 5},
                }
            },
        ),
        "orders": base(
            "orders",
            {"checkout_mode": "maintenance"},
            db=db("orders"),
            max_items_per_order=10,
            outbox_interval_s=0.3,
            auth={"trusted_keys": [key]},
            upstreams={
                "auth": auth_up,
                "catalog": {
                    "url": url("catalog"),
                    "timeout_ms": 1500,
                    "retries": 1,
                    "pool_size": 8,
                },
                "inventory": {"url": url("inventory"), "timeout_ms": 2000, "pool_size": 8},
                "payments": {"url": url("payments"), "timeout_ms": 6000, "pool_size": 8},
                "broker": broker_up,
            },
        ),
        "fulfillment": base(
            "fulfillment",
            {},
            upstreams={
                "orders": {"url": url("orders"), "timeout_ms": 2000, "retries": 1, "pool_size": 2},
                "broker": broker_up,
            },
            consumer={"queue": "fulfillment.order-paid", "topic": "order.paid", "batch": 10},
        ),
        "notify": base(
            "notify",
            {},
            outbox_file="data/outbox/emails.jsonl",
            smtp_file="etc/external/smtp.yaml",
            upstreams={"broker": broker_up},
            consumer={"queue": "notify.events", "topics": ["order.paid", "order.shipped"]},
        ),
        "scheduler": base(
            "scheduler",
            {},
            databases={
                "inventory": "data/inventory.db",
                "payments": "data/payments.db",
                "orders": "data/orders.db",
            },
            upstreams={
                "payments": {"url": url("payments"), "timeout_ms": 5000},
                "orders": {"url": url("orders"), "timeout_ms": 2000},
            },
            jobs={
                "expire_reservations": {"interval_s": 5, "batch": 200},
                "reconcile_payments": {"interval_s": 10, "grace_s": 30},
                "rotate_logs": {"interval_s": 15, "max_bytes": 2_000_000, "keep": 2},
            },
        ),
        "edge": base(
            "edge",
            upstreams={
                "auth": {**auth_up, "retries": 0},
                "catalog": {"url": url("catalog"), "timeout_ms": 2000, "pool_size": 16},
                "orders": {"url": url("orders"), "timeout_ms": 8000, "pool_size": 16},
            },
            routes=[
                {"prefix": "/api/auth/login", "upstream": "auth", "rewrite": "/login"},
                {"prefix": "/api/catalog/", "upstream": "catalog", "rewrite": "/"},
                {"prefix": "/api/orders/", "upstream": "orders", "rewrite": "/orders/"},
                {"prefix": "/api/checkout", "upstream": "orders", "rewrite": "/checkout"},
            ],
            rate_limit={"rps": 20, "burst": 40},
        ),
        "telemetry": base(
            "telemetry",
            window_s=30,
            scrape_interval_s=2,
            log_interval_s=1,
            max_traces=5000,
            targets={
                n: (
                    {"url": url(n, "https"), "ca_file": "etc/tls/ca.pem"}
                    if n == "auth"
                    else {"url": url(n)}
                )
                for n in SERVICES
                if n != "telemetry"
            },
        ),
        "trafficgen": base(
            "trafficgen",
            seed=seed,
            workers=3,
            think_time_ms=[400, 1400],
            seed_file="data/seed/manifest.json",
            mix={"browse": 5, "search": 3, "checkout": 2, "bad_input": 1},
            upstreams={"edge": {"url": url("edge"), "timeout_ms": 10000, "pool_size": 4}},
        ),
    }
    cfgs["auth"]["listen"]["tls"] = {
        "cert_file": "etc/tls/auth.pem",
        "key_file": "etc/tls/auth.key",
    }
    return cfgs


FLAGS = {
    "global": {"maintenance_mode": False},
    "services": {
        "auth": {"token_ttl_s": 900},
        "catalog": {"promo_discount_pct": 10, "promo_categories": ["kitchen"]},
        "orders": {"checkout_mode": "on"},
    },
}


def supervisor_config(p: dict) -> dict:
    return {
        "listen": {"host": "127.0.0.1", "port": p["supervisor"]},
        "layers": layers(),
        "services": {n: {"memory_limit_mb": MEMORY_LIMIT_MB[n], "port": p[n]} for n in SERVICES},
        "restart": {"backoff_s": 1, "max_backoff_s": 20, "stable_after_s": 30},
        "layer_ready_timeout_s": 8,
    }


def topology_doc(p: dict) -> dict:
    return {
        "services": [
            {"name": n, "port": p[n], "depends_on": s[2], "role": s[3]} for n, s in SERVICES.items()
        ],
        "databases": {svc: f"data/{dbn}.db" for svc, dbn in DATABASES.items()}
        | {"broker": "data/broker.db"},
        "supervisor_port": p["supervisor"],
    }


def table_hash(conn, table: str) -> str:
    h = hashlib.sha256()
    for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):
        h.update(repr(tuple(row)).encode())
    return h.hexdigest()


def provision(home: str, private: str, base_port: int, seed: int = 0) -> dict:
    from .lib.db import connect

    rng = random.Random(seed)
    p = ports(base_port)
    for d in ("etc/env", "etc/keys", "etc/tls", "data/outbox", "data/seed", "logs", "run"):
        os.makedirs(os.path.join(home, d), exist_ok=True)
    os.makedirs(private, mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)

    for name, cfg in service_configs(p, seed).items():
        with open(os.path.join(home, f"etc/{name}.yaml"), "w") as f:
            f.write(f"# {name} {VERSION[name]} - managed by deploy tooling\n")
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        with open(os.path.join(home, f"etc/env/{name}.env"), "w") as f:
            f.write(f"NW_ENV=production\nNW_SERVICE={name}\nMALLOC_ARENA_MAX=2\n")
    # behaviour of third-party systems (not ours to change; hash-protected by the verifier)
    os.makedirs(os.path.join(home, "etc/external"), exist_ok=True)
    with open(os.path.join(home, "etc/external/smtp.yaml"), "w") as f:
        f.write("# external mail relay (third party)\n")
        yaml.safe_dump({"relay": "smtp.mailrelay.example:587", "error_rate": 0.2}, f)
    with open(os.path.join(home, "etc/feature-flags.yaml"), "w") as f:
        yaml.safe_dump(FLAGS, f, sort_keys=False)
    with open(os.path.join(home, "etc/supervisor.yaml"), "w") as f:
        yaml.safe_dump(supervisor_config(p), f, sort_keys=False)
    with open(os.path.join(home, "etc/topology.json"), "w") as f:
        json.dump(topology_doc(p), f, indent=1)

    # service-to-service token: readable only by the stack owner, never by the agent user
    fd = os.open(os.path.join(home, "run", "internal.token"), os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(os.urandom(16).hex() + "\n")
    with open(os.path.join(home, "etc/keys/jwt-k-2026-10.key"), "w") as f:
        f.write(rng.randbytes(32).hex() + "\n")
    tls = tls_material()
    for fname in ("ca.pem", "auth.pem", "auth.key"):
        shutil.copy(os.path.join(tls, fname), os.path.join(home, "etc/tls", fname))

    seeded_at = 1767225600.0  # fixed (2026-01-01) so seed data is identical for a seed
    conns = {}
    for dbn in ("users", "catalog", "inventory", "orders", "payments", "provider"):
        c = connect(os.path.join(home, f"data/{dbn}.db"))
        schema.migrate(c, dbn)
        conns[dbn] = c

    users = []
    for i in range(1, N_USERS + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        email = f"{name.split()[0].lower()}.{i}@example.com"
        # probe users get unguessable passwords (not derivable from the public seed)
        password = os.urandom(6).hex() if i > N_USERS - N_PROBE_USERS else rng.randbytes(6).hex()
        salt = rng.randbytes(4).hex()
        addr = f"{rng.randint(1, 400)} {rng.choice(STREETS)}, Springfield"
        users.append({"id": i, "email": email, "password": password})
        conns["users"].execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, email, name, hash_password(password, salt), salt, addr, seeded_at),
        )
    skus = []
    for cat, names in CATEGORIES.items():
        for j, pname in enumerate(names):
            sku = f"{cat[:3].upper()}-{100 + j}"
            skus.append(sku)
            price = rng.randint(4, 180) * 100 + rng.choice([0, 49, 99])
            desc = f"{pname} from the Northwind {cat} range"
            conns["catalog"].execute(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, 1)", (sku, pname, cat, desc, price)
            )
            conns["inventory"].execute("INSERT INTO stock VALUES (?, ?, 0)", (sku, STOCK_ON_HAND))

    seed_hashes = {
        "users.users": table_hash(conns["users"], "users"),
        "catalog.products": table_hash(conns["catalog"], "products"),
    }
    for c in conns.values():
        c.close()

    traffic_users = users[: N_USERS - N_PROBE_USERS]
    probe_users = users[N_USERS - N_PROBE_USERS :]
    with open(os.path.join(home, "data/seed/manifest.json"), "w") as f:
        json.dump({"seed": seed, "traffic_users": traffic_users, "skus": skus}, f)
    with open(
        os.open(
            os.path.join(private, "verifier.json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        ),
        "w",
    ) as f:
        json.dump(
            {
                "seed": seed,
                "probe_users": probe_users,
                "skus": skus,
                "seed_hashes": seed_hashes,
                "stock_on_hand": STOCK_ON_HAND,
                "ports": p,
                "home": home,
            },
            f,
            indent=1,
        )
    return {"ports": p, "skus": skus}
