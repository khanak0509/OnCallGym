"""Database schemas as ordered migrations per database file.

Stack creation applies every migration and records it in schema_migrations. Services do
not migrate on start (like most production setups, migrations are a deploy step), but they
check that the schema version they need is present and report not-ready otherwise.
"""

MIGRATIONS: dict[str, list[tuple[int, str]]] = {
    "users": [
        (
            1,
            """
CREATE TABLE users (
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  password_hash TEXT NOT NULL, salt TEXT NOT NULL, address TEXT NOT NULL,
  created_at REAL NOT NULL);
""",
        ),
        (
            2,
            "CREATE TABLE login_events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, "
            "email TEXT, ok INTEGER NOT NULL, ts REAL NOT NULL);",
        ),
    ],
    "catalog": [
        (
            1,
            """
CREATE TABLE products (
  sku TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
  description TEXT NOT NULL, price_cents INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1);
""",
        ),
        (2, "CREATE INDEX products_category ON products (category);"),
    ],
    "inventory": [
        (
            1,
            """
CREATE TABLE stock (sku TEXT PRIMARY KEY, on_hand INTEGER NOT NULL, reserved INTEGER NOT NULL
  DEFAULT 0, CHECK (on_hand >= 0), CHECK (reserved >= 0));
CREATE TABLE reservations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, sku TEXT NOT NULL,
  qty INTEGER NOT NULL, status TEXT NOT NULL, expires_at REAL NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX reservations_order ON reservations (order_id);
CREATE INDEX reservations_status ON reservations (status, expires_at);
""",
        ),
    ],
    "orders": [
        (
            1,
            """
CREATE TABLE orders (
  id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, email TEXT NOT NULL, status TEXT NOT NULL,
  total_cents INTEGER NOT NULL, idempotency_key TEXT UNIQUE, created_at REAL NOT NULL,
  updated_at REAL NOT NULL, failure_reason TEXT);
CREATE TABLE order_items (order_id TEXT NOT NULL, sku TEXT NOT NULL, qty INTEGER NOT NULL,
  price_cents INTEGER NOT NULL);
CREATE INDEX order_items_order ON order_items (order_id);
CREATE INDEX orders_status ON orders (status, updated_at);
CREATE TABLE outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL,
  payload TEXT NOT NULL, request_id TEXT, created_at REAL NOT NULL, published_at REAL);
CREATE INDEX outbox_pending ON outbox (published_at, id);
""",
        ),
        (
            2,
            """
ALTER TABLE orders ADD COLUMN shipping_address TEXT;
CREATE TABLE shipments (order_id TEXT PRIMARY KEY, tracking TEXT NOT NULL,
  created_at REAL NOT NULL);
""",
        ),
    ],
    "payments": [
        (
            1,
            """
CREATE TABLE charges (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
  amount_cents INTEGER NOT NULL, status TEXT NOT NULL, provider_charge_id TEXT,
  error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX charges_order ON charges (order_id);
CREATE TABLE refunds (id TEXT PRIMARY KEY, charge_id TEXT NOT NULL, order_id TEXT NOT NULL,
  amount_cents INTEGER NOT NULL, provider_refund_id TEXT, created_at REAL NOT NULL);
""",
        ),
    ],
    "provider": [
        (
            1,
            """
CREATE TABLE charges (
  id TEXT PRIMARY KEY, idempotency_key TEXT, merchant_ref TEXT NOT NULL,
  amount_cents INTEGER NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL);
CREATE UNIQUE INDEX charges_idem ON charges (idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE refunds (id TEXT PRIMARY KEY, charge_id TEXT NOT NULL, amount_cents INTEGER NOT NULL,
  created_at REAL NOT NULL);
""",
        ),
    ],
}

# version each service requires (services report not-ready below this)
REQUIRED = {"users": 2, "catalog": 2, "inventory": 1, "orders": 2, "payments": 1, "provider": 1}


def migrate(conn, db: str, upto: int | None = None) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, "
        "applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    current = max(done, default=0)
    for version, sql in MIGRATIONS[db]:
        if version in done or (upto is not None and version > upto):
            continue
        conn.executescript(
            "BEGIN;" + sql + f"INSERT INTO schema_migrations (version) VALUES ({version}); COMMIT;"
        )
        current = version
    return current


def schema_version(pool_or_conn) -> int:
    try:
        if hasattr(pool_or_conn, "one"):
            r = pool_or_conn.one("SELECT MAX(version) AS v FROM schema_migrations")
            return int(r["v"] or 0)
        r = pool_or_conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(r[0] or 0)
    except Exception:
        return 0
