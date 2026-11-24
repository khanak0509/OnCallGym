import contextlib
import queue
import sqlite3
import threading
import time


class PoolTimeout(Exception):
    pass


class Pool:
    def __init__(
        self,
        path: str,
        size: int = 4,
        acquire_timeout_ms: int = 1000,
        busy_timeout_ms: int = 2000,
        metrics=None,
        name: str = "db",
    ):
        self.path = path
        self.size = int(size)
        self.acquire_timeout = acquire_timeout_ms / 1000.0
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._q: queue.Queue = queue.Queue()
        self._created = 0
        self._lock = threading.Lock()
        self.in_use = 0
        self.waits = 0
        self.timeouts = 0
        self.metrics = metrics
        if metrics:
            metrics.gauge_fn("db_pool_in_use", lambda: self.in_use, db=name)
            metrics.gauge_fn("db_pool_size", lambda: self.size, db=name)
            metrics.gauge_fn("db_pool_timeouts_total", lambda: self.timeouts, db=name)

    def _connect(self):
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    def _get(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self.size:
                self._created += 1
                try:
                    return self._connect()
                except Exception:
                    self._created -= 1
                    raise
        self.waits += 1
        try:
            return self._q.get(timeout=self.acquire_timeout)
        except queue.Empty:
            self.timeouts += 1
            raise PoolTimeout(
                f"no connection within {self.acquire_timeout * 1000:.0f}ms (pool size {self.size})"
            ) from None

    @contextlib.contextmanager
    def conn(self):
        t0 = time.monotonic()
        c = self._get()
        if self.metrics:
            self.metrics.observe("db_pool_wait_ms", (time.monotonic() - t0) * 1000)
        with self._lock:
            self.in_use += 1
        try:
            yield c
        finally:
            with self._lock:
                self.in_use -= 1
            if c.in_transaction:
                c.rollback()
            self._q.put(c)

    @contextlib.contextmanager
    def tx(self):
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def query(self, sql, args=()):
        with self.conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        with self.conn() as c:
            r = c.execute(sql, args).fetchone()
            return dict(r) if r else None

    def execute(self, sql, args=()):
        with self.conn() as c:
            return c.execute(sql, args)


def connect(path: str, threads: bool = False) -> sqlite3.Connection:
    c = sqlite3.connect(path, timeout=5.0, isolation_level=None, check_same_thread=not threads)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c
