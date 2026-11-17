import sqlite3
import threading
import time

import pytest

from northwind.lib import jwt
from northwind.lib.db import Pool, PoolTimeout
from northwind.lib.log import format_line, parse_line
from northwind.lib.metrics import Registry, parse_text
from northwind.lib.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, TokenBucket


def test_logfmt_roundtrip():
    line = format_line(
        "warn",
        "orders",
        'checkout "failed"',
        {"req": "abc", "reason": "a b=c", "n": 3, "skip": None},
    )
    rec = parse_line(line)
    assert rec["level"] == "warn" and rec["svc"] == "orders" and rec["req"] == "abc"
    assert rec["msg"] == 'checkout "failed"'
    assert rec["reason"] == "a b=c" and rec["n"] == "3" and "skip" not in rec


def test_metrics_render_and_parse():
    m = Registry()
    m.inc("http_requests_total", route="/x", status="200")
    m.inc("http_requests_total", route="/x", status="200")
    m.observe("lat_ms", 7, route="/x")
    m.gauge_fn("pool_in_use", lambda: 3, db="orders")
    samples = parse_text(m.render())
    got = {(n, tuple(sorted(lab.items()))): v for n, lab, v in samples}
    assert got[("http_requests_total", (("route", "/x"), ("status", "200")))] == 2
    assert got[("pool_in_use", (("db", "orders"),))] == 3
    assert got[("lat_ms_bucket", (("le", "10"), ("route", "/x")))] == 1
    assert got[("lat_ms_bucket", (("le", "5"), ("route", "/x")))] == 0


def test_breaker_opens_and_recovers():
    b = CircuitBreaker(failure_threshold=3, window_s=10, reset_timeout_s=0.1)
    for _ in range(3):
        assert b.allow()
        b.failure()
    assert b.state == OPEN and not b.allow()
    time.sleep(0.12)
    assert b.allow() and b.state == HALF_OPEN
    assert not b.allow()  # only one trial call
    b.success()
    assert b.state == CLOSED


def test_breaker_half_open_failure_reopens():
    b = CircuitBreaker(failure_threshold=1, reset_timeout_s=0.05)
    b.failure()
    time.sleep(0.06)
    assert b.allow()
    b.failure()
    assert b.state == OPEN


def test_token_bucket():
    tb = TokenBucket(rate_per_s=0.0, burst=2)
    assert tb.take() and tb.take() and not tb.take()


def test_jwt_sign_verify_and_rotation():
    k1, k2 = b"a" * 32, b"b" * 32
    tok = jwt.sign({"sub": 1, "exp": time.time() + 60}, k1, "k1")
    assert jwt.verify(tok, {"k1": k1})["sub"] == 1
    with pytest.raises(jwt.TokenError, match="unknown signing key"):
        jwt.verify(tok, {"k2": k2})
    with pytest.raises(jwt.TokenError, match="bad signature"):
        jwt.verify(tok, {"k1": k2})
    old = jwt.sign({"sub": 1, "exp": time.time() - 1}, k1, "k1")
    with pytest.raises(jwt.TokenError, match="expired"):
        jwt.verify(old, {"k1": k1})


def test_pool_times_out_when_exhausted(tmp_path):
    path = str(tmp_path / "x.db")
    sqlite3.connect(path).close()
    pool = Pool(path, size=1, acquire_timeout_ms=50)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with pool.conn():
            held.set()
            release.wait(2)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(2)
    with pytest.raises(PoolTimeout):
        with pool.conn():
            pass
    release.set()
    t.join()
    with pool.conn() as c:
        assert c.execute("SELECT 1").fetchone()[0] == 1
    assert pool.timeouts == 1


def test_pool_tx_rolls_back(tmp_path):
    pool = Pool(str(tmp_path / "y.db"), size=2)
    pool.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(RuntimeError):
        with pool.tx() as c:
            c.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("boom")
    assert pool.one("SELECT COUNT(*) AS n FROM t")["n"] == 0
