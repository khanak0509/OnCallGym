"""Minimal JSON HTTP server and resilient HTTP client (stdlib only).

Every request carries X-Request-Id. The server stores it in a context variable so that
outgoing calls made while handling the request propagate the same id. Each handled
request writes one access log line (type=access) that telemetry uses to build traces.
"""

import contextvars
import http.client
import json
import queue
import re
import ssl
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .resilience import STATE_CODE, CircuitBreaker, backoff_delays

current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_request_id", default=None
)

QUIET_PATHS = {"/health", "/ready", "/metrics"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class Request:
    def __init__(self, method, path, query, headers, body, client):
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body
        self.client = client
        self.params: dict = {}
        self.request_id = None
        self._json = None

    @property
    def json(self):
        if self._json is None:
            if not self.body:
                self._json = {}
            else:
                self._json = json.loads(self.body)
        return self._json

    def arg(self, name, default=None):
        return self.query.get(name, default)


class HTTPError(Exception):
    def __init__(self, status: int, error: str, **extra):
        super().__init__(error)
        self.status = status
        self.body = {"error": error, **extra}


class App:
    def __init__(self, svc: str, log, metrics):
        self.svc = svc
        self.log = log
        self.metrics = metrics
        self.routes: list[tuple[str, re.Pattern, callable, str]] = []
        self.fallback = None  # fn(req) -> (route_label, result), used by the edge proxy
        self.quiet_paths = set(QUIET_PATHS)

    def route(self, method: str, pattern: str):
        """Register a handler; pattern uses {name} placeholders, e.g. /orders/{id}."""
        regex = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$")

        def deco(fn):
            self.routes.append((method, regex, fn, pattern))
            return fn

        return deco

    def dispatch(self, req: Request):
        allowed = False
        for method, regex, fn, pattern in self.routes:
            m = regex.match(req.path)
            if not m:
                continue
            allowed = True
            if method != req.method:
                continue
            req.params = m.groupdict()
            return pattern, fn(req)
        if allowed:
            return req.path, (405, {"error": "method not allowed"})
        if self.fallback:
            return self.fallback(req)
        return "unmatched", (404, {"error": "not found"})


def _make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True  # headers and body are separate writes
        server_version = f"northwind-{app.svc}"

        def log_message(self, *args):
            pass

        def _handle(self):
            t0 = time.monotonic()
            parts = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(parts.query).items()}
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            req = Request(self.command, parts.path, query, self.headers, body, self.client_address)
            req.request_id = self.headers.get("X-Request-Id") or new_request_id()
            token = current_request_id.set(req.request_id)
            route = "unmatched"
            try:
                route, result = app.dispatch(req)
            except HTTPError as e:
                result = (e.status, e.body)
            except json.JSONDecodeError:
                result = (400, {"error": "invalid json"})
            except Exception as e:
                app.log.error(
                    "unhandled error",
                    req=req.request_id,
                    path=req.path,
                    error=f"{type(e).__name__}: {e}",
                )
                result = (500, {"error": "internal error"})
            finally:
                current_request_id.reset(token)
            status, payload = result[0], result[1]
            extra_headers = result[2] if len(result) > 2 else {}
            if isinstance(payload, (dict, list)):
                data = json.dumps(payload, separators=(",", ":")).encode()
                ctype = "application/json"
            else:
                data = str(payload).encode()
                ctype = "text/plain; charset=utf-8"
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Request-Id", req.request_id)
                for k, v in extra_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            dur_ms = (time.monotonic() - t0) * 1000
            if req.path not in app.quiet_paths:
                app.metrics.inc("http_requests_total", route=route, status=str(status))
                app.metrics.observe("http_request_duration_ms", dur_ms, route=route)
                app.log.info(
                    "request",
                    req=req.request_id,
                    type="access",
                    method=req.method,
                    path=req.path,
                    status=status,
                    dur_ms=round(dur_ms, 1),
                    caller=self.headers.get("X-Caller", "client"),
                )

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


def make_server(app: App, host: str, port: int, tls: dict | None = None) -> Server:
    srv = Server((host, port), _make_handler(app))
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(tls["cert_file"], tls["key_file"])
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    return srv


def serve_in_thread(srv: Server) -> threading.Thread:
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    return t


class UpstreamError(Exception):
    def __init__(self, upstream: str, kind: str, detail: str = ""):
        super().__init__(f"{upstream}: {kind} {detail}".strip())
        self.upstream = upstream
        self.kind = kind
        self.detail = detail


class Response:
    def __init__(self, status: int, body: bytes, headers):
        self.status = status
        self.body = body
        self.headers = headers

    @property
    def ok(self):
        return 200 <= self.status < 300

    def json(self):
        try:
            return json.loads(self.body) if self.body else {}
        except json.JSONDecodeError:
            return {"error": "invalid json from upstream"}


class Upstream:
    """Client for one dependency: keep-alive pool, per-attempt timeout, retries with
    exponential backoff, optional circuit breaker. Config keys (all optional except url):
      url, timeout_ms, retries, backoff_ms, backoff_max_ms, pool_size, pool_timeout_ms,
      retry_on_status (list), breaker: {failure_threshold, window_s, reset_timeout_s}, ca_file
    """

    def __init__(self, caller: str, name: str, cfg: dict, metrics=None, log=None):
        self.caller = caller
        self.name = name
        self.cfg = cfg
        parts = urlsplit(cfg["url"])
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port
        self.base_path = parts.path.rstrip("/")
        self.timeout = float(cfg.get("timeout_ms", 2000)) / 1000.0
        self.retries = int(cfg.get("retries", 0))
        self.backoff_ms = float(cfg.get("backoff_ms", 50))
        self.backoff_max_ms = float(cfg.get("backoff_max_ms", 1000))
        self.retry_on_status = set(cfg.get("retry_on_status", [502, 503, 504]))
        self.pool_size = int(cfg.get("pool_size", 8))
        self.pool_timeout = float(cfg.get("pool_timeout_ms", cfg.get("timeout_ms", 2000))) / 1000
        self.metrics = metrics
        self.log = log
        self._pool: queue.LifoQueue = queue.LifoQueue()
        self._slots = threading.BoundedSemaphore(self.pool_size)
        self._in_use = 0
        self._in_use_lock = threading.Lock()
        self._ssl_ctx = None
        if self.scheme == "https":
            self._ssl_ctx = ssl.create_default_context(cafile=cfg.get("ca_file"))
        b = cfg.get("breaker")
        self.breaker = CircuitBreaker(**b) if b else None
        if self.breaker and log:
            self.breaker.on_change = lambda old, new: log.warn(
                "circuit breaker state change", upstream=name, old=old, new=new
            )
        if metrics:
            metrics.gauge_fn("upstream_pool_in_use", lambda: self._in_use, upstream=name)
            metrics.gauge_fn("upstream_pool_size", lambda: self.pool_size, upstream=name)
            if self.breaker:
                metrics.gauge_fn(
                    "circuit_breaker_state", lambda: STATE_CODE[self.breaker.state], upstream=name
                )

    def _new_conn(self):
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout, context=self._ssl_ctx
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _attempt(self, method, path, data, headers) -> Response:
        if not self._slots.acquire(timeout=self.pool_timeout):
            raise UpstreamError(self.name, "pool_exhausted", f"pool_size={self.pool_size}")
        with self._in_use_lock:
            self._in_use += 1
        conn = None
        try:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                conn = self._new_conn()
            try:
                conn.request(method, self.base_path + path, body=data, headers=headers)
                r = conn.getresponse()
                body = r.read()
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                # stale keep-alive connection: retry once on a fresh one
                conn.close()
                conn = self._new_conn()
                conn.request(method, self.base_path + path, body=data, headers=headers)
                r = conn.getresponse()
                body = r.read()
            resp = Response(r.status, body, r.headers)
            if r.will_close:
                conn.close()
            else:
                self._pool.put(conn)
            conn = None
            return resp
        except TimeoutError as e:
            raise UpstreamError(self.name, "timeout", f"after {self.timeout * 1000:.0f}ms") from e
        except (ConnectionRefusedError, ConnectionError, OSError, http.client.HTTPException) as e:
            raise UpstreamError(self.name, "connect", type(e).__name__) from e
        finally:
            if conn is not None:
                conn.close()
            with self._in_use_lock:
                self._in_use -= 1
            self._slots.release()

    def request(self, method: str, path: str, body=None, headers=None) -> Response:
        hdrs = {"Content-Type": "application/json", "X-Caller": self.caller}
        rid = current_request_id.get()
        if rid:
            hdrs["X-Request-Id"] = rid
        if headers:
            hdrs.update(headers)
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
        delays = list(backoff_delays(self.retries, self.backoff_ms, self.backoff_max_ms))
        attempt = 0
        while True:
            if self.breaker and not self.breaker.allow():
                self._count("breaker_open")
                raise UpstreamError(self.name, "breaker_open")
            t0 = time.monotonic()
            try:
                resp = self._attempt(method, path, data, hdrs)
            except UpstreamError as e:
                self._observe(t0)
                if e.kind != "pool_exhausted" and self.breaker:
                    self.breaker.failure()
                self._count(e.kind)
                if attempt < len(delays) and e.kind in ("timeout", "connect"):
                    self._retry_log(attempt, e.kind)
                    time.sleep(delays[attempt])
                    attempt += 1
                    continue
                raise
            self._observe(t0)
            if resp.status >= 500:
                if self.breaker:
                    self.breaker.failure()
                self._count(f"http_{resp.status}")
                if attempt < len(delays) and resp.status in self.retry_on_status:
                    self._retry_log(attempt, f"http_{resp.status}")
                    time.sleep(delays[attempt])
                    attempt += 1
                    continue
                return resp
            if self.breaker:
                self.breaker.success()
            self._count("ok")
            return resp

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)

    def _count(self, outcome):
        if self.metrics:
            self.metrics.inc("upstream_requests_total", upstream=self.name, outcome=outcome)

    def _observe(self, t0):
        if self.metrics:
            self.metrics.observe(
                "upstream_duration_ms", (time.monotonic() - t0) * 1000, upstream=self.name
            )

    def _retry_log(self, attempt, reason):
        if self.metrics:
            self.metrics.inc("upstream_retries_total", upstream=self.name)
        if self.log:
            self.log.warn(
                "retrying upstream call",
                req=current_request_id.get(),
                upstream=self.name,
                attempt=attempt + 1,
                reason=reason,
            )
