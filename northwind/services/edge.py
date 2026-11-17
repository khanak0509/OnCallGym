"""Edge reverse proxy: routing by path prefix, per-client rate limiting, request ids,
per-route timeouts, TLS to auth.

etc/edge.yaml:
  routes: [{prefix: /api/auth/login, upstream: auth, rewrite: /login}]
  rate_limit: {rps: 40, burst: 80}   # per client (X-Client-Id header, else peer address)
"""

import threading
from urllib.parse import urlencode

from northwind.lib.http import UpstreamError
from northwind.lib.resilience import TokenBucket
from northwind.lib.service import Service

FORWARD_HEADERS = ("Authorization", "Idempotency-Key", "Content-Type")


def main():
    svc = Service("edge")
    cfg = svc.config
    ups = {name: svc.upstream(name) for name in cfg["upstreams"]}
    routes = sorted(cfg["routes"], key=lambda r: -len(r["prefix"]))
    rl = cfg.get("rate_limit", {})
    rps, burst = float(rl.get("rps", 40)), float(rl.get("burst", 80))
    buckets: dict[str, TokenBucket] = {}
    lock = threading.Lock()

    def bucket(client):
        with lock:
            b = buckets.get(client)
            if b is None:
                if len(buckets) > 10_000:
                    buckets.clear()
                b = buckets[client] = TokenBucket(rps, burst)
            return b

    def proxy(req):
        route = next((r for r in routes if req.path.startswith(r["prefix"])), None)
        if route is None:
            return "unmatched", (404, {"error": "no route"})
        label = route["prefix"]
        client = req.headers.get("X-Client-Id") or req.client[0]
        if not bucket(client).take():
            svc.metrics.inc("edge_rate_limited_total", route=label)
            svc.log.warn("rate limit exceeded", req=req.request_id, client=client, route=label)
            return label, (429, {"error": "rate limit exceeded"}, {"Retry-After": "1"})
        path = route.get("rewrite", "/") + req.path[len(route["prefix"]) :]
        if req.query:
            path += "?" + urlencode(req.query)
        headers = {h: req.headers[h] for h in FORWARD_HEADERS if req.headers.get(h)}
        up = ups[route["upstream"]]
        try:
            r = up.request(req.method, path, body=req.body or None, headers=headers)
        except UpstreamError as e:
            status = 504 if e.kind == "timeout" else 502
            svc.log.error(
                "upstream request failed",
                req=req.request_id,
                upstream=up.name,
                reason=e.kind,
                path=req.path,
            )
            return label, (status, {"error": f"upstream {up.name} {e.kind}"})
        body = r.json()
        extra = {}
        if r.headers.get("Retry-After"):
            extra["Retry-After"] = r.headers["Retry-After"]
        return label, (r.status, body, extra)

    svc.app.fallback = proxy
    svc.run()


if __name__ == "__main__":
    main()
