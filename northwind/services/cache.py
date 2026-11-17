"""Small KV cache with a memory budget and an eviction policy.

etc/cache.yaml:
  max_memory_bytes: 4000000
  eviction_policy: allkeys-lru   # or noeviction (writes fail with 507 when full)
  default_ttl_s: 60
"""

import threading
import time
from collections import OrderedDict

from northwind.lib.service import Service, require


def main():
    svc = Service("cache")
    cfg = svc.config
    max_bytes = int(cfg.get("max_memory_bytes", 4_000_000))
    policy = cfg.get("eviction_policy", "allkeys-lru")
    default_ttl = float(cfg.get("default_ttl_s", 60))
    data: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
    lock = threading.Lock()
    stats = {"used": 0, "hits": 0, "misses": 0, "evictions": 0, "rejected": 0, "expired": 0}

    def size(k, v):
        return len(k) + len(v) + 64

    def drop(k):
        v, _ = data.pop(k)
        stats["used"] -= size(k, v)

    def expire_now():
        now = time.time()
        with lock:
            for k in [k for k, (_, exp) in data.items() if exp <= now]:
                drop(k)
                stats["expired"] += 1

    m = svc.metrics
    m.gauge_fn("cache_used_bytes", lambda: stats["used"])
    m.gauge_fn("cache_max_bytes", lambda: max_bytes)
    m.gauge_fn("cache_keys", lambda: len(data))
    for key in ("hits", "misses", "evictions", "rejected", "expired"):
        m.gauge_fn(f"cache_{key}_total", lambda key=key: stats[key])
    svc.every(5.0, expire_now, "expire")

    @svc.app.route("GET", "/kv/{key}")
    def get(req):
        k = req.params["key"]
        with lock:
            item = data.get(k)
            if item is None or item[1] <= time.time():
                if item is not None:
                    drop(k)
                stats["misses"] += 1
                return 404, {"error": "miss"}
            data.move_to_end(k)
            stats["hits"] += 1
            return 200, {"key": k, "value": item[0].decode()}

    @svc.app.route("PUT", "/kv/{key}")
    def put(req):
        k = req.params["key"]
        body = req.json
        require("value" in body, 400, "missing value")
        v = str(body["value"]).encode()
        ttl = float(body.get("ttl_s", default_ttl))
        need = size(k, v)
        with lock:
            if k in data:
                drop(k)
            if need > max_bytes:
                stats["rejected"] += 1
                return 507, {"error": "value larger than max_memory_bytes"}
            while stats["used"] + need > max_bytes:
                if policy == "noeviction" or not data:
                    stats["rejected"] += 1
                    svc.log.warn(
                        "OOM command not allowed when used memory > max_memory_bytes",
                        used=stats["used"],
                        max=max_bytes,
                        policy=policy,
                    )
                    return 507, {"error": "cache full", "policy": policy}
                oldest = next(iter(data))
                drop(oldest)
                stats["evictions"] += 1
            data[k] = (v, time.time() + ttl)
            stats["used"] += need
        return 200, {"stored": True}

    @svc.app.route("DELETE", "/kv/{key}")
    def delete(req):
        with lock:
            if req.params["key"] in data:
                drop(req.params["key"])
        return 200, {"deleted": True}

    @svc.app.route("GET", "/stats")
    def get_stats(req):
        return 200, {**stats, "keys": len(data), "max_memory_bytes": max_bytes, "policy": policy}

    svc.run()


if __name__ == "__main__":
    main()
