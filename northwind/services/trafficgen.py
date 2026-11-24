import json
import random
import threading
import time

from northwind.lib.http import Upstream, UpstreamError, current_request_id, new_request_id
from northwind.lib.service import Service

SEARCH_WORDS = ["lamp", "mug", "desk", "chair", "pro", "mini", "steel", "cotton", "kit", "x"]


def main():
    svc = Service("trafficgen")
    cfg = svc.config
    edge = Upstream("trafficgen", "edge", {**cfg["upstreams"]["edge"]}, svc.metrics, None)
    with open(svc.path(cfg.get("seed_file", "data/seed/manifest.json"))) as f:
        manifest = json.load(f)
    users = manifest["traffic_users"]
    skus = manifest["skus"]
    mix = cfg.get("mix", {"browse": 5, "search": 3, "checkout": 2, "bad_input": 1})
    tmin, tmax = cfg.get("think_time_ms", [300, 1200])
    stats: dict[str, dict] = {k: {"ok": 0, "failed": 0} for k in mix}
    lock = threading.Lock()

    def call(client, method, path, body=None, token=None, retry=True):
        headers = {"X-Client-Id": client}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if method == "POST" and path == "/api/checkout":
            headers["Idempotency-Key"] = new_request_id()
        for attempt in range(2 if retry else 1):
            tok = current_request_id.set(new_request_id())
            try:
                r = edge.request(method, path, body=body, headers=headers)
            except UpstreamError:
                r = None
            finally:
                current_request_id.reset(tok)
            if r is not None and r.status < 500 and r.status != 429:
                return r
            if attempt == 0 and retry:
                time.sleep(0.3)
        return r

    def journey(rng, kind, user):
        client = f"user-{user['id']}"
        if kind == "browse":
            r = call(client, "GET", f"/api/catalog/products?page={rng.randint(1, 3)}")
            if r is None or not r.ok:
                return False
            r = call(client, "GET", f"/api/catalog/products/{rng.choice(skus)}")
            return r is not None and r.ok
        if kind == "search":
            r = call(client, "GET", f"/api/catalog/search?q={rng.choice(SEARCH_WORDS)}")
            return r is not None and r.ok
        if kind == "checkout":
            r = call(
                client,
                "POST",
                "/api/auth/login",
                {"email": user["email"], "password": user["password"]},
            )
            if r is None or not r.ok:
                return False
            token = r.json()["token"]
            items = [
                {"sku": s, "qty": rng.randint(1, 2)} for s in rng.sample(skus, rng.randint(1, 3))
            ]
            r = call(client, "POST", "/api/checkout", {"items": items}, token=token)
            if r is None or r.status not in (200, 201):
                return False
            oid = r.json().get("order_id")
            r = call(client, "GET", f"/api/orders/{oid}", token=token)
            return r is not None and r.ok
        if kind == "bad_input":
            choice = rng.randint(0, 2)
            if choice == 0:
                r = call(
                    client,
                    "POST",
                    "/api/auth/login",
                    {"email": user["email"], "password": "wrong-password"},
                    retry=False,
                )
                return r is not None and r.status == 401
            if choice == 1:
                r = call(client, "GET", "/api/catalog/products/SKU-DOES-NOT-EXIST", retry=False)
                return r is not None and r.status == 404
            r = call(
                client,
                "POST",
                "/api/checkout",
                {"items": [{"sku": skus[0], "qty": 0}]},
                token="not-a-token",
                retry=False,
            )
            return r is not None and r.status in (400, 401)
        return False

    def worker(i):
        rng = random.Random(int(cfg.get("seed", 0)) * 1000 + i)
        kinds = list(mix)
        weights = [mix[k] for k in kinds]
        svc.stopping.wait(rng.uniform(0, 1.0))
        while not svc.stopping.is_set():
            if not cfg.get("enabled", True):
                svc.stopping.wait(1.0)
                continue
            kind = rng.choices(kinds, weights)[0]
            user = rng.choice(users)
            try:
                ok = journey(rng, kind, user)
            except Exception as e:
                ok = False
                svc.log.error("journey crashed", journey=kind, error=f"{type(e).__name__}: {e}")
            with lock:
                stats[kind]["ok" if ok else "failed"] += 1
            svc.metrics.inc(
                "trafficgen_journeys_total", journey=kind, outcome="ok" if ok else "failed"
            )
            if not ok:
                svc.log.info("journey failed", journey=kind, user=user["id"])
            svc.stopping.wait(rng.uniform(tmin, tmax) / 1000)

    for i in range(int(cfg.get("workers", 3))):
        svc.add_background(worker, i)

    @svc.app.route("GET", "/stats")
    def get_stats(req):
        with lock:
            return 200, {"journeys": stats}

    svc.run()


if __name__ == "__main__":
    main()
