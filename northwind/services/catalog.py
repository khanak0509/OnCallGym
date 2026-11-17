"""Products, search and pricing, with a read-through cache in front of catalog.db."""

import json

from northwind.lib.http import UpstreamError
from northwind.lib.service import Service, require


def main():
    svc = Service("catalog")
    svc.use_flags()
    db = svc.database("catalog")
    cache = svc.upstream("cache")
    ccfg = svc.config.get("cache", {})
    cache_enabled = bool(ccfg.get("enabled", True))
    ttl = int(ccfg.get("ttl_s", 30))
    cols = "sku, name, category, description, price_cents"

    def price(p):
        discount = float(svc.flag("promo_discount_pct", 0) or 0)
        if discount and p["category"] in (svc.flag("promo_categories") or []):
            p = {**p, "price_cents": int(p["price_cents"] * (100 - discount) / 100)}
        return p

    def cached(key, load):
        if cache_enabled:
            try:
                r = cache.get(f"/kv/{key}")
                if r.status == 200:
                    svc.metrics.inc("catalog_cache_total", outcome="hit")
                    return json.loads(r.json()["value"])
            except UpstreamError as e:
                svc.metrics.inc("catalog_cache_total", outcome="error")
                svc.log.warn("cache read failed, falling back to db", error=str(e)[:120])
        svc.metrics.inc("catalog_cache_total", outcome="miss")
        value = load()
        if cache_enabled and value is not None:
            try:
                r = cache.request(
                    "PUT", f"/kv/{key}", body={"value": json.dumps(value), "ttl_s": ttl}
                )
                if not r.ok:
                    svc.metrics.inc("catalog_cache_write_errors_total")
                    svc.log.warn(
                        "cache write rejected",
                        key=key,
                        status=r.status,
                        error=r.json().get("error"),
                    )
            except UpstreamError as e:
                svc.log.warn("cache write failed", error=str(e)[:120])
        return value

    @svc.app.route("GET", "/products")
    def list_products(req):
        page = max(1, int(req.arg("page", 1)))
        category = req.arg("category")

        def load():
            if category:
                return db.query(
                    f"SELECT {cols} FROM products WHERE active=1 AND category=? "
                    "ORDER BY sku LIMIT 20 OFFSET ?",
                    (category, (page - 1) * 20),
                )
            return db.query(
                f"SELECT {cols} FROM products WHERE active=1 ORDER BY sku LIMIT 20 OFFSET ?",
                ((page - 1) * 20,),
            )

        rows = cached(f"products:{category or 'all'}:{page}", load)
        return 200, {"page": page, "products": [price(p) for p in rows]}

    @svc.app.route("GET", "/products/{sku}")
    def get_product(req):
        sku = req.params["sku"]
        p = cached(
            f"product:{sku}",
            lambda: db.one(f"SELECT {cols} FROM products WHERE sku=? AND active=1", (sku,)),
        )
        require(p, 404, "product not found", sku=sku)
        return 200, price(p)

    @svc.app.route("GET", "/search")
    def search(req):
        q = (req.arg("q") or "").strip().lower()
        require(q, 400, "missing q")
        require(len(q) <= 64, 400, "query too long")
        rows = db.query(
            f"SELECT {cols} FROM products WHERE active=1 AND "
            "(lower(name) LIKE ? OR lower(description) LIKE ?) ORDER BY sku LIMIT 20",
            (f"%{q}%", f"%{q}%"),
        )
        return 200, {"q": q, "results": [price(p) for p in rows]}

    svc.run()


if __name__ == "__main__":
    main()
