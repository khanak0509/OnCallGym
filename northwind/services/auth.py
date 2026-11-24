import hashlib
import hmac
import secrets
import time

from northwind.lib import jwt
from northwind.lib.http import UpstreamError
from northwind.lib.service import Service, require


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + ":" + password).encode()).hexdigest()


def main():
    svc = Service("auth")
    svc.use_flags()
    db = svc.database("users")
    sessions = svc.upstream("cache")
    sk = svc.config["signing_key"]
    keys = jwt.load_keys([{"kid": sk["kid"], "file": svc.path(sk["file"])}])
    signing_key = keys.get(sk["kid"])
    svc.ready_checks.append(
        lambda: (signing_key is not None, f"signing key file missing: {sk['file']}")
    )
    verify_keys = jwt.load_keys(
        [{"kid": k["kid"], "file": svc.path(k["file"])} for k in svc.config.get("verify_keys", [])]
        or []
    )
    verify_keys.update(keys)

    @svc.app.route("POST", "/login")
    def login(req):
        b = req.json
        email, password = b.get("email"), b.get("password")
        require(email and password, 400, "email and password required")
        user = db.one("SELECT * FROM users WHERE email=?", (email,))
        ok = user is not None and hmac.compare_digest(
            user["password_hash"], hash_password(password, user["salt"])
        )
        with db.tx() as c:
            c.execute(
                "INSERT INTO login_events (user_id, email, ok, ts) VALUES (?, ?, ?, ?)",
                (user["id"] if user else None, email, int(ok), time.time()),
            )
        if not ok:
            svc.metrics.inc("auth_logins_total", outcome="bad_credentials")
            svc.log.info("login failed", req=req.request_id, reason="bad credentials")
            return 401, {"error": "invalid credentials"}
        ttl = int(svc.flag("token_ttl_s", 900))
        now = int(time.time())
        sid = secrets.token_hex(12)
        try:
            r = sessions.request(
                "PUT", f"/kv/session:{sid}", body={"value": str(user["id"]), "ttl_s": ttl}
            )
            stored = r.ok
            err = None if r.ok else r.json().get("error")
        except UpstreamError as e:
            stored, err = False, e.kind
        if not stored:
            svc.metrics.inc("auth_logins_total", outcome="session_store_error")
            svc.log.error("login failed: cannot store session", req=req.request_id, error=err)
            return 503, {"error": "session store unavailable"}
        token = jwt.sign(
            {
                "sub": user["id"],
                "email": user["email"],
                "iat": now,
                "exp": now + ttl,
                "iss": "northwind-auth",
            },
            signing_key,
            sk["kid"],
        )
        svc.metrics.inc("auth_logins_total", outcome="ok")
        return 200, {"token": token, "expires_in": ttl, "user_id": user["id"]}

    @svc.app.route("POST", "/verify", internal=True)
    def verify(req):
        token = req.json.get("token", "")
        try:
            claims = jwt.verify(token, verify_keys)
        except jwt.TokenError as e:
            return 401, {"valid": False, "error": str(e)}
        return 200, {"valid": True, "claims": claims}

    @svc.app.route("GET", "/users/{id}", internal=True)
    def get_user(req):
        user = db.one(
            "SELECT id, email, name, address FROM users WHERE id=?", (int(req.params["id"]),)
        )
        require(user, 404, "user not found")
        return 200, user

    svc.run()


if __name__ == "__main__":
    main()
