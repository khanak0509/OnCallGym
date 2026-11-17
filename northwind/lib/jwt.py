"""HS256 JSON Web Tokens with a key id (kid) header, stdlib only."""

import base64
import hashlib
import hmac
import json
import time


class TokenError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign(claims: dict, key: bytes, kid: str) -> str:
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    h = _b64(json.dumps(header, separators=(",", ":")).encode())
    p = _b64(json.dumps(claims, separators=(",", ":")).encode())
    sig = hmac.new(key, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64(sig)}"


def verify(token: str, keys: dict[str, bytes], now: float | None = None) -> dict:
    """Verify signature against the key named by the token's kid; check exp."""
    try:
        h, p, s = token.split(".")
        header = json.loads(_unb64(h))
        claims = json.loads(_unb64(p))
    except Exception as e:
        raise TokenError("malformed token") from e
    kid = header.get("kid")
    key = keys.get(kid)
    if key is None:
        raise TokenError(f"unknown signing key kid={kid}")
    expected = hmac.new(key, f"{h}.{p}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _unb64(s)):
        raise TokenError("bad signature")
    if claims.get("exp", 0) < (now or time.time()):
        raise TokenError("token expired")
    return claims


def load_keys(entries: list[dict]) -> dict[str, bytes]:
    """entries: [{kid, file}] -> {kid: key bytes}. Missing files are skipped."""
    keys = {}
    for e in entries or []:
        try:
            with open(e["file"], "rb") as f:
                keys[e["kid"]] = f.read().strip()
        except OSError:
            continue
    return keys
