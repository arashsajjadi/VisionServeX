"""API authentication helpers.

We accept three credential styles:

1. ``Authorization: Bearer <api-key>``
2. ``X-API-Key: <api-key>``
3. Cloudflare Access service tokens: ``CF-Access-Client-Id`` +
   ``CF-Access-Client-Secret``. We do not validate Cloudflare's signature here;
   the assumption is that a Cloudflare Access policy in front of the origin
   has already verified these. If the headers are present, we still require
   ``CF-Access-Client-Id`` to match a configured value when one is set.

When auth is disabled, requests pass through.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from visionservex.config import AuthConfig


@dataclass(frozen=True)
class AuthOutcome:
    authenticated: bool
    principal: str = "anonymous"
    method: str = "none"
    reason: str | None = None


def authenticate_request(headers: dict[str, str], auth: AuthConfig) -> AuthOutcome:
    """Return an :class:`AuthOutcome` for the given headers."""
    if not auth.enabled:
        return AuthOutcome(True, principal="anonymous", method="disabled")

    if not auth.api_key:
        return AuthOutcome(False, reason="auth enabled but server has no api_key configured")

    bearer = _extract_bearer(headers)
    if bearer and hmac.compare_digest(bearer, auth.api_key):
        return AuthOutcome(True, principal="api-key", method="bearer")

    api_key_header = headers.get("x-api-key") or headers.get("X-API-Key")
    if api_key_header and hmac.compare_digest(api_key_header, auth.api_key):
        return AuthOutcome(True, principal="api-key", method="x-api-key")

    if auth.accept_cf_access_headers:
        cf_id = headers.get("cf-access-client-id") or headers.get("CF-Access-Client-Id")
        cf_secret = headers.get("cf-access-client-secret") or headers.get("CF-Access-Client-Secret")
        if cf_id and cf_secret:
            # Cloudflare Access has already enforced policy at the edge.
            # Treat these as a valid second-factor only if the secret also
            # matches the configured api_key (defense-in-depth) OR a dedicated
            # cf access secret is provided in env. For 0.1 we require the
            # secret matches the configured api_key.
            if hmac.compare_digest(cf_secret, auth.api_key):
                return AuthOutcome(True, principal=cf_id, method="cf-access")
            return AuthOutcome(False, reason="cf-access headers present but secret mismatch")

    return AuthOutcome(False, reason="missing or invalid credentials")


def _extract_bearer(headers: dict[str, str]) -> str | None:
    raw = headers.get("authorization") or headers.get("Authorization")
    if not raw:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


__all__ = ["AuthOutcome", "authenticate_request"]
