"""SSRF protection for remote URL inputs."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class URLValidationError(ValueError):
    pass


_FORBIDDEN_HOSTS = {"localhost", "metadata.google.internal", "metadata"}
_FORBIDDEN_SCHEMES_DEFAULT = {"file", "ftp", "gopher", "data"}


def validate_remote_url(
    url: str,
    *,
    allow_http: bool = False,
    allowlist_hosts: list[str] | None = None,
) -> str:
    """Reject URLs that point to private or sensitive resources.

    Returns the canonical URL on success; raises :class:`URLValidationError`
    otherwise.
    """
    if not url or len(url) > 2048:
        raise URLValidationError("url is empty or too long")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme in _FORBIDDEN_SCHEMES_DEFAULT or not scheme:
        raise URLValidationError(f"scheme {scheme!r} is not allowed")

    if scheme == "http" and not allow_http:
        raise URLValidationError("http is not allowed (require https)")
    if scheme not in {"http", "https"}:
        raise URLValidationError(f"scheme {scheme!r} is not allowed")

    if "@" in (parsed.netloc or ""):
        raise URLValidationError("userinfo in URL is not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise URLValidationError("missing host")

    if host in _FORBIDDEN_HOSTS:
        raise URLValidationError(f"host {host!r} is not allowed")

    if allowlist_hosts and host not in {h.lower() for h in allowlist_hosts}:
        raise URLValidationError(f"host {host!r} is not in the allowlist")

    # Resolve all addresses and reject if any is private/loopback/link-local.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise URLValidationError(f"could not resolve host {host!r}") from exc

    for _family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise URLValidationError(f"host {host!r} resolves to disallowed address {ip}")

    return parsed.geturl()


__all__ = ["URLValidationError", "validate_remote_url"]
