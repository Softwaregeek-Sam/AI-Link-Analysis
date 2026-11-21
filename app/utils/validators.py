import ipaddress
import socket
from urllib.parse import urlparse

MAX_URL_LENGTH = 2000
ALLOWED_PORTS = {80, 443}


class InvalidURLError(Exception):
    """Bad syntax, scheme or port, or the domain does not exist."""


class BlockedURLError(Exception):
    """The URL points to a private or internal address."""


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%")[0])  # drop IPv6 zone id if present
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 must be judged as 127.0.0.1
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local  # includes 169.254.169.254 (cloud metadata)
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


def _host_status(host: str) -> str:
    """Return "ok", "not_found" or "blocked" for a hostname or IP literal."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return "not_found"
    if not infos:
        return "not_found"
    for info in infos:
        if not _is_public_ip(info[4][0]):
            return "blocked"  # any non-public answer blocks the whole host
    return "ok"


def is_safe_host(url: str) -> bool:
    """True only if the URL's host resolves and every resolved IP is public."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return bool(host) and _host_status(host) == "ok"


def validate_url(raw: str) -> str:
    """Return a normalized URL, or raise InvalidURLError / BlockedURLError."""
    url = (raw or "").strip()
    if not url:
        raise InvalidURLError("URL is empty")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError("URL is too long")
    if "://" not in url:
        url = "https://" + url

    # Backslashes and user:pass@ make parsers disagree about the real host.
    if "\\" in url:
        raise InvalidURLError("Invalid URL")
    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError:
        raise InvalidURLError("Invalid URL")
    if parts.scheme not in ("http", "https"):
        raise InvalidURLError("Only http and https URLs are allowed")
    if not parts.hostname:
        raise InvalidURLError("URL has no hostname")
    if parts.username or parts.password:
        raise InvalidURLError("URLs with credentials are not allowed")
    if port is not None and port not in ALLOWED_PORTS:
        raise InvalidURLError("Only ports 80 and 443 are allowed")

    # Note: DNS is resolved here and again by httpx later, so DNS rebinding
    # is not fully prevented (no IP pinning, by design).
    status = _host_status(parts.hostname)
    if status == "not_found":
        raise InvalidURLError("Domain not found")
    if status == "blocked":
        raise BlockedURLError("URL points to a private or internal address")
    return url