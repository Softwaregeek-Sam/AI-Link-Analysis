import asyncio
import json
import sys
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.utils.validators import BlockedURLError, is_safe_host

MAX_REDIRECTS = 3
MAX_BYTES = 2 * 1024 * 1024  # 2 MB
REQUEST_TIMEOUT = 10  # seconds, per network operation
TOTAL_TIMEOUT = 15  # seconds, whole fetch including redirects
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
BLOCKED_STATUSES = {403, 429, 503}
HTML_TYPES = {"text/html", "application/xhtml+xml"}
MAX_HEADINGS = 30
MAX_LINKS = 30
MAX_JSON_LD = 15
MAX_BODY_CHARS = 12000
THIN_CONTENT_CHARS = 300  # less visible text than this usually means a JavaScript-rendered page
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class FetchError(Exception):
    """Base class for problems while fetching a website."""


class FetchTimeout(FetchError):
    pass


class FetchConnectionError(FetchError):
    pass


class SiteBlocked(FetchError):
    """The site refused automated access (403, 429 or 503)."""


class BadStatus(FetchError):
    """Any other non-2xx response."""


class NotHTML(FetchError):
    pass


class TooManyRedirects(FetchError):
    pass


async def _ensure_allowed(url: str) -> None:
    """Scheme, port and address checks. Runs for the first URL and every redirect."""
    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError:
        raise BlockedURLError("URL is not valid")
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURLError("Only http and https URLs are allowed")
    if "\\" in url or parts.username or parts.password:
        raise BlockedURLError("URL is not allowed")
    if port not in (None, 80, 443):
        raise BlockedURLError("Only ports 80 and 443 are allowed")
    if not await asyncio.to_thread(is_safe_host, url):
        raise BlockedURLError("URL points to a private or internal address")


async def _read_capped(resp: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():  # decoded bytes, so the cap also limits gzip bombs
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_BYTES:
            break
    return b"".join(chunks)[:MAX_BYTES]


def _decode(body: bytes, resp: httpx.Response) -> str:
    try:
        return body.decode(resp.charset_encoding or "utf-8", errors="replace")
    except LookupError:  # unknown charset name
        return body.decode("utf-8", errors="replace")


def _check_response(resp: httpx.Response) -> None:
    status = resp.status_code
    if status in BLOCKED_STATUSES:
        raise SiteBlocked(f"The site blocked automated access (HTTP {status})")
    if not 200 <= status < 300:
        raise BadStatus(f"The site returned HTTP {status}")
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type and content_type not in HTML_TYPES:
        raise NotHTML(f"Expected an HTML page but got {content_type}")


async def _fetch(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=REQUEST_TIMEOUT, headers=HEADERS
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _ensure_allowed(url)
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in REDIRECT_STATUSES:
                        location = resp.headers.get("location")
                        if not location:
                            raise BadStatus("Redirect without a Location header")
                        url = urljoin(url, location)
                        continue
                    _check_response(resp)
                    body = await _read_capped(resp)
                    return url, _decode(body, resp)
            except httpx.TimeoutException:
                raise FetchTimeout("The website took too long to respond")
            except (httpx.RequestError, httpx.InvalidURL) as exc:
                raise FetchConnectionError(f"Could not connect to the website ({type(exc).__name__})")
    raise TooManyRedirects(f"More than {MAX_REDIRECTS} redirects")


async def fetch_html(url: str) -> tuple[str, str]:
    """Fetch a page safely. Returns (final_url, html) or raises a FetchError / BlockedURLError."""
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT):
            return await _fetch(url)
    except TimeoutError:
        raise FetchTimeout("The website took too long to respond")


# ---------------------------------------------------------------- extraction


def _clean(text: str | None) -> str | None:
    text = (text or "").strip()
    return text or None


def _json_ld_blocks(soup: BeautifulSoup) -> list:
    blocks: list = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text())
        except (json.JSONDecodeError, ValueError):
            continue  # broken JSON-LD is common, just skip it
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                blocks.extend(item["@graph"])  # e.g. Yoast puts everything in @graph
            else:
                blocks.append(item)
    return blocks[:MAX_JSON_LD]


def _links(soup: BeautifulSoup, selector: str, base_url: str, keep_contact: bool) -> list[dict]:
    links: list[dict] = []
    seen: set = set()
    for a in soup.select(selector):
        href = (a.get("href") or "").strip()
        lowered = href.lower()
        if not href or href == "#" or lowered.startswith("javascript:"):
            continue
        is_contact = lowered.startswith(("mailto:", "tel:"))
        if is_contact and not keep_contact:
            continue
        # icon-only links (social media) have no text, so fall back to their labels
        text = a.get_text(" ", strip=True) or (a.get("aria-label") or a.get("title") or "").strip()
        if not text:
            continue
        if base_url and not is_contact:
            href = urljoin(base_url, href)
        if (text, href) in seen:
            continue
        seen.add((text, href))
        links.append({"text": text, "href": href})
        if len(links) >= MAX_LINKS:
            break
    return links


def extract(html: str, base_url: str = "") -> dict:
    """Turn raw HTML into a compact dict for the LLM. Never raises on odd or empty HTML."""
    soup = BeautifulSoup(html or "", "lxml")

    # meta tags, keyed by lowercase name/property (first one wins)
    metas: dict = {}
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or "").strip().lower()
        content = _clean(tag.get("content"))
        if key and content and key not in metas:
            metas[key] = content

    title = _clean(soup.title.get_text()) if soup.title else None
    html_tag = soup.find("html")
    canonical_tag = soup.find("link", rel="canonical")
    canonical = _clean(canonical_tag.get("href")) if canonical_tag else None
    if canonical and base_url:
        canonical = urljoin(base_url, canonical)

    json_ld = _json_ld_blocks(soup)  # must run before scripts are removed

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    headings = [h for h in headings if h][:MAX_HEADINGS]

    lines: list[str] = []
    seen: set = set()
    for line in (soup.body or soup).get_text("\n", strip=True).split("\n"):
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    body_text = "\n".join(lines)[:MAX_BODY_CHARS]

    images = soup.find_all("img")
    meta_description = metas.get("description")

    return {
        "title": title,
        "meta_description": meta_description,
        "canonical": canonical,
        "lang": _clean(html_tag.get("lang")) if html_tag else None,
        "og_site_name": metas.get("og:site_name"),
        "og_title": metas.get("og:title"),
        "og_description": metas.get("og:description"),
        "json_ld": json_ld,
        "headings": headings,
        "nav_links": _links(soup, "nav a[href], header a[href]", base_url, keep_contact=False),
        "footer_links": _links(soup, "footer a[href]", base_url, keep_contact=True),
        "body_text": body_text,
        "seo_signals": {
            "title_length": len(title) if title else 0,
            "has_meta_description": bool(meta_description),
            "h1_count": len(soup.find_all("h1")),
            "images_total": len(images),
            "images_missing_alt": sum(1 for img in images if not (img.get("alt") or "").strip()),
            "has_schema": bool(json_ld),
        },
        "thin_content": len(body_text) < THIN_CONTENT_CHARS,
    }


if __name__ == "__main__":
    # Manual check: python -m app.services.website_extractor https://example.com
    from app.utils.validators import validate_url

    async def _show(raw_url: str) -> None:
        url = validate_url(raw_url)
        final_url, html = await fetch_html(url)
        print(json.dumps(extract(html, final_url), indent=2, ensure_ascii=False))

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on non-English text otherwise
    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.services.website_extractor <url>")
    try:
        asyncio.run(_show(sys.argv[1]))
    except Exception as exc:
        sys.exit(f"{type(exc).__name__} -> {exc}")  # errors go to stderr so JSON redirects stay clean