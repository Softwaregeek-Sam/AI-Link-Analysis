import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import llm_service, website_extractor
from app.services.llm_service import LLMError
from app.services.website_extractor import (
    BadStatus,
    FetchConnectionError,
    FetchTimeout,
    NotHTML,
    SiteBlocked,
    TooManyRedirects,
    extract,
    fetch_html,
)
from app.utils.validators import BlockedURLError, InvalidURLError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://127.0.0.1",
        "http://169.254.169.254",
        "http://10.0.0.1",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",
    ],
)
def test_rejects_private_and_internal_urls(url):
    with pytest.raises(BlockedURLError):
        validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "http://example.com:8080",
        "https://example.com/" + "a" * 3000,
        "https://nonexistent-domain-xyz.invalid",
    ],
)
def test_rejects_invalid_urls(url):
    with pytest.raises(InvalidURLError):
        validate_url(url)


def _mock_http(monkeypatch, handler):
    """Make fetch_html talk to a fake server instead of the network."""
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        website_extractor.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )


def test_redirect_to_internal_ip_is_blocked(monkeypatch):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    _mock_http(monkeypatch, handler)
    monkeypatch.setattr(website_extractor, "is_safe_host", lambda url: "169.254.169.254" not in url)
    with pytest.raises(BlockedURLError):
        asyncio.run(fetch_html("https://public.example/"))


@pytest.mark.parametrize(
    "status, headers, error",
    [
        (403, {}, SiteBlocked),
        (404, {}, BadStatus),
        (200, {"content-type": "application/pdf"}, NotHTML),
    ],
)
def test_fetch_error_mapping(monkeypatch, status, headers, error):
    _mock_http(monkeypatch, lambda request: httpx.Response(status, headers=headers, content=b"x"))
    monkeypatch.setattr(website_extractor, "is_safe_host", lambda url: True)
    with pytest.raises(error):
        asyncio.run(fetch_html("https://public.example/"))


@pytest.mark.parametrize(
    "error, status",
    [
        (InvalidURLError("x"), 422),
        (BlockedURLError("x"), 400),
        (FetchTimeout("x"), 504),
        (FetchConnectionError("x"), 502),
        (SiteBlocked("x"), 502),
        (BadStatus("x"), 502),
        (TooManyRedirects("x"), 502),
        (NotHTML("x"), 415),
        (LLMError("x"), 502),
        (RuntimeError("boom"), 500),
    ],
)
def test_error_status_mapping(monkeypatch, error, status):
    async def fake_analyze(url):
        raise error

    monkeypatch.setattr(main, "analyze_url", fake_analyze)
    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post("/link-analysis", json={"url": "https://example.com"})
    body = response.json()
    assert response.status_code == status
    assert body["success"] is False and body["error"] and body["detail"]


def test_missing_url_returns_422():
    response = TestClient(main.app).post("/link-analysis", json={})
    assert response.status_code == 422
    assert response.json()["success"] is False


NORMAL_PAGE = """
<html lang="en"><head>
  <title>Acme Web Studio | Websites for Small Businesses</title>
  <meta name="Description" content="We build affordable websites for small businesses.">
  <meta property="og:site_name" content="Acme Web Studio">
  <meta property="og:title" content="Acme Web Studio">
  <link rel="canonical" href="/home">
  <script type="application/ld+json">{"@type": "Organization", "name": "Acme Web Studio"}</script>
  <script type="application/ld+json">{ this is broken json </script>
  <script>var secret = "tracking code";</script>
</head><body>
  <header><nav>
    <a href="/services">Services</a> <a href="https://other.example/blog">Blog</a>
    <a href="javascript:void(0)">Menu</a> <a href="mailto:hi@acme.example">Mail</a>
  </nav></header>
  <h1>Websites that bring customers</h1>
  <h2>Our services</h2>
  <p>Acme Web Studio designs and builds fast, affordable websites for local shops, clinics and restaurants.</p>
  <p>We also run monthly maintenance plans, so your site stays secure, quick and easy to update.</p>
  <p>Every project starts with a free call where we learn about your customers and your goals.</p>
  <p>Our team has launched more than a hundred sites across Bihar and neighbouring states.</p>
  <p>Contact us today</p>
  <p>Contact us today</p>
  <img src="a.png" alt="Team photo"><img src="b.png">
  <footer><a href="mailto:hi@acme.example">hi@acme.example</a>
    <a href="https://linkedin.com/company/acme" aria-label="LinkedIn"><svg></svg></a></footer>
</body></html>
"""


def test_extract_normal_page():
    data = extract(NORMAL_PAGE, "https://acme.example/")
    assert data["title"] == "Acme Web Studio | Websites for Small Businesses"
    assert data["meta_description"] == "We build affordable websites for small businesses."
    assert data["og_site_name"] == "Acme Web Studio"
    assert data["lang"] == "en"
    assert data["canonical"] == "https://acme.example/home"
    assert data["json_ld"] == [{"@type": "Organization", "name": "Acme Web Studio"}]  # broken block skipped
    assert data["headings"] == ["Websites that bring customers", "Our services"]
    assert {"text": "Services", "href": "https://acme.example/services"} in data["nav_links"]
    assert all(not link["href"].startswith(("javascript:", "mailto:")) for link in data["nav_links"])
    assert {"text": "LinkedIn", "href": "https://linkedin.com/company/acme"} in data["footer_links"]
    assert "tracking code" not in data["body_text"]
    assert data["body_text"].count("Contact us today") == 1  # duplicate line removed
    assert data["seo_signals"]["images_total"] == 2
    assert data["seo_signals"]["images_missing_alt"] == 1
    assert data["seo_signals"]["h1_count"] == 1
    assert data["thin_content"] is False


def test_extract_spa_shell():
    shell = (
        '<html><head><title>My App</title><meta name="description" content="Track your tasks">'
        '</head><body><div id="root"></div><script src="/bundle.js"></script></body></html>'
    )
    data = extract(shell)
    assert data["thin_content"] is True
    assert data["title"] == "My App"
    assert data["meta_description"] == "Track your tasks"
    assert data["body_text"] == ""


VALID_LLM_RESULT = {
    "brand": {"name": "Acme", "category": "Web design", "summary": "Builds websites."},
    "analysis": {section: {} for section in llm_service.ANALYSIS_SECTIONS},
}


def _fake_llm(monkeypatch, replies):
    """Replace the OpenAI client. Each reply is the message text the model returns."""
    queue = list(replies)

    async def create(**kwargs):
        message = SimpleNamespace(content=queue.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_service, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(llm_service, "RETRY_WAIT_SECONDS", 0)
    monkeypatch.setattr(llm_service.config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.config, "LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setattr(llm_service.config, "LLM_MODEL", "test-model")


def test_llm_parses_fenced_json(monkeypatch):
    import json

    _fake_llm(monkeypatch, ["```json\n" + json.dumps(VALID_LLM_RESULT) + "\n```"])
    result = asyncio.run(llm_service.analyze({"title": "Acme"}, "https://acme.example"))
    assert result["brand"]["name"] == "Acme"
    assert set(result["analysis"]) == set(llm_service.ANALYSIS_SECTIONS)


def test_llm_invalid_json_twice_raises(monkeypatch):
    _fake_llm(monkeypatch, ["not json at all", "still { not json"])
    with pytest.raises(LLMError):
        asyncio.run(llm_service.analyze({"title": "Acme"}, "https://acme.example"))


def test_llm_payload_drops_misleading_input():
    extracted = {
        "title": "Acme",
        "og_title": "Acme",
        "headings": ["Our services", "Our services", "x" * 300],
        "nav_links": [
            {"text": "Login", "href": "https://acme.example/login"},
            {"text": "Services", "href": "https://acme.example/services"},
        ],
        "footer_links": [{"text": "Services", "href": "https://acme.example/services"}],
        "json_ld": [
            {
                "@type": "Organization",
                "name": "Acme",
                "contactPoint": {"contactType": "customer support"},
                "address": {"streetAddress": "1 Main St", "addressLocality": "Miami"},
            }
        ],
    }
    payload = llm_service._trim_payload(extracted)
    assert "og_title" not in payload  # same as the title
    assert payload["headings"] == ["Our services", "x" * 100]  # duplicate removed, long one cut
    assert [link["text"] for link in payload["nav_links"]] == ["Services"]  # login link removed
    assert "footer_links" not in payload  # only a duplicate of the nav
    block = payload["json_ld"][0]
    assert "contactPoint" not in block
    assert block["address"] == {"addressLocality": "Miami"}  # no street address