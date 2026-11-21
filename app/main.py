import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import analyze_url
from app.services.llm_service import LLMError
from app.services.website_extractor import (
    BadStatus,
    FetchConnectionError,
    FetchTimeout,
    NotHTML,
    SiteBlocked,
    TooManyRedirects,
)
from app.utils.validators import BlockedURLError, InvalidURLError

logging.basicConfig(level=logging.INFO)  # makes the LLM token/time log line visible
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Link Analysis API")

# exception -> (HTTP status, short error code)
ERROR_MAP = {
    InvalidURLError: (422, "invalid_url"),
    BlockedURLError: (400, "url_not_allowed"),
    FetchTimeout: (504, "timeout"),
    FetchConnectionError: (502, "connection_failed"),
    SiteBlocked: (502, "site_blocked"),
    BadStatus: (502, "bad_status"),
    TooManyRedirects: (502, "too_many_redirects"),
    NotHTML: (415, "not_html"),
    LLMError: (502, "llm_error"),
}


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": code, "detail": detail},
    )


async def _handle_known_error(request: Request, exc: Exception) -> JSONResponse:
    for exc_class, (status, code) in ERROR_MAP.items():
        if isinstance(exc, exc_class):
            return _error(status, code, str(exc))
    return _error(500, "internal_error", "Something went wrong while analysing the website")


for _exc_class in ERROR_MAP:
    app.add_exception_handler(_exc_class, _handle_known_error)


@app.exception_handler(RequestValidationError)
async def _handle_bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "invalid_request", 'Send a JSON body like {"url": "https://example.com"}')


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected error")
    return _error(500, "internal_error", "Something went wrong while analysing the website")


@app.post("/link-analysis", response_model=LinkAnalysisResponse)
async def link_analysis(body: LinkAnalysisRequest):
    return await analyze_url(body.url)