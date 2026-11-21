import asyncio
import logging

from pydantic import ValidationError

from app.schemas import Analysis, Brand, LinkAnalysisResponse
from app.services import llm_service
from app.services.llm_service import LLMError
from app.services.website_extractor import extract, fetch_html
from app.utils.validators import validate_url

logger = logging.getLogger(__name__)


def _build_warnings(extracted: dict, analysis: Analysis) -> list[str]:
    warnings = []
    if extracted.get("thin_content"):
        warnings.append(
            "Page content was limited (likely a JavaScript-rendered site); "
            "results are based mostly on metadata."
        )
    if len(analysis.target_search_queries) < 5:
        warnings.append(f"Only {len(analysis.target_search_queries)} search queries were returned (5 expected).")
    if len(analysis.top_services) < 5:
        warnings.append(f"Only {len(analysis.top_services)} services were returned (5 expected).")
    if len(analysis.competitor_analysis.competitors) < 3:
        warnings.append("Fewer than 3 competitors were returned.")
    return warnings


async def analyze_url(raw_url: str) -> dict:
    """validate -> fetch -> extract -> LLM -> validate the LLM output -> final response dict."""
    url = await asyncio.to_thread(validate_url, raw_url)  # DNS lookup blocks, keep it off the event loop
    final_url, html = await fetch_html(url)
    extracted = await asyncio.to_thread(extract, html, final_url)  # parsing is CPU work
    result = await llm_service.analyze(extracted, final_url)

    try:
        brand = Brand.model_validate(result["brand"])
        analysis = Analysis.model_validate(result["analysis"])
    except ValidationError as exc:
        # log where it failed, never the content itself
        problems = [".".join(str(part) for part in err["loc"]) + ": " + err["msg"] for err in exc.errors()[:5]]
        logger.warning("LLM output failed validation: %s", problems)
        raise LLMError("The LLM returned an invalid response")

    analysis.seo_improvements.current_signals = extracted["seo_signals"]
    response = LinkAnalysisResponse(
        website=url,
        brand=brand,
        analysis=analysis,
        warnings=_build_warnings(extracted, analysis),
    )
    return response.model_dump()