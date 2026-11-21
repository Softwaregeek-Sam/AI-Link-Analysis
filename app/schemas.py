from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PLATFORM_ALIASES = {
    "google": "Google",
    "google search": "Google",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "x": "X",
    "twitter": "X",
    "x (twitter)": "X",
    "twitter (x)": "X",
    "x/twitter": "X",
    "twitter/x": "X",
}


class LinkAnalysisRequest(BaseModel):
    url: str


class _Model(BaseModel):
    """Base for everything the LLM produces. Unknown fields (for example an invented
    search volume) are ignored, and nulls fall back to the field's default."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


class Brand(_Model):
    name: str
    category: str
    summary: str


class Competitor(_Model):
    name: str
    main_service: str = ""
    target_market: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    source: Literal["ai_suggested"] = "ai_suggested"

    @field_validator("source", mode="before")
    @classmethod
    def _always_ai_suggested(cls, value):
        return "ai_suggested"  # competitors always come from the model's general knowledge


class CompetitorAnalysis(_Model):
    brand_overview: str = ""
    main_products_services: list[str] = []
    competitors: list[Competitor] = []
    brand_advantages: list[str] = []
    opportunities: list[str] = []


class SearchQuery(_Model):
    query: str
    platforms: list[Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]] = Field(min_length=1)
    intent: str = ""
    business_relevance: str = ""
    target_audience: str = ""

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalize_platforms(cls, value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        names: list = []
        for item in value:
            name = PLATFORM_ALIASES.get(str(item).strip().lower(), item)  # "Twitter" -> "X"
            if name not in names:
                names.append(name)
        return names


class SeoImprovements(_Model):
    issues_found: list[str] = []
    primary_keyword: str
    secondary_keywords: list[str] = []
    long_tail_keywords: list[str] = []
    business_keywords: list[str] = []
    recommended_title: str
    title_structure: str = ""
    meta_description: str
    supporting_content: list[str] = []
    current_signals: dict[str, Any] = {}  # measured by our code, not written by the LLM


class Service(_Model):
    name: str
    description: str
    target_industry: str = ""
    target_customer: str = ""
    main_benefit: str = ""
    source: Literal["from_website", "inferred"] = "inferred"

    @field_validator("source", mode="before")
    @classmethod
    def _unknown_source_means_inferred(cls, value):
        return value if value in ("from_website", "inferred") else "inferred"


class RecommendedSegment(_Model):
    audience: str
    business_type: str = ""
    requirement: str = ""


class TargetAudience(_Model):
    primary_audience: list[str] = []
    secondary_audience: list[str] = []
    industries: list[str] = []
    business_types: list[str] = []
    customer_profiles: list[str] = []
    geographic_focus: str | None = None
    customer_needs: list[str] = []
    customer_interests: list[str] = []
    recommended_segments: list[RecommendedSegment] = []


class Analysis(_Model):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=1)
    seo_improvements: SeoImprovements
    top_services: list[Service] = Field(min_length=1)
    target_audience: TargetAudience

    @field_validator("target_search_queries", "top_services")
    @classmethod
    def _keep_first_five(cls, value):
        return value[:5]


class LinkAnalysisResponse(_Model):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis
    warnings: list[str] = []