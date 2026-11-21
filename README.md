# AI Link Analysis API

Analyze a public website URL for brand, competitor, SEO, search-query, service, and audience insights.  
The API is built with FastAPI and uses an OpenAI-compatible LLM endpoint for structured analysis.  
Website HTML is validated, fetched, reduced to relevant content, and validated again after LLM inference.

> **Live demo:** Not deployed. Replace the placeholder host below when a deployment is available.

```bash
curl -X POST "https://<live-demo-host>/link-analysis" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://fireflies.ai"}'
```

Trimmed response:

```json
{
  "success": true,
  "website": "https://fireflies.ai",
  "brand": {
    "name": "Fireflies.ai",
    "category": "AI meeting transcription and assistant platform"
  },
  "analysis": {
    "target_search_queries": ["..."],
    "seo_improvements": {"primary_keyword": "AI meeting transcription"},
    "top_services": ["..."],
    "target_audience": {"primary_audience": ["Remote teams", "Sales organizations"]}
  }
}
```

The request pipeline is:

```text
validate → fetch → extract → LLM → response
```

## Project structure

```text
app/
  main.py                         FastAPI app, endpoint, and error handlers
  config.py                       Environment-backed LLM configuration
  schemas.py                      Request and response Pydantic models
  services/
    analysis_service.py            End-to-end analysis orchestration
    llm_service.py                 LLM prompt, call, parsing, and retries
    website_extractor.py           Fetching and HTML extraction
  utils/
    validators.py                  URL and public-host validation
samples/
  extracted_sample.json            Saved extractor output
  fireflies_ai_200.json            Saved complete response example
  logs.txt                         Saved local run log
tests/
  test_link_analysis.py            Pytest suite
```

## Features

- Validates HTTP(S) URLs, allowed ports, DNS resolution, and public-address requirements.
- Fetches HTML with redirect, timeout, response-size, content-type, and blocked-site handling.
- Extracts page metadata, canonical URL, language, JSON-LD, headings, navigation/footer links, visible text, and SEO signals.
- Sends cleaned structured website data to one LLM call.
- Returns validated brand and analysis data, including competitors, five search queries when returned by the model, SEO improvements, five services when returned by the model, and target-audience information.
- Exposes OpenAPI/Swagger documentation through FastAPI.

## Requirements

- Python 3.11 or newer. The application uses `asyncio.timeout`, which requires Python 3.11+.
- An API key and model for an OpenAI-compatible LLM provider.
- Network access to the website being analyzed and to the configured LLM provider.

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Set at least the following values in `.env`:

```dotenv
LLM_API_KEY=your-provider-api-key
LLM_BASE_URL=BASE_URL_OF_PROVIDER
LLM_MODEL=openai/gpt-oss-120b
```

`LLM_BASE_URL` is also required by the application configuration check. Optional settings already present in `.env.example` are `LLM_JSON_MODE=false` and `LLM_TIMEOUT_SECONDS=60`.

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- API endpoint: `POST http://localhost:4000/link-analysis`
- Swagger UI: `http://localhost:4000/docs`
- For Live check : `https://ai-link-analysis-gv0y.onrender.com/docs`

### 5. Run the tests

```bash
pytest
```

The pytest configuration uses automatic asyncio mode and discovers tests under `tests/`.

## API usage

### Request

```http
POST /link-analysis
Content-Type: application/json
```

```json
{
  "url": "https://fireflies.ai"
}
```

The `url` field accepts a URL with or without a scheme; URLs are normalized to HTTPS when no scheme is supplied. Only `http` and `https` URLs on ports 80 or 443 are accepted.

### Response

The saved example [`samples/fireflies_ai_200.json`](/samples/fireflies_ai_200.json) contains a complete response. Its top-level response fields are:

```json
{
  "success": true,
  "website": "https://fireflies.ai",
  "brand": {
    "name": "Fireflies.ai",
    "category": "AI meeting transcription and assistant platform",
    "summary": "AI teammate that records, transcribes, summarizes and analyzes meetings, emails, messages and tasks, integrating with 240+ apps."
  },
  "analysis": {
    "competitor_analysis": {},
    "target_search_queries": [],
    "seo_improvements": {},
    "top_services": [],
    "target_audience": {}
  }
}
```

The example contains five target search queries, five top services, and three AI-suggested competitors. The extraction-only example is [`samples/extracted_sample.json`](/samples/extracted_sample.json). It shows the compact data produced by the website extractor before LLM analysis. Additional run information is recorded in [`samples/logs.txt`](/samples/logs.txt).

## Error handling

Known failures use the following JSON shape:

```json
{
  "success": false,
  "error": "error_code",
  "detail": "A human-readable error detail"
}
```

| HTTP status | Error code(s) | Cause and behavior |
|---:|---|---|
| 400 | `url_not_allowed` | The URL resolves to a private or internal address. |
| 415 | `not_html` | The response is not HTML/XHTML. |
| 422 | `invalid_url`, `invalid_request` | The URL or request body is invalid. |
| 502 | `connection_failed`, `site_blocked`, `bad_status`, `too_many_redirects`, `llm_error` | The website or LLM provider cannot complete the operation. |
| 504 | `timeout` | The website exceeds the configured fetch timeout. |
| 500 | `internal_error` | An unexpected exception occurs; a generic error is returned. |

Error mapping is implemented in [`app/main.py`](/app/main.py), with URL validation in [`app/utils/validators.py`](/app/utils/validators.py) and website fetching in [`app/services/website_extractor.py`](/app/services/website_extractor.py).

## LLM -Reseach -Recomendation

### Research findings

For this workload, the main paid API options are OpenAI, Google, and Anthropic. OpenAI `gpt-4.1-mini` is the strongest cost and structured-output candidate. Google `gemini-2.5-flash` offers a very large context window and structured JSON support, making it useful for unusually large websites. Anthropic `claude-haiku-4-5` offers strong instruction following and long-context comprehension, but has the highest estimated cost of the compared models.

For an estimated 4K input and 2K output tokens, the expected costs are approximately `$0.0048` per OpenAI analysis, `$0.0062` per Gemini analysis, and `$0.0140` per Claude analysis. These are provider-price-based estimates, not application benchmarks. Open-weight models can reduce deployment cost and provide hosting flexibility, but require separate quality, latency, infrastructure, and schema-validity evaluation.

| Provider and model | Estimated cost per analysis | Strengths | Trade-offs |
|---|---:|---|---|
| OpenAI `gpt-4.1-mini` | `$0.0048` at 4K input + 2K output; `$0.0112` at 4K input + 6K output | Native Structured Outputs; strong fit for structured website and SEO analysis | Validate quality and latency on representative sites |
| Google `gemini-2.5-flash` | `$0.0062` at 4K input + 2K output; `$0.0162` at 4K input + 6K output | Very large context; structured JSON support; suitable for complex pages | Reasoning tokens can increase cost and latency |
| Anthropic `claude-haiku-4-5` | `$0.014` at 4K input + 2K output; `$0.034` at 4K input + 6K output | Strong instruction following and long-context comprehension | Highest estimated cost for this workload |

### Model selected for this project

The current project configuration uses **Groq** with the open-weight model `openai/gpt-oss-120b`. The model is accessed through Groq's OpenAI-compatible API and its response is parsed and validated against the application's Pydantic schema.

**Groq testing limitation:** Groq provides an 8K-token-per-minute limit for the configured usage pattern. One analysis frequently consumes about 7K tokens, so the limit can be exhausted by a single request. During testing, send only one request and wait for the token window to recover before sending another.

### Final LLM Recommendation

| Item | Recommendation |
|---|---|
| Recommended LLM provider | OpenAI |
| Recommended model name | `gpt-4.1-mini` |
| Estimated input token usage | 3,000–4,000 tokens |
| Estimated output token usage | 1,500–2,500 visible output tokens |
| Estimated API cost per website analysis | `$0.0048` for approximately 4K input and 2K output tokens; up to approximately `$0.0112` when output/reasoning usage reaches 6K tokens |
| Expected response time | Approximately 3–10 seconds as an estimate; measure with representative payloads |
| Reason for selecting the model | Best reported balance of low cost, fast mini-class performance, sufficient website-analysis quality, and native Structured Outputs |
| Alternative model for fallback | Google `gemini-2.5-flash` |
| Recommended approach | Use cleaned and deduplicated website JSON, require structured output, validate every response, cache unchanged analyses when available, and route unusually large or complex sites to Gemini after measuring quality, latency, and cost |

This recommendation is a production candidate, not a benchmark claim. Validate it on representative websites before replacing the current Groq configuration.

## Cost optimization

### Already implemented

- **HTML cleanup:** [`extract`](/app/services/website_extractor.py) removes scripts, styles, noscript content, SVGs, and iframes, deduplicates visible text, and extracts compact fields and SEO signals.
- **Payload trimming:** [`_trim_payload`](/app/services/llm_service.py) removes empty/duplicate fields, limits headings, removes CTA and duplicate links, cleans JSON-LD, and caps JSON-LD content.
- **Website text truncation:** the extractor caps visible body text at 12,000 characters.
- **One analysis call:** [`analyze`](/app/services/llm_service.py) generates the required brand and analysis sections in one LLM call rather than five separate calls.
- **Output/retry bounds:** the service sets `max_tokens=4096` and allows at most two attempts for recoverable invalid JSON, 429, and 5xx responses.

## License

No license file or licensing terms were provided in the repository.
