import os

from dotenv import load_dotenv

load_dotenv()




LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "false").strip().lower() in ("1", "true", "yes")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))