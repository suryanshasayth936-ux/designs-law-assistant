"""Shared configuration: API key loader, Gemini backend auto-detection, model constants."""
import os
import random
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Active 2026 models (retired: embedding-001, text-embedding-004, gemini-1.5-flash)
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")  # gemini-2.5-flash retired; 3.8-flash times out on this key
LLM_TEMPERATURE = 0

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "manual_of_designs.pdf"
DB_DIR = str(BASE_DIR / "db")
COLLECTION_NAME = "designs_manual"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 300
EMBED_BATCH_SIZE = 25
RETRIEVE_K = 6

NOT_FOUND = "Not found"

EVAL_QUESTIONS = [
    "How long does a proprietor have copyright in a registered design, and can it be extended?",
    "Which classification system is used to register designs?",
    "What is the government fee to register a trademark?",
    "Who is the current Controller General of Patents?",
    "What is the fee to register a design?",
]


def require_api_key() -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and put your key in it."
        )
    return GEMINI_API_KEY


def with_backoff(fn, *args, max_retries: int = 12, base_delay: float = 10.0, **kwargs):
    """Call fn, retrying with exponential backoff on HTTP 429 / quota errors."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = any(
                k in msg
                for k in ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit", "503", "unavailable")
            )
            if not transient or attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  [rate limit] retry {attempt + 1}/{max_retries - 1} in {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 65)


def _make_embeddings(vertexai: bool):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=require_api_key(), vertexai=vertexai
    )


@lru_cache(maxsize=1)
def detect_backend() -> bool:
    """Return True if Vertex AI Express Mode must be used, False for the Gemini Developer API.

    `AQ.`-prefixed keys are issued for both AI Studio and Vertex AI Express Mode, so we probe
    vertexai=False first and fall back to vertexai=True. Override with GENAI_BACKEND=aistudio|vertex.
    """
    forced = os.getenv("GENAI_BACKEND", "").strip().lower()
    if forced in ("vertex", "vertexai"):
        return True
    if forced in ("aistudio", "gemini", "developer"):
        return False

    errors = {}
    for vertex in (False, True):
        try:
            _make_embeddings(vertex).embed_query("ping")
            print(f"[config] Gemini backend: {'Vertex AI Express Mode' if vertex else 'Gemini Developer API (AI Studio)'}")
            return vertex
        except Exception as exc:  # noqa: BLE001
            errors[vertex] = exc
            if "429" in str(exc) or "quota" in str(exc).lower():
                # Key is valid for this backend, just rate limited.
                return vertex
    raise RuntimeError(
        "Could not authenticate against either backend.\n"
        f"  AI Studio: {errors.get(False)}\n  Vertex Express: {errors.get(True)}"
    )


def get_embeddings():
    return _make_embeddings(detect_backend())


def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=require_api_key(),
        vertexai=detect_backend(),
        timeout=60,
        max_retries=1,
    )
