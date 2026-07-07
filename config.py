"""
config.py — Shared configuration for BVRIT FAQ Chatbot
======================================================
Single source of truth for paths, models, and constants.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
KB_DIR            = BASE_DIR / "bvrith_knowledge_base"
SUMMARY_FILE      = KB_DIR / "run_summary.json"
CHUNKS_STRUCTURED = KB_DIR / "chunks_structured.jsonl"
CHUNKS_RAW        = KB_DIR / "chunks.jsonl"
CHROMA_DIR        = str(BASE_DIR / "chroma_db")
COLLECTION_NAME   = "langchain"
DB_PATH           = str(BASE_DIR / "chat_history.db")

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL       = "text-embedding-3-small"
EMBED_DIM         = 1536
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBED_DIM   = 384

# ── Generation models (display name → OpenRouter model id) ───────────────────
# NOTE: Free tiers on OpenRouter change frequently. Update when models get deprecated.
MODEL_MAP = {
    "DeepSeek R1":          "deepseek/deepseek-r1:free",
    "Gemma 3 12B":          "google/gemma-3-12b-it:free",
    "Llama 3.1 8B":         "meta-llama/llama-3.1-8b-instruct:free",
}
FALLBACK_MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-12b-it:free",
]

# ── Retrieval ────────────────────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 0.28
DEFAULT_TOP_K      = 5
MAX_TOP_K          = 25

# ── App limits ───────────────────────────────────────────────────────────────
MAX_INPUT_CHARS          = 500
MAX_QUERIES_PER_SESSION  = 40
GLOBAL_QUERY_LIMIT       = 200   # total across all sessions before reset

# ── Cost estimation (approximate, varies by model) ────────────────────────────
COST_PER_1K_TOKENS_IN  = 0.0      # free tier = $0
COST_PER_1K_TOKENS_OUT = 0.0
ESTIMATED_COST_NOTE    = "Free tier — no charge"

# ── Security ─────────────────────────────────────────────────────────────────
ALLOWED_DOMAIN     = "bvrithyderabad.edu.in"

# ── API key helpers ──────────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

def get_api_base() -> str:
    custom = os.environ.get("OPENAI_API_BASE")
    if custom:
        return custom
    return "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else "https://api.openai.com/v1"
