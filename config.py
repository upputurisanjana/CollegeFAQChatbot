"""
config.py — Shared configuration for BVRIT FAQ Chatbot
======================================================
Single source of truth for paths, models, and constants.
All other modules import from here.
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
COLLECTION_NAME   = "bvrith_faq"
DB_PATH           = str(BASE_DIR / "chat_history.db")

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"
EMBED_DIM         = 384

# ── Generation models (display name → OpenRouter model id) ───────────────────
MODEL_MAP = {
    "GPT-4o Mini":   "openai/gpt-4o-mini",
    "Gemma 4 31B":   "google/gemma-4-31b-it:free",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
}

# ── Retrieval ────────────────────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 0.28
DEFAULT_TOP_K       = 8
MAX_TOP_K           = 25
ALLOWED_DOMAIN      = "bvrithyderabad.edu.in"

# ── Chunking ─────────────────────────────────────────────────────────────────
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
CHUNK_SIZE     = 450
CHUNK_OVERLAP  = 60
MIN_CHUNK_CHARS = 40

# ── App limits ───────────────────────────────────────────────────────────────
MAX_INPUT_CHARS          = 500
MAX_QUERIES_PER_SESSION  = 40
GLOBAL_QUERY_LIMIT       = 200

# ── API key helpers ──────────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

def get_api_base() -> str:
    custom = os.environ.get("OPENAI_API_BASE")
    if custom:
        return custom
    return "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else "https://api.openai.com/v1"
