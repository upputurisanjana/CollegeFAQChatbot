"""
ingest.py — Pipeline B (raw crawl) ingest for BVRIT FAQ Chatbot
================================================================
Loads chunks.jsonl → embeds via text-embedding-3-small (OpenRouter) →
persists to ChromaDB at ./chroma_bvrith.

Usage:
    python ingest.py                         # default path
    python ingest.py --chunks path/to/chunks.jsonl
    python ingest.py --reset                 # wipe and re-index from scratch
    python ingest.py --verify-only           # skip indexing, just verify counts

Spec reference: spec.md §4 (Phase 1 — Ingest and index)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import openai

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CHUNKS_PATH = Path("bvrith_knowledge_base/chunks.jsonl")
CHROMA_DIR = "./chroma_bvrith"
COLLECTION_NAME = "bvrith_faq"

# spec §4: chunk_size=500, overlap=50 was the *curated-doc* target; Pipeline B
# uses the scraper's native 1000-char chunks. These are stored as metadata so
# the UI can display them.
SCRAPER_CHUNK_SIZE = 1000
SCRAPER_CHUNK_OVERLAP = 150

EMBED_MODEL = "text-embedding-3-small"   # spec §4 — same model at index + query time
BATCH_SIZE = 500                          # chunks per embedding API call

# Section keywords → normalised section name (for metadata "section" field)
# IMPORTANT: evaluated in priority order — most specific patterns first.
# Each entry is (url_pattern, section_name). First match wins.
SECTION_RULES = [
    # Admissions — check before "about" to avoid /admission/about matching About BVRIT
    ("admission",                   "Admissions"),
    ("intake",                      "Admissions"),
    ("eamcet",                      "Admissions"),
    ("b-category",                  "Admissions"),
    ("documents-to-submit",         "Admissions"),
    # Fee structure
    ("fee",                         "Fee Structure"),
    # Placements
    ("placement",                   "Placements"),
    ("internship",                  "Placements"),
    ("recruiter",                   "Placements"),
    # Faculty — individual profile pages
    ("/dr-",                        "Faculty"),
    ("/mr-",                        "Faculty"),
    ("/ms-",                        "Faculty"),
    ("/mrs-",                       "Faculty"),
    ("faculty",                     "Faculty"),
    ("about-hod",                   "Faculty"),
    # Departments — specific dept paths (before generic "about")
    ("computer-science",            "Departments"),
    ("cse-artificial",              "Departments"),
    ("electronics-and-communication","Departments"),
    ("electrical-and-electronics",  "Departments"),
    ("information-technology",      "Departments"),
    ("basic-sciences",              "Departments"),
    ("under-graduate",              "Departments"),
    ("post-graduate",               "Departments"),
    # Campus & Facilities
    ("hostel",                      "Campus & Facilities"),
    ("library",                     "Campus & Facilities"),
    ("transport",                   "Campus & Facilities"),
    ("laboratory",                  "Campus & Facilities"),
    ("/labs",                       "Campus & Facilities"),
    ("gym",                         "Campus & Facilities"),
    ("yoga",                        "Campus & Facilities"),
    ("food",                        "Campus & Facilities"),
    ("cafeteria",                   "Campus & Facilities"),
    ("pcs-facilities",              "Campus & Facilities"),
    ("differentiator",              "Campus & Facilities"),
    # Contact
    ("contact",                     "Contact"),
    # Research
    ("research",                    "About BVRIT"),
    ("naac",                        "About BVRIT"),
    ("nba",                         "About BVRIT"),
    ("nirf",                        "About BVRIT"),
    ("committee",                   "About BVRIT"),
    ("principal",                   "About BVRIT"),
    ("management",                  "About BVRIT"),
    ("about-bvrith",                "About BVRIT"),
    ("organogram",                  "About BVRIT"),
    ("nisp",                        "About BVRIT"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_section(chunk: dict) -> str:
    """Infer the spec §3 section from URL only (breadcrumbs contain site nav and are unreliable)."""
    url = (chunk.get("source_url") or "").lower()
    for pattern, section in SECTION_RULES:
        if pattern in url:
            return section
    return "General"


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    errors = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Line %d: JSON decode error: %s", i, e)
                errors += 1
    log.info("Loaded %d chunks (%d decode errors) from %s", len(chunks), errors, path)
    return chunks


def build_openrouter_ef() -> embedding_functions.OpenAIEmbeddingFunction:
    """
    ChromaDB's built-in OpenAI embedding function works with OpenRouter
    by overriding the api_base to OpenRouter's endpoint.
    spec §4: text-embedding-3-small, 1536-dim.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
        )
        sys.exit(1)

    # If using OpenRouter, override api_base
    api_base = os.environ.get("OPENAI_API_BASE", None)
    if not api_base and os.environ.get("OPENROUTER_API_KEY"):
        api_base = "https://openrouter.ai/api/v1"

    kwargs = dict(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )
    if api_base:
        kwargs["api_base"] = api_base

    return embedding_functions.OpenAIEmbeddingFunction(**kwargs)


def embed_texts(texts: list[str], timeout: float = 60.0) -> list[list[float]]:
    """
    Embed texts using requests directly (bypasses openai SDK response parsing issues
    with OpenRouter). Returns list of embedding vectors.
    """
    import requests as _requests
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", None)
    if not api_base and os.environ.get("OPENROUTER_API_KEY"):
        api_base = "https://openrouter.ai/api/v1"
    if not api_base:
        api_base = "https://api.openai.com/v1"

    url = api_base.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": EMBED_MODEL, "input": texts}
    resp = _requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if "data" not in body:
        raise ValueError(f"No 'data' in response: {json.dumps(body)[:300]}")
    # Sort by index to ensure order matches input
    items = sorted(body["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest(chunks_path: Path, reset: bool = False) -> int:
    """
    Embed and persist all chunks to ChromaDB.
    Returns total chunk count after indexing.
    spec §4 verification: prints count before and after reload.
    """
    # Validate API key early
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        log.info("--reset: deleting existing collection '%s'", COLLECTION_NAME)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # Create collection WITHOUT embedding_function — we supply embeddings directly
    # so we can control timeouts and retries on the API call.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    log.info("Collection '%s' — existing chunks: %d", COLLECTION_NAME, existing_count)

    chunks = load_chunks(chunks_path)
    if not chunks:
        log.error("No chunks loaded — aborting.")
        sys.exit(1)

    # Determine which chunk_ids are new (for idempotent re-runs)
    existing_ids: set[str] = set()
    if existing_count > 0 and not reset:
        offset = 0
        batch = 100
        while True:
            result = collection.get(limit=batch, offset=offset, include=[])
            ids = result.get("ids", [])
            if not ids:
                break
            existing_ids.update(ids)
            offset += len(ids)
            if len(ids) < batch:
                break
        log.info("Found %d existing IDs — will skip duplicates.", len(existing_ids))

    # Prepare batches
    new_chunks = [c for c in chunks if c.get("chunk_id") and c["chunk_id"] not in existing_ids]
    log.info("%d new chunks to embed and index.", len(new_chunks))

    if not new_chunks:
        log.info("Nothing new to index.")
        final_count = collection.count()
        log.info("Verification — chunks in store: %d", final_count)
        return final_count

    total_added = 0
    for batch_start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_start: batch_start + BATCH_SIZE]
        ids, documents, metadatas = [], [], []
        for c in batch:
            chunk_id = c["chunk_id"]
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ids.append(chunk_id)
            documents.append(text)
            metadatas.append({
                "source_url":        c.get("source_url") or "",
                "page_title":        (c.get("page_title") or "")[:256],
                "section":           infer_section(c),
                "breadcrumb":        " > ".join(c.get("breadcrumb") or [])[:256],
                "chunk_index":       c.get("chunk_index", 0),
                "retrieved_at_utc":  c.get("retrieved_at_utc") or "",
                "pdf_page_number":   c.get("pdf_page_number", -1),
                "pipeline":          "B",
            })

        if not ids:
            continue

        # Embed with direct openai client (has timeout)
        for attempt in range(3):
            try:
                embeddings = embed_texts(documents, timeout=90.0)
                break
            except Exception as e:
                log.warning("Embedding attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    log.error("Batch %d–%d: embedding failed 3 times — skipping.", batch_start, batch_start + len(ids) - 1)
                    embeddings = None
                    break
                time.sleep(10 * (attempt + 1))  # 10s, then 20s

        if embeddings is None:
            continue

        try:
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            total_added += len(ids)
            log.info(
                "Indexed batch %d–%d (%d/%d total)",
                batch_start, batch_start + len(batch) - 1,
                total_added, len(new_chunks),
            )
            time.sleep(1)  # gentle rate-limit pause between batches
        except Exception as e:
            log.error("Batch %d add failed: %s — retrying once...", batch_start, e)
            time.sleep(3)
            try:
                collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                total_added += len(ids)
            except Exception as e2:
                log.error("Batch %d failed again: %s — skipping.", batch_start, e2)

    # --- spec §4 verification step ---
    count_after = collection.count()
    log.info("=" * 60)
    log.info("Indexing complete — %d new chunks added.", total_added)
    log.info("Chunks in store immediately after indexing: %d", count_after)

    # Reload in a fresh client to confirm persistence
    client2 = chromadb.PersistentClient(path=CHROMA_DIR)
    collection2 = client2.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count_reload = collection2.count()
    log.info("Chunks after fresh-client reload:            %d", count_reload)

    if count_after == count_reload:
        log.info("✓ Persistence verified — counts match.")
    else:
        log.error(
            "✗ Persistence mismatch! Before reload: %d, after: %d",
            count_after, count_reload,
        )

    return count_reload


def verify_only() -> int:
    """Just print current collection stats without indexing."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        log.info("Collection '%s' — %d chunks indexed.", COLLECTION_NAME, count)

        # Sample 3 chunks to confirm metadata structure
        sample = collection.get(limit=3, include=["metadatas", "documents"])
        for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
            log.info("Sample %d: section=%s | url=%s | text=%s…",
                     i, meta.get("section"), meta.get("source_url"), doc[:80])
        return count
    except Exception as e:
        log.error("Could not open collection: %s", e)
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest BVRIT chunks into ChromaDB.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.jsonl")
    parser.add_argument("--reset", action="store_true", help="Wipe collection before re-indexing")
    parser.add_argument("--verify-only", action="store_true", help="Print current counts, no indexing")
    args = parser.parse_args()

    if args.verify_only:
        verify_only()
        return

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        log.error("chunks.jsonl not found at %s — run the scraper first.", chunks_path)
        sys.exit(1)

    total = ingest(chunks_path, reset=args.reset)
    print(f"\nDone. Total chunks in store: {total}")


if __name__ == "__main__":
    main()
