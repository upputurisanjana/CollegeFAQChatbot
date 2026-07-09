"""
chunk_and_index.py
-------------------
Chunk scraped BVRIT markdown files (header-aware) and load them into a
Chroma vector database, with metadata for filtering and evaluation.

Usage:
    python chunk_and_index.py --input_dir ./scraped_data --db_dir ./chroma_db

Requirements:
    pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers
"""

import os
import glob
import json
import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

from config import (
    HEADERS_TO_SPLIT_ON, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS,
    EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME, KB_DIR, SUMMARY_FILE,
    CHUNKS_RAW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_page_type(filename: str, content: str) -> str:
    """Rough heuristic tagging based on filename/content patterns.
    Adjust these keywords to match your actual scraped filenames."""
    name = filename.lower()
    text = content.lower()

    if "faculty" in name or "profile" in name:
        return "faculty_profile"
    if "fee" in name or "fee structure" in text:
        return "fee_structure"
    if "deadline" in name or "counselling" in text or "eamcet" in text:
        return "deadline"
    if "faq" in name:
        return "faq"
    return "general"


def build_header_path(metadata: dict) -> str:
    """Combine h1/h2/h3 metadata from MarkdownHeaderTextSplitter into
    a single readable path, e.g. 'Admissions > EAMCET Counselling'."""
    parts = [metadata.get(k) for k in ("h1", "h2", "h3") if metadata.get(k)]
    return " > ".join(parts) if parts else "Untitled Section"


def make_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Deterministic ID so re-running the script updates rather than
    duplicates existing chunks."""
    raw = f"{source_file}-{chunk_index}-{text[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text)."""
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip().strip('"').strip("'")
            return meta, parts[2].strip()
    return meta, text


def chunk_markdown_file(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    filename = os.path.basename(filepath)
    frontmatter, body_text = _parse_frontmatter(raw_text)
    page_type = classify_page_type(filename, raw_text)
    source_url = frontmatter.get("url", filepath)

    # Stage 1: split by header structure
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text inside the chunk for context
    )
    header_sections = header_splitter.split_text(body_text)

    # Stage 2: sub-split any section that's too large
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks = []
    chunk_index = 0

    for section in header_sections:
        text = section.page_content.strip()

        if len(text) < MIN_CHUNK_CHARS:
            # Too small on its own — merge into previous chunk if one exists
            if final_chunks:
                final_chunks[-1].page_content += "\n\n" + text
            continue

        if len(text) <= CHUNK_SIZE:
            sub_chunks = [text]
        else:
            sub_chunks = recursive_splitter.split_text(text)

        header_path = build_header_path(section.metadata)

        for sub_text in sub_chunks:
            metadata = {
                "source_file": filename,
                "source_path": source_url,
                "source_url": source_url,
                "section_heading": header_path,
                "page_type": page_type,
                "chunk_index": chunk_index,
                "ingested_date": datetime.now().strftime("%Y-%m-%d"),
            }
            doc = Document(page_content=sub_text, metadata=metadata)
            final_chunks.append(doc)
            chunk_index += 1

    return final_chunks


def chunk_all_files(input_dir: str) -> list[Document]:
    md_files = glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True)
    if not md_files:
        raise FileNotFoundError(f"No .md files found under {input_dir}")

    all_chunks = []
    for filepath in md_files:
        chunks = chunk_markdown_file(filepath)
        all_chunks.extend(chunks)
        print(f"  {os.path.basename(filepath):40s} -> {len(chunks)} chunks")

    return all_chunks


# ---------------------------------------------------------------------------
# Vector DB ingestion
# ---------------------------------------------------------------------------

def build_vector_db(chunks: list[Document], db_dir: str):
    print(f"\nEmbedding {len(chunks)} chunks with '{EMBED_MODEL}'...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    ids = [
        make_chunk_id(doc.metadata.get("source_file", "unknown"), doc.metadata.get("chunk_index", i), doc.page_content)
        for i, doc in enumerate(chunks)
    ]

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist()
    print(f"Vector DB persisted to: {db_dir} (collection: {COLLECTION_NAME})")
    return vectordb


# ---------------------------------------------------------------------------
# chunks.jsonl ingestion (Pipeline B — direct from scraper output)
# ---------------------------------------------------------------------------

def chunk_from_jsonl(jsonl_path: str) -> list[Document]:
    """Ingest pre-chunked records from scraper's chunks.jsonl."""
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if len(text) < MIN_CHUNK_CHARS:
                continue
            source_url = rec.get("source_url", "")
            page_title = rec.get("page_title", "Untitled")
            retrieved_at = rec.get("retrieved_at_utc", "")
            metadata = {
                "source_file": os.path.basename(source_url.rstrip("/")) + ".md",
                "source_path": source_url,
                "source_url": source_url,
                "section_heading": page_title or "General",
                "page_type": "general",
                "chunk_index": i,
                "ingested_date": retrieved_at[:10] if retrieved_at else datetime.now().strftime("%Y-%m-%d"),
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Run summary (for UI freshness display)
# ---------------------------------------------------------------------------

def write_run_summary(chunk_count: int, source_dirs: list[str]):
    summary = {
        "crawl_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": chunk_count,
        "sources": source_dirs,
        "embedding_model": EMBED_MODEL,
    }
    KB_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run summary written to {SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def spot_check(vectordb, queries: list[str], k: int = 3):
    print("\n--- Spot check retrieval ---")
    for q in queries:
        print(f"\nQuery: {q}")
        results = vectordb.similarity_search(q, k=k)
        for i, r in enumerate(results, 1):
            heading = r.metadata.get("section_heading", "Unknown")
            source = r.metadata.get("source_file", "Unknown")
            snippet = r.page_content[:120].replace("\n", " ")
            print(f"  [{i}] ({source} | {heading}) {snippet}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunk markdown files into a Chroma vector DB")
    parser.add_argument("--input_dir", default="./scraped_site/pages", help="Folder containing .md files")
    parser.add_argument("--db_dir", default=CHROMA_DIR, help="Where to persist the vector DB")
    parser.add_argument("--chunks_jsonl", default=None, help="Optional chunks.jsonl from scraper for supplementary ingestion")
    parser.add_argument("--no_spot_check", action="store_true", help="Skip the sample retrieval test")
    args = parser.parse_args()

    all_chunks = []

    # 1. Ingest from .md files
    if os.path.isdir(args.input_dir):
        print(f"Scanning '{args.input_dir}' for markdown files...")
        md_chunks = chunk_all_files(args.input_dir)
        print(f"From .md files: {len(md_chunks)} chunks")
        all_chunks.extend(md_chunks)
    else:
        print(f"Input directory '{args.input_dir}' not found. Skipping .md ingestion.")

    # 2. Optionally ingest from chunks.jsonl
    if args.chunks_jsonl and os.path.isfile(args.chunks_jsonl):
        print(f"Ingesting from '{args.chunks_jsonl}'...")
        jsonl_chunks = chunk_from_jsonl(args.chunks_jsonl)
        print(f"From chunks.jsonl: {len(jsonl_chunks)} chunks")
        all_chunks.extend(jsonl_chunks)
    elif args.chunks_jsonl:
        print(f"chunks.jsonl '{args.chunks_jsonl}' not found. Skipping.")

    if not all_chunks:
        print("ERROR: No chunks to index. Provide --input_dir and/or --chunks_jsonl.")
        return

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    vectordb = build_vector_db(all_chunks, args.db_dir)

    # Write run summary
    sources = []
    if os.path.isdir(args.input_dir):
        sources.append(args.input_dir)
    if args.chunks_jsonl and os.path.isfile(args.chunks_jsonl):
        sources.append(args.chunks_jsonl)
    write_run_summary(len(all_chunks), sources)

    if not args.no_spot_check:
        sample_queries = [
            "What is the last date for EAMCET counselling?",
            "What is the annual tuition fee for CSE?",
            "Who are the faculty in the CSE department?",
        ]
        spot_check(vectordb, sample_queries)


if __name__ == "__main__":
    main()