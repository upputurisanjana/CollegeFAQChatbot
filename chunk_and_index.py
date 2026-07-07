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
import argparse
import hashlib
from datetime import datetime

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

CHUNK_SIZE = 450          # tokens/chars target for sub-splitting large sections
CHUNK_OVERLAP = 60        # ~15% overlap
MIN_CHUNK_CHARS = 40      # merge/skip chunks smaller than this (near-empty headers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, no API key needed


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

def chunk_markdown_file(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    filename = os.path.basename(filepath)
    page_type = classify_page_type(filename, raw_text)

    # Stage 1: split by header structure
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text inside the chunk for context
    )
    header_sections = header_splitter.split_text(raw_text)

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
                "source_path": filepath,
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
    print(f"\nEmbedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    ids = [
        make_chunk_id(doc.metadata["source_file"], doc.metadata["chunk_index"], doc.page_content)
        for doc in chunks
    ]

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=db_dir,
    )
    vectordb.persist()
    print(f"Vector DB persisted to: {db_dir}")
    return vectordb


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
    parser.add_argument("--input_dir", default="./scraped_data", help="Folder containing .md files")
    parser.add_argument("--db_dir", default="./chroma_db", help="Where to persist the vector DB")
    parser.add_argument("--no_spot_check", action="store_true", help="Skip the sample retrieval test")
    args = parser.parse_args()

    print(f"Scanning '{args.input_dir}' for markdown files...")
    chunks = chunk_all_files(args.input_dir)
    print(f"\nTotal chunks created: {len(chunks)}")

    vectordb = build_vector_db(chunks, args.db_dir)

    if not args.no_spot_check:
        sample_queries = [
            "What is the last date for EAMCET counselling?",
            "What is the annual tuition fee for CSE?",
            "Who are the faculty in the CSE department?",
        ]
        spot_check(vectordb, sample_queries)


if __name__ == "__main__":
    main()