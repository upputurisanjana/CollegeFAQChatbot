"""
Run exactly one batch of 5 chunks through the full ingest pipeline and show errors.
"""
import json, os, sys, logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

import chromadb
from chromadb.utils import embedding_functions

CHUNKS_PATH = Path("bvrith_knowledge_base/chunks.jsonl")
CHROMA_DIR = "./chroma_bvrith"

chunks = []
with open(CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
            if len(chunks) >= 5:
                break

print(f"Testing with {len(chunks)} chunks")

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
api_base = "https://openrouter.ai/api/v1"

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
    api_base=api_base,
)

client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("bvrith_faq_test")
except:
    pass
col = client.get_or_create_collection("bvrith_faq_test", embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})

ids = [c["chunk_id"] for c in chunks]
docs = [c["text"] for c in chunks]
metas = [{"source_url": c.get("source_url", ""), "page_title": (c.get("page_title") or "")[:200]} for c in chunks]

print("Calling collection.add() ...")
try:
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"SUCCESS — count: {col.count()}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# cleanup
try:
    client.delete_collection("bvrith_faq_test")
except:
    pass
