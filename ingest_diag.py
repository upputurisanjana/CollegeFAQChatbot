"""Quick ingest diagnostic — runs one batch and prints everything to stdout."""
import json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_bvrith"
CHUNKS_PATH = Path("bvrith_knowledge_base/chunks.jsonl")

# Load chunks
chunks = []
with open(CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
print(f"Loaded {len(chunks)} chunks from chunks.jsonl")

# Check chunk_id presence
with_id = sum(1 for c in chunks if c.get("chunk_id"))
print(f"Chunks with chunk_id: {with_id}")
print(f"Sample chunk_id: {chunks[0].get('chunk_id')}")
print(f"Sample text (50 chars): {chunks[0].get('text','')[:50]}")

# Build embedding function
api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
api_base = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
print(f"\nAPI key present: {bool(api_key)}")
print(f"API base: {api_base}")

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
    api_base=api_base,
)

# Try embedding just 2 chunks
print("\nTesting embedding of 2 chunks...")
try:
    test_texts = [chunks[0]["text"][:500], chunks[1]["text"][:500]]
    result = ef(test_texts)
    print(f"Embedding OK — got {len(result)} vectors of dim {len(result[0])}")
except Exception as e:
    print(f"Embedding FAILED: {e}")
    sys.exit(1)

# Create collection and add first batch of 10
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("bvrith_faq")
    print("Deleted existing collection")
except:
    pass

col = client.get_or_create_collection("bvrith_faq", embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})

batch = chunks[:10]
ids = [c["chunk_id"] for c in batch]
docs = [c["text"][:1000] for c in batch]
metas = [{"source_url": c.get("source_url",""), "page_title": c.get("page_title","")[:200]} for c in batch]

print(f"\nAdding first 10 chunks...")
try:
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Success! Collection count: {col.count()}")
except Exception as e:
    print(f"Add FAILED: {e}")
