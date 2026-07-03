"""Find the max batch size OpenRouter accepts for embeddings."""
import os, json, requests
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

key = os.environ.get("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/embeddings"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

chunks = []
with open("bvrith_knowledge_base/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
        if len(chunks) >= 110:
            break

texts = [c["text"] for c in chunks]

for size in [10, 20, 50, 100]:
    payload = {"model": "text-embedding-3-small", "input": texts[:size]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        body = r.json()
        if "data" in body:
            print(f"Size {size:3d}: OK — got {len(body['data'])} embeddings")
        else:
            print(f"Size {size:3d}: FAIL — keys={list(body.keys())} | {json.dumps(body)[:200]}")
    except Exception as e:
        print(f"Size {size:3d}: EXCEPTION — {e}")
