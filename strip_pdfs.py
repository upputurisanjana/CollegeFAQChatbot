"""
Remove all PDF-sourced entries from chunks.jsonl and clean checkpoint.json.
"""
import json
from pathlib import Path

kb = Path("bvrith_knowledge_base")

# --- Strip PDF chunks from chunks.jsonl ---
chunks_path = kb / "chunks.jsonl"
kept, removed = [], 0
with open(chunks_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        if "pdf_page_number" in chunk:
            removed += 1
        else:
            kept.append(line)

chunks_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"chunks.jsonl: removed {removed} PDF chunks, kept {len(kept)}")

# --- Clean checkpoint ---
cp_path = kb / "checkpoint.json"
cp = json.loads(cp_path.read_text(encoding="utf-8"))
cp.pop("pdf_referrers", None)
cp.pop("visited_pdfs", None)
cp_path.write_text(json.dumps(cp), encoding="utf-8")
print("checkpoint.json: removed pdf_referrers and visited_pdfs")
