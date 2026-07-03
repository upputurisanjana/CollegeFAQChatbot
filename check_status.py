import json
from pathlib import Path

kb = Path("bvrith_knowledge_base")

summary_path = kb / "run_summary.json"
if summary_path.exists():
    print("=== run_summary.json ===")
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))
else:
    print("No run_summary.json yet")

print()
for fname in ["pages.jsonl", "chunks.jsonl", "images_manifest.jsonl", "pdf_documents.jsonl"]:
    path = kb / fname
    if path.exists():
        lines = sum(1 for l in path.open(encoding="utf-8") if l.strip())
        size = path.stat().st_size
        print(f"{fname}: {lines} records, {size:,} bytes")

img_dir = kb / "images"
imgs = list(img_dir.iterdir()) if img_dir.exists() else []
print(f"Images on disk: {len(imgs)}")

cp_path = kb / "checkpoint.json"
cp = json.loads(cp_path.read_text(encoding="utf-8"))
print(f"Checkpoint visited_pages: {len(cp.get('visited_pages', []))}")
