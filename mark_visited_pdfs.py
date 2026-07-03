"""
Mark the 16 already-successfully-downloaded PDFs as visited in checkpoint.json
so they aren't re-downloaded on resume.
"""
import csv
import json
from pathlib import Path

kb = Path("bvrith_knowledge_base")
checkpoint_path = kb / "checkpoint.json"
log_path = kb / "crawl_log.csv"

cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
visited_pdfs = set(cp.get("visited_pdfs", []))

with open(log_path, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("type", "").lower() == "pdf" and row.get("status", "") == "200":
            visited_pdfs.add(row["url"].strip())

cp["visited_pdfs"] = list(visited_pdfs)
checkpoint_path.write_text(json.dumps(cp), encoding="utf-8")
print(f"Marked {len(visited_pdfs)} PDFs as already visited (won't re-download).")
remaining = len(cp["pdf_referrers"]) - len(visited_pdfs)
print(f"PDFs still to process: {remaining}")
