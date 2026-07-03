"""
Patch checkpoint.json with all PDF URLs discovered during the previous crawl
(extracted from crawl_log.csv — both failed and succeeded ones, minus already visited).
Run once before resuming the scraper.
"""
import csv
import json
from pathlib import Path

kb = Path("bvrith_knowledge_base")
checkpoint_path = kb / "checkpoint.json"
log_path = kb / "crawl_log.csv"

cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))

# Collect all PDF URLs from crawl log
pdf_referrers = dict(cp.get("pdf_referrers", {}))  # keep any already found
visited_pdfs = set(cp.get("visited_pdfs", []))

with open(log_path, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("type", "").lower() == "pdf":
            url = row["url"].strip()
            if url and url not in visited_pdfs:
                # Use empty referrer list if not already tracked
                if url not in pdf_referrers:
                    pdf_referrers[url] = [{"page": "crawl_log_recovery", "link_text": ""}]

print(f"Total pdf_referrers after patch: {len(pdf_referrers)}")
print(f"Already visited PDFs: {len(visited_pdfs)}")
print(f"PDFs to process on next resume: {len(pdf_referrers) - len(visited_pdfs)}")

cp["pdf_referrers"] = pdf_referrers
checkpoint_path.write_text(json.dumps(cp), encoding="utf-8")
print("checkpoint.json updated — ready to resume with PDF processing.")
