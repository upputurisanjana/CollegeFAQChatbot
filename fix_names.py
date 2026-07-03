"""
Copy context_heading -> semantic_name for images where semantic_name is empty.
This ensures the display name shows correctly.
"""
import json
from pathlib import Path

MANIFEST = Path("bvrith_knowledge_base/images_manifest.jsonl")
lines = MANIFEST.read_text(encoding="utf-8").splitlines()
updated = []
fixed = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    img = json.loads(line)
    if not img.get("semantic_name") and img.get("context_heading"):
        img["semantic_name"] = img["context_heading"]
        fixed += 1
    updated.append(json.dumps(img, ensure_ascii=False))

MANIFEST.write_text("\n".join(updated) + "\n", encoding="utf-8")
with open("fix_names_out.txt", "w") as out:
    out.write(f"Fixed {fixed} images — copied context_heading to semantic_name\n")
    named = sum(1 for line in updated if json.loads(line).get("semantic_name") and json.loads(line).get("status") == "saved")
    total = sum(1 for l in updated if json.loads(l).get("status") == "saved")
    out.write(f"Images with semantic_name: {named} / {total}\n")
