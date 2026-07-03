"""
Re-tag images_manifest.jsonl:
- Set category, department, semantic_name from page_url for all faculty profile images
- More complete name extraction from URL slug
"""
import json, re
from pathlib import Path

MANIFEST = Path("bvrith_knowledge_base/images_manifest.jsonl")

DEPT_URL_MAP = [
    ("computer-science-and-engineering",            "Computer Science and Engineering"),
    ("cse-artificial-intelligence-and-machine",     "CSE (Artificial Intelligence & Machine Learning)"),
    ("electronics-and-communication-engineering",   "Electronics and Communication Engineering"),
    ("electrical-and-electronics-engineering",      "Electrical and Electronics Engineering"),
    ("information-technology",                      "Information Technology"),
    ("basic-sciences-and-humanities",               "Basic Sciences and Humanities"),
]

TITLE_PREFIXES = {"dr", "mr", "ms", "mrs", "prof", "er"}

def slug_to_name(slug: str) -> str:
    """Convert URL slug like 'dr-j-naga-vishnu-vardhan' to 'Dr. J Naga Vishnu Vardhan'"""
    parts = slug.split("-")
    if not parts:
        return ""
    result = []
    for i, part in enumerate(parts):
        if i == 0 and part.lower() in TITLE_PREFIXES:
            result.append(part.capitalize() + ".")
        else:
            result.append(part.capitalize())
    return " ".join(result)

def is_faculty_url(url: str) -> bool:
    url = url.lower().rstrip("/")
    last = url.split("/")[-1]
    parts = last.split("-")
    return parts and parts[0].lower() in TITLE_PREFIXES or any(
        p in url for p in ["/dr-", "/mr-", "/ms-", "/mrs-", "/prof-"]
    )

lines = MANIFEST.read_text(encoding="utf-8").splitlines()
updated_lines = []
changed = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    img = json.loads(line)
    page_url = (img.get("page_url") or "").lower().rstrip("/")
    updated = dict(img)
    modified = False

    # Infer department from page_url
    if not img.get("department"):
        for pattern, dept_name in DEPT_URL_MAP:
            if pattern in page_url:
                updated["department"] = dept_name
                modified = True
                break

    # Infer category and name for faculty profile pages
    slug = page_url.split("/")[-1] if page_url else ""
    slug_parts = slug.split("-")
    is_faculty_page = slug_parts and slug_parts[0] in TITLE_PREFIXES

    if is_faculty_page:
        if img.get("category") in ("other", None, ""):
            updated["category"] = "faculty"
            modified = True
        if not img.get("semantic_name"):
            updated["semantic_name"] = slug_to_name(slug)
            modified = True
        # Also set context_heading if missing
        if not img.get("context_heading") and updated.get("semantic_name"):
            updated["context_heading"] = updated["semantic_name"]
            modified = True
    elif updated.get("department") and img.get("category") in ("other", None, ""):
        updated["category"] = "department"
        modified = True

    if modified:
        changed += 1
    updated_lines.append(json.dumps(updated, ensure_ascii=False))

MANIFEST.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
print(f"Updated {changed} of {len(updated_lines)} images")

# Summary
from collections import Counter
depts = Counter()
cats = Counter()
named = 0
for line in updated_lines:
    img = json.loads(line)
    if img.get("status") == "saved":
        depts[img.get("department") or "None"] += 1
        cats[img.get("category") or "other"] += 1
        if img.get("semantic_name"):
            named += 1

print(f"\nImages with semantic_name: {named}")
print("\nCategory breakdown:")
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print("\nDepartment breakdown:")
for dept, count in sorted(depts.items(), key=lambda x: -x[1]):
    print(f"  {dept}: {count}")
