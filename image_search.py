"""
image_search.py — Semantic image retrieval for BVRIT FAQ Chatbot
=================================================================
Uses the enriched images_manifest.jsonl produced by the v2 scraper,
which includes: category, semantic_name, department, searchable_text.
"""

import json
import re
from pathlib import Path
from typing import Optional

IMAGES_FILE = Path("bvrith_knowledge_base/images_manifest.jsonl")

DEPT_KEYWORDS = {
    "cse": "Computer Science and Engineering",
    "computer science": "Computer Science and Engineering",
    "ece": "Electronics and Communication Engineering",
    "ecec": "Electronics and Communication Engineering",   # common typo
    "electronics": "Electronics and Communication Engineering",
    "eee": "Electrical and Electronics Engineering",
    "electrical": "Electrical and Electronics Engineering",
    "it": "Information Technology",
    "information technology": "Information Technology",
    "csm": "CSE (Artificial Intelligence & Machine Learning)",
    "ai": "CSE (Artificial Intelligence & Machine Learning)",
    "aiml": "CSE (Artificial Intelligence & Machine Learning)",
    "artificial intelligence": "CSE (Artificial Intelligence & Machine Learning)",
    "machine learning": "CSE (Artificial Intelligence & Machine Learning)",
    "basic sciences": "Basic Sciences and Humanities",
    "humanities": "Basic Sciences and Humanities",
}


# Known role overrides — these people have specific roles beyond "faculty"
ROLE_OVERRIDES = {
    "kvn sunitha": "principal",
    "k.v.n. sunitha": "principal",
    "dr. kvn sunitha": "principal",
    "dr kvn sunitha": "principal",
}


def _get_role(img: dict) -> str:
    """Return role override if known, else the image category."""
    name = (img.get("semantic_name") or "").lower().strip()
    return ROLE_OVERRIDES.get(name, img.get("category", "other"))


def _normalize_dept(query: str) -> Optional[str]:
    """Return canonical department name if query mentions a known department."""
    q = query.lower()
    for key, val in DEPT_KEYWORDS.items():
        if key in q:
            return val
    return None


def detect_image_request(question: str) -> bool:
    """Check if the user is asking for images/photos."""
    q_lower = question.lower()
    image_keywords = [
        "image", "photo", "picture", "pic", "show", "gallery",
        "looks like", "view", "visual", "see", "display", "give"
    ]
    if any(kw in q_lower for kw in image_keywords):
        return True
    # "list/show/give N faculty/staff/teachers" implicitly wants photos
    faculty_keywords = ["faculty", "teacher", "professor", "staff", "hod", "principal"]
    action_keywords = ["give", "list", "show", "find", "get", "any", "some"]
    has_faculty = any(kw in q_lower for kw in faculty_keywords)
    has_action = any(kw in q_lower for kw in action_keywords)
    return has_faculty and has_action


def is_faculty_image_request(question: str) -> bool:
    """Check if user is asking specifically for faculty/teacher images."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in [
        "teacher", "faculty", "professor", "staff", "hod",
        "instructor", "lecturer", "principal"
    ])


def is_campus_image_request(question: str) -> bool:
    """Check if user is asking for campus/facility images."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in [
        "campus", "building", "college", "hostel", "library",
        "lab", "facility", "ground", "infrastructure"
    ])


def _load_images() -> list[dict]:
    if not IMAGES_FILE.exists():
        return []
    images = []
    with open(IMAGES_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                img = json.loads(line)
                images.append(img)
            except json.JSONDecodeError:
                continue
    return images


def search_images(query: str, limit: int = 5) -> list[dict]:
    """
    Search for images matching the query using semantic fields.
    Priority order:
    1. Category match (faculty/campus/event/department)
    2. Department match
    3. Semantic name match (faculty name)
    4. Keyword match in searchable_text
    """
    images = _load_images()
    if not images:
        return []

    query_lower = query.lower()
    query_words = set(re.sub(r'[^\w\s]', '', query_lower).split())

    # Detect intent
    want_faculty = is_faculty_image_request(query)
    want_campus = is_campus_image_request(query)
    want_principal = any(kw in query_lower for kw in ["principal", "kvn sunitha", "sunitha"])
    target_dept = _normalize_dept(query)

    results = []
    for img in images:
        score = 0
        role = _get_role(img)
        category = img.get("category", "other")
        semantic_name = (img.get("semantic_name") or "").lower()
        department = (img.get("department") or "").lower()
        searchable = (img.get("searchable_text") or "").lower()

        # Principal query — boost principal, skip if strictly faculty dept query
        if want_principal:
            if role == "principal":
                score += 30
            elif not want_faculty:
                continue

        # Category match — strongest signal
        if want_faculty and not want_principal:
            if role == "principal":
                continue  # principal is not a faculty result
            if role in ("faculty", "department"):
                score += 20 if role == "faculty" else 10
            else:
                continue  # strictly exclude non-faculty/dept images

        if want_campus and category == "campus":
            score += 20
        elif want_campus and not want_faculty and category not in ("campus", "department"):
            score -= 5

        # Department match
        if target_dept and target_dept.lower() in department:
            score += 15
        elif target_dept and target_dept.lower() not in department and want_faculty and not want_principal:
            continue  # user asked for specific dept, skip others

        # Semantic name match (faculty name in query)
        if semantic_name and semantic_name in query_lower:
            score += 25

        # Keyword scoring against searchable_text
        searchable_words = set(searchable.split())
        matches = query_words & searchable_words
        score += len(matches) * 2

        # Boost if faculty name appears in query words
        if semantic_name:
            name_words = set(semantic_name.replace('.', '').split())
            name_matches = query_words & name_words
            score += len(name_matches) * 5

        if score > 0:
            img["_score"] = score
            results.append(img)

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return results[:limit]


if __name__ == "__main__":
    # Test
    test_queries = [
        "show me pictures of CSE teachers",
        "images of faculty from ECE department",
        "college campus photos",
        "give any 5 teacher images",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        images = search_images(q, limit=3)
        print(f"Results: {len(images)}")
        for img in images:
            print(f"  [{img.get('_score')}] {img.get('category')} | {img.get('semantic_name') or img.get('context_heading','')[:50]}")
            print(f"       Dept: {img.get('department','')}")
            print(f"       URL: {img.get('url') or img.get('src','')[:80]}")
