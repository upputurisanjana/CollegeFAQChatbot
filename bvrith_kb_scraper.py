#!/usr/bin/env python3
"""
================================================================================
 BVRIT HYDERABAD — Hardened Knowledge Base Scraper (v2)
================================================================================
Builds on the original bvrith_kb_scraper.py design (verbatim text, source_url +
retrieved_at_utc + content_hash on everything, PDF page-level citations) and
adds two things that matter for a real unattended run against a live WordPress
site: it cannot get stuck, and it actually captures images properly instead of
just logging <img src> tags.

WHY THE ORIGINAL VERSION CAN STALL OR RUN FOREVER ON A SITE LIKE THIS
----------------------------------------------------------------------
WordPress sites generate near-infinite URL spaces that a naive same-domain BFS
will happily crawl forever: comment-reply links (?replytocom=N), calendar
widgets (?day=/?month=), tag/category cross-products, feed URLs, attachment
pages (one per image, linking back to itself), and unbounded pagination
(/page/2/, /page/3/, ... /page/9999/) on category archives. A single hanging
TCP connection (bad proxy, DNS blackhole, slow-loris server) can also block a
sequential crawler indefinitely if there's no hard wall-clock cap. This
version defends against all of that. See the numbered fixes below.

FIX 1 — Trap-URL filtering (never queue known infinite patterns)
FIX 2 — Max crawl depth (bounds any pattern the filter misses)
FIX 3 — Global wall-clock budget + per-request hard timeout via a future,
         so even a connection that ignores requests' own timeout can't hang
         the process forever
FIX 4 — Per-host circuit breaker (N consecutive failures -> stop, don't spin)
FIX 5 — Periodic checkpointing + --resume, so a run that's killed partway
         through doesn't lose progress or have to restart from zero
FIX 6 — Duplicate-content detection (content_hash) to stop re-processing
         paginated archives that repeat the same posts

IMAGE HANDLING — WHY THE ORIGINAL VERSION MISSES MOST IMAGES
--------------------------------------------------------------
The original only reads <img src>. Modern WordPress themes lazy-load images
(data-src / data-lazy-src / data-original), serve responsive variants via
srcset, and use CSS background-image in inline styles for hero banners and
department cards. All of that is now captured, downloaded, verified as a real
image (not an HTML error page served with a 200 status), deduplicated by
content hash (so a logo used in the header on all 300 pages isn't downloaded
300 times), and filtered by minimum size so 1x1 tracking pixels and nav icons
don't pollute the output.

RUN
---
    pip install -r requirements.txt   # requests, beautifulsoup4, lxml, Pillow
    python bvrith_kb_scraper_v2.py --max-runtime-minutes 45 --download-images

    # if it gets killed or you Ctrl+C partway through, resume where it left off:
    python bvrith_kb_scraper_v2.py --max-runtime-minutes 45 --download-images --resume

OUTPUTS (default: ./bvrith_knowledge_base/)
    pages.jsonl, chunks.jsonl, pdf_documents.jsonl   — same schema as v1
    images_manifest.jsonl   — one record per unique image (deduplicated),
                              with every page that referenced it
    images/                 — downloaded image files, named by content hash
                              (so identical images never get saved twice)
    crawl_log.csv           — every URL visited, status, notes
    checkpoint.json         — visited-URL state, used by --resume
    run_summary.json        — final stats, including why the run stopped
================================================================================
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, Tag

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False  # degrade to byte-size-only filtering, don't hard-fail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_START_URL = "https://bvrithyderabad.edu.in/"
DEFAULT_ALLOWED_DOMAINS = {"bvrithyderabad.edu.in", "www.bvrithyderabad.edu.in"}
USER_AGENT = "BVRITH-Chatbot-KB-Builder/2.0 (+college knowledge-base crawler)"

DOCUMENT_EXTENSIONS = {".pdf"}
SKIP_EXTENSIONS = {
    ".css", ".js", ".mp4", ".mp3", ".avi", ".mov", ".zip", ".rar",
    ".woff", ".woff2", ".ttf", ".eot", ".ics",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".avif"}
CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
    "image/gif": ".gif", "image/webp": ".webp", "image/svg+xml": ".svg",
    "image/bmp": ".bmp", "image/avif": ".avif",
}

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}

# FIX 1 — regex patterns that identify infinite/low-value WordPress URL spaces.
# Matched against the path+query of a normalized URL; a match means "don't queue."
TRAP_PATTERNS = [
    re.compile(r"/feed/?$"),
    re.compile(r"/wp-json/"),
    re.compile(r"/wp-admin/"),
    re.compile(r"/attachment/"),
    re.compile(r"[?&]replytocom="),
    re.compile(r"[?&](day|month)="),                # calendar widget traps
    re.compile(r"/comment-page-\d+"),
    re.compile(r"/page/(\d+)/?$"),                   # handled specially, see is_trap_url
    re.compile(r"/tag/.+/tag/"),                     # tag cross-product loops
    re.compile(r"\.(ics|xml)(\?|$)"),
]
MAX_PAGINATION_PAGE = 15   # /page/16/ and beyond on any archive is diminishing returns
MAX_QUERY_PARAMS = 4
MAX_URL_LENGTH = 300
MAX_CRAWL_DEPTH = 8        # FIX 2 — hard ceiling regardless of trap-pattern coverage

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Image filtering thresholds — tuned to drop nav icons / spacer gifs / social
# icons while keeping real campus/department/faculty photos.
MIN_IMAGE_BYTES = 3_000        # ~3KB — below this is almost always an icon
MIN_IMAGE_DIMENSION = 150      # px, either side — only checked if Pillow is available

STOP_EVENT_CHECK_INTERVAL = 1  # seconds, how often the watchdog re-checks the clock


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str, base: str = None) -> str:
    if base:
        url = urljoin(base, url)
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(parsed.query) if k not in TRACKING_PARAMS]
    q.sort()
    query = urlencode(q)
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def get_extension(url: str) -> str:
    import os
    path = urlparse(url).path.lower()
    return os.path.splitext(path)[1]


def is_probably_html(content_type: str) -> bool:
    return "text/html" in (content_type or "").lower()


def is_trap_url(url: str) -> bool:
    """FIX 1 — reject known infinite/low-value URL patterns before they're
    ever queued. This is checked in addition to (not instead of) depth
    limiting and the per-host circuit breaker."""
    if len(url) > MAX_URL_LENGTH:
        return True
    parsed = urlparse(url)
    n_params = len(parse_qsl(parsed.query))
    if n_params > MAX_QUERY_PARAMS:
        return True
    page_match = re.search(r"/page/(\d+)/?$", parsed.path)
    if page_match and int(page_match.group(1)) > MAX_PAGINATION_PAGE:
        return True
    full = parsed.path + ("?" + parsed.query if parsed.query else "")
    return any(p.search(full) for p in TRAP_PATTERNS)


def parse_srcset(srcset: str):
    """Return list of URLs from a srcset attribute, largest-descriptor first."""
    candidates = []
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        url = bits[0]
        width = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                width = int(bits[1][:-1])
            except ValueError:
                width = 0
        candidates.append((width, url))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [url for _, url in candidates]


BACKGROUND_IMAGE_RE = re.compile(r"background(-image)?\s*:\s*url\((['\"]?)(.*?)\2\)", re.I)


# --------------------------------------------------------------------------
# HTTP session with retry/backoff mounted (handles transient 5xx/429 without
# the crawler having to hand-roll retry loops everywhere)
# --------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def hard_timeout_get(session, url, timeout_connect_read, hard_ceiling_seconds, executor):
    """FIX 3 — wraps session.get in a future with an outer hard timeout, so a
    connection that hangs past what `requests`' own timeout tuple should
    enforce (rare, but happens with some proxies/misbehaving servers) still
    can't block the crawler indefinitely."""
    future = executor.submit(session.get, url, timeout=timeout_connect_read)
    try:
        return future.result(timeout=hard_ceiling_seconds)
    except FutureTimeoutError:
        logging.warning("Hard timeout ceiling hit for %s (>%ss) — abandoning", url, hard_ceiling_seconds)
        return None
    except requests.RequestException as e:
        logging.warning("Request failed for %s: %s", url, e)
        return None


# --------------------------------------------------------------------------
# Page content extraction (text side — same shape as v1, condensed here)
# --------------------------------------------------------------------------

class PageContent:
    def __init__(self, url):
        self.url = url
        self.title = ""
        self.meta_description = ""
        self.breadcrumb = []
        self.headings = []
        self.blocks = []
        self.tables = []
        self.image_refs = []   # raw candidate image URLs + context, before download
        self.doc_links = []
        self.outlinks = []
        self.full_text = ""
        self.retrieved_at = now_iso()
        self.content_hash = ""


def collect_image_candidates(main_scope, page_url):
    """Gathers every plausible image reference: <img src>, lazy-load attrs,
    srcset (all resolutions -> take the largest), <picture><source>, and
    inline CSS background-image. Returns a de-duplicated (by URL) list."""
    found = {}  # url -> {"alt", "context_heading", "caption"}
    last_heading = ""

    lazy_attrs = ["src", "data-src", "data-lazy-src", "data-original", "data-srcset"]

    for el in main_scope.descendants:
        if not isinstance(el, Tag):
            continue
        if re.match(r"h[1-6]$", el.name or ""):
            last_heading = el.get_text(" ", strip=True)

        if el.name == "img":
            candidate_urls = []
            for attr in lazy_attrs:
                val = el.get(attr)
                if not val:
                    continue
                if "srcset" in attr:
                    candidate_urls.extend(parse_srcset(val))
                else:
                    candidate_urls.append(val)
            srcset = el.get("srcset")
            if srcset:
                candidate_urls.extend(parse_srcset(srcset))

            chosen = next((u for u in candidate_urls if u and not u.startswith("data:")), None)
            if chosen:
                abs_url = normalize_url(chosen, base=page_url)
                caption = ""
                fig = el.find_parent("figure")
                if fig:
                    cap_tag = fig.find("figcaption")
                    if cap_tag:
                        caption = cap_tag.get_text(" ", strip=True)
                found.setdefault(abs_url, {
                    "alt": (el.get("alt") or "").strip(),
                    "context_heading": last_heading,
                    "caption": caption,
                })

        if el.name == "source" and el.get("srcset"):
            for u in parse_srcset(el["srcset"]):
                abs_url = normalize_url(u, base=page_url)
                found.setdefault(abs_url, {"alt": "", "context_heading": last_heading, "caption": ""})

        style = el.get("style") if el.has_attr("style") else None
        if style:
            m = BACKGROUND_IMAGE_RE.search(style)
            if m:
                abs_url = normalize_url(m.group(3), base=page_url)
                found.setdefault(abs_url, {"alt": "", "context_heading": last_heading, "caption": "background-image"})

    return [{"src": url, **meta} for url, meta in found.items()]


# ==========================================================================
# SEMANTIC IMAGE TAGGING — extract faculty names, departments, categories
# ==========================================================================

def extract_faculty_name_from_url(url: str) -> str:
    """
    Extract faculty name from URL patterns like:
    /computer-science-and-engineering/ms-k-vineela → "Ms. K Vineela"
    /ece/dr-sarath-babu → "Dr. Sarath Babu"
    """
    match = re.search(r'/(ms|mr|dr|prof)-([a-z-]+)/?$', url.lower())
    if match:
        title = match.group(1).upper()
        name_parts = match.group(2).split('-')
        name = ' '.join(word.capitalize() for word in name_parts)
        # Handle title formatting
        if title == 'MS':
            return f"Ms. {name}"
        elif title == 'MR':
            return f"Mr. {name}"
        elif title == 'DR':
            return f"Dr. {name}"
        elif title == 'PROF':
            return f"Prof. {name}"
    return ""


def extract_department_from_url(url: str) -> str:
    """Map URL path to department name."""
    dept_map = {
        'computer-science-and-engineering': 'Computer Science and Engineering',
        'cse-artificial-intelligence': 'CSE (Artificial Intelligence & Machine Learning)',
        'electronics-and-communication': 'Electronics and Communication Engineering',
        'electrical-and-electronics': 'Electrical and Electronics Engineering',
        'information-technology': 'Information Technology',
        'basic-sciences-and-humanities': 'Basic Sciences and Humanities',
    }
    url_lower = url.lower()
    for key, val in dept_map.items():
        if key in url_lower:
            return val
    return ""


def categorize_image(src_url: str, page_url: str, page_title: str, context_heading: str, alt_text: str) -> dict:
    """
    Assign semantic category and searchable metadata to an image.
    Returns dict with: category, semantic_name, department, searchable_text
    """
    url_lower = page_url.lower()
    title_lower = page_title.lower()
    context_lower = context_heading.lower()
    
    # FACULTY: Individual faculty bio pages
    dept_patterns = ['/computer-science-and-engineering/', '/cse-artificial', 
                     '/electronics-and-communication/', '/electrical-and-electronics/',
                     '/information-technology/', '/basic-sciences']
    name_patterns = ['ms-', 'mr-', 'dr-', 'prof-']
    
    is_faculty_page = any(dept in url_lower for dept in dept_patterns) and \
                      any(pattern in url_lower for pattern in name_patterns)
    
    if is_faculty_page:
        faculty_name = extract_faculty_name_from_url(page_url)
        department = extract_department_from_url(page_url)
        searchable = f"{faculty_name} faculty teacher professor staff {department} {context_heading}".lower()
        return {
            "category": "faculty",
            "semantic_name": faculty_name,
            "department": department,
            "searchable_text": searchable.strip()
        }
    
    # CAMPUS: Building/facility photos
    campus_keywords = ['campus', 'building', 'hostel', 'library', 'lab', 'facility', 'classroom', 'auditorium']
    if any(kw in title_lower or kw in context_lower for kw in campus_keywords):
        searchable = f"campus building facility {context_heading} {alt_text}".lower()
        return {
            "category": "campus",
            "semantic_name": "",
            "department": "",
            "searchable_text": searchable.strip()
        }
    
    # EVENT: News/award/event photos
    event_keywords = ['/category/news', '/category/awards', '/category/events']
    if any(kw in url_lower for kw in event_keywords) or 'event' in context_lower or 'award' in context_lower:
        searchable = f"event award news {context_heading} {alt_text}".lower()
        return {
            "category": "event",
            "semantic_name": "",
            "department": "",
            "searchable_text": searchable.strip()
        }
    
    # DEPARTMENT: Department-level pages (not individual faculty)
    if any(dept in url_lower for dept in dept_patterns) and not any(p in url_lower for p in name_patterns):
        department = extract_department_from_url(page_url)
        searchable = f"department {department} {context_heading} {alt_text}".lower()
        return {
            "category": "department",
            "semantic_name": "",
            "department": department,
            "searchable_text": searchable.strip()
        }
    
    # Default: OTHER
    searchable = f"{context_heading} {alt_text}".lower()
    return {
        "category": "other",
        "semantic_name": "",
        "department": "",
        "searchable_text": searchable.strip()
    }


def extract_page(html_text: str, url: str) -> PageContent:
    pc = PageContent(url)
    soup = BeautifulSoup(html_text, "lxml")

    if soup.title and soup.title.string:
        pc.title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        pc.title = h1.get_text(strip=True) if h1 else url

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        pc.meta_description = meta_desc["content"].strip()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = normalize_url(href, base=url)
        ext = get_extension(abs_url)
        if ext in DOCUMENT_EXTENSIONS:
            pc.doc_links.append({"url": abs_url, "link_text": a.get_text(" ", strip=True), "ext": ext})
        else:
            pc.outlinks.append(abs_url)

    bc = soup.find(class_=re.compile("breadcrumb", re.I))
    if bc:
        pc.breadcrumb = [x.get_text(strip=True) for x in bc.find_all(["a", "span"]) if x.get_text(strip=True)]

    main = (
        soup.find("main") or soup.find("article") or soup.find(id="main")
        or soup.find(class_=re.compile(r"\belementor\b", re.I)) or soup.body or soup
    )

    # Harvest images BEFORE stripping nav/footer/header — a lot of real
    # content photos live inside theme "main content" wrappers that vary,
    # but we still want hero images even if a template puts them oddly.
    pc.image_refs = collect_image_candidates(main, url)

    for tag_name in ["script", "style", "noscript", "iframe", "form", "svg", "header", "footer", "nav"]:
        for t in main.find_all(tag_name):
            t.decompose()

    def block_text_of(tag):
        return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()

    for el in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        if el.name != "table" and el.find_parent("table") is not None:
            continue
        if el.name.startswith("h"):
            text = block_text_of(el)
            if text:
                level = int(el.name[1])
                pc.headings.append({"level": level, "text": text})
                pc.blocks.append(f"{'#' * level} {text}")
        elif el.name == "p":
            text = block_text_of(el)
            if text:
                pc.blocks.append(text)
        elif el.name == "li":
            text = block_text_of(el)
            if text:
                pc.blocks.append(f"- {text}")
        elif el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [block_text_of(td) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(cells)
            if rows:
                pc.tables.append(rows)
                pc.blocks.append("\n".join("| " + " | ".join(r) + " |" for r in rows))

    pc.full_text = "\n\n".join(pc.blocks).strip()
    if not pc.full_text:
        pc.full_text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))
    pc.content_hash = sha256_text(pc.full_text)
    return pc


def chunk_text(text, source_url, title, retrieved_at, breadcrumb, extra=None):
    chunks = []
    if not text:
        return chunks
    start, idx, n = 0, 0, len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start + 200:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            record = {
                "chunk_id": sha256_text(f"{source_url}|{idx}|{chunk[:50]}")[:16],
                "source_url": source_url, "page_title": title, "breadcrumb": breadcrumb,
                "chunk_index": idx, "text": chunk, "retrieved_at_utc": retrieved_at,
            }
            if extra:
                record.update(extra)
            chunks.append(record)
            idx += 1
        start = end - CHUNK_OVERLAP if end < n else n
        if start <= 0:
            start = end
    return chunks


def extract_pdf_text(pdf_bytes: bytes):
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages
    except ImportError:
        pass
    except Exception as e:
        logging.warning("pdfplumber failed (%s), trying pypdf", e)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception as e:
        logging.error("PDF text extraction failed entirely: %s", e)
    return pages


# --------------------------------------------------------------------------
# Image downloading (separate, concurrent, deduplicated pipeline)
# --------------------------------------------------------------------------

class ImageDownloader:
    """Downloads, verifies, deduplicates (by content hash, not URL — the same
    logo served from 5 different URLs is still only saved once), and filters
    out icon-sized noise. Runs on its own thread pool so a slow image host
    never blocks the HTML crawl."""

    def __init__(self, session, output_dir: Path, max_workers=8, min_bytes=MIN_IMAGE_BYTES,
                 min_dimension=MIN_IMAGE_DIMENSION, timeout=(5, 15)):
        self.session = session
        self.dir = output_dir / "images"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.min_bytes = min_bytes
        self.min_dimension = min_dimension
        self.timeout = timeout
        self.seen_hashes = set()   # dedup by content, across the whole run
        self.seen_urls = set()     # dedup by URL, avoid re-downloading the same link twice
        self.lock = __import__("threading").Lock()

    def _passes_dimension_filter(self, data: bytes) -> bool:
        if not PIL_AVAILABLE:
            return True  # degrade gracefully — byte-size filter still applies
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                return w >= self.min_dimension and h >= self.min_dimension
        except Exception:
            return True  # not a format Pillow can read (e.g. some SVGs) — keep it, don't drop silently

    def _download_one(self, url: str):
        try:
            resp = self.session.get(url, timeout=self.timeout, stream=True)
        except requests.RequestException as e:
            return {"url": url, "status": "error", "error": str(e)}
        if resp.status_code != 200:
            return {"url": url, "status": resp.status_code}

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return {"url": url, "status": "not_an_image", "content_type": content_type}

        data = resp.content
        if len(data) < self.min_bytes:
            return {"url": url, "status": "too_small_bytes", "size": len(data)}
        if not self._passes_dimension_filter(data):
            return {"url": url, "status": "too_small_dimensions"}

        content_hash = sha256_bytes(data)
        with self.lock:
            if content_hash in self.seen_hashes:
                return {"url": url, "status": "duplicate_content", "content_hash": content_hash}
            self.seen_hashes.add(content_hash)

        ext = CONTENT_TYPE_TO_EXT.get(content_type, get_extension(url) or ".jpg")
        local_path = self.dir / f"{content_hash[:20]}{ext}"
        with open(local_path, "wb") as fh:
            fh.write(data)

        return {
            "url": url, "status": "saved", "local_path": str(local_path),
            "content_hash": content_hash, "bytes": len(data), "content_type": content_type,
        }

    def submit(self, url: str):
        with self.lock:
            if url in self.seen_urls:
                return None
            self.seen_urls.add(url)
        return self.executor.submit(self._download_one, url)

    def shutdown(self):
        self.executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# Main crawler
# --------------------------------------------------------------------------

class KnowledgeBaseCrawler:
    def __init__(self, start_url, allowed_domains, output_dir, max_pages=1000, max_pdfs=500,
                 delay=0.5, connect_read_timeout=(5, 15), hard_timeout_ceiling=30,
                 max_runtime_minutes=60, max_depth=MAX_CRAWL_DEPTH,
                 download_images=True, download_pdfs=True, respect_robots=True,
                 max_consecutive_errors=15, resume=False, image_workers=8):
        self.start_url = normalize_url(start_url)
        self.allowed_domains = allowed_domains
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.max_pdfs = max_pdfs
        self.delay = delay
        self.connect_read_timeout = connect_read_timeout
        self.hard_timeout_ceiling = hard_timeout_ceiling
        self.deadline = time.monotonic() + max_runtime_minutes * 60   # FIX 3
        self.max_depth = max_depth
        self.download_images = download_images
        self.download_pdfs = download_pdfs
        self.respect_robots = respect_robots
        self.max_consecutive_errors = max_consecutive_errors          # FIX 4
        self.resume = resume

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = build_session()
        self.request_executor = ThreadPoolExecutor(max_workers=1)  # used only for the hard-timeout wrapper
        self.image_downloader = ImageDownloader(self.session, self.output_dir, max_workers=image_workers,
                                                 timeout=connect_read_timeout) if download_images else None

        self.visited_pages = set()
        self.visited_pdfs = set()
        self.content_hashes_seen = set()   # FIX 6 — skip near-duplicate archive pages
        self.queue = deque([(self.start_url, 0)])   # (url, depth)
        self.pdf_referrers = {}
        self.consecutive_errors = 0
        self.stop_reason = None

        self.robots = None
        if respect_robots:
            self.robots = RobotFileParser()
            try:
                self.robots.set_url(urljoin(self.start_url, "/robots.txt"))
                self.robots.read()
            except Exception as e:
                logging.warning("Could not read robots.txt (%s); proceeding without robots gating", e)
                self.robots = None

        checkpoint_path = self.output_dir / "checkpoint.json"
        if resume and checkpoint_path.exists():
            self._load_checkpoint(checkpoint_path)

        mode = "a" if (resume and (self.output_dir / "pages.jsonl").exists()) else "w"
        self.f_pages = open(self.output_dir / "pages.jsonl", mode, encoding="utf-8")
        self.f_chunks = open(self.output_dir / "chunks.jsonl", mode, encoding="utf-8")
        self.f_images = open(self.output_dir / "images_manifest.jsonl", mode, encoding="utf-8")
        self.f_pdfs = open(self.output_dir / "pdf_documents.jsonl", mode, encoding="utf-8")
        self.f_log = open(self.output_dir / "crawl_log.csv", mode, newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.f_log)
        if mode == "w":
            self.log_writer.writerow(["timestamp_utc", "url", "type", "status", "notes"])

        self.stats = {"pages": 0, "images_saved": 0, "images_skipped": 0, "pdfs": 0, "chunks": 0, "errors": 0}

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self._interrupted = False

    def _handle_interrupt(self, signum, frame):
        logging.warning("Interrupt received — finishing current page, then checkpointing and exiting cleanly.")
        self._interrupted = True
        self.stop_reason = "interrupted"

    def _load_checkpoint(self, path):
        try:
            data = json.loads(path.read_text())
            self.visited_pages = set(data.get("visited_pages", []))
            self.visited_pdfs = set(data.get("visited_pdfs", []))
            self.pdf_referrers = data.get("pdf_referrers", {})
            logging.info("Resumed from checkpoint: %d pages, %d pdfs already visited, %d pdf referrers",
                         len(self.visited_pages), len(self.visited_pdfs), len(self.pdf_referrers))
        except Exception as e:
            logging.warning("Could not load checkpoint (%s); starting fresh", e)

        # Pre-seed image dedup hashes from files already saved on disk so we
        # don't re-download content we already have.
        if self.image_downloader:
            images_dir = self.output_dir / "images"
            if images_dir.exists():
                pre_seeded = 0
                for img_file in images_dir.iterdir():
                    # Filenames are <content_hash[:20]><ext> — extract the hash prefix
                    stem = img_file.stem  # first 20 hex chars of the sha256
                    if len(stem) == 20:
                        with self.image_downloader.lock:
                            self.image_downloader.seen_hashes.add(stem)
                        pre_seeded += 1
                logging.info("Pre-seeded %d image hashes from existing files on disk", pre_seeded)

    def _save_checkpoint(self):
        data = {
            "visited_pages": list(self.visited_pages),
            "visited_pdfs": list(self.visited_pdfs),
            "pdf_referrers": self.pdf_referrers,
        }
        (self.output_dir / "checkpoint.json").write_text(json.dumps(data))

    def _log(self, url, type_, status, notes=""):
        self.log_writer.writerow([now_iso(), url, type_, status, notes])
        self.f_log.flush()

    def _allowed_by_domain(self, url):
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        return netloc in {d.replace("www.", "") for d in self.allowed_domains}

    def _allowed_by_robots(self, url):
        if not self.respect_robots or self.robots is None:
            return True
        try:
            return self.robots.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def _time_remaining(self) -> bool:
        return time.monotonic() < self.deadline

    def _get(self, url):
        return hard_timeout_get(self.session, url, self.connect_read_timeout, self.hard_timeout_ceiling, self.request_executor)

    def _get_pdf(self, url):
        """Like _get but with a much more generous timeout for large PDF files."""
        pdf_read_timeout = max(120.0, self.connect_read_timeout[1] if isinstance(self.connect_read_timeout, tuple) else self.connect_read_timeout)
        pdf_timeout = (self.connect_read_timeout[0] if isinstance(self.connect_read_timeout, tuple) else 5.0, pdf_read_timeout)
        pdf_hard_ceiling = max(150.0, self.hard_timeout_ceiling)
        return hard_timeout_get(self.session, url, pdf_timeout, pdf_hard_ceiling, self.request_executor)

    # ---- main loop ---------------------------------------------------
    def crawl(self):
        logging.info("Starting crawl at %s | deadline in %.0fs | max_pages=%d | max_depth=%d",
                     self.start_url, self.deadline - time.monotonic(), self.max_pages, self.max_depth)

        last_checkpoint = time.monotonic()

        while self.queue and self.stats["pages"] < self.max_pages:
            if self._interrupted:
                self.stop_reason = self.stop_reason or "interrupted"
                break
            if not self._time_remaining():
                self.stop_reason = "max_runtime_reached"
                logging.warning("Wall-clock budget exhausted — stopping crawl gracefully.")
                break
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.stop_reason = "circuit_breaker_tripped"
                logging.error("Too many consecutive errors (%d) — site may be down or blocking us. Stopping.",
                              self.consecutive_errors)
                break

            url, depth = self.queue.popleft()
            if url in self.visited_pages:
                continue
            self.visited_pages.add(url)

            if depth > self.max_depth:
                self._log(url, "page", "skipped_max_depth")
                continue
            if not self._allowed_by_domain(url):
                continue
            if is_trap_url(url):
                self._log(url, "page", "skipped_trap_pattern")
                continue
            if not self._allowed_by_robots(url):
                self._log(url, "page", "skipped_robots")
                continue

            resp = self._get(url)
            time.sleep(self.delay)

            if resp is None:
                self.stats["errors"] += 1
                self.consecutive_errors += 1
                self._log(url, "page", "request_failed_or_timed_out")
                continue
            if resp.status_code != 200:
                self._log(url, "page", resp.status_code)
                if resp.status_code >= 500:
                    self.consecutive_errors += 1
                continue
            if not is_probably_html(resp.headers.get("Content-Type", "")):
                self._log(url, "page", "skipped_non_html")
                continue

            self.consecutive_errors = 0  # reset on any success

            try:
                pc = extract_page(resp.text, url)
            except Exception as e:
                logging.exception("Failed to parse %s", url)
                self.stats["errors"] += 1
                self._log(url, "page", "parse_error", str(e))
                continue

            if pc.content_hash in self.content_hashes_seen:
                self._log(url, "page", "skipped_duplicate_content")
                continue
            self.content_hashes_seen.add(pc.content_hash)

            self._store_page(pc)
            self._log(url, "page", 200, f"depth={depth} blocks={len(pc.blocks)} images={len(pc.image_refs)} docs={len(pc.doc_links)}")
            self.stats["pages"] += 1
            logging.info("[%d/%d] depth=%d %s — %s", self.stats["pages"], self.max_pages, depth, url, pc.title[:60])

            for link in pc.outlinks:
                ext = get_extension(link)
                if ext in SKIP_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                    continue
                if self._allowed_by_domain(link) and link not in self.visited_pages and not is_trap_url(link):
                    self.queue.append((link, depth + 1))

            for doc in pc.doc_links:
                self.pdf_referrers.setdefault(doc["url"], []).append({"page": url, "link_text": doc["link_text"]})

            if time.monotonic() - last_checkpoint > 20:   # FIX 5 — checkpoint every ~20s of work
                self._save_checkpoint()
                last_checkpoint = time.monotonic()

        if not self.stop_reason:
            self.stop_reason = "queue_exhausted" if not self.queue else "max_pages_reached"

        if self.download_pdfs:
            self._process_pdfs()
        self._drain_image_downloads()
        self._save_checkpoint()
        self._write_summary()
        self._close()

    def _store_page(self, pc: PageContent):
        record = {
            "url": pc.url, "title": pc.title, "meta_description": pc.meta_description,
            "breadcrumb": pc.breadcrumb, "headings": pc.headings, "tables": pc.tables,
            "num_image_refs": len(pc.image_refs), "content_hash": pc.content_hash,
            "retrieved_at_utc": pc.retrieved_at, "full_text": pc.full_text,
        }
        self.f_pages.write(json.dumps(record, ensure_ascii=False) + "\n")

        chunks = chunk_text(pc.full_text, pc.url, pc.title, pc.retrieved_at, pc.breadcrumb)
        for c in chunks:
            self.f_chunks.write(json.dumps(c, ensure_ascii=False) + "\n")
        self.stats["chunks"] += len(chunks)

        if self.download_images and self.image_downloader:
            for img in pc.image_refs:
                future = self.image_downloader.submit(img["src"])
                if future is not None:
                    # Build semantic tags for this image based on the page it came from
                    sem = categorize_image(
                        src_url=img["src"],
                        page_url=pc.url,
                        page_title=pc.title,
                        context_heading=img.get("context_heading", ""),
                        alt_text=img.get("alt", ""),
                    )
                    future.page_context = {   # type: ignore[attr-defined]
                        "page_url": pc.url,
                        "page_title": pc.title,
                        "alt": img["alt"],
                        "context_heading": img["context_heading"],
                        "caption": img["caption"],
                        "retrieved_at_utc": pc.retrieved_at,
                        # Semantic fields
                        "category": sem["category"],
                        "semantic_name": sem["semantic_name"],
                        "department": sem["department"],
                        "searchable_text": sem["searchable_text"],
                    }
                    if not hasattr(self, "_pending_image_futures"):
                        self._pending_image_futures = []
                    self._pending_image_futures.append(future)

    def _drain_image_downloads(self):
        futures = getattr(self, "_pending_image_futures", [])
        logging.info("Waiting on %d queued image downloads...", len(futures))
        for future in futures:
            try:
                result = future.result(timeout=60)
            except Exception as e:
                result = {"status": "error", "error": str(e)}
            ctx = getattr(future, "page_context", {})
            record = {**ctx, **result}
            self.f_images.write(json.dumps(record, ensure_ascii=False) + "\n")
            if result.get("status") == "saved":
                self.stats["images_saved"] += 1
            else:
                self.stats["images_skipped"] += 1
        if self.image_downloader:
            self.image_downloader.shutdown()

    def _process_pdfs(self):
        pdf_urls = [u for u in self.pdf_referrers if u not in self.visited_pdfs][: self.max_pdfs]
        logging.info("Processing %d linked PDF documents...", len(pdf_urls))
        for pdf_url in pdf_urls:
            if not self._time_remaining():
                logging.warning("Wall-clock budget exhausted during PDF processing — stopping early.")
                break
            self.visited_pdfs.add(pdf_url)
            if not self._allowed_by_robots(pdf_url):
                self._log(pdf_url, "pdf", "skipped_robots")
                continue
            resp = self._get_pdf(pdf_url)
            time.sleep(self.delay)
            if resp is None or resp.status_code != 200:
                self._log(pdf_url, "pdf", resp.status_code if resp else "request_failed")
                self.stats["errors"] += 1
                continue

            if self.download_pdfs:
                fname = re.sub(r"[^A-Za-z0-9._-]", "_", urlparse(pdf_url).path.rsplit("/", 1)[-1]) or "document.pdf"
                (self.output_dir / "pdfs").mkdir(exist_ok=True)
                with open(self.output_dir / "pdfs" / fname, "wb") as fh:
                    fh.write(resp.content)

            pages_text = extract_pdf_text(resp.content)
            full_text = "\n\n".join(pages_text).strip()
            retrieved_at = now_iso()
            record = {
                "pdf_url": pdf_url, "referenced_from": self.pdf_referrers.get(pdf_url, []),
                "num_pages": len(pages_text), "content_hash": sha256_text(full_text),
                "retrieved_at_utc": retrieved_at, "full_text": full_text,
            }
            self.f_pdfs.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.stats["pdfs"] += 1
            self._log(pdf_url, "pdf", 200, f"{len(pages_text)} pages extracted")

            for page_num, page_text in enumerate(pages_text, start=1):
                page_text = (page_text or "").strip()
                if not page_text:
                    continue
                title = urlparse(pdf_url).path.rsplit("/", 1)[-1]
                chunks = chunk_text(page_text, pdf_url, title, retrieved_at, [], extra={"pdf_page_number": page_num})
                for c in chunks:
                    self.f_chunks.write(json.dumps(c, ensure_ascii=False) + "\n")
                self.stats["chunks"] += len(chunks)

    def _write_summary(self):
        summary = {
            "start_url": self.start_url, "allowed_domains": sorted(self.allowed_domains),
            "finished_at_utc": now_iso(), "stop_reason": self.stop_reason,
            "pages_crawled": self.stats["pages"], "images_saved": self.stats["images_saved"],
            "images_skipped": self.stats["images_skipped"], "pdfs_processed": self.stats["pdfs"],
            "total_chunks": self.stats["chunks"], "errors": self.stats["errors"],
            "unique_image_hashes": len(self.image_downloader.seen_hashes) if self.image_downloader else 0,
        }
        (self.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        logging.info("=" * 60)
        logging.info("DONE (%s). %s", self.stop_reason, json.dumps(summary, indent=2))

    def _close(self):
        for f in [self.f_pages, self.f_chunks, self.f_images, self.f_pdfs, self.f_log]:
            f.close()
        self.request_executor.shutdown(wait=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hardened BVRIT Hyderabad knowledge-base scraper (v2).")
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--allowed-domains", nargs="*", default=list(DEFAULT_ALLOWED_DOMAINS))
    parser.add_argument("--output-dir", default="bvrith_knowledge_base")
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-pdfs", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=MAX_CRAWL_DEPTH)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests (politeness).")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=15.0)
    parser.add_argument("--hard-timeout-ceiling", type=float, default=30.0,
                         help="Absolute max seconds to wait for any single request, no matter what.")
    parser.add_argument("--max-runtime-minutes", type=float, default=60.0,
                         help="Whole-run wall-clock budget. Crawl stops gracefully and writes output when hit.")
    parser.add_argument("--max-consecutive-errors", type=int, default=15,
                         help="Stop the whole crawl if this many requests in a row fail (site likely down/blocking).")
    parser.add_argument("--download-images", action="store_true", default=True)
    parser.add_argument("--no-download-images", dest="download_images", action="store_false")
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--download-pdfs", action="store_true", default=False)
    parser.add_argument("--no-download-pdfs", dest="download_pdfs", action="store_false")
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint.json in --output-dir.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    if not PIL_AVAILABLE:
        logging.warning("Pillow not installed — image dimension filtering disabled, falling back to byte-size only. "
                         "Install with: pip install Pillow")

    crawler = KnowledgeBaseCrawler(
        start_url=args.start_url,
        allowed_domains=set(args.allowed_domains),
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_pdfs=args.max_pdfs,
        delay=args.delay,
        connect_read_timeout=(args.connect_timeout, args.read_timeout),
        hard_timeout_ceiling=args.hard_timeout_ceiling,
        max_runtime_minutes=args.max_runtime_minutes,
        max_depth=args.max_depth,
        download_images=args.download_images,
        download_pdfs=args.download_pdfs,
        respect_robots=not args.ignore_robots,
        max_consecutive_errors=args.max_consecutive_errors,
        resume=args.resume,
        image_workers=args.image_workers,
    )
    crawler.crawl()


if __name__ == "__main__":
    main()