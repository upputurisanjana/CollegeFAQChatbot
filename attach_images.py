#!/usr/bin/env python3
"""
attach_images.py

Standalone image-attachment tool for a RAG knowledge base already built by a
separate text scraper (scrape_site.py). It does NOT re-crawl or re-generate
any of your existing text.

What it does
------------
1. Builds its list of (url, md_file) pairs one of two ways:
     a) --scan-dir DIR  (recommended, no manifest.csv needed): recursively
        finds every *.md file under DIR and reads the `url:` field out of
        each file's YAML frontmatter (the frontmatter your text scraper
        already writes at the top of every page). This is the path to use
        since you only have the scraped_site/ folder of .md files.
     b) --manifest manifest.csv: if you do have a manifest CSV with url and
        md_path columns, that still works and takes priority over --scan-dir.
   Files with no `url:` frontmatter field are skipped and reported at the end
   so you can see exactly what was missed and why.
2. Re-visits each URL (network only, no changes to the site).
3. Walks the page in document order and pulls out every image it can find:
      - lazy-load attrs checked before plain src: data-src, data-lazy-src,
        data-original (WordPress/Elementor lazy loading pattern)
      - srcset, picks the highest-resolution candidate
      - <figure>/<figcaption> pairs (caption becomes the image's alt/caption)
      - inline CSS background-image, including shorthand
        `background: #fff url(...) no-repeat`
   For each image it records the nearest preceding heading text on the page,
   so "which section is this image under" survives.
4. Downloads each image:
      - verified by real Content-Type (rejects an HTML error page served
        with 200)
      - size-filtered (--min-bytes / --max-bytes)
      - deduplicated by content hash -> images/{hash}.ext (a logo repeated on
        300 pages is saved once)
      - hard timeout per request, so one bad image can't stall the run
5. Opens the existing .md file for that page and inserts
   `![alt](../images/hash.ext "optional caption")`
   right after the matching heading line. If no heading matches (or the
   image had no nearby heading), it appends a final "## Images" section
   instead of silently dropping it.
   Existing text is never rewritten -- only new lines are inserted.

This keeps alt text + tags intact so a downstream chatbot can literally
answer "show me a picture of X" by matching alt text / heading context to
the stored image path.

Usage
-----
No manifest.csv? Just point at the folder of .md files:

    python3 attach_images.py --scan-dir scraped_site

    # dry run first (no downloads, no file writes, just prints what it would do)
    python3 attach_images.py --scan-dir scraped_site --dry-run

Have a manifest.csv instead? Use that:

    python3 attach_images.py --manifest manifest.csv --root ./kb

(On Windows cmd.exe, put the whole command on one line, or use the caret (^)
character instead of a backslash for line continuation -- backslash line
continuation is bash-only; cmd.exe passes a trailing backslash through
literally as an argument, which is what happened above.)

Manifest format (manifest.csv), if you use --manifest
-------------------------------------------------------
url,md_path
https://bvrithyderabad.edu.in/admission/fee-details,pages/admission-fee-details.md
https://bvrithyderabad.edu.in/computer-science-and-engineering/ms-k-vineela,pages/faculty-ms-k-vineela.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BG_URL_RE = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", re.IGNORECASE)
SKIP_EXT_HINTS = (".svg",)  # icons/spacers are rarely useful KB images; drop with --no-svg


@dataclass
class FoundImage:
    src_url: str
    alt: str
    caption: str = ""
    heading: str = ""  # nearest preceding heading text on the page


@dataclass
class DownloadResult:
    ok: bool
    local_path: Optional[Path] = None
    reason: str = ""


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def best_srcset_candidate(srcset: str) -> Optional[str]:
    """Pick the highest-resolution URL out of a srcset attribute."""
    best_url, best_width = None, -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split()
        url = pieces[0]
        width = 0
        if len(pieces) > 1 and pieces[1].endswith("w"):
            try:
                width = int(pieces[1][:-1])
            except ValueError:
                width = 0
        elif len(pieces) > 1 and pieces[1].endswith("x"):
            try:
                width = int(float(pieces[1][:-1]) * 1000)
            except ValueError:
                width = 0
        if width >= best_width:
            best_url, best_width = url, width
    return best_url


def resolve_img_src(tag: Tag) -> Optional[str]:
    # Lazy-load attributes take priority: a lazy page usually leaves a blank
    # placeholder in src and puts the real URL in one of these.
    for attr in ("data-src", "data-lazy-src", "data-original"):
        if tag.get(attr):
            return tag.get(attr)
    for attr in ("data-srcset", "srcset"):
        if tag.get(attr):
            candidate = best_srcset_candidate(tag.get(attr))
            if candidate:
                return candidate
    if tag.get("src"):
        return tag.get("src")
    return None


def extract_background_images(tag: Tag) -> list[str]:
    style = tag.get("style", "")
    if "url(" not in style:
        return []
    return BG_URL_RE.findall(style)


def extract_images_in_order(soup: BeautifulSoup) -> list[FoundImage]:
    """Walk the parsed page in document order, tracking the nearest heading,
    and collect every image reference found."""
    # Strip obvious chrome so nav/footer logos don't get attributed to content
    for junk in soup.select("nav, header, footer, script, style, noscript"):
        junk.decompose()

    body = soup.body or soup
    found: list[FoundImage] = []
    current_heading = ""

    def walk(node):
        nonlocal current_heading
        if not isinstance(node, Tag):
            return

        name = node.name.lower() if node.name else ""

        if name in HEADING_TAGS:
            text = node.get_text(strip=True)
            if text:
                current_heading = text

        if name == "figure":
            img = node.find("img")
            cap_tag = node.find("figcaption")
            caption = cap_tag.get_text(strip=True) if cap_tag else ""
            if img is not None:
                src = resolve_img_src(img)
                if src:
                    found.append(FoundImage(
                        src_url=src,
                        alt=(img.get("alt") or caption or "").strip(),
                        caption=caption,
                        heading=current_heading,
                    ))
            # also catch a figure with only a CSS background image
            for bg in extract_background_images(node):
                found.append(FoundImage(src_url=bg, alt=caption, caption=caption, heading=current_heading))
            return  # don't double-descend into the figure's own <img>

        if name == "img":
            src = resolve_img_src(node)
            if src:
                found.append(FoundImage(
                    src_url=src,
                    alt=(node.get("alt") or "").strip(),
                    heading=current_heading,
                ))

        for bg in extract_background_images(node):
            found.append(FoundImage(src_url=bg, alt="", heading=current_heading))

        for child in node.children:
            walk(child)

    walk(body)
    return found


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------

def guess_ext(content_type: str, url: str) -> str:
    ct_map = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
        "image/webp": ".webp", "image/gif": ".gif", "image/svg+xml": ".svg",
        "image/avif": ".avif", "image/bmp": ".bmp", "image/x-icon": ".ico",
    }
    if content_type in ct_map:
        return ct_map[content_type]
    path_ext = Path(urlparse(url).path).suffix.lower()
    return path_ext if path_ext else ".jpg"


def download_image(
    session: requests.Session,
    url: str,
    images_dir: Path,
    seen_hashes: dict[str, Path],
    min_bytes: int,
    max_bytes: int,
    timeout: float,
    allow_svg: bool,
) -> DownloadResult:
    try:
        resp = session.get(url, timeout=timeout, stream=True)
    except requests.RequestException as exc:
        return DownloadResult(False, reason=f"request failed: {exc}")

    if resp.status_code != 200:
        return DownloadResult(False, reason=f"HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if not content_type.startswith("image/"):
        return DownloadResult(False, reason=f"not an image (Content-Type: {content_type or 'unknown'})")
    if content_type == "image/svg+xml" and not allow_svg:
        return DownloadResult(False, reason="svg skipped (--allow-svg to include)")

    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        total += len(chunk)
        if total > max_bytes:
            return DownloadResult(False, reason=f"exceeds max-bytes ({max_bytes})")
        buf.write(chunk)

    if total < min_bytes:
        return DownloadResult(False, reason=f"below min-bytes ({min_bytes}), likely an icon/spacer")

    data = buf.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]

    if digest in seen_hashes:
        return DownloadResult(True, local_path=seen_hashes[digest], reason="dedup hit")

    ext = guess_ext(content_type, url)
    local_path = images_dir / f"{digest}{ext}"
    if not local_path.exists():
        local_path.write_bytes(data)
    seen_hashes[digest] = local_path
    return DownloadResult(True, local_path=local_path)


# --------------------------------------------------------------------------
# Markdown insertion
# --------------------------------------------------------------------------

def md_image_line(alt: str, rel_path: str, caption: str = "") -> str:
    alt_text = (alt or "image").replace("[", "").replace("]", "")
    if caption and caption != alt:
        return f'![{alt_text}]({rel_path} "{caption}")'
    return f"![{alt_text}]({rel_path})"


def insert_images_into_md(md_path: Path, images_by_heading: dict[str, list[str]], unmatched: list[str]) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: list[str] = []
    used_headings: set[str] = set()

    heading_re = re.compile(r"^#{1,6}\s+(.*)$")

    for line in lines:
        out.append(line)
        m = heading_re.match(line.strip())
        if not m:
            continue
        heading_text = m.group(1).strip()
        # normalize file's own H1 (page title) shouldn't slurp every image
        matches = [h for h in images_by_heading if h.strip() == heading_text and h not in used_headings]
        for h in matches:
            used_headings.add(h)
            out.append("")
            for img_md in images_by_heading[h]:
                out.append(img_md)
            out.append("")

    # anything whose heading never matched (page text changed, or heading-less image)
    leftover = [images_by_heading[h] for h in images_by_heading if h not in used_headings]
    leftover_flat = [img for group in leftover for img in group] + unmatched
    if leftover_flat:
        out.append("")
        out.append("## Images")
        out.append("")
        out.extend(leftover_flat)

    # collapse any triple+ blank lines left over from insertion
    cleaned: list[str] = []
    blank_run = 0
    for line in out:
        if line.strip() == "":
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        cleaned.append(line)

    md_path.write_text("\n".join(cleaned), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def read_manifest(path: Path, url_col: str, md_col: str) -> list[tuple[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if url_col not in reader.fieldnames or md_col not in reader.fieldnames:
            raise SystemExit(
                f"manifest.csv is missing expected columns. "
                f"Found: {reader.fieldnames}. "
                f"Expected '{url_col}' and '{md_col}' (override with --url-col/--md-col)."
            )
        for row in reader:
            url = (row.get(url_col) or "").strip()
            md_path = (row.get(md_col) or "").strip()
            if url and md_path:
                rows.append((url, md_path))
    return rows


FRONTMATTER_URL_RE = re.compile(r'^url:\s*["\']?(.+?)["\']?\s*$', re.IGNORECASE)


def extract_frontmatter_url(md_file: Path) -> Optional[str]:
    """Pull the `url:` field out of a page's YAML frontmatter (the block
    between the first two `---` lines that scrape_site.py writes at the top
    of every page)."""
    try:
        text = md_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    frontmatter = text[3:end]
    for line in frontmatter.splitlines():
        m = FRONTMATTER_URL_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def scan_dir_for_pages(scan_dir: Path) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Recursively find every *.md file under scan_dir and pair it with the
    URL from its own frontmatter -- no manifest.csv required."""
    found: list[tuple[str, Path]] = []
    skipped: list[Path] = []
    for md_file in sorted(scan_dir.rglob("*.md")):
        url = extract_frontmatter_url(md_file)
        if url:
            found.append((url, md_file))
        else:
            skipped.append(md_file)
    return found, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None, help="Path to manifest.csv from the earlier text scrape (optional -- if omitted, use --scan-dir instead)")
    ap.add_argument("--scan-dir", default=None, help="Folder to recursively scan for *.md files, pairing each with the `url:` field in its own frontmatter. Use this if you don't have a manifest.csv.")
    ap.add_argument("--root", default=None, help="KB root directory for the images/ folder (default: --scan-dir itself, or current dir when using --manifest)")
    ap.add_argument("--url-col", default="url", help="Manifest column name for the page URL (default: url)")
    ap.add_argument("--md-col", default="md_path", help="Manifest column name for the .md file path, relative to --root (default: md_path)")
    ap.add_argument("--min-bytes", type=int, default=3_000, help="Skip images smaller than this (icons/spacers), default 3000")
    ap.add_argument("--max-bytes", type=int, default=8_000_000, help="Skip images larger than this, default 8MB")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout in seconds (page fetch + each image), default 15")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between page fetches, seconds (politeness), default 0.5")
    ap.add_argument("--allow-svg", action="store_true", help="Include SVG images (excluded by default: usually icons)")
    ap.add_argument("--user-agent", default="Mozilla/5.0 (compatible; BVRIT-KB-ImageBot/1.0)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen; no downloads, no file writes")
    args = ap.parse_args()

    if not args.manifest and not args.scan_dir:
        raise SystemExit("Provide either --manifest manifest.csv or --scan-dir <folder of .md files>.")

    # Build the (url, md_path) list either from the manifest or by scanning
    # frontmatter directly -- this is the no-manifest path for you.
    pages: list[tuple[str, Path]] = []
    if args.manifest:
        root = Path(args.root).resolve() if args.root else Path(".").resolve()
        manifest_rows = read_manifest(Path(args.manifest), args.url_col, args.md_col)
        pages = [(url, root / md_rel) for url, md_rel in manifest_rows]
        log(f"Loaded {len(pages)} pages from manifest.")
    else:
        scan_dir = Path(args.scan_dir).resolve()
        if not scan_dir.exists():
            raise SystemExit(f"--scan-dir not found: {scan_dir}")
        root = Path(args.root).resolve() if args.root else scan_dir
        found, skipped = scan_dir_for_pages(scan_dir)
        pages = found
        log(f"Scanned {scan_dir}: {len(found)} page(s) with a url: in frontmatter, {len(skipped)} skipped.")
        if skipped:
            log("Skipped (no `url:` found in frontmatter -- check these manually):")
            for p in skipped:
                log(f"    {p}")

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    seen_hashes: dict[str, Path] = {}
    total_images_attached = 0
    total_pages_touched = 0

    for i, (url, md_path) in enumerate(pages, start=1):
        if not md_path.exists():
            log(f"[{i}/{len(pages)}] SKIP {url} -> {md_path} (md file not found)")
            continue

        log(f"[{i}/{len(pages)}] {url}")
        try:
            resp = session.get(url, timeout=args.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log(f"    failed to fetch page: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        images = extract_images_in_order(soup)
        if not images:
            log("    no images found")
            continue

        images_by_heading: dict[str, list[str]] = {}
        unmatched: list[str] = []
        attached_this_page = 0

        rel_images_dir = Path(os.path.relpath(images_dir, md_path.parent)).as_posix()

        for img in images:
            abs_url = urljoin(url, img.src_url)

            if args.dry_run:
                log(f"    would fetch: {abs_url} (heading: {img.heading or 'none'}, alt: {img.alt or 'none'})")
                continue

            result = download_image(
                session, abs_url, images_dir, seen_hashes,
                args.min_bytes, args.max_bytes, args.timeout, args.allow_svg,
            )
            if not result.ok:
                log(f"    skip {abs_url}: {result.reason}")
                continue

            rel_from_page = f"{rel_images_dir}/{result.local_path.name}"
            img_md = md_image_line(img.alt, rel_from_page, img.caption)
            attached_this_page += 1

            if img.heading:
                images_by_heading.setdefault(img.heading, []).append(img_md)
            else:
                unmatched.append(img_md)

        if args.dry_run:
            continue

        if attached_this_page:
            insert_images_into_md(md_path, images_by_heading, unmatched)
            total_images_attached += attached_this_page
            total_pages_touched += 1
            log(f"    attached {attached_this_page} image(s) -> {md_path}")
        else:
            log("    no images passed filters")

        time.sleep(args.delay)

    if not args.dry_run:
        log(f"\nDone. {total_images_attached} image(s) attached across {total_pages_touched} page(s).")
        log(f"Images stored in: {images_dir}")


if __name__ == "__main__":
    main()