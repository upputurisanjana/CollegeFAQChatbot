"""
Website Markdown scraper for building a RAG knowledge base.

What it does:
- Starts at a seed URL
- Crawls every internal link on the same domain (breadth-first)
- Converts each page's content to PROPERLY STRUCTURED MARKDOWN — real
  headings (#, ##, ###), nested bullet/numbered lists, and tables with a
  valid header-separator row — instead of flattening everything to plain
  text lines. This matters because a downstream chunker that splits on
  markdown heading markers (very common in RAG pipelines) needs those
  markers to actually exist in the source.
- Strips nav/header/footer/aside chrome so the same menu and footer text
  isn't duplicated at the top of every single page's file
- Adds a small YAML frontmatter block (url, title, retrieved_at) to every
  page — the metadata a citation-grounded chatbot needs to say "according
  to <url>, fetched on <date>"
- Saves each page as its own .md file AND one combined.md with all pages
- Writes manifest.csv (url, filename, char count) for quick auditing

Usage:
    python scrape_site.py https://bvrithyderabad.edu.in/ --max-pages 200 --delay 0.5

Output:
    ./scraped_site/pages/*.md     (one Markdown file per page, with frontmatter)
    ./scraped_site/combined.md    (every page, in one structured Markdown file)
    ./scraped_site/manifest.csv   (url, filename, num_chars)
"""

import argparse
import csv
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGKnowledgeBaseBot/1.0; +for-college-chatbot)"
}

# Non-content junk — never worth extracting text from.
STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "form"]
# Layout chrome — real HTML, but repeated identically on every page, so it
# would otherwise duplicate the same nav/footer text across the whole corpus.
CHROME_TAGS = ["header", "footer", "nav", "aside"]

MIN_CONTENT_CHARS = 20
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "blockquote"]


def is_same_domain(url: str, base_netloc: str) -> bool:
    return urlparse(url).netloc in ("", base_netloc)


def normalize_url(base: str, link: str) -> str | None:
    if not link:
        return None
    link = link.strip()
    if link.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    full = urljoin(base, link)
    full, _ = urldefrag(full)
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|zip|rar|mp4|mp3|docx?|xlsx?|pptx?)$", full, re.I):
        return None
    return full


# ---------------------------------------------------------------------------
# HTML -> Markdown conversion
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _li_own_text(li: Tag) -> str:
    """Text belonging to this <li> only — excludes any nested <ul>/<ol> text,
    so a sub-list doesn't get smashed into its parent item's line."""
    parts = []
    for child in li.children:
        if isinstance(child, Tag) and child.name in ("ul", "ol"):
            continue
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(child.get_text(" ", strip=True))
    return _clean(" ".join(parts))


def _render_list(tag: Tag, depth: int = 0) -> list[str]:
    lines = []
    ordered = tag.name == "ol"
    idx = 1
    for li in tag.find_all("li", recursive=False):
        text = _li_own_text(li)
        indent = "  " * depth
        if text:
            marker = f"{idx}." if ordered else "-"
            lines.append(f"{indent}{marker} {text}")
            if ordered:
                idx += 1
        for nested in li.find_all(["ul", "ol"], recursive=False):
            lines.extend(_render_list(nested, depth + 1))
    return lines


def _render_table(tag: Tag) -> list[str]:
    rows = []
    for tr in tag.find_all("tr"):
        cells = [_clean(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        r = (r + [""] * width)[:width]  # pad/truncate so every row matches the header width
        lines.append("| " + " | ".join(r) + " |")
    return lines


def html_to_markdown(scope: Tag, level_offset: int = 1) -> str:
    """Walks block-level elements in document order and renders real Markdown.
    level_offset shifts heading levels down (e.g. offset=1 turns an in-page
    <h1> into '##') so a synthesized page-title '#' stays the only H1 in the
    file — one coherent heading hierarchy per document."""
    blocks = []
    for el in scope.find_all(BLOCK_TAGS):
        if el.name in ("ul", "ol") and el.find_parent(["ul", "ol"]) is not None:
            continue  # nested lists are rendered by their parent list's recursion
        if el.find_parent("table") is not None:
            continue
        if re.fullmatch(r"h[1-6]", el.name):
            level = min(6, int(el.name[1]) + level_offset)
            text = _clean(el.get_text(" ", strip=True))
            if text:
                blocks.append("#" * level + " " + text)
        elif el.name == "p":
            text = _clean(el.get_text(" ", strip=True))
            if text:
                blocks.append(text)
        elif el.name in ("ul", "ol"):
            lines = _render_list(el)
            if lines:
                blocks.append("\n".join(lines))
        elif el.name == "table":
            lines = _render_table(el)
            if lines:
                blocks.append("\n".join(lines))
        elif el.name == "blockquote":
            text = _clean(el.get_text(" ", strip=True))
            if text:
                blocks.append("> " + text)
    return "\n\n".join(blocks)


def extract_markdown(html: str) -> tuple[str, str]:
    """Returns (title, markdown_body). Body headings start at H2 (offset=1),
    reserving H1 for the page title in the caller."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    main = soup.find("main") or soup.find("article") or soup.find(id="main") or soup.body or soup
    for chrome in CHROME_TAGS:
        for t in main.find_all(chrome):
            t.decompose()

    body = html_to_markdown(main, level_offset=1)
    return title, body


def frontmatter(url: str, title: str, retrieved_at: str) -> str:
    safe_title = title.replace('"', '\\"')
    return f'---\nurl: {url}\ntitle: "{safe_title}"\nretrieved_at: {retrieved_at}\n---\n\n'


def safe_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        path = "home"
    name = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")
    return (name or "page")[:150] + ".md"


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

def crawl(seed_url: str, max_pages: int, delay: float, out_dir: Path):
    base_netloc = urlparse(seed_url).netloc
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    visited = set()
    queue = deque([seed_url])
    manifest_rows = []
    combined_parts = []

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
        except requests.RequestException as e:
            print(f"[skip] {url} -> {e}")
            continue

        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        title, body = extract_markdown(resp.text)

        if len(body) < MIN_CONTENT_CHARS:
            print(f"[thin] {url} ({len(body)} chars)")
        else:
            fname = safe_filename(url)
            i = 1
            candidate = fname
            while (pages_dir / candidate).exists():
                candidate = fname.replace(".md", f"_{i}.md")
                i += 1
            fname = candidate

            page_md = frontmatter(url, title, retrieved_at) + f"# {title}\n\n" + body + "\n"
            (pages_dir / fname).write_text(page_md, encoding="utf-8")

            manifest_rows.append((url, fname, len(body)))
            combined_parts.append(
                f"\n\n---\n\n# {title}\n\n> Source: {url} · Retrieved: {retrieved_at}\n\n{body}"
            )
            print(f"[ok]   {url}  ({len(body)} chars)")

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = normalize_url(url, a["href"])
            if link and is_same_domain(link, base_netloc) and link not in visited:
                queue.append(link)

        time.sleep(delay)

    (out_dir / "combined.md").write_text("".join(combined_parts).lstrip("\n"), encoding="utf-8")

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "filename", "num_chars"])
        writer.writerows(manifest_rows)

    print(f"\nDone. Crawled {len(visited)} URLs, saved {len(manifest_rows)} pages with text.")
    print(f"Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl a website and extract structured Markdown for a RAG knowledge base.")
    parser.add_argument("url", help="Seed URL to start crawling from")
    parser.add_argument("--max-pages", type=int, default=200, help="Max number of pages to crawl")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (seconds) between requests")
    parser.add_argument("--out", type=str, default="scraped_site", help="Output directory")
    args = parser.parse_args()

    crawl(args.url, args.max_pages, args.delay, Path(args.out))