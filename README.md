# BVRIT Hyderabad — Chatbot Knowledge Base Builder

A crawler that walks the **entire** `bvrithyderabad.edu.in` site — every menu
item, every dropdown sub-link, every department/admission/placement/research
page, every news post — plus every PDF linked anywhere on the site (NAAC/NBA/
AICTE approvals, committees, fee details, policies, syllabi, etc.), and turns
it into a clean, citation-grounded knowledge base your chatbot can retrieve
from.

## Why it won't hallucinate

- **No rewriting.** Every stored piece of text is copied verbatim from the
  live page/PDF — the script never summarizes or "improves" wording.
- **Every fact is traceable.** Every page record, chunk, image entry, and PDF
  page carries `source_url`, `retrieved_at_utc`, and a `content_hash`. Your
  chatbot should *always* surface the `source_url` next to its answer.
- **No invented links.** The crawler only follows `<a href>` targets it finds
  literally in the HTML — this is how it discovers every dropdown item
  automatically, since BVRITH's menus are plain server-rendered `<ul><li>`
  links (not JavaScript-only), so nothing is missed.
- **Downstream rule of thumb:** if your chatbot's retrieval step finds no
  relevant chunk for a question, it should say so rather than guess.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Run the crawl

```bash
python bvrith_kb_scraper.py \
    --max-pages 1000 \
    --max-pdfs 300 \
    --delay 1.0 \
    --download-images \
    --download-pdfs \
    -v
```

Run this from a machine/server with real internet access to
`bvrithyderabad.edu.in` — a polite crawl of the whole site (~a few hundred
pages + PDFs) typically takes 15–45 minutes with the default 1-second delay.
Increase `--delay` if you want to be extra gentle on their server, or ask the
college IT team for permission first — it's good practice even though the
site's own `robots.txt` is respected automatically.

Useful flags:
| Flag | Purpose |
|---|---|
| `--max-pages` | Safety cap on number of HTML pages crawled |
| `--max-pdfs` | Safety cap on number of PDFs downloaded & OCR'd |
| `--delay` | Seconds between requests (politeness / rate-limit) |
| `--download-images` / `--download-pdfs` | Save the actual binary files, not just metadata/text |
| `--ignore-robots` | Not recommended — only if you have explicit permission |
| `-v` | Verbose logging, prints every page as it's scraped |

## 3. Output files (in `bvrith_knowledge_base/`)

| File | Contents |
|---|---|
| `pages.jsonl` | One JSON object per crawled page: title, headings, tables, full text, hash |
| `chunks.jsonl` | ~1000-character overlapping chunks, **this is what you feed to your RAG retriever** |
| `images_manifest.jsonl` | Every image: `src`, `alt`, `context_heading` (nearest heading), `caption`, page it appeared on |
| `pdf_documents.jsonl` | Full extracted text of every linked PDF, page-by-page |
| `crawl_log.csv` | Audit trail — every URL visited, HTTP status, timestamp (use this to double check nothing important 404'd) |
| `run_summary.json` | Totals: pages/images/PDFs/chunks/errors |

### `chunks.jsonl` schema (what your chatbot will actually query)

```json
{
  "chunk_id": "0356a8aad63337a4",
  "source_url": "https://bvrithyderabad.edu.in/admission/fee-details/",
  "page_title": "Fee Details – BVRIT HYDERABAD",
  "breadcrumb": ["Home", "Admissions", "Fee Details"],
  "chunk_index": 0,
  "text": "verbatim scraped text ...",
  "retrieved_at_utc": "2026-07-02T10:15:00+00:00",
  "pdf_page_number": 3          // only present for chunks sourced from a PDF
}
```

## 4. Wiring it into a chatbot (next step, not included here)

This script deliberately stops at "clean, citation-tagged knowledge base" —
that's the part where hallucination risk is highest if done carelessly.
For the retrieval layer:

1. Embed each row of `chunks.jsonl` (e.g. with `sentence-transformers` or the
   Claude/OpenAI embeddings API) and store vectors in something like
   Chroma/FAISS/Qdrant, keeping `chunk_id`/`source_url` as metadata.
2. At query time, retrieve top-k chunks, pass them to the LLM as context, and
   instruct it: *"Only answer using the provided excerpts. Cite the
   `source_url` for every claim. If the excerpts don't contain the answer,
   say you don't know and suggest contacting the college."*
3. Re-run this scraper periodically (e.g. weekly via cron) since the site has
   a live "Announcements" ticker and news section — compare `content_hash`
   values to detect what actually changed instead of re-embedding everything.

## 5. Scope note

By default the crawler stays on `bvrithyderabad.edu.in`. It will discover
links to sister-campus/portal domains (e.g. `bvrithyderabad.ac.in` the
admissions portal, `vjoc.in`, `bvritnext.com`) but **won't follow them**
unless you add them to `--allowed-domains`. Add them explicitly if you want
those in scope too:

```bash
python bvrith_kb_scraper.py --allowed-domains bvrithyderabad.edu.in bvrithyderabad.ac.in
```
