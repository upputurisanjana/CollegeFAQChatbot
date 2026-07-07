# BVRIT Hyderabad — FAQ Chatbot

A grounded RAG (Retrieval Augmented Generation) chatbot for BVRIT Hyderabad
College of Engineering for Women. Answers questions about admissions, fees,
departments, faculty, placements, campus facilities, and more — using only
content retrieved from the college's official website.

## Architecture

```
User query → Section filter → Embedding (all-MiniLM-L6-v2)
    → ChromaDB retrieval (384-dim, cosine)
    → Context + System prompt → OpenRouter LLM (free tier)
    → Tool calling (fee/date/percentage calculators)
    → Governance (audit log, rate limit, content monitor)
    → Conversation memory (entity extraction)
    → Answer + Citations + Images
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` with three tabs:
- **Chat** — ask questions, filter by section, select model
- **Evaluation** — run test suite across 8 dimensions
- **Admin** — audit log, rate limits, usage stats

## Files

| File | Purpose |
|---|---|
| `rag.py` | Core RAG pipeline: retrieval, grounding, generation, tool calling |
| `app.py` | Streamlit frontend (chat UI, eval dashboard, admin panel) |
| `tools.py` | Function-calling tools: fee calculator, date checker, percentage calculator |
| `memory.py` | Conversation memory and entity extraction (name, departments, topics) |
| `governance.py` | Audit logging, rate limiting, content monitoring, prompt versioning |
| `eval.py` | 8-dimension evaluation suite with LLM judge |
| `image_search.py` | Faculty/campus image retrieval from scraped images |
| `chunk_and_index.py` | Chunk pages + build Chroma vector store |
| `scrape_site.py` | Crawl bvrithyderabad.edu.in |
| `config.py` | Shared configuration |
| `requirements.txt` | Python dependencies |

## Models

Uses **OpenRouter** free tier. Configured models (in `rag.py`):

| Display name | Model ID |
|---|---|
| Free Router | `openrouter/free` (auto-selects best available) |
| Gemma 4 31B | `google/gemma-4-31b-it:free` |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` |

Set `OPENROUTER_API_KEY` in `.env` (get one at [openrouter.ai/keys](https://openrouter.ai/keys)).

## Vector store

The ChromaDB vector store (`chroma_db/`) contains ~2100 chunks with 384-dim
embeddings from `all-MiniLM-L6-v2`. Rebuild it:

```bash
python chunk_and_index.py
```

## Evaluation

```bash
python eval.py                     # full suite
python eval.py --dim 04            # single dimension
python eval.py --no-ragas          # skip RAGAS metrics
python eval.py --out report.json   # save JSON report
```
