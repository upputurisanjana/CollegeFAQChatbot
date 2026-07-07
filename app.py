"""
app.py — BVRIT Hyderabad FAQ Chatbot (full Streamlit UI)
=========================================================
spec.md §6 — two tabs: Chat + Evaluation Dashboard
Run:  streamlit run app.py
Requires: ingest.py run first (python ingest.py) to build ./chroma_bvrith
"""

import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants / config  (spec §6.1)
# ---------------------------------------------------------------------------

SECTIONS = [
    "All Sections",
    "About BVRIT",
    "Departments",
    "Admissions",
    "Fee Structure",
    "Placements",
    "Campus & Facilities",
    "Faculty",
    "Contact",
]

from config import MODEL_MAP
GENERATION_MODELS = list(MODEL_MAP.keys())

QUICK_PROMPTS = [
    "What departments are offered?",
    "What is the hostel fee?",
    "How do I apply for admission?",
    "What is the fee structure?",
]

MAX_INPUT_CHARS       = 500   # spec §9.1 — prompt-based DoS guard
MAX_QUERIES_PER_SESSION = 40  # spec §9.4 — per-session cost cap
ALLOWED_DOMAIN        = "bvrithyderabad.edu.in"  # spec §9.3 — citation spoofing guard

from config import KB_DIR, CHROMA_DIR
SUMMARY_FILE = KB_DIR / "run_summary.json"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BVRIT Hyderabad — FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #faf7f2; }

/* ── header ── */
.bvrith-header {
    display:flex; align-items:baseline; justify-content:space-between;
    padding-bottom:0.75rem; border-bottom:1px solid #e2dccf; margin-bottom:1.25rem;
}
.bvrith-wordmark {
    font-family:'Source Serif 4',serif; font-size:1.5rem;
    font-weight:500; color:#4a1b28;
}
.bvrith-tagline  { font-size:0.85rem; color:#7a776e; margin-top:2px; }
.bvrith-status   { font-size:0.75rem; color:#7a776e; white-space:nowrap; }
.bvrith-status .dot {
    display:inline-block; width:6px; height:6px; border-radius:50%;
    background:#4a8f5c; margin-right:6px;
}

/* ── chat bubbles ── */
.user-msg {
    background:#f1dbe1; color:#4a1b28; border-radius:14px;
    padding:10px 16px; max-width:72%; margin-left:auto;
    margin-bottom:4px; font-size:0.92rem;
}
.assistant-answer {
    font-size:0.92rem; line-height:1.65; color:#2c2c2a;
    padding:4px 2px 0 2px;
}
.citation {
    border-top:1px solid #e2dccf; margin-top:10px; padding-top:6px;
    font-size:0.75rem; letter-spacing:0.02em; color:#7a776e;
}
.badge {
    display:inline-block; font-size:0.68rem; font-weight:500;
    padding:2px 8px; border-radius:10px; margin-left:6px;
    vertical-align:middle;
}
.badge-green  { background:#d4edda; color:#155724; }
.badge-red    { background:#f8d7da; color:#721c24; }
.badge-amber  { background:#fff3cd; color:#856404; }

/* ── sidebar ── */
section[data-testid="stSidebar"] { background:#f1ebe0; border-right:1px solid #e2dccf; }
section[data-testid="stSidebar"] h3 { font-size:0.8rem; color:#7a776e; letter-spacing:0.03em; }

/* ── eval dashboard ── */
.dim-card {
    background:#fff; border:1px solid #e2dccf; border-radius:10px;
    padding:14px 16px; margin-bottom:10px;
}
.dim-title  { font-size:0.9rem; font-weight:600; color:#4a1b28; }
.dim-counts { font-size:0.78rem; color:#7a776e; margin-top:2px; }
.fail-card  {
    background:#fff8f8; border-left:3px solid #c0392b;
    padding:10px 14px; margin-bottom:8px; border-radius:4px;
    font-size:0.82rem;
}
.summary-banner {
    background:#4a1b28; color:#fff; border-radius:10px;
    padding:16px 22px; margin-bottom:20px;
    display:flex; gap:32px; align-items:center;
}
.sb-metric { text-align:center; }
.sb-num    { font-size:1.6rem; font-weight:700; }
.sb-label  { font-size:0.72rem; opacity:0.75; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Security helpers  (spec §9.3)
# ---------------------------------------------------------------------------

def safe(text: str) -> str:
    """Escape before injecting into unsafe_allow_html blocks."""
    return html.escape(str(text))


def citation_trusted(citation: str) -> bool:
    """Only surface citations pointing at the real source domain."""
    return bool(citation) and ALLOWED_DOMAIN in citation

# ---------------------------------------------------------------------------
# Knowledge-base metadata helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_run_summary() -> dict:
    if SUMMARY_FILE.exists():
        try:
            return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data(ttl=300)
def get_chroma_stats() -> dict:
    """Return chunk count + index status without re-loading on every rerun."""
    try:
        from rag import get_collection_stats
        return get_collection_stats()
    except Exception as e:
        return {"total_chunks": 0, "indexed": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "messages":       [],
        "pending_prompt": None,
        "last_latency":   None,
        "last_chunks":    None,
        "last_tokens_in": None,
        "last_tokens_out":None,
        "eval_report":    None,
        "eval_running":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------------------------------------------------------
# Session identity & Memory
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    raw = f"{time.time()}-{os.urandom(8).hex()}"
    st.session_state.session_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

if "memory" not in st.session_state:
    from memory import ConversationMemory
    st.session_state.memory = ConversationMemory(st.session_state.session_id)

# ---------------------------------------------------------------------------
# Sidebar  (spec §6.1 — every field)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📚 Knowledge Base")

    summary   = load_run_summary()
    kb_stats  = get_chroma_stats()
    chunk_count = kb_stats.get("total_chunks", 0)

    # Index status badge
    if chunk_count > 0:
        index_badge = "🟢 LIVE"
    elif summary:
        index_badge = "🟡 STALE — run ingest.py"
    else:
        index_badge = "🔴 NOT INDEXED — run ingest.py"

    st.markdown(f"**Status:** {index_badge}")
    st.caption(f"Chunks indexed: **{chunk_count}**")
    st.caption("Corpus mode: Raw Crawl (Pipeline B) ⚠ content not manually reviewed")

    # Crawl freshness (from run_summary.json)
    if summary:
        finished = summary.get("finished_at_utc", "unknown")[:19].replace("T", " ")
        st.markdown("### 🕐 Crawl Freshness")
        st.caption(f"Source crawled: {finished} UTC")
        st.caption(
            f"Pages: {summary.get('pages_crawled', '?')} · "
            f"PDFs: {summary.get('pdfs_processed', '?')} · "
            f"Images: {summary.get('images_found', '?')}"
        )

    st.divider()

    st.markdown("### ⚙️ Retrieval Settings")

    # Chunk size / overlap — read-only (set at index time)
    col1, col2 = st.columns(2)
    col1.metric("Chunk Size", "1000", help="Set at index time — changing requires re-embedding")
    col2.metric("Overlap", "150", help="Set at index time")

    top_k = st.slider("Top-K Results", min_value=3, max_value=10, value=5)

    section_filter = st.selectbox("Section Filter", SECTIONS, index=0)
    effective_filter = None if section_filter == "All Sections" else section_filter

    model = st.selectbox("Generation Model", GENERATION_MODELS, index=0)

    st.divider()

    # RAGAS panel (populated after eval run)
    st.markdown("### 📊 RAGAS Evaluation")
    report = st.session_state.get("eval_report")
    if report and report.get("ragas"):
        r = report["ragas"]
        metrics = [
            ("Faithfulness",       r.get("faithfulness")),
            ("Answer Relevancy",   r.get("answer_relevancy")),
            ("Context Precision",  r.get("context_precision")),
            ("Context Recall",     r.get("context_recall")),
        ]
        scores = [v for _, v in metrics if v is not None]
        mean_score = sum(scores) / len(scores) if scores else 0
        for label, val in metrics:
            if val is not None:
                st.progress(val, text=f"{label}: {val:.2f}")
        verdict = "✅ Good" if mean_score >= 0.8 else "⚠️ Needs work"
        st.caption(f"Overall: **{verdict}** (mean {mean_score:.2f})")
    else:
        st.caption("Run evaluation (tab 2) to see scores.")

    st.divider()

    # Last query metrics
    st.markdown("### ⏱ Last Query")
    if st.session_state.last_latency is not None:
        st.caption(f"Latency: {st.session_state.last_latency:.2f}s")
        st.caption(f"Chunks retrieved: {st.session_state.last_chunks}")
        st.caption(
            f"Tokens: {st.session_state.last_tokens_in} in / "
            f"{st.session_state.last_tokens_out} out"
        )
    else:
        st.caption("No queries yet.")

    queries_used = len(st.session_state.messages) // 2
    st.caption(f"Queries this session: {queries_used}/{MAX_QUERIES_PER_SESSION}")

    st.divider()

    # ── Remembered context (from memory module) ──
    st.markdown("### 🧠 Remembered Context")
    memory = st.session_state.get("memory")
    if memory:
        blurb = memory.entities.get_context_blurb()
        if blurb:
            st.caption(blurb)
        else:
            st.caption("No context remembered yet.")
    else:
        st.caption("Memory not initialized.")

    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_latency   = None
        st.session_state.last_chunks    = None
        st.session_state.last_tokens_in = None
        st.session_state.last_tokens_out= None
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_eval, tab_admin = st.tabs(["💬 Chat", "📋 Evaluation Dashboard", "🔐 Admin"])

# ===========================================================================
# TAB 1 — CHAT
# ===========================================================================

with tab_chat:

    # Header
    crawl_date = (summary.get("finished_at_utc") or "unknown")[:10]
    st.markdown(
        f"""
        <div class="bvrith-header">
          <div>
            <div class="bvrith-wordmark">BVRIT Hyderabad</div>
            <div class="bvrith-tagline">Ask about admissions, fees, placements, hostel and more</div>
          </div>
          <div class="bvrith-status">
            <span class="dot"></span>source current as of {safe(crawl_date)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Warn if not indexed
    if chunk_count == 0:
        st.warning(
            "⚠️ Knowledge base not indexed. Run `python ingest.py` first, "
            "then refresh this page.",
            icon="⚠️",
        )

    # Quick-prompt chips
    chip_cols = st.columns(len(QUICK_PROMPTS))
    for col, label in zip(chip_cols, QUICK_PROMPTS):
        if col.button(label, use_container_width=True):
            st.session_state.pending_prompt = label

    st.markdown("---")

    # Message thread
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{safe(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            refused  = msg.get("refused", False)
            no_pred  = msg.get("no_prediction", False)

            if refused:
                badge = '<span class="badge badge-red">REFUSED</span>'
            elif no_pred:
                badge = '<span class="badge badge-amber">⚠ Documented stats only</span>'
            else:
                badge = '<span class="badge badge-green">Cited</span>'

            st.markdown(
                f'<div class="assistant-answer">'
                f'{safe(msg["content"])}{badge}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Citations (only trust ALLOWED_DOMAIN)
            for cit in msg.get("citations", []):
                if citation_trusted(cit):
                    st.markdown(
                        f'<div class="citation">📄 {safe(cit)}</div>',
                        unsafe_allow_html=True,
                    )

            # Images (NEW: display images if available)
            images = msg.get("images", [])
            if images:
                st.markdown("---")
                n_cols = min(3, len(images))
                cols = st.columns(n_cols)
                for idx, img in enumerate(images[:6]):
                    col = cols[idx % n_cols]
                    with col:
                        name = img.get("semantic_name") or img.get("context_heading") or "Image"
                        img_url = img.get("url") or img.get("src", "")
                        if img_url:
                            st.markdown(
                                f'<img src="{html.escape(img_url)}" style="width:100%;border-radius:6px;">',
                                unsafe_allow_html=True,
                            )
                            st.caption(name)
                        else:
                            st.caption(f"🖼️ {name}")

    # Chat input
    chat_input = st.chat_input(
        "Ask about BVRIT Hyderabad…",
        max_chars=MAX_INPUT_CHARS,
    )

    # Resolve pending prompt (from quick chips)
    prompt = chat_input
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        prompt = re.sub(r"\s+", " ", prompt).strip()
        queries_used = len(st.session_state.messages) // 2

        if queries_used >= MAX_QUERIES_PER_SESSION:
            st.warning("Session query limit reached. Reset the conversation to continue.")
        elif prompt:
            # Show user message immediately before waiting for the answer
            st.markdown(
                f'<div class="user-msg">{safe(prompt)}</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({"role": "user", "content": prompt})

            if chunk_count == 0:
                # Not indexed — return a canned response
                answer_msg = {
                    "role": "assistant",
                    "content": (
                        "The knowledge base hasn't been indexed yet. "
                        "Please run `python ingest.py` to build the vector store, "
                        "then refresh and try again."
                    ),
                    "citations": [],
                    "refused": True,
                }
            else:
                with st.spinner("Searching knowledge base…"):
                    try:
                        from rag import answer_question
                        result = answer_question(
                            question=prompt,
                            section_filter=effective_filter,
                            top_k=top_k,
                            model=model,
                            history=[
                                m for m in st.session_state.messages[:-1]
                                if m["role"] in ("user", "assistant")
                            ],
                            session_id=st.session_state.session_id,
                            memory=st.session_state.memory,
                        )
                        # Update last-query metrics in sidebar
                        st.session_state.last_latency    = result.latency_s
                        st.session_state.last_chunks     = result.chunks_retrieved
                        st.session_state.last_tokens_in  = result.tokens_in
                        st.session_state.last_tokens_out = result.tokens_out

                        no_pred = any(
                            kw in result.answer.lower()
                            for kw in ("individual outcome", "cannot predict", "cannot guarantee")
                        )
                        answer_msg = {
                            "role":          "assistant",
                            "content":       result.answer,
                            "citations":     result.citations,
                            "images":        result.images,  # NEW: include images
                            "refused":       result.refused,
                            "no_prediction": no_pred,
                        }
                    except Exception as e:
                        answer_msg = {
                            "role":      "assistant",
                            "content":   f"Sorry, an error occurred: {e}",
                            "citations": [],
                            "refused":   True,
                        }

            st.session_state.messages.append(answer_msg)
            st.rerun()


# ===========================================================================
# TAB 2 — EVALUATION DASHBOARD  (spec §6.3)
# ===========================================================================

with tab_eval:
    st.markdown("## 📋 Evaluation Dashboard")
    st.caption(
        "Runs the 8-dimension test suite against the live chatbot. "
        "Uses a separate judge model (Gemma 3 12B) to score each answer."
    )

    col_run, col_opts = st.columns([2, 3])
    with col_opts:
        skip_ragas  = st.checkbox("Skip RAGAS (faster)", value=False)
        dim_options = ["All"] + [f"{k} — {v}" for k, v in {
            "01":"Functional","02":"Quality","03":"Safety","04":"Security",
            "05":"Robustness","06":"Performance","07":"Context","08":"RAGAS",
        }.items()]
        dim_select = st.selectbox("Run dimension", dim_options, index=0)
    with col_run:
        run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        dim_filter = None if dim_select == "All" else dim_select[:2]
        with st.spinner("Running test suite… this may take 1–2 minutes."):
            try:
                from eval import run_suite
                report = run_suite(dim_filter=dim_filter, skip_ragas=skip_ragas)
                st.session_state.eval_report = report
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                report = None

    report = st.session_state.get("eval_report")

    if not report:
        st.info("Click **▶ Run Evaluation** to start. Results appear here.")
    else:
        s = report["summary"]
        pass_pct = int(s["pass_rate"] * 100)

        # ── Summary banner ───────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="summary-banner">
              <div class="sb-metric"><div class="sb-num">{s['total']}</div><div class="sb-label">TOTAL</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#90ee90">{s['passed']}</div><div class="sb-label">PASSED</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#ff6b6b">{s['failed']}</div><div class="sb-label">FAILED</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#ffd700">{s['warning']}</div><div class="sb-label">WARNINGS</div></div>
              <div class="sb-metric"><div class="sb-num">{pass_pct}%</div><div class="sb-label">PASS RATE</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Per-dimension 4×2 grid ───────────────────────────────────────────
        st.markdown("### Dimension Results")
        dims = report.get("dimensions", [])
        rows = [dims[i:i+2] for i in range(0, len(dims), 2)]

        for row in rows:
            cols = st.columns(2)
            for col, dim in zip(cols, row):
                with col:
                    total_d = dim["total"]
                    passed_d = dim["passed"]
                    failed_d = dim["failed"]
                    warned_d = dim["warned"]
                    color = "#155724" if failed_d == 0 else "#721c24"
                    st.markdown(
                        f"""
                        <div class="dim-card">
                          <div class="dim-title" style="color:{color}">
                            {safe(dim['id'])} {safe(dim['name'])}
                          </div>
                          <div class="dim-counts">
                            ✓ {passed_d} passed &nbsp;|&nbsp;
                            ✗ {failed_d} failed &nbsp;|&nbsp;
                            ⚠ {warned_d} warnings &nbsp;|&nbsp;
                            {total_d} total
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Show checks
                    for case in dim.get("cases", []):
                        icon = {"PASS":"✅","FAIL":"❌","WARN":"⚠️"}.get(case["verdict"],"❓")
                        st.caption(f"{icon} {case['id']}: {case['reason'][:80]}")

        # ── Failed test details ──────────────────────────────────────────────
        all_cases = [c for d in dims for c in d.get("cases", [])]
        failures  = [c for c in all_cases if c["verdict"] == "FAIL"]
        if failures:
            st.markdown("### ❌ Failed Test Details")
            for c in failures:
                with st.expander(f"{c['id']} — {c['reason'][:60]}"):
                    st.markdown(f"**Question:** {safe(c['question'])}")
                    st.markdown(f"**Answer:** {safe(c['answer'][:400])}")
                    st.markdown(f"**Root cause:** {safe(c.get('root_cause',''))}")
                    st.markdown(f"**Suggested fix:** {safe(c.get('fix',''))}")

        # ── Weakest dimension callout ────────────────────────────────────────
        wd = report.get("weakest_dimension")
        if wd:
            st.markdown("### 🔧 Weakest Dimension — Recommended Fix")
            st.warning(
                f"**{wd['id']} {wd['name']}** "
                f"({int(wd['pass_rate']*100)}% pass rate)  \n"
                f"{wd['fix']}"
            )

        # ── RAGAS bars ───────────────────────────────────────────────────────
        ragas = report.get("ragas", {})
        if ragas and "error" not in ragas:
            st.markdown("### 📈 RAGAS Metrics")
            metric_info = [
                ("faithfulness",      "Faithfulness",      "Are all claims in the answer supported by retrieved context?"),
                ("answer_relevancy",  "Answer Relevancy",  "Does the answer address what was asked?"),
                ("context_precision", "Context Precision", "Were the retrieved chunks actually useful?"),
                ("context_recall",    "Context Recall",    "Did retrieval find all the relevant chunks?"),
            ]
            scores = [ragas[k] for k, _, _ in metric_info if ragas.get(k) is not None]
            mean_score = sum(scores) / len(scores) if scores else 0

            for key, label, tooltip in metric_info:
                val = ragas.get(key)
                if val is not None:
                    st.progress(val, text=f"{label}: {val:.3f}")
                    st.caption(f"↳ {tooltip}")

            # Diagnosis line
            if scores:
                weakest_metric = min(metric_info, key=lambda t: ragas.get(t[0]) or 1.0)
                diagnoses = {
                    "context_precision": "Context Precision is the weakest metric — consider reducing chunk_size or adding section metadata filters.",
                    "context_recall":    "Context Recall is the weakest metric — consider increasing top_k or chunk overlap.",
                    "faithfulness":      "Faithfulness is the weakest metric — tighten the GROUNDING RULE in the system prompt.",
                    "answer_relevancy":  "Answer Relevancy is the weakest metric — check that retrieved chunks are not too noisy (Pipeline B mode).",
                }
                st.info(diagnoses.get(weakest_metric[0], "Review retrieval configuration."))

        elif ragas.get("error"):
            st.caption(f"RAGAS: {ragas['error']}")

        # ── Residual risk statement (spec §9.5) ──────────────────────────────
        st.markdown("### 🔒 Security — Residual Risk Statement")
        st.info(report.get("residual_risk_statement", ""))

        # ── Download report ──────────────────────────────────────────────────
        st.download_button(
            label="⬇ Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name="bvrit_eval_report.json",
            mime="application/json",
        )


# ===========================================================================
# TAB 3 — ADMIN DASHBOARD  (governance + audit)
# ===========================================================================

with tab_admin:
    st.markdown("## 🔐 Admin Dashboard")
    st.caption("Governance, audit logging, and usage statistics.")

    try:
        from governance import AuditLog, RateLimiter
        audit = AuditLog()
        stats = audit.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", stats["total_queries"])
        col2.metric("Today", stats["today_queries"])
        col3.metric("Avg Latency", f"{stats['avg_latency_s']:.2f}s")
        col4.metric("Total Tokens", stats["total_tokens"])

        st.divider()

        # Model distribution
        st.markdown("### 📊 Model Distribution")
        dist = stats.get("model_distribution", {})
        if dist:
            for model_name, count in dist.items():
                st.caption(f"{model_name}: {count} queries")
        else:
            st.caption("No data yet.")

        st.divider()

        # Rate limit info
        st.markdown("### ⚡ Rate Limiting")
        limiter = RateLimiter(max_per_session=40, max_per_minute=10)
        usage = limiter.get_usage(st.session_state.get("session_id", "default"))
        st.caption(f"Session queries: {usage['session_queries']} / {usage['max_per_session']}")
        st.caption(f"Remaining: {usage['remaining']}")
        st.caption("Max per minute: 10")

        st.divider()

        # Recent audit log
        st.markdown("### 📋 Recent Activity")
        recent = audit.get_recent(limit=20)
        if recent:
            for entry in recent:
                flags = json.loads(entry.get("flags", "[]"))
                flag_badge = ""
                if flags:
                    flag_badge = f' <span class="badge badge-amber">{" ".join(flags)}</span>'
                st.markdown(
                    f"**{entry['timestamp'][:19]}** | {entry['model']} | "
                    f"{entry['latency_s']:.1f}s | {entry['tokens_in'] + entry['tokens_out']} tokens"
                    f"{flag_badge}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Q: {entry['query'][:120]}")
                st.caption(f"A: {entry['response'][:120]}")
                st.markdown("---")
        else:
            st.caption("No queries logged yet. Start chatting to see activity here.")

        st.divider()

        # Prompt version
        st.markdown("### 📝 Prompt Version")
        last_entry = recent[0] if recent else None
        if last_entry and last_entry.get("prompt_version"):
            st.caption(f"Last used prompt version: `{last_entry['prompt_version']}`")
            st.caption("Prompt versions are SHA256 hashes of the system prompt template.")
        else:
            st.caption("No prompt version tracked yet.")

    except Exception as e:
        st.warning(f"Admin dashboard unavailable: {e}")
