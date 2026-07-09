
"""
app.py — BVRIT Hyderabad FAQ Chatbot
=====================================
Run:  streamlit run app.py
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
# Config
# ---------------------------------------------------------------------------

SECTIONS = [
    "All Sections",
    "About BVRITH",
    "Faculty",
    "Placement",
    "Laboratories",
    "Sports",
    "Transportation",
    "Library",
    "CSE – AI&ML",
    "Electronics and Communication Engineering",
    "Computer Science and Engineering",
    "Basic Sciences and Humanities",
    "Electrical and Electronics Engineering",
    "Information Technology",
    "Patents",
    "Entrepreneurship",
]

GENERATION_MODELS = ["Free Router", "Gemma 4 31B", "Llama 3.3 70B"]

QUICK_PROMPTS = [
    "What departments are offered?",
    "What is the fee structure?",
    "How do I apply for admission?",
    "Who are the ECE faculty?",
]

MAX_INPUT_CHARS         = 500
MAX_QUERIES_PER_SESSION = 40
ALLOWED_DOMAIN          = "bvrithyderabad.edu.in"

KB_DIR       = Path("bvrith_knowledge_base")
SUMMARY_FILE = KB_DIR / "run_summary.json"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BVRIT Hyderabad — FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — hide deploy bar, sidebar toggle, toolbar; clean typography
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600;700&display=swap');

/* ── Hide Streamlit chrome ── */
.stApp > header,
.stApp [data-testid="stToolbar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
#MainMenu,
footer { display: none !important; }

/* ── Hide sidebar entirely ── */
section[data-testid="stSidebar"] { display: none !important; }

:root {
    --bg:           #f6f1e8;
    --panel:        rgba(255,255,255,0.80);
    --panel-strong: #ffffff;
    --ink:          #1d1b16;
    --muted:        #6f675e;
    --line:         #ded4c3;
    --brand:        #5b2333;
    --brand-2:      #8b5e34;
    --accent:       #1f6f63;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(circle at top left,  rgba(91,35,51,0.10),  transparent 24%),
        radial-gradient(circle at top right, rgba(31,111,99,0.10), transparent 20%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(91,35,51,0.98), rgba(139,94,52,0.92));
    color: white;
    border-radius: 20px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 14px 36px rgba(41,20,26,0.16);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255,255,255,0.11), transparent 45%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.85rem;
    font-weight: 600;
    line-height: 1.1;
    margin: 0;
}
.hero-subtitle {
    margin-top: 0.4rem;
    color: rgba(255,255,255,0.86);
    font-size: 0.94rem;
}
.hero-pillrow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.85rem;
}
.hero-pill {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.18);
    color: white;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.78rem;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #90e1a2;
    margin-right: 0.4rem;
    box-shadow: 0 0 0 3px rgba(144,225,162,0.2);
}

/* ── Chat area ── */
.chat-wrap {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(222,212,195,0.9);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(65,45,27,0.05);
    min-height: 120px;
}
.chat-bubble {
    border: 1px solid rgba(222,212,195,0.9);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.65rem;
    background: var(--panel-strong);
    box-shadow: 0 6px 16px rgba(43,30,18,0.04);
}
.chat-bubble.user {
    background: linear-gradient(180deg, #f7e7eb, #f3d8df);
    border-color: rgba(91,35,51,0.12);
    margin-left: 12%;
}
.chat-bubble.assistant {
    background: linear-gradient(180deg, #ffffff, #fffdf9);
    margin-right: 8%;
}
.chat-label {
    font-size: 0.71rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.chat-text { font-size: 0.95rem; line-height: 1.7; }

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
}
.badge-green { background:#d7f1de; color:#165b2c; }
.badge-red   { background:#f9dbdf; color:#7f1d2d; }
.badge-amber { background:#fff0c9; color:#8a5a00; }

/* ── Citation box ── */
.citation-box {
    border: 1px solid var(--line);
    background: rgba(249,245,238,0.92);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
}

/* ── Eval summary banner ── */
.summary-banner {
    background: linear-gradient(135deg, rgba(91,35,51,0.96), rgba(31,111,99,0.94));
    color: #fff;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.65rem;
}
.sb-metric {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 0.7rem 0.5rem;
    text-align: center;
}
.sb-num   { font-size: 1.55rem; font-weight: 800; line-height: 1.1; }
.sb-label { font-size: 0.7rem;  opacity: 0.84; letter-spacing: 0.07em; }

/* ── Eval dimension cards ── */
.dim-card {
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.dim-title  { font-size: 0.93rem; font-weight: 700; color: var(--brand); }
.dim-counts { font-size: 0.78rem; color: var(--muted); margin-top: 0.18rem; }

/* ── Controls strip ── */
.controls-strip {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(222,212,195,0.85);
    border-radius: 16px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.85rem;
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
    flex-wrap: wrap;
}

/* ── Buttons ── */
.stButton button {
    border-radius: 10px;
    border: 1px solid rgba(91,35,51,0.15);
    background: linear-gradient(180deg, #ffffff, #f3ece1);
    color: var(--brand);
    font-weight: 600;
}
.stButton button:hover {
    border-color: rgba(91,35,51,0.28);
    box-shadow: 0 6px 14px rgba(91,35,51,0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def safe(text: str) -> str:
    return html.escape(str(text))


def citation_trusted(citation: str) -> bool:
    return bool(citation) and ALLOWED_DOMAIN in citation


def strip_inline_citations(answer: str) -> str:
    text = re.sub(r"\s*\[(?:[^\[\]]{2,200}?)\]\s*", " ", answer)
    text = re.sub(
        r"\s*[\u2014-]\s*https?://\S+\s*\(?(?:retrieved\s+\d{4}-\d{2}-\d{2}|page\s+\d+)\)?\s*",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_hero(title: str, subtitle: str, pills: list[str], status_line: str):
    pill_html = "".join(f'<span class="hero-pill">{safe(p)}</span>' for p in pills)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">{safe(title)}</div>
          <div class="hero-subtitle">{safe(subtitle)}</div>
          <div class="hero-pillrow">{pill_html}</div>
          <div style="margin-top:0.6rem;font-size:0.78rem;opacity:0.7">
            <span class="status-dot"></span>{safe(status_line)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(msg: dict):
    role = msg["role"]
    content = msg.get("content", "")
    citations = msg.get("citations", [])
    images = msg.get("images", [])
    refused = msg.get("refused", False)
    no_prediction = msg.get("no_prediction", False)

    label = "You" if role == "user" else "Assistant"
    bubble_class = "chat-bubble user" if role == "user" else "chat-bubble assistant"

    st.markdown(
        f"""
        <div class="{bubble_class}">
          <div class="chat-label">{label}</div>
          <div class="chat-text">{safe(content)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if citations and role == "assistant":
        st.markdown(
            f"""<div class="citation-box"><strong>Sources:</strong><br>{'<br>'.join(safe(c) for c in citations)}</div>""",
            unsafe_allow_html=True,
        )

    if images and role == "assistant":
        cols = st.columns(min(len(images), 5))
        for col, img in zip(cols, images[:5]):
            col.image(img.get("url", ""), use_container_width=True)

    if role == "assistant":
        badges = []
        if refused:
            badges.append('<span class="badge badge-red">Refused</span>')
        if no_prediction:
            badges.append('<span class="badge badge-amber">No Prediction</span>')
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KB helpers
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
    try:
        from rag import get_collection_stats
        return get_collection_stats()
    except Exception as e:
        return {"total_chunks": 0, "indexed": False, "error": str(e)}


run_summary = load_run_summary()
crawl_date = run_summary.get("crawl_date", "latest crawl")
chunk_count = run_summary.get("total_chunks", 0)
chroma_stats = get_chroma_stats()
if chunk_count == 0:
    chunk_count = chroma_stats.get("total_chunks", 0)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "messages":        [],
        "pending_prompt":  None,
        "last_latency":    None,
        "last_chunks":     None,
        "last_tokens_in":  None,
        "last_tokens_out": None,
        "eval_report":     None,
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
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_eval, tab_admin = st.tabs(["Chat", "Evaluation", "Admin"])

# ===========================================================================
# TAB 1 — CHAT
# ===========================================================================

with tab_chat:

    # Hero
    render_hero(
        title    = "BVRIT Hyderabad FAQ Assistant",
        subtitle = "Grounded answers for admissions, departments, faculty, placements, facilities and more.",
        pills    = ["Admissions", "Departments", "Faculty", "Placements", "Campus", "Research"],
        status_line = f"Source current as of {crawl_date}",
    )

    # Not-indexed warning
    if chunk_count == 0:
        st.warning("Knowledge base not indexed. Run `python ingest.py` first, then refresh.", icon="⚠️")

    # ── Controls strip ───────────────────────────────────────────────────────
    with st.container():
        col_filter, col_model, col_topk, col_reset = st.columns([2, 2, 1, 1])

        with col_filter:
            section_filter_label = st.selectbox("Section", SECTIONS, index=0, label_visibility="collapsed")
            effective_filter = None if section_filter_label == "All Sections" else section_filter_label

        with col_model:
            model = st.selectbox("Model", GENERATION_MODELS, index=0, label_visibility="collapsed")

        with col_topk:
            top_k = st.number_input("Top-K", min_value=3, max_value=10, value=5, label_visibility="collapsed")

        with col_reset:
            if st.button("Reset", use_container_width=True):
                st.session_state.messages        = []
                st.session_state.last_latency    = None
                st.session_state.last_chunks     = None
                st.session_state.last_tokens_in  = None
                st.session_state.last_tokens_out = None
                st.rerun()

    # ── Remembered context (from memory module) ──
    memory = st.session_state.get("memory")
    if memory:
        blurb = memory.entities.get_context_blurb()
        if blurb:
            st.caption(f"🧠 {blurb}")

    # ── Quick-prompt chips ───────────────────────────────────────────────────
    chip_cols = st.columns(len(QUICK_PROMPTS))
    for col, label in zip(chip_cols, QUICK_PROMPTS):
        if col.button(label, use_container_width=True):
            st.session_state.pending_prompt = label

    # ── Chat history ─────────────────────────────────────────────────────────
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.caption("Ask about admissions, fees, departments, faculty, placements, campus facilities or contact details.")
    else:
        for msg in st.session_state.messages:
            render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Last query stats (compact, below chat) ───────────────────────────────
    if st.session_state.last_latency is not None:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Latency", f"{st.session_state.last_latency:.2f}s")
        sc2.metric("Chunks used", st.session_state.last_chunks)
        sc3.metric("Tokens in",  st.session_state.last_tokens_in)
        sc4.metric("Tokens out", st.session_state.last_tokens_out)

    # ── Chat input ───────────────────────────────────────────────────────────
    chat_input = st.chat_input("Ask about BVRIT Hyderabad…", max_chars=MAX_INPUT_CHARS)

    prompt = chat_input
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        prompt = re.sub(r"\s+", " ", prompt).strip()
        queries_used = len(st.session_state.messages) // 2

        # Input sanitization
        if len(prompt) == 0 or len(prompt.split()) == 0:
            answer_msg = {
                "role": "assistant",
                "content": "Please ask a clear question about BVRIT Hyderabad — admissions, departments, fees, placements, faculty, or campus facilities.",
                "citations": [],
                "refused": True,
            }
            st.session_state.messages.append(answer_msg)
            st.rerun()

        if queries_used >= MAX_QUERIES_PER_SESSION:
            st.warning("Session query limit reached. Reset the conversation to continue.")
        elif prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            if chunk_count == 0:
                answer_msg = {
                    "role": "assistant",
                    "content": "The knowledge base hasn't been indexed yet. Please run `python ingest.py` to build the vector store, then refresh.",
                    "citations": [],
                    "refused": True,
                }
            else:
                with st.spinner("Searching knowledge base…"):
                    try:
                        from rag import answer_question
                        result = answer_question(
                            question      = prompt,
                            section_filter= effective_filter,
                            top_k         = top_k,
                            model         = model,
                            history       = [
                                m for m in st.session_state.messages[:-1]
                                if m["role"] in ("user", "assistant")
                            ],
                            session_id    = st.session_state.session_id,
                            memory        = st.session_state.memory,
                        )
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
                            "content":       strip_inline_citations(result.answer),
                            "citations":     result.citations,
                            "images":        result.images,
                            "refused":       result.refused,
                            "no_prediction": no_pred,
                        }
                    except Exception as e:
                        answer_msg = {
                            "role": "assistant",
                            "content": f"An error occurred: {e}",
                            "citations": [],
                            "refused": True,
                        }

            st.session_state.messages.append(answer_msg)
            st.rerun()


# ===========================================================================
# TAB 2 — EVALUATION
# ===========================================================================

with tab_eval:

    render_hero(
        title    = "Evaluation Dashboard",
        subtitle = "Functional checks, safety probes, retrieval quality, and RAGAS metrics.",
        pills    = ["Functional", "Safety", "Retrieval", "RAGAS"],
        status_line = "Live chatbot under test",
    )

    # ── Run controls ─────────────────────────────────────────────────────────
    ctrl_a, ctrl_b, ctrl_c = st.columns([1, 1, 2])
    with ctrl_a:
        run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)
    with ctrl_b:
        skip_ragas = st.checkbox("Skip RAGAS", value=False)
    with ctrl_c:
        dim_options = ["All"] + [
            f"{k} — {v}" for k, v in {
                "01": "Functional", "02": "Quality",  "03": "Safety",
                "04": "Security",   "05": "Robustness","06": "Performance",
                "07": "Context",    "08": "RAGAS",
            }.items()
        ]
        dim_select = st.selectbox("Run dimension", dim_options, index=0)

    if run_btn:
        dim_filter = None if dim_select == "All" else dim_select[:2]
        with st.spinner("Running test suite… this may take 1–2 minutes."):
            try:
                from eval import run_suite
                report = run_suite(dim_filter=dim_filter, skip_ragas=skip_ragas)
                st.session_state.eval_report = report
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    report = st.session_state.get("eval_report")

    if not report:
        st.info("Click **Run Evaluation** to start. Results appear here.")
    else:
        s        = report["summary"]
        pass_pct = int(s["pass_rate"] * 100)

        # ── Summary banner ────────────────────────────────────────────────────
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

        # ── Dimension results ─────────────────────────────────────────────────
        st.markdown("### Dimension Results")
        dims = report.get("dimensions", [])
        rows = [dims[i:i+2] for i in range(0, len(dims), 2)]
        for row in rows:
            cols = st.columns(2)
            for col, dim in zip(cols, row):
                with col:
                    color = "#155724" if dim["failed"] == 0 else "#721c24"
                    st.markdown(
                        f"""
                        <div class="dim-card">
                          <div class="dim-title" style="color:{color}">
                            {safe(dim['id'])} {safe(dim['name'])}
                          </div>
                          <div class="dim-counts">
                            {dim['passed']} passed &nbsp;|&nbsp;
                            {dim['failed']} failed &nbsp;|&nbsp;
                            {dim['warned']} warnings &nbsp;|&nbsp;
                            {dim['total']} total
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for case in dim.get("cases", []):
                        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(case["verdict"], "❓")
                        st.caption(f"{icon} {case['id']}: {case['reason'][:80]}")

        # ── Failed test details ───────────────────────────────────────────────
        all_cases = [c for d in dims for c in d.get("cases", [])]
        failures  = [c for c in all_cases if c["verdict"] == "FAIL"]
        if failures:
            st.markdown("### Failed Test Details")
            for c in failures:
                with st.expander(f"{c['id']} — {c['reason'][:60]}"):
                    st.markdown(f"**Question:** {safe(c['question'])}")
                    st.markdown(f"**Answer:** {safe(c['answer'][:400])}")
                    st.markdown(f"**Root cause:** {safe(c.get('root_cause', ''))}")
                    st.markdown(f"**Suggested fix:** {safe(c.get('fix', ''))}")

        # ── Weakest dimension ─────────────────────────────────────────────────
        wd = report.get("weakest_dimension")
        if wd:
            st.markdown("### Weakest Dimension")
            st.warning(
                f"**{wd['id']} {wd['name']}** ({int(wd['pass_rate']*100)}% pass rate)\n\n{wd['fix']}"
            )

        # ── RAGAS metrics ─────────────────────────────────────────────────────
        ragas = report.get("ragas", {})
        if ragas and "error" not in ragas:
            st.markdown("### RAGAS Metrics")
            metric_info = [
                ("faithfulness",      "Faithfulness",      "Are all claims supported by retrieved context?"),
                ("answer_relevancy",  "Answer Relevancy",  "Does the answer address what was asked?"),
                ("context_precision", "Context Precision", "Were the retrieved chunks actually useful?"),
                ("context_recall",    "Context Recall",    "Did retrieval find all the relevant chunks?"),
            ]
            mcols = st.columns(2)
            for idx, (key, label, tooltip) in enumerate(metric_info):
                val = ragas.get(key)
                if val is not None:
                    with mcols[idx % 2]:
                        st.markdown(
                            f'<div class="dim-card"><div class="dim-title">{label}</div>'
                            f'<div class="dim-counts">{tooltip}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(val, text=f"{val:.3f}")

            scores = [ragas[k] for k, _, _ in metric_info if ragas.get(k) is not None]
            if scores:
                weakest = min(metric_info, key=lambda t: ragas.get(t[0]) or 1.0)
                diagnoses = {
                    "context_precision": "Consider reducing chunk_size or adding section metadata filters.",
                    "context_recall":    "Consider increasing top_k or chunk overlap.",
                    "faithfulness":      "Tighten the GROUNDING RULE in the system prompt.",
                    "answer_relevancy":  "Check that retrieved chunks are not too noisy.",
                }
                st.info(diagnoses.get(weakest[0], "Review retrieval configuration."))
        elif ragas.get("error"):
            st.caption(f"RAGAS skipped: {ragas['error']}")

        # ── Security residual risk ────────────────────────────────────────────
        if report.get("residual_risk_statement"):
            st.markdown("### Security — Residual Risk")
            st.info(report["residual_risk_statement"])

        # ── Download ──────────────────────────────────────────────────────────
        st.download_button(
            label     = "Download JSON Report",
            data      = json.dumps(report, indent=2),
            file_name = "bvrit_eval_report.json",
            mime      = "application/json",
        )


# ===========================================================================
# TAB 3 — ADMIN DASHBOARD (governance + audit)
# ===========================================================================

with tab_admin:
    st.markdown("## Admin Dashboard")
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
        st.markdown("### Model Distribution")
        dist = stats.get("model_distribution", {})
        if dist:
            for model_name, count in dist.items():
                st.caption(f"{model_name}: {count} queries")
        else:
            st.caption("No data yet.")

        st.divider()

        # Rate limit info
        st.markdown("### Rate Limiting")
        limiter = RateLimiter(max_per_session=40, max_per_minute=10)
        usage = limiter.get_usage(st.session_state.get("session_id", "default"))
        st.caption(f"Session queries: {usage['session_queries']} / {usage['max_per_session']}")
        st.caption(f"Remaining: {usage['remaining']}")

        st.divider()

        # Recent audit log
        st.markdown("### Recent Activity")
        recent = audit.get_recent(limit=20)
        if recent:
            for entry in recent:
                flags_str = entry.get("flags", "[]")
                try:
                    import json as _j
                    flags = _j.loads(flags_str) if isinstance(flags_str, str) else flags_str
                except Exception:
                    flags = []
                flag_badge = f" [{', '.join(flags)}]" if flags else ""
                st.markdown(
                    f"**{entry['timestamp'][:19]}** | {entry['model']} | "
                    f"{entry['latency_s']:.1f}s | {entry['tokens_in'] + entry['tokens_out']} tokens"
                    f"{flag_badge}",
                )
                st.caption(f"Q: {entry['query'][:120]}")
                st.caption(f"A: {entry['response'][:120]}")
                st.markdown("---")
        else:
            st.caption("No queries logged yet. Start chatting to see activity here.")

        st.divider()

        # Prompt version
        st.markdown("### Prompt Version")
        last_entry = recent[0] if recent else None
        if last_entry and last_entry.get("prompt_version"):
            st.caption(f"Last used prompt version: `{last_entry['prompt_version']}`")
        else:
            st.caption("No prompt version tracked yet.")

    except Exception as e:
        st.warning(f"Admin dashboard unavailable: {e}")


