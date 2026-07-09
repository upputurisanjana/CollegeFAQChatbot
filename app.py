"""
app.py — BVRIT Hyderabad FAQ Chatbot
=====================================
Run:  streamlit run app.py
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui_components import (
    inject_css,
    render_hero,
    render_message,
    render_empty_state,
    safe,
    strip_inline_citations,
    ALLOWED_DOMAIN,
)

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

KB_DIR       = Path("bvrith_knowledge_base")
SUMMARY_FILE = KB_DIR / "run_summary.json"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BVRIT Hyderabad — FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

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


run_summary  = load_run_summary()
chroma_stats = get_chroma_stats()
crawl_date   = run_summary.get("crawl_date", chroma_stats.get("crawl_date", "latest crawl"))
chunk_count  = run_summary.get("total_chunks", 0) or chroma_stats.get("total_chunks", 0)

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

# Session identity
if "session_id" not in st.session_state:
    raw = f"{time.time()}-{os.urandom(8).hex()}"
    st.session_state.session_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

# Conversation memory
if "memory" not in st.session_state:
    from memory import ConversationMemory
    st.session_state.memory = ConversationMemory(st.session_state.session_id)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Branding
    st.markdown(
        """
        <div class="sidebar-logo">
            🎓 BVRIT Hyderabad<br>FAQ Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Knowledge Base stats
    st.markdown('<div class="sidebar-section-header">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-num">{chunk_count:,}</div>
            <div>Indexed chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div style="font-size:0.7rem;opacity:0.7;margin-bottom:0.1rem">Crawl date</div>
            <div style="font-weight:600">{crawl_date}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_ready = chunk_count > 0
    status_color = "#90e1a2" if is_ready else "#ff6b6b"
    status_label = "Ready" if is_ready else "Not indexed"
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div style="font-size:0.7rem;opacity:0.7;margin-bottom:0.1rem">Status</div>
            <div style="font-weight:600">
                <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                    background:{status_color};margin-right:6px;vertical-align:middle"></span>
                {status_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Session stats
    st.markdown('<div class="sidebar-section-header">Session</div>', unsafe_allow_html=True)
    queries_used = len([m for m in st.session_state.messages if m["role"] == "user"])
    remaining    = MAX_QUERIES_PER_SESSION - queries_used

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-num">{queries_used}</div>
            <div>Queries this session</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div style="font-size:0.7rem;opacity:0.7;margin-bottom:0.1rem">Remaining</div>
            <div style="font-weight:600">{remaining} / {MAX_QUERIES_PER_SESSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div style="font-size:0.7rem;opacity:0.7;margin-bottom:0.1rem">Session ID</div>
            <div style="font-weight:600;font-size:0.78rem;word-break:break-all">
                {st.session_state.session_id}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_eval, tab_admin = st.tabs(["Chat", "Evaluation", "Admin"])

# ===========================================================================
# TAB 1 — CHAT
# ===========================================================================

with tab_chat:

    # Hero banner
    render_hero(
        title       = "BVRIT Hyderabad FAQ Assistant",
        subtitle    = "Grounded answers for admissions, departments, faculty, placements, facilities and more.",
        pills       = ["Admissions", "Departments", "Faculty", "Placements", "Campus", "Research"],
        status_line = f"Source current as of {crawl_date}",
    )

    if chunk_count == 0:
        st.warning("Knowledge base not indexed. Run `python chunk_and_index.py` first, then refresh.", icon="⚠️")

    # ── Controls strip ───────────────────────────────────────────────────────
    col_filter, col_model, col_topk, col_reset = st.columns([2, 2, 1, 1])

    with col_filter:
        st.markdown('<div class="ctrl-label">Section Filter</div>', unsafe_allow_html=True)
        section_filter_label = st.selectbox(
            "Section", SECTIONS, index=0, label_visibility="collapsed"
        )
        effective_filter = None if section_filter_label == "All Sections" else section_filter_label

    with col_model:
        st.markdown('<div class="ctrl-label">Model</div>', unsafe_allow_html=True)
        model = st.selectbox("Model", GENERATION_MODELS, index=0, label_visibility="collapsed")

    with col_topk:
        st.markdown('<div class="ctrl-label">Top-K</div>', unsafe_allow_html=True)
        top_k = st.number_input(
            "Top-K", min_value=3, max_value=20, value=5, label_visibility="collapsed"
        )

    with col_reset:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("Reset chat", use_container_width=True):
            st.session_state.messages        = []
            st.session_state.last_latency    = None
            st.session_state.last_chunks     = None
            st.session_state.last_tokens_in  = None
            st.session_state.last_tokens_out = None
            st.rerun()

    # ── Memory context blurb ─────────────────────────────────────────────────
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

    # ── Chat messages ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    # ── Last query metrics (compact) ─────────────────────────────────────────
    if st.session_state.last_latency is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Latency",    f"{st.session_state.last_latency:.2f}s")
        mc2.metric("Chunks",     st.session_state.last_chunks)
        mc3.metric("Tokens in",  st.session_state.last_tokens_in)
        mc4.metric("Tokens out", st.session_state.last_tokens_out)

    # ── Chat input ───────────────────────────────────────────────────────────
    chat_input = st.chat_input("Ask about BVRIT Hyderabad…", max_chars=MAX_INPUT_CHARS)

    prompt = chat_input
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        prompt = re.sub(r"\s+", " ", prompt).strip()

        if not prompt:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please ask a question about BVRIT Hyderabad — admissions, departments, fees, placements, faculty, or campus facilities.",
                "citations": [],
                "refused": False,
            })
            st.rerun()

        elif queries_used >= MAX_QUERIES_PER_SESSION:
            st.warning("Session query limit reached. Reset the conversation to continue.")

        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            if chunk_count == 0:
                answer_msg = {
                    "role": "assistant",
                    "content": "The knowledge base hasn't been indexed yet. Please run `python chunk_and_index.py` to build the vector store, then refresh.",
                    "citations": [],
                    "refused": True,
                }
            else:
                with st.spinner("Searching knowledge base…"):
                    try:
                        from rag import answer_question
                        result = answer_question(
                            question       = prompt,
                            section_filter = effective_filter,
                            top_k          = int(top_k),
                            model          = model,
                            history        = [
                                m for m in st.session_state.messages[:-1]
                                if m["role"] in ("user", "assistant")
                            ],
                            session_id     = st.session_state.session_id,
                            memory         = st.session_state.memory,
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
        title       = "Evaluation Dashboard",
        subtitle    = "Functional checks, safety probes, retrieval quality, and RAGAS metrics.",
        pills       = ["Functional", "Safety", "Retrieval", "RAGAS"],
        status_line = "Live chatbot under test",
    )

    ctrl_a, ctrl_b, ctrl_c = st.columns([1, 1, 2])
    with ctrl_a:
        run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)
    with ctrl_b:
        skip_ragas = st.checkbox("Skip RAGAS", value=True)
    with ctrl_c:
        dim_options = ["All"] + [
            f"{k} — {v}" for k, v in {
                "01": "Functional", "02": "Quality",   "03": "Safety",
                "04": "Security",   "05": "Robustness","06": "Performance",
                "07": "Context",    "08": "RAGAS",
            }.items()
        ]
        dim_select = st.selectbox("Run dimension", dim_options, index=0)

    # Test case source controls
    tc_col1, tc_col2 = st.columns([2, 1])
    with tc_col1:
        use_saved = st.checkbox(
            "Use previously saved test cases",
            value=False,
            help="Unchecked = generate fresh test cases via LLM before running. "
                 "Checked = reuse the last generated set saved in generated_test_cases.json.",
        )
    with tc_col2:
        generate_n = st.number_input(
            "Cases to generate",
            min_value=8, max_value=40, value=16, step=4,
            disabled=use_saved,
            help="How many test cases the LLM generator should create (ignored when using saved cases).",
        )

    if not use_saved:
        st.caption("🔄 Fresh test cases will be generated before this run.")
    else:
        import os as _os
        saved_exists = _os.path.exists("generated_test_cases.json")
        if saved_exists:
            st.caption("📂 Using saved test cases from `generated_test_cases.json`.")
        else:
            st.caption("⚠️ No saved file found — will generate fresh cases automatically.")

    if run_btn:
        dim_filter = None if dim_select == "All" else dim_select[:2]
        with st.spinner("Running test suite… this may take 1–2 minutes."):
            try:
                from eval import run_suite
                report = run_suite(
                    dim_filter = dim_filter,
                    skip_ragas = skip_ragas,
                    use_saved  = use_saved,
                    generate_n = int(generate_n),
                )
                st.session_state.eval_report = report
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    report = st.session_state.get("eval_report")

    if not report:
        st.info("Click **Run Evaluation** to start. Results appear here.")
    else:
        s        = report["summary"]
        pass_pct = int(s["pass_rate"] * 100)
        tc_source = report.get("test_case_source", "static")
        tc_count  = report.get("test_case_count", s["total"])
        source_label = "📂 Saved cases" if tc_source == "saved" else "🔄 Generated cases"
        st.caption(f"{source_label} · {tc_count} test cases run")

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

        wd = report.get("weakest_dimension")
        if wd:
            st.markdown("### Weakest Dimension")
            st.warning(
                f"**{wd['id']} {wd['name']}** ({int(wd['pass_rate']*100)}% pass rate)\n\n{wd['fix']}"
            )

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
        elif ragas.get("error"):
            st.caption(f"RAGAS skipped: {ragas['error']}")

        if report.get("residual_risk_statement"):
            st.markdown("### Security — Residual Risk")
            st.info(report["residual_risk_statement"])

        st.download_button(
            label     = "Download JSON Report",
            data      = json.dumps(report, indent=2),
            file_name = "bvrit_eval_report.json",
            mime      = "application/json",
        )


# ===========================================================================
# TAB 3 — ADMIN
# ===========================================================================

with tab_admin:
    st.markdown("## Admin Dashboard")
    st.caption("Governance, audit logging, and usage statistics.")

    try:
        from governance import AuditLog, RateLimiter

        audit = AuditLog()
        stats = audit.get_stats()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries",  stats["total_queries"])
        c2.metric("Today",          stats["today_queries"])
        c3.metric("Avg Latency",    f"{stats['avg_latency_s']:.2f}s")
        c4.metric("Total Tokens",   stats["total_tokens"])

        st.divider()
        st.markdown("### Model Distribution")
        dist = stats.get("model_distribution", {})
        if dist:
            for m_name, count in dist.items():
                st.caption(f"{m_name}: {count} queries")
        else:
            st.caption("No data yet.")

        st.divider()
        st.markdown("### Rate Limiting")
        limiter = RateLimiter(max_per_session=MAX_QUERIES_PER_SESSION, max_per_minute=10)
        usage   = limiter.get_usage(st.session_state.get("session_id", "default"))
        st.caption(f"Session queries: {usage['session_queries']} / {usage['max_per_session']}")
        st.caption(f"Remaining: {usage['remaining']}")

        st.divider()
        st.markdown("### Recent Activity")
        recent = audit.get_recent(limit=20)
        if recent:
            for entry in recent:
                flags_raw = entry.get("flags", "[]")
                try:
                    flags = json.loads(flags_raw) if isinstance(flags_raw, str) else flags_raw
                except Exception:
                    flags = []
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                st.markdown(
                    f'<div class="log-entry">'
                    f'<div class="log-meta">'
                    f'<span class="log-time">{entry["timestamp"][:19]}</span>'
                    f'<span class="log-pill">{entry["model"]}</span>'
                    f'<span class="log-pill">{entry["latency_s"]:.1f}s</span>'
                    f'<span class="log-pill">{entry["tokens_in"] + entry["tokens_out"]} tok</span>'
                    f'{flag_str}'
                    f'</div>'
                    f'<div class="log-q">Q: {safe(entry["query"][:140])}</div>'
                    f'<div class="log-a">A: {safe(entry["response"][:140])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No queries logged yet.")

        st.divider()
        st.markdown("### Prompt Version")
        last_entry = recent[0] if recent else None
        if last_entry and last_entry.get("prompt_version"):
            st.caption(f"Last used: `{last_entry['prompt_version']}`")
        else:
            st.caption("No prompt version tracked yet.")

    except Exception as e:
        st.warning(f"Admin dashboard unavailable: {e}")
