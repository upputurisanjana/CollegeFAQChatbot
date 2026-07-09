"""
app.py — BVRIT Hyderabad FAQ Chatbot
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
    inject_css, render_hero, render_typing_indicator,
    render_message, render_empty_state, render_citations,
    render_metric_card, render_log_entry, safe, strip_inline_citations,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTIONS = [
    "All Sections", "About BVRITH", "Faculty", "Placement", "Laboratories",
    "Sports", "Transportation", "Library", "CSE – AI&ML",
    "Electronics and Communication Engineering",
    "Computer Science and Engineering", "Basic Sciences and Humanities",
    "Electrical and Electronics Engineering", "Information Technology",
    "Patents", "Entrepreneurship",
]
SECTION_ICONS = {
    "All Sections":"🏫","About BVRITH":"ℹ️","Faculty":"👩‍🏫","Placement":"💼",
    "Laboratories":"🔬","Sports":"🏅","Transportation":"🚌","Library":"📚",
    "CSE – AI&ML":"🤖","Electronics and Communication Engineering":"📡",
    "Computer Science and Engineering":"💻","Basic Sciences and Humanities":"🔭",
    "Electrical and Electronics Engineering":"⚡","Information Technology":"🖥️",
    "Patents":"📜","Entrepreneurship":"🚀",
}
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
KB_DIR                  = Path("bvrith_knowledge_base")
SUMMARY_FILE            = KB_DIR / "run_summary.json"

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
crawl_date   = run_summary.get("crawl_date", "latest crawl")
chunk_count  = run_summary.get("total_chunks", 0)
chroma_stats = get_chroma_stats()
if chunk_count == 0:
    chunk_count = chroma_stats.get("total_chunks", 0)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "messages": [], "pending_prompt": None,
        "last_latency": None, "last_chunks": None,
        "last_tokens_in": None, "last_tokens_out": None,
        "eval_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

if "session_id" not in st.session_state:
    raw = f"{time.time()}-{os.urandom(8).hex()}"
    st.session_state.session_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

if "memory" not in st.session_state:
    from memory import ConversationMemory
    st.session_state.memory = ConversationMemory(st.session_state.session_id)

queries_used = len(st.session_state.messages) // 2
remaining    = MAX_QUERIES_PER_SESSION - queries_used

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎓 BVRIT Hyderabad<br>FAQ Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section-header">Knowledge Base</div>', unsafe_allow_html=True)
    kb_ok = chunk_count > 0
    st.markdown(
        f'<div class="sidebar-stat"><div class="sidebar-stat-num">{chunk_count:,}</div><div>Indexed chunks</div></div>'
        f'<div class="sidebar-stat"><div style="font-size:0.78rem;opacity:0.8">Crawl date</div><div style="font-weight:600">{safe(crawl_date)}</div></div>'
        f'<div class="sidebar-stat"><div style="font-size:0.78rem;opacity:0.8">Status</div><div style="font-weight:600">{"✅ Ready" if kb_ok else "⚠️ Not indexed"}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-header">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-stat"><div class="sidebar-stat-num">{queries_used}</div><div>Queries this session</div></div>'
        f'<div class="sidebar-stat"><div style="font-size:0.78rem;opacity:0.8">Remaining</div><div style="font-weight:600">{remaining} / {MAX_QUERIES_PER_SESSION}</div></div>'
        f'<div class="sidebar-stat"><div style="font-size:0.78rem;opacity:0.8">Session ID</div><div style="font-weight:600;font-size:0.75rem;word-break:break-all">{st.session_state.session_id}</div></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(queries_used / MAX_QUERIES_PER_SESSION, 1.0))

    st.markdown("---")
    st.markdown('<div class="sidebar-section-header">Quick Settings</div>', unsafe_allow_html=True)
    show_stats  = st.toggle("Show query stats", value=True)
    show_images = st.toggle("Show images",      value=True)

    st.markdown("---")
    st.caption("Powered by ChromaDB + OpenRouter · all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_eval, tab_admin = st.tabs(["💬 Chat", "📊 Evaluation", "🛡 Admin"])

# ===========================================================================
# TAB 1 — CHAT
# ===========================================================================
with tab_chat:
    render_hero(
        title       = "BVRIT Hyderabad FAQ Assistant",
        subtitle    = "Grounded answers for admissions, departments, faculty, placements, facilities and more.",
        pills       = ["Admissions", "Departments", "Faculty", "Placements", "Campus", "Research"],
        status_line = f"Source current as of {crawl_date}",
    )

    if chunk_count == 0:
        st.warning("Knowledge base not indexed. Run `python chunk_and_index.py` first, then refresh.", icon="⚠️")

    # Controls strip
    col_filter, col_model, col_topk, col_reset = st.columns([2, 2, 1, 1])
    with col_filter:
        section_label = st.selectbox(
            "Section",
            [f"{SECTION_ICONS.get(s,'')} {s}" for s in SECTIONS],
            index=0, label_visibility="collapsed",
        )
        # strip icon prefix back to raw section name
        raw_section = section_label.split(" ", 1)[-1].strip() if " " in section_label else section_label
        effective_filter = None if raw_section == "All Sections" else raw_section

    with col_model:
        model = st.selectbox("Model", GENERATION_MODELS, index=0, label_visibility="collapsed")

    with col_topk:
        top_k = st.number_input("Top-K", min_value=3, max_value=10, value=5, label_visibility="collapsed")

    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ("messages","last_latency","last_chunks","last_tokens_in","last_tokens_out"):
                st.session_state[k] = [] if k == "messages" else None
            st.rerun()

    # Memory context blurb
    memory = st.session_state.get("memory")
    if memory:
        blurb = memory.entities.get_context_blurb()
        if blurb:
            st.caption(f"🧠 {blurb}")

    # Quick-prompt chips
    chip_cols = st.columns(len(QUICK_PROMPTS))
    for col, label in zip(chip_cols, QUICK_PROMPTS):
        if col.button(label, use_container_width=True):
            st.session_state.pending_prompt = label

    # Chat history
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            render_message(msg, show_images=show_images)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scroll anchor
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
    st.markdown(
        '<script>document.getElementById("chat-bottom")?.scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True,
    )

    # Query stats row
    if show_stats and st.session_state.last_latency is not None:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Latency",    f"{st.session_state.last_latency:.2f}s")
        sc2.metric("Chunks",     st.session_state.last_chunks)
        sc3.metric("Tokens in",  st.session_state.last_tokens_in)
        sc4.metric("Tokens out", st.session_state.last_tokens_out)

    # Chat input
    chat_input = st.chat_input("Ask about BVRIT Hyderabad…", max_chars=MAX_INPUT_CHARS)
    prompt = chat_input or st.session_state.pop("pending_prompt", None)

    if prompt:
        prompt = re.sub(r"\s+", " ", prompt).strip()
        if not prompt:
            st.rerun()

        if queries_used >= MAX_QUERIES_PER_SESSION:
            st.warning("Session query limit reached. Reset the conversation to continue.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            if chunk_count == 0:
                answer_msg = {
                    "role": "assistant",
                    "content": "The knowledge base hasn't been indexed yet. Run `python chunk_and_index.py` to build the vector store, then refresh.",
                    "citations": [], "refused": True,
                }
            else:
                # Show typing indicator while generating
                typing_placeholder = st.empty()
                with typing_placeholder:
                    render_typing_indicator()

                try:
                    from rag import answer_question
                    result = answer_question(
                        question       = prompt,
                        section_filter = effective_filter,
                        top_k          = top_k,
                        model          = model,
                        history        = [m for m in st.session_state.messages[:-1] if m["role"] in ("user","assistant")],
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
                        "role": "assistant",
                        "content": strip_inline_citations(result.answer),
                        "citations": result.citations,
                        "images": result.images,
                        "refused": result.refused,
                        "no_prediction": no_pred,
                    }
                except Exception as e:
                    answer_msg = {
                        "role": "assistant",
                        "content": f"An error occurred: {e}",
                        "citations": [], "refused": True,
                    }
                finally:
                    typing_placeholder.empty()

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
        run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
    with ctrl_b:
        skip_ragas = st.checkbox("Skip RAGAS", value=False)
    with ctrl_c:
        dim_options = ["All"] + [
            f"{k} — {v}" for k, v in {
                "01":"Functional","02":"Quality","03":"Safety",
                "04":"Security","05":"Robustness","06":"Performance",
                "07":"Context","08":"RAGAS",
            }.items()
        ]
        dim_select = st.selectbox("Dimension", dim_options, index=0)

    if run_btn:
        dim_filter = None if dim_select == "All" else dim_select[:2]
        with st.spinner("Running test suite… this may take 1–2 minutes."):
            try:
                from eval import run_suite
                st.session_state.eval_report = run_suite(dim_filter=dim_filter, skip_ragas=skip_ragas)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    report = st.session_state.get("eval_report")
    if not report:
        st.info("Click **▶ Run Evaluation** to start. Results appear here.")
    else:
        s = report["summary"]
        pass_pct = int(s["pass_rate"] * 100)

        st.markdown(
            f"""<div class="summary-banner">
              <div class="sb-metric"><div class="sb-num">{s['total']}</div><div class="sb-label">TOTAL</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#90ee90">{s['passed']}</div><div class="sb-label">PASSED</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#ff6b6b">{s['failed']}</div><div class="sb-label">FAILED</div></div>
              <div class="sb-metric"><div class="sb-num" style="color:#ffd700">{s['warning']}</div><div class="sb-label">WARNINGS</div></div>
              <div class="sb-metric"><div class="sb-num">{pass_pct}%</div><div class="sb-label">PASS RATE</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Dimension Results")
        dims = report.get("dimensions", [])
        for row in [dims[i:i+2] for i in range(0, len(dims), 2)]:
            cols = st.columns(2)
            for col, dim in zip(cols, row):
                with col:
                    color = "#155724" if dim["failed"] == 0 else "#721c24"
                    st.markdown(
                        f'<div class="dim-card"><div class="dim-title" style="color:{color}">{safe(dim["id"])} {safe(dim["name"])}</div>'
                        f'<div class="dim-counts">{dim["passed"]} passed | {dim["failed"]} failed | {dim["warned"]} warnings | {dim["total"]} total</div></div>',
                        unsafe_allow_html=True,
                    )
                    for case in dim.get("cases", []):
                        icon = {"PASS":"✅","FAIL":"❌","WARN":"⚠️"}.get(case["verdict"],"❓")
                        st.caption(f"{icon} {case['id']}: {case['reason'][:80]}")

        all_cases = [c for d in dims for c in d.get("cases", [])]
        failures  = [c for c in all_cases if c["verdict"] == "FAIL"]
        if failures:
            st.markdown("### Failed Test Details")
            for c in failures:
                with st.expander(f"{c['id']} — {c['reason'][:60]}"):
                    st.markdown(f"**Question:** {safe(c['question'])}")
                    st.markdown(f"**Answer:** {safe(c['answer'][:400])}")
                    st.markdown(f"**Root cause:** {safe(c.get('root_cause',''))}")
                    st.markdown(f"**Suggested fix:** {safe(c.get('fix',''))}")

        wd = report.get("weakest_dimension")
        if wd:
            st.markdown("### Weakest Dimension")
            st.warning(f"**{wd['id']} {wd['name']}** ({int(wd['pass_rate']*100)}% pass rate)\n\n{wd['fix']}")

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
                            f'<div class="dim-card"><div class="dim-title">{label}</div><div class="dim-counts">{tooltip}</div></div>',
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

        if report.get("residual_risk_statement"):
            st.markdown("### Security — Residual Risk")
            st.info(report["residual_risk_statement"])

        st.download_button(
            label="⬇ Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name="bvrit_eval_report.json",
            mime="application/json",
        )

# ===========================================================================
# TAB 3 — ADMIN
# ===========================================================================
with tab_admin:
    render_hero(
        title       = "Admin Dashboard",
        subtitle    = "Governance, audit logging, rate limits, and usage statistics.",
        pills       = ["Audit Log", "Rate Limits", "Usage", "Prompt Version"],
        status_line = "Live governance data",
    )

    try:
        from governance import AuditLog, RateLimiter
        audit = AuditLog()
        stats = audit.get_stats()

        # Metric cards
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: render_metric_card(stats["total_queries"],          "Total Queries")
        with mc2: render_metric_card(stats["today_queries"],          "Today")
        with mc3: render_metric_card(f"{stats['avg_latency_s']:.2f}s","Avg Latency")
        with mc4: render_metric_card(stats["total_tokens"],           "Total Tokens")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Model Distribution")
            dist = stats.get("model_distribution", {})
            if dist:
                for mn, cnt in dist.items():
                    pct = cnt / max(stats["total_queries"], 1)
                    st.markdown(f"**{safe(mn)}** — {cnt} queries")
                    st.progress(pct)
            else:
                st.caption("No data yet.")

        with col_right:
            st.markdown("#### Rate Limiting")
            limiter = RateLimiter(max_per_session=40, max_per_minute=10)
            usage   = limiter.get_usage(st.session_state.get("session_id", "default"))
            st.markdown(
                f'<div class="sidebar-stat" style="background:rgba(91,35,51,0.06);border-color:rgba(91,35,51,0.15)">'
                f'<div class="sidebar-stat-num" style="color:#5b2333">{usage["session_queries"]}</div>'
                f'<div>Session queries / {usage["max_per_session"]}</div></div>'
                f'<div class="sidebar-stat" style="background:rgba(31,111,99,0.06);border-color:rgba(31,111,99,0.15)">'
                f'<div class="sidebar-stat-num" style="color:#1f6f63">{usage["remaining"]}</div>'
                f'<div>Remaining queries</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Recent Activity")
        recent = audit.get_recent(limit=20)
        if recent:
            for entry in recent:
                render_log_entry(entry)
        else:
            st.caption("No queries logged yet. Start chatting to see activity here.")

        st.markdown("---")
        st.markdown("#### Prompt Version")
        last_entry = recent[0] if recent else None
        if last_entry and last_entry.get("prompt_version"):
            st.code(last_entry["prompt_version"], language=None)
        else:
            st.caption("No prompt version tracked yet.")

    except Exception as e:
        st.warning(f"Admin dashboard unavailable: {e}")
