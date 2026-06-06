"""Streamlit demo — shows the live routing decision + answer + cost/latency.

Run locally:
    streamlit run demo_app/streamlit_app.py

On Hugging Face Spaces: set API_BASE_URL to your Render/Railway API endpoint.
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

TIER_COLORS = {
    "long_context": "#4CAF50",
    "hybrid":       "#2196F3",
    "graph_rag":    "#FF9800",
    "agentic":      "#9C27B0",
}

TIER_ICONS = {
    "long_context": "📄",
    "hybrid":       "🔍",
    "graph_rag":    "🕸️",
    "agentic":      "🤖",
}

TIER_DESCRIPTIONS = {
    "long_context": "Full corpus in-context — fast, cheap, single-hop.",
    "hybrid":       "FAISS + BM25 + rerank — best for factual lookups.",
    "graph_rag":    "Entity-graph traversal — relational / multi-document.",
    "agentic":      "Plan → retrieve → critique loop — complex multi-step.",
}

EXAMPLE_QUESTIONS = [
    "What was Apple's total net revenue in FY2024?",
    "List NVIDIA's main reportable business segments.",
    "Compare Apple's supply-chain risks with the concerns its CFO raised on the Q3 earnings call.",
    "How did JPMorgan's stated interest-rate sensitivity assumptions relate to its actual Q2 NII result?",
    "Synthesise how rising AI compute demand affects NVIDIA revenue, Apple capex, and JPMorgan technology investment.",
    "What does the lunar calendar have to do with JPMorgan's Tier 1 capital ratio?",
]

st.set_page_config(
    page_title="Adaptive RAG — Financial Filings",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Adaptive RAG")
    st.caption("Routes each query to the cheapest tier that can answer it faithfully.")

    router_ver = st.radio("Router version", ["v1 (heuristic)", "v2 (LLM-judge)"], index=1)
    router_param = "v2" if router_ver.startswith("v2") else "v1"

    st.markdown("---")
    st.markdown("**Upload your own document**")
    st.caption("Upload a .txt or .pdf file to query it instead of the built-in SEC filings.")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Only re-upload if the file changed
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            with st.spinner("Indexing document…"):
                try:
                    api_url_sidebar = st.session_state.get("api_url", API_BASE)
                    resp = requests.post(
                        f"{api_url_sidebar}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state["session_id"] = data["session_id"]
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["uploaded_chunks"] = data["chunks"]
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")
                    st.session_state.pop("session_id", None)
                    st.session_state.pop("uploaded_filename", None)

    if st.session_state.get("session_id"):
        fname = st.session_state.get("uploaded_filename", "file")
        chunks = st.session_state.get("uploaded_chunks", "?")
        st.success(f"**{fname}** indexed ({chunks} chunks). Queries will use this document.")
        if st.button("Clear — use SEC filings instead"):
            st.session_state.pop("session_id", None)
            st.session_state.pop("uploaded_filename", None)
            st.session_state.pop("uploaded_chunks", None)
            st.rerun()

    st.markdown("---")
    st.markdown("**Tier guide**")
    for tier, desc in TIER_DESCRIPTIONS.items():
        color = TIER_COLORS[tier]
        icon = TIER_ICONS[tier]
        st.markdown(
            f'<span style="color:{color};font-weight:bold">{icon} {tier}</span><br>'
            f'<span style="font-size:0.85em">{desc}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Example questions**")
    for eq in EXAMPLE_QUESTIONS:
        if st.button(eq, key=eq):
            st.session_state["question_input"] = eq


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 Adaptive Retrieval Router")

if st.session_state.get("session_id"):
    fname = st.session_state.get("uploaded_filename", "your document")
    st.caption(
        f"Per-query routing across four tiers: long-context · hybrid · GraphRAG · agentic. "
        f"Querying **{fname}** (uploaded). Clear it in the sidebar to switch back to SEC filings."
    )
else:
    st.caption(
        "Per-query routing across four tiers: long-context · hybrid · GraphRAG · agentic. "
        "Every answer is cited back to SEC filings (AAPL, NVDA, JPM). "
        "Upload your own document in the sidebar."
    )

question_label = (
    f"Ask a question about {st.session_state.get('uploaded_filename', 'your document')}:"
    if st.session_state.get("session_id")
    else "Ask a question about AAPL, NVDA, or JPM SEC filings:"
)
question_placeholder = (
    "e.g. What are the main risks described in this document?"
    if st.session_state.get("session_id")
    else "e.g. Compare Apple's supply-chain risks with what the CFO said on the Q3 call."
)

question = st.text_area(
    question_label,
    value=st.session_state.get("question_input", ""),
    height=80,
    key="question_input",
    placeholder=question_placeholder,
)

col_btn, col_api = st.columns([1, 3])
with col_btn:
    run = st.button("Ask", type="primary", use_container_width=True)
with col_api:
    api_url = st.text_input("API URL", value=API_BASE, label_visibility="collapsed")
    st.session_state["api_url"] = api_url

if run and question.strip():
    with st.spinner("Routing and retrieving…"):
        t0 = time.perf_counter()
        payload: dict = {"question": question}
        if st.session_state.get("session_id"):
            payload["session_id"] = st.session_state["session_id"]
        try:
            resp = requests.post(
                f"{api_url}/query?router={router_param}",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            wall_ms = (time.perf_counter() - t0) * 1000
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot reach API at `{api_url}`. "
                "Start it with `uvicorn app.main:app --reload` or set API_BASE_URL."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    route = data["route"]
    result = data["result"]
    tier = result["tier"]
    color = TIER_COLORS.get(tier, "#888")
    icon = TIER_ICONS.get(tier, "❓")

    # ── Routing badge ──
    st.markdown("### Routing decision")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tier chosen", f"{icon} {tier}")
    with col2:
        st.metric("Est. hops", route["estimated_hops"])
    with col3:
        st.metric("Cost", f"${result['cost_usd']:.5f}")
    with col4:
        st.metric("Latency", f"{result['latency_ms']:.0f} ms  (wall {wall_ms:.0f} ms)")

    conf = route["features"].get("confidence")
    router_label = f"{router_param}"
    if conf is not None:
        router_label += f"  confidence={conf:.2f}"
    st.caption(f"Router {router_label} · {route['reasoning']}")

    # ── Tier cost comparison bar ──
    tier_order = ["long_context", "hybrid", "graph_rag", "agentic"]
    tier_base_costs = [0.0040, 0.0009, 0.0035, 0.0110]
    bar_data = {t: c for t, c in zip(tier_order, tier_base_costs)}
    bar_data[tier] = result["cost_usd"]

    import pandas as pd
    df = pd.DataFrame({"tier": list(bar_data.keys()), "cost_usd": list(bar_data.values())})
    df["selected"] = df["tier"] == tier
    st.bar_chart(df.set_index("tier")["cost_usd"], color=color, height=140)
    st.caption("Approximate tier cost comparison (selected tier shows actual cost).")

    # ── Answer ──
    st.markdown("### Answer")
    st.markdown(result["answer"])

    # ── Citations ──
    if result.get("citations"):
        with st.expander(f"Citations ({len(result['citations'])})"):
            for c in result["citations"]:
                doc_id = c.get("doc_id", "")
                snippet = c.get("snippet", "")
                score = c.get("score")
                score_str = f"  _(score: {score:.2f})_" if score else ""
                st.markdown(f"**`{doc_id}`**{score_str}")
                if snippet:
                    st.caption(snippet[:300])
                st.markdown("---")

    # ── Notes ──
    if result.get("notes"):
        with st.expander("Internal notes (debug)"):
            st.code(result["notes"])

elif run:
    st.warning("Please enter a question.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Research tool — not financial advice. "
    "Answers sourced exclusively from SEC EDGAR filings (AAPL, NVDA, JPM). "
    "[Source code](https://github.com/your-org/adaptive-rag)"
)
