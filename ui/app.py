"""
Author: Pranay Hedau
Purpose: Streamlit chat interface for TechDocs QA Engine
Talks to the FastAPI backend at localhost:8000.
Date Created: 03/05/2026

Upgrades (v2):
- Hybrid search option added to retrieval method selector
- BM25 index status shown in sidebar stats
- Workers slider for parallel ingestion control
"""

import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"

# --- Page config ---
st.set_page_config(
    page_title="TechDocs QA Engine",
    page_icon="📚",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=10, value=5,
        help="More chunks = more context for the LLM, but slower"
    )

    # [UPGRADE: hybrid search] — added "hybrid" as third retrieval option
    # Previously: options=["similarity", "mmr"] — dense search only
    # Now: options=["similarity", "mmr", "hybrid"]
    # hybrid = BM25 keyword search + dense vector search merged with RRF
    # Fixes context recall weakness (0.76) by catching keyword matches
    # that pure semantic search misses when query wording differs from docs
    method = st.selectbox(
        "Retrieval method",
        options=["similarity", "mmr", "hybrid"],
        index=2,  # [UPGRADE] default to hybrid — best recall performance
        help=(
            "similarity: pure dense vector search\n"
            "mmr: diverse dense search (reduces redundant chunks)\n"
            "hybrid: BM25 + dense merged with RRF — best recall ✨"
        )
    )

    # [UPGRADE: hybrid search] — visual indicator of active method
    if method == "hybrid":
        st.success("🔀 Hybrid: BM25 + Dense + RRF")
    elif method == "mmr":
        st.info("🔄 MMR: Diverse dense retrieval")
    else:
        st.info("🔍 Similarity: Pure vector search")

    st.divider()
    st.subheader("📥 Ingest Documents")
    strategy = st.selectbox(
        "Chunking strategy",
        options=["recursive", "fixed", "semantic"],
    )
    force_recreate = st.checkbox("Force recreate collection", value=False)

    # [UPGRADE: parallel ingestion] — expose worker count in UI
    # Previously: no control over parallelism
    # Now: engineers can tune workers based on available CPU cores
    max_workers = st.slider(
        "Parallel ingest workers",
        min_value=1, max_value=8, value=4,
        help="Number of documents processed simultaneously"
    )

    if st.button("🔄 Ingest", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                res = requests.post(f"{API_BASE}/ingest", json={
                    "strategy": strategy,
                    "force_recreate": force_recreate,
                    # [UPGRADE: parallel ingestion] — pass workers to API
                    "max_workers": max_workers,
                })
                data = res.json()
                if res.status_code == 200:
                    st.success(
                        f"✅ {data['documents_loaded']} docs → "
                        f"{data['chunks_created']} chunks "
                        f"({data['latency_ms']}ms)"
                    )
                    # [UPGRADE: parallel ingestion] — show worker count used
                    st.caption(
                        f"Processed with {data.get('workers_used', max_workers)} "
                        f"parallel workers"
                    )
                else:
                    st.error(data.get("detail", "Ingestion failed"))
            except Exception as e:
                st.error(f"API error: {e}")

    st.divider()

    # Collection stats
    try:
        stats = requests.get(f"{API_BASE}/stats").json()
        if stats.get("exists"):
            st.metric("Vectors stored", stats["total_vectors"])
            st.metric("Dimensions", stats["dimensions"])
            st.caption(f"Distance: {stats['distance_metric']}")

            # [UPGRADE: hybrid search] — show BM25 index status
            # Tells engineers whether hybrid search is available or
            # if they need to re-ingest to rebuild the BM25 index
            bm25_ready = stats.get("bm25_index_built", False)
            bm25_size = stats.get("bm25_corpus_size", 0)
            if bm25_ready:
                st.success(f"🔀 BM25 index ready ({bm25_size} chunks)")
            else:
                st.warning(
                    "⚠️ BM25 index not built.\n"
                    "Hybrid search will fall back to dense only.\n"
                    "Re-ingest to rebuild."
                )
        else:
            st.warning("No collection found. Click Ingest first.")
    except:
        st.error(
            "⚠️ API not reachable. Start the FastAPI server first:\n\n"
            "`uvicorn api.main:app --port 8000`"
        )


# --- Main chat area ---
st.title("📚 TechDocs QA Engine")
st.caption(
    "Ask questions about your technical documentation — "
    "powered by RAG + llama3.2 · "
    # [UPGRADE] — surface current retrieval method in caption
    f"retrieval: **{method if 'method' in dir() else 'hybrid'}**"
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** — chunk {src['chunk_index']}")
                    st.code(src["content"], language=None)
        if msg.get("latency_ms"):
            # [UPGRADE] — show retrieval method alongside latency
            retrieval = msg.get("retrieval_method", "")
            st.caption(f"⏱ {msg['latency_ms']}ms · method: {retrieval}")

# Chat input
if question := st.chat_input("Ask a question about your docs..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving ({method}) and generating..."):
            try:
                res = requests.post(f"{API_BASE}/ask", json={
                    "question": question,
                    "k": k,
                    "method": method,
                }, timeout=120)

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    latency = data["latency_ms"]
                    retrieval_method = data["retrieval_method"]

                    st.markdown(answer)

                    with st.expander(
                        f"📎 Sources ({len(sources)} chunks · "
                        f"{retrieval_method} retrieval)"
                    ):
                        for src in sources:
                            st.caption(
                                f"**{src['source']}** — chunk {src['chunk_index']}"
                            )
                            st.code(src["content"], language=None)

                    # [UPGRADE] — show method badge alongside latency
                    method_badge = {
                        "hybrid": "🔀 hybrid",
                        "mmr": "🔄 mmr",
                        "similarity": "🔍 similarity",
                    }.get(retrieval_method, retrieval_method)

                    st.caption(f"⏱ {latency}ms · {method_badge}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency,
                        "retrieval_method": retrieval_method,
                    })

                else:
                    error = res.json().get("detail", "Unknown error")
                    st.error(f"API error: {error}")

            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out — llama3.2 is still generating. "
                    "Try a shorter question or restart Ollama."
                )
            except Exception as e:
                st.error(f"Could not reach API: {e}")