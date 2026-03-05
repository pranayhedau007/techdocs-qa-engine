"""
Author: Pranay Hedau
Purpose: Streamlit chat interface for TechDocs QA Engine
Talks to the FastAPI backend at localhost:8000.
Date Created: 03/05/2026
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

    method = st.selectbox(
        "Retrieval method",
        options=["similarity", "mmr"],
        help="MMR reduces redundancy by picking diverse chunks"
    )

    st.divider()
    st.subheader("📥 Ingest Documents")
    strategy = st.selectbox(
        "Chunking strategy",
        options=["recursive", "fixed", "semantic"],
    )
    force_recreate = st.checkbox("Force recreate collection", value=False)

    if st.button("🔄 Ingest", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                res = requests.post(f"{API_BASE}/ingest", json={
                    "strategy": strategy,
                    "force_recreate": force_recreate,
                })
                data = res.json()
                if res.status_code == 200:
                    st.success(
                        f"✅ {data['documents_loaded']} docs → "
                        f"{data['chunks_created']} chunks "
                        f"({data['latency_ms']}ms)"
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
        else:
            st.warning("No collection found. Click Ingest first.")
    except:
        st.error("⚠️ API not reachable. Start the FastAPI server first:\n\n`python -m uvicorn api.main:app --port 8000`")


# --- Main chat area ---
st.title("📚 TechDocs QA Engine")
st.caption("Ask questions about your technical documentation — powered by RAG + llama3.2")

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
            st.caption(f"⏱ {msg['latency_ms']}ms")

# Chat input
if question := st.chat_input("Ask a question about your docs..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
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

                    st.markdown(answer)

                    with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                        for src in sources:
                            st.caption(f"**{src['source']}** — chunk {src['chunk_index']}")
                            st.code(src["content"], language=None)

                    st.caption(f"⏱ {latency}ms")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency,
                    })

                else:
                    error = res.json().get("detail", "Unknown error")
                    st.error(f"API error: {error}")

            except requests.exceptions.Timeout:
                st.error("Request timed out — llama3.2 is still generating. Try a shorter question or restart Ollama.")
            except Exception as e:
                st.error(f"Could not reach API: {e}")