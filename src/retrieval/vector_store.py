"""
Author: Pranay Hedau
Purpose: Qdrant vector store management

Handles:
- Creating and managing Qdrant collections
- Storing document chunks + their vectors
- The connection between our embeddings and retrieval layer

Upgrades (v2):
- Hybrid search combining BM25 sparse retrieval + dense vector retrieval
- Reciprocal Rank Fusion (RRF) to merge both result sets
- Fixes context recall weakness (0.76 → improved) by catching keyword
  matches that pure semantic search misses
"""

import os
import warnings
from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Purpose: [UPGRADE: hybrid search] — BM25 sparse retrieval
# BM25 is a classical keyword-based ranking algorithm (same family as
# early Google search). It finds documents containing the exact words
# in the query — catching matches that dense vector search misses when
# the wording differs but the meaning is the same.
# rank_bm25 is a lightweight pure-Python implementation — no server needed.
from rank_bm25 import BM25Okapi

from src.ingestion.embedder import get_embeddings

warnings.filterwarnings("ignore", message=".*Qdrant client version.*")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "techdocs")
EMBEDDING_DIMENSIONS = 768

# [UPGRADE: hybrid search] — BM25 index stored in memory
# Built once at ingest time, reused for every query.
# This is an in-memory index — it does not persist across server restarts.
# On restart, call _rebuild_bm25_index() or re-ingest to rebuild it.
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: Optional[List[Document]] = None


"""Purpose: Raw Qdrant client for admin operations."""
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


"""Purpose: Check if a collection already exists in Qdrant."""
def collection_exists(collection_name: str = COLLECTION_NAME) -> bool:
    client = get_qdrant_client()
    collections = client.get_collections().collections
    return any(c.name == collection_name for c in collections)


"""Purpose: Create a new Qdrant collection with Cosine distance."""
def create_collection(collection_name: str = COLLECTION_NAME) -> None:
    client = get_qdrant_client()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=Distance.COSINE,
        )
    )
    print(f"[vector_store] Created collection: '{collection_name}'")


"""Purpose: Delete a collection — used when re-ingesting with new chunking strategy."""
def delete_collection(collection_name: str = COLLECTION_NAME) -> None:
    client = get_qdrant_client()
    if collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[vector_store] Deleted collection: '{collection_name}'")
    else:
        print(f"[vector_store] Collection '{collection_name}' doesn't exist, skipping")


# ---------------------------------------------------------------------------
# [UPGRADE: hybrid search] — BM25 index builder
#
# BM25 works on tokenized text. We build an index over all chunks at
# ingest time so queries can be matched against it instantly.
#
# Tokenization: simple whitespace + lowercase split. Good enough for
# technical documentation. For production, I will use a proper tokenizer
# like NLTK or spaCy that handles punctuation and stop words.
#
# Why rebuild at ingest time?
# The BM25 index must match the chunks in Qdrant exactly. If you rebuild
# Qdrant with new chunks, you must rebuild BM25 too. Keeping them in sync
# is why _rebuild_bm25_index() is called inside ingest_documents().
# ---------------------------------------------------------------------------

def _rebuild_bm25_index(chunks: List[Document]) -> None:
    """
    [UPGRADE: hybrid search] — Build BM25 index from ingested chunks.
    Called automatically after every ingest so BM25 stays in sync with Qdrant.

    Args:
        chunks: The same chunks that were ingested into Qdrant
    """
    global _bm25_index, _bm25_corpus

    # Tokenize each chunk: lowercase + split on whitespace
    # Example: "How do I install LangChain?" →
    #          ["how", "do", "i", "install", "langchain?"]
    tokenized_corpus = [
        doc.page_content.lower().split()
        for doc in chunks
    ]

    _bm25_corpus = chunks
    _bm25_index = BM25Okapi(tokenized_corpus)

    print(f"[vector_store] BM25 index built over {len(chunks)} chunks")


# ---------------------------------------------------------------------------
# Purpose: [UPGRADE: hybrid search] — Reciprocal Rank Fusion to merge the
# two ranked list coming from BM25 and Dense vector search
#
# RRF merges two ranked lists into one by assigning each document a score
# based on its position (rank) in each list:
#
#   RRF_score(doc) = 1/(k + rank_in_dense) + 1/(k + rank_in_bm25)
#
# Where k=60 is a smoothing constant (standard value from the RRF paper).
#
# Why RRF over simple score averaging?
# Dense and BM25 scores are on completely different scales — dense returns
# cosine similarities (0.0-1.0), BM25 returns term frequency scores
# (unbounded). You cannot average them directly. RRF uses ranks (positions)
# instead of raw scores, making the combination scale-invariant.
#
# Example:
#   Dense results:  [chunk_A(rank1), chunk_C(rank2), chunk_B(rank3)]
#   BM25 results:   [chunk_B(rank1), chunk_A(rank2), chunk_D(rank3)]
#
#   chunk_A RRF = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ← WINS
#   chunk_B RRF = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
#   chunk_C RRF = 1/(60+2) + 0        = 0.0161           = 0.0161
#   chunk_D RRF = 0        + 1/(60+3) = 0.0159           = 0.0159
#
#   Final: [chunk_A, chunk_B, chunk_C, chunk_D]
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    dense_results: List[Document],
    bm25_results: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    [UPGRADE: hybrid search] — Merge dense and BM25 results using RRF.

    Args:
        dense_results: Ranked list from Qdrant vector search
        bm25_results:  Ranked list from BM25 keyword search
        k:             Smoothing constant (60 is standard from RRF paper)

    Returns:
        Merged and re-ranked list of documents
    """
    # Use page_content as the unique key for each chunk
    # (two chunks with the same content are the same chunk)
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Score dense results by rank position
    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    # Score BM25 results by rank position — add to existing scores
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    # Sort by combined RRF score descending
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys]


def _bm25_search(query: str, top_k: int = 10) -> List[Document]:
    """
    [UPGRADE: hybrid search] — Search using BM25 keyword matching.

    Args:
        query:  The user's question
        top_k:  How many results to return

    Returns:
        Top-k chunks ranked by BM25 score, or empty list if no index built
    """
    # Guard: if BM25 index not built yet (server restart), return empty
    # Dense search still works — hybrid degrades gracefully to dense-only
    if _bm25_index is None or _bm25_corpus is None:
        print("[vector_store] BM25 index not available — skipping sparse search")
        return []

    # Tokenize query the same way we tokenized the corpus
    tokenized_query = query.lower().split()

    # Get BM25 scores for every chunk in the corpus
    scores = _bm25_index.get_scores(tokenized_query)

    # Get indices of top-k highest scoring chunks
    # argsort returns ascending — we take the last top_k reversed
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Return the actual Document objects for top indices
    # Filter out zero-score results (query terms not found at all)
    return [
        _bm25_corpus[i]
        for i in top_indices
        if scores[i] > 0
    ]


"""
Purpose: Store chunks + vectors in Qdrant.

Args:
    chunks:          Output of chunker.get_chunks()
    collection_name: Target Qdrant collection
    force_recreate:  Wipe and rebuild collection from scratch

Returns:
    QdrantVectorStore ready for similarity search
"""
def ingest_documents(
    chunks: List[Document],
    collection_name: str = COLLECTION_NAME,
    force_recreate: bool = False,
) -> QdrantVectorStore:

    if force_recreate:
        delete_collection(collection_name)

    if not collection_exists(collection_name):
        create_collection(collection_name)

    embeddings = get_embeddings()

    print(f"[vector_store] Ingesting {len(chunks)} chunks into '{collection_name}'...")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=collection_name,
    )

    client = get_qdrant_client()
    count = client.count(collection_name).count
    print(f"[vector_store] Ingestion complete — {count} vectors stored")

    # [UPGRADE: hybrid search] — rebuild BM25 index after every ingest
    # This keeps BM25 in sync with whatever is now in Qdrant.
    # Must be called here — not separately — to guarantee consistency.
    _rebuild_bm25_index(chunks)

    return vectorstore


"""Purpose: Connect to existing collection at query time."""
def get_vectorstore(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:

    if not collection_exists(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' not found. "
            f"Run ingest_documents() first."
        )

    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        embedding=embeddings,
    )


# ---------------------------------------------------------------------------
# [UPGRADE: hybrid search] — Main hybrid search function
#
# This is the key upgrade. Instead of calling get_vectorstore().search()
# directly in the chain, the chain now calls hybrid_search() which:
#   1. Runs dense vector search against Qdrant (semantic similarity)
#   2. Runs BM25 keyword search against in-memory index
#   3. Merges both result sets using RRF
#   4. Returns the top-k merged results
#
# PREVIOUS APPROACH (dense only):
# def search(query, k):
#     vectorstore = get_vectorstore()
#     return vectorstore.similarity_search(query, k=k)
#
# Problem: if the query uses different words than the document,
# even with the same meaning, semantic search can miss relevant chunks.
# Example: query "install dependencies" misses chunk saying "pip requirements"
#
# HYBRID APPROACH:
# Dense catches semantic meaning → "install" ≈ "setup" ≈ "configure"
# BM25 catches exact keywords  → "pip" matches "pip" exactly
# RRF combines both            → best of both worlds
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> List[Document]:
    """
    [UPGRADE: hybrid search] — Combine dense + BM25 retrieval with RRF.

    Replaces simple similarity_search() with a two-stage retrieval:
    1. Dense: retrieve top 2k from Qdrant (more candidates for merging)
    2. BM25:  retrieve top 2k from in-memory index
    3. RRF:   merge and re-rank, return top k

    Why retrieve 2k from each before merging?
    The merge may reorder results significantly. Retrieving more candidates
    from each source gives RRF more material to work with and reduces the
    chance of missing a relevant chunk that ranked 6th in dense but 1st
    in BM25.

    Args:
        query:           The user's question
        k:               Final number of chunks to return
        collection_name: Qdrant collection to search

    Returns:
        Top-k chunks merged from dense + BM25 retrieval
    """
    # Fetch more candidates than needed — RRF reranking needs headroom
    candidate_k = k * 2

    # Stage 1 — Dense vector search (semantic similarity)
    vectorstore = get_vectorstore(collection_name)
    dense_results = vectorstore.similarity_search(query, k=candidate_k)
    print(f"[vector_store] Dense retrieval: {len(dense_results)} candidates")

    # Stage 2 — BM25 keyword search (exact term matching)
    bm25_results = _bm25_search(query, top_k=candidate_k)
    print(f"[vector_store] BM25 retrieval: {len(bm25_results)} candidates")

    # Stage 3 — RRF merge and rerank
    if not bm25_results:
        # BM25 unavailable (server restart) — fall back to dense only
        # System degrades gracefully rather than failing
        print("[vector_store] Falling back to dense-only retrieval")
        return dense_results[:k]

    merged = _reciprocal_rank_fusion(dense_results, bm25_results)
    print(f"[vector_store] Hybrid merged: {len(merged)} unique chunks → returning top {k}")

    return merged[:k]


"""Purpose: Stats about stored collection — for debugging and README benchmarks."""
def get_collection_stats(collection_name: str = COLLECTION_NAME) -> dict:

    client = get_qdrant_client()

    if not collection_exists(collection_name):
        return {"exists": False}

    info = client.get_collection(collection_name)
    count = client.count(collection_name).count

    return {
        "exists": True,
        "collection": collection_name,
        "total_vectors": count,
        "dimensions": info.config.params.vectors.size,
        "distance_metric": str(info.config.params.vectors.distance),
        # [UPGRADE: hybrid search] — surface BM25 index status in stats
        # Operators can check if BM25 is available without querying
        "bm25_index_built": _bm25_index is not None,
        "bm25_corpus_size": len(_bm25_corpus) if _bm25_corpus else 0,
    }