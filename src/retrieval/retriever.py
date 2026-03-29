"""
Author: Pranay Hedau
Purpose: Semantic retrieval module

Takes a user query, converts it to a vector, and finds the
most similar chunks in Qdrant. This is the "R" in RAG.

Three retrieval modes:
- similarity:      Pure vector similarity (default)
- mmr:             Maximal Marginal Relevance — balances relevance
                   with diversity, avoids returning 5 near-identical chunks
- hybrid:          [UPGRADE] BM25 keyword search + dense vector search
                   merged with Reciprocal Rank Fusion (RRF).
                   Fixes context recall weakness by catching keyword
                   matches that pure semantic search misses.
"""

from typing import List

from langchain.schema import Document

from src.retrieval.vector_store import get_vectorstore

# [UPGRADE: hybrid search] — import the new hybrid_search function
# Previously: only get_vectorstore was imported — dense search only
# Now: hybrid_search combines BM25 + dense retrieval via RRF
from src.retrieval.vector_store import get_vectorstore, hybrid_search


"""
    Purpose: Retrieve the top-k most relevant chunks for a query.

    Args:
        query:           User's question in plain English
        k:               Number of chunks to return (default 5)
        method:          "similarity", "mmr", or "hybrid" [UPGRADE]
        collection_name: Qdrant collection to search

    Returns:
        List of Document objects, most relevant first
    """
def retrieve(
    query: str,
    k: int = 5,
    method: str = "similarity",
    collection_name: str = "techdocs",
) -> List[Document]:

    # -------------------------------------------------------------------
    # [UPGRADE: hybrid search] — new "hybrid" method added
    #
    # PREVIOUS APPROACH (v1) — only two methods:
    #   "similarity" → pure dense vector search via Qdrant
    #   "mmr"        → diverse dense search via Qdrant
    #
    # Problem: both methods rely entirely on semantic similarity.
    # When query wording differs from document wording, relevant chunks
    # get missed. Example:
    #   Query:    "how do I install dependencies"
    #   Document: "run pip install -r requirements.txt"
    #   Dense search may miss this because "install dependencies" ≠
    #   "pip install requirements" semantically.
    #   BM25 catches it because "install" appears in both.
    #
    # NEW APPROACH (v2) — three methods:
    #   "similarity" → unchanged, pure dense
    #   "mmr"        → unchanged, diverse dense
    #   "hybrid"     → BM25 + dense merged with RRF [UPGRADE]
    # -------------------------------------------------------------------

    if method == "hybrid":
        # [UPGRADE: hybrid search] — use BM25 + dense + RRF
        # hybrid_search handles both retrieval stages and merging internally
        results = hybrid_search(
            query=query,
            k=k,
            collection_name=collection_name,
        )

    else:
        # Original dense retrieval — unchanged from v1
        vectorstore = get_vectorstore(collection_name)

        if method == "mmr":
            # MMR: fetch 2x candidates, then pick k diverse ones
            results = vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k * 2,
            )
        else:
            # Default: pure cosine similarity search
            results = vectorstore.similarity_search(
                query=query,
                k=k,
            )

    print(f"[retriever] Query: '{query}'")
    print(f"[retriever] Retrieved {len(results)} chunks via '{method}'")

    return results


"""
    Purpose: Same as retrieve() but also returns similarity scores.
    Score is a float between 0 and 1 — higher = more similar.
    Useful for debugging retrieval quality and setting score thresholds.
    Note: scores only available for dense similarity search — not hybrid.
    """
def retrieve_with_scores(
    query: str,
    k: int = 5,
    collection_name: str = "techdocs",
) -> List[tuple[Document, float]]:

    vectorstore = get_vectorstore(collection_name)

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
    )

    print(f"[retriever] Query: '{query}'")
    for doc, score in results:
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        print(f"  Score: {score:.4f} | Source: {source} | "
              f"Preview: {doc.page_content[:60].strip()}...")

    return results