"""
Author: Pranay Hedau
Purpose: Semantic retrieval module

Takes a user query, converts it to a vector, and finds the
most similar chunks in Qdrant. This is the "R" in RAG.

Two retrieval modes:
- similarity:      Pure vector similarity (default)
- mmr:             Maximal Marginal Relevance — balances relevance
                   with diversity, avoids returning 5 near-identical chunks
"""

from typing import List

from langchain.schema import Document

from src.retrieval.vector_store import get_vectorstore


"""
    Purpose: Retrieve the top-k most relevant chunks for a query.

    Args:
        query:           User's question in plain English
        k:               Number of chunks to return (default 5)
        method:          "similarity" or "mmr"
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
    
    vectorstore = get_vectorstore(collection_name)

    if method == "mmr":
        # MMR: fetch 2x candidates, then pick k diverse ones
        results = vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 2,
        )
    else:
        results = vectorstore.similarity_search(
            query=query,
            k=k,
        )

    print(f"[retriever] Query: '{query}'")
    print(f"[retriever] Retrieved {len(results)} chunks via {method}")

    return results


"""
    Purpose: Same as retrieve() but also returns similarity scores.
    Score is a float between 0 and 1 — higher = more similar.
    Useful for debugging retrieval quality and setting score thresholds.
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