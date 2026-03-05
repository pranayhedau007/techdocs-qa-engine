"""
Author: Pranay Hedau
Purpose: Embedding module for TechDocs QA Engine
         Converts text chunks into vector representations (embeddings).
         These vectors capture semantic meaning — similar text → similar vectors.
"""

import time
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document


# Default embedding model — matches what we pulled with ollama pull
EMBEDDING_MODEL = "nomic-embed-text"


"""
    Purpose: Returns an OllamaEmbeddings instance.
    
    This is a lightweight object — it doesn't load the model into memory yet.
    The model only activates when you call embed_documents() or embed_query().
    
    We return the embeddings object rather than raw vectors so LangChain's
    vector store can use it directly for both ingestion and retrieval.
    """
def get_embeddings(model: str = EMBEDDING_MODEL) -> OllamaEmbeddings:
    
    embeddings = OllamaEmbeddings(
        model=model,
        base_url="http://localhost:11434",  # Ollama default port
    )
    return embeddings


"""
    Purpose: Embed a single query string into a vector.
    Used at query time — converts user question to vector for similarity search.
    
    Returns: List of 768 floats (the vector representation of the query)
    """
def embed_query(query: str, model: str = EMBEDDING_MODEL) -> List[float]:
    
    embeddings = get_embeddings(model)
    vector = embeddings.embed_query(query)
    return vector


"""
    Purpose: Embed a list of Document chunks into vectors.
    Used during ingestion — converts all chunks to vectors for storage.
    
    Returns:
        vectors:        List of embedding vectors (one per chunk)
        elapsed_time:   How long embedding took (useful for benchmarking)
    """
def embed_documents(
    chunks: List[Document],
    model: str = EMBEDDING_MODEL
) -> tuple[List[List[float]], float]:
    
    embeddings = get_embeddings(model)
    texts = [chunk.page_content for chunk in chunks]

    print(f"[embedder] Embedding {len(texts)} chunks with {model}...")
    start = time.time()

    vectors = embeddings.embed_documents(texts)

    elapsed = round(time.time() - start, 2)
    print(f"[embedder] ✅ Done in {elapsed}s — {len(vectors)} vectors, "
          f"{len(vectors[0])} dimensions each")

    return vectors, elapsed


"""
    Purpose: Return stats about the generated embeddings.
    Useful for README benchmarks and debugging.
    """
def get_embedding_stats(vectors: List[List[float]]) -> dict:
    
    if not vectors:
        return {"total_vectors": 0}

    return {
        "total_vectors": len(vectors),
        "dimensions": len(vectors[0]),
        "model": EMBEDDING_MODEL,
    }