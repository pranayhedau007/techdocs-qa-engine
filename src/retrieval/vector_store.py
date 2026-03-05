"""
Author: Pranay Hedau
Purpose: Qdrant vector store management

Handles:
- Creating and managing Qdrant collections
- Storing document chunks + their vectors
- The connection between our embeddings and retrieval layer
"""

import os
import warnings
from typing import List

from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.ingestion.embedder import get_embeddings

warnings.filterwarnings("ignore", message=".*Qdrant client version.*")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "techdocs")
EMBEDDING_DIMENSIONS = 768

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
    print(f"[vector_store] ✅ Created collection: '{collection_name}'")


"""Purpose: Delete a collection — used when re-ingesting with new chunking strategy."""
def delete_collection(collection_name: str = COLLECTION_NAME) -> None:
    
    client = get_qdrant_client()
    if collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[vector_store] 🗑️  Deleted collection: '{collection_name}'")
    else:
        print(f"[vector_store] Collection '{collection_name}' doesn't exist, skipping")


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
    print(f"[vector_store] ✅ Ingestion complete — {count} vectors stored")

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
    }
