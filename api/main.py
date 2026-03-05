"""
Author: Pranay Hedau
Purpose: FastAPI backend for TechDocs QA Engine
Exposes the RAG pipeline as REST endpoints.

Endpoints:
  POST /ask          — Ask a question, get an answer
  POST /ingest       — Ingest documents into the vector store
  GET  /health       — Health check (used by Docker/k8s)
  GET  /stats        — Collection stats
  Date created: 03/05/2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from src.ingestion.loader import load_all_docs
from src.ingestion.chunker import get_chunks
from src.retrieval.vector_store import ingest_documents, get_collection_stats
from src.generation.chain import ask


# --- App setup ---

app = FastAPI(
    title="TechDocs QA Engine",
    description="RAG pipeline for technical documentation Q&A",
    version="1.0.0",
)

# Allow Streamlit UI (port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response schemas ---

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="The question to ask the RAG pipeline",
        example="How do I install LangChain?"
    )
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    method: Optional[str] = Field(
        default="similarity",
        description="Retrieval method: 'similarity' or 'mmr'"
    )


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_index: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    retrieval_method: str
    latency_ms: float


class IngestRequest(BaseModel):
    strategy: Optional[str] = Field(
        default="recursive",
        description="Chunking strategy: 'fixed', 'recursive', or 'semantic'"
    )
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=50, ge=0, le=200)
    force_recreate: Optional[bool] = Field(
        default=False,
        description="Wipe and rebuild the collection from scratch"
    )


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    strategy_used: str
    latency_ms: float


# --- Endpoints ---

"""
    Purpose: Get call - Liveness probe — used by Docker, k8s, and load balancers.
    Returns 200 if the service is up.
    """
@app.get("/health")
def health_check():
    
    return {"status": "ok", "service": "techdocs-qa-engine"}


"""
    Purpose: Get call - Returns stats about the current vector store collection.
    Useful for debugging and monitoring.
    """
@app.get("/stats")
def get_stats():
    
    stats = get_collection_stats()
    return stats


"""
    Purpose: Main endpoint — ask a question, get a grounded answer.

    The pipeline:
    1. Embed the question using nomic-embed-text
    2. Retrieve top-k similar chunks from Qdrant
    3. Build prompt with retrieved context
    4. Generate answer with llama3.2
    5. Return answer + source chunks
    """
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    
    start = time.time()

    try:
        result = ask(
            question=request.question,
            k=request.k,
            method=request.method,
        )
    except ValueError as e:
        # Collection doesn't exist — needs ingestion first
        raise HTTPException(
            status_code=400,
            detail=f"Vector store not ready: {str(e)}. Call POST /ingest first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    # Format source chunks for response
    sources = []
    for i, doc in enumerate(result["sources"]):
        sources.append(SourceChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown").split("/")[-1],
            chunk_index=i,
        ))

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        retrieval_method=request.method,
        latency_ms=latency,
    )


"""
    Purpose: Ingest all documents from data/docs/ into the vector store.
    Call this once after adding new documents, or to rebuild with
    a different chunking strategy.
    """
@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    
    start = time.time()

    try:
        docs = load_all_docs()
        chunks = get_chunks(
            docs,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        ingest_documents(chunks, force_recreate=request.force_recreate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        strategy_used=request.strategy,
        latency_ms=latency,
    )