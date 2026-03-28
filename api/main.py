"""
Author: Pranay Hedau
Purpose: FastAPI backend for TechDocs QA Engine
Exposes the RAG pipeline as REST endpoints.

Endpoints:
  POST /ask          — Ask a question, get an answer
  POST /ingest       — Ingest documents into the vector store
  GET  /health       — Health check (used by Docker/k8s)
  GET  /stats        — Collection stats
  GET  /metrics      — Prometheus metrics scrape endpoint [UPGRADE: observability]
  Date created: 03/05/2026

Upgrades (v2):
  - Async endpoints for concurrent query handling
  - Parallel document ingestion with ThreadPoolExecutor
  - Prometheus metrics for production observability
  - Structured logging for CloudWatch-compatible output
"""
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging
import json
import threading

# [UPGRADE: parallel ingestion] — ThreadPoolExecutor for processing
# multiple documents simultaneously instead of one at a time
from concurrent.futures import ThreadPoolExecutor

# [UPGRADE: observability] — Prometheus metrics for production monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.ingestion.loader import load_all_docs
from src.ingestion.chunker import get_chunks
from src.retrieval.vector_store import ingest_documents, get_collection_stats
from src.generation.chain import ask


# --- App setup ---

app = FastAPI(
    title="TechDocs QA Engine",
    description="RAG pipeline for technical documentation Q&A",
    version="2.0.0",  # [UPGRADE] bumped to v2 to reflect production upgrades
)

# Allow Streamlit UI (port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Purpose: For prod observability — Prometheus metrics definitions
#   - Query latency (p50, p95, p99) to monitor retrieval performance
#   - Chunks retrieved per query to detect retrieval quality degradation
#   - Ingestion duration to track pipeline performance over time
#   - Active ingestion jobs to detect concurrent load
# ---------------------------------------------------------------------------

query_counter = Counter(
    'techdocs_queries_total',
    'Total number of questions asked to the RAG pipeline'
)

query_latency_histogram = Histogram(
    'techdocs_query_latency_seconds',
    'End-to-end query latency from request to response',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

chunks_retrieved_histogram = Histogram(
    'techdocs_chunks_retrieved_per_query',
    'Number of context chunks retrieved per query',
    buckets=[1, 2, 3, 5, 8, 10, 15, 20]
)

ingestion_counter = Counter(
    'techdocs_ingestions_total',
    'Total number of ingestion jobs completed'
)

ingestion_latency_histogram = Histogram(
    'techdocs_ingestion_latency_seconds',
    'Total time to ingest and embed all documents',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

active_ingestion_gauge = Gauge(
    'techdocs_active_ingestion_jobs',
    'Number of ingestion jobs currently running'
)

query_error_counter = Counter(
    'techdocs_query_errors_total',
    'Total number of failed query requests'
)


# ---------------------------------------------------------------------------
# Purpose: Structured logging — JSON structured logs for CloudWatch
# Plain print() statements are unqueryable in production. Structured JSON
# logs can be ingested by CloudWatch Logs Insights and queried with SQL-like
# syntax — e.g. "find all queries with latency_ms > 2000 in the last hour"
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    Emits JSON-structured log lines compatible with CloudWatch Logs Insights.
    """

    def query_event(self, question: str, latency_ms: float,
                    chunks_found: int, method: str):
        print(json.dumps({
            "event": "query",
            "question_length": len(question),
            "latency_ms": latency_ms,
            "chunks_found": chunks_found,
            "retrieval_method": method,
            "timestamp": time.time()
        }))

    def ingest_event(self, docs_loaded: int, chunks_created: int,
                     strategy: str, latency_ms: float, workers: int):
        print(json.dumps({
            "event": "ingest",
            "docs_loaded": docs_loaded,
            "chunks_created": chunks_created,
            "strategy": strategy,
            "latency_ms": latency_ms,
            "parallel_workers": workers,
            "timestamp": time.time()
        }))

    def error_event(self, endpoint: str, error: str):
        print(json.dumps({
            "event": "error",
            "endpoint": endpoint,
            "error": error,
            "timestamp": time.time()
        }))


logger = StructuredLogger()


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
    # [UPGRADE: parallel ingestion] — configurable worker count
    # Exposes parallelism as an API parameter so callers can tune
    # based on available CPU cores. Default 4 is safe for most machines.
    max_workers: Optional[int] = Field(
        default=4,
        ge=1,
        le=16,
        description="[UPGRADE] Number of parallel workers for document ingestion"
    )


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    strategy_used: str
    latency_ms: float
    # [UPGRADE: parallel ingestion] — surface parallelism in response
    # so callers can verify how many workers processed their documents
    workers_used: int


# --- Endpoints ---

@app.get("/health")
# Purpose:  async get call for health check — converted from sync def to async def
# Previously: def health_check() — blocked the event loop on every call
# Now: async def — non-blocking, allows concurrent requests to be served
async def health_check():
    """
    Liveness probe — used by Docker, k8s, and load balancers.
    Returns 200 if the service is up.
    """
    return {"status": "ok", "service": "techdocs-qa-engine", "version": "2.0.0"}


@app.get("/stats")
# Purpose: async get call for stats about the current vector store collection — converted from sync def to async def
# Same reasoning as health_check — non-blocking for concurrent access
async def get_stats():
    """
    Returns stats about the current vector store collection.
    Useful for debugging and monitoring.
    """
    stats = get_collection_stats()
    return stats


# ---------------------------------------------------------------------------
# Purpose: Prometheus scrape endpoint to get the metrics info
# Exposes all metrics defined above at GET /metrics.
# A Prometheus server (or Amazon Managed Prometheus) scrapes this endpoint
# on a configurable interval (e.g. every 15 seconds) and stores the
# time-series data. Grafana then reads from Prometheus to build dashboards.
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics scrape endpoint.
    [UPGRADE] — New in v2. Exposes query latency, throughput,
    chunk retrieval counts, and ingestion job metrics.
    Connect to Amazon Managed Prometheus or local Prometheus instance.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/ask", response_model=AskResponse)
# Purpose — Main End Point to ask query
# Previously: def ask_question() — each request blocked the entire server
# thread, meaning concurrent queries from multiple engineers would queue up
# and wait. With async, FastAPI can handle multiple simultaneous queries.
async def ask_question(request: AskRequest):
    """
    Main endpoint — ask a question, get a grounded answer.

    The pipeline:
    1. Embed the question using nomic-embed-text
    2. Retrieve top-k similar chunks from Qdrant
    3. Build prompt with retrieved context
    4. Generate answer with llama3.2
    5. Return answer + source chunks + latency
    """
    start = time.time()

    # incrementing query counter on every request
    query_counter.inc()

    try:
        result = ask(
            question=request.question,
            k=request.k,
            method=request.method,
        )
    except ValueError as e:
        # Collection doesn't exist — needs ingestion first
        # tracking errors in Prometheus + structured log
        query_error_counter.inc()
        logger.error_event("/ask", str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Vector store not ready: {str(e)}. Call POST /ingest first."
        )
    except Exception as e:
        # tracking errors in Prometheus + structured log
        query_error_counter.inc()
        logger.error_event("/ask", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)
    latency_seconds = latency / 1000

    # record latency and chunk count in Prometheus
    # These two metrics answer: "is the system fast enough?" and
    # "is retrieval quality holding up?" — the two most important
    # production questions for a RAG pipeline.
    query_latency_histogram.observe(latency_seconds)
    chunks_retrieved_histogram.observe(len(result.get("sources", [])))

    # emit structured log for CloudWatch
    logger.query_event(
        question=request.question,
        latency_ms=latency,
        chunks_found=len(result.get("sources", [])),
        method=request.method
    )

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


# ---------------------------------------------------------------------------
# Purpose: Helper function for ThreadPoolExecutor
# Each document is processed independently — chunked and embedded in its
# own thread. 
# thread_lock protects the shared chunks list from race conditions —
# without it, two threads appending simultaneously could corrupt the list.
# ---------------------------------------------------------------------------

thread_lock = threading.Lock()


def _process_single_document(doc_and_params: tuple) -> list:
    """
    [UPGRADE: parallel ingestion] — processes one document in its own thread.
    Accepts a tuple of (doc, strategy, chunk_size, chunk_overlap) so it
    works cleanly with ThreadPoolExecutor.map() which passes one argument.
    Returns the list of chunks created from this document.
    """
    doc, strategy, chunk_size, chunk_overlap = doc_and_params
    chunks = get_chunks(
        [doc],  # process single document
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks


@app.post("/ingest", response_model=IngestResponse)
# Purpose: To ingest any technical doc — converted from sync def to async def
# Previously: def ingest() — blocked the server for the entire ingestion
# duration (could be 30-60 seconds for large document sets). Now async
# so health checks and stats endpoints remain responsive during ingestion.
async def ingest(request: IngestRequest):
    """
    Ingest all documents from data/docs/ into the vector store.
    Call this once after adding new documents, or to rebuild with
    a different chunking strategy.

    [UPGRADE] v2: Documents are processed in parallel using ThreadPoolExecutor.
    max_workers parameter controls parallelism (default: 4).
    """
    start = time.time()

    # [UPGRADE: observability] — track active ingestion jobs
    # If this gauge is > 1 something is wrong — ingestion should be idempotent
    active_ingestion_gauge.inc()

    try:
        docs = load_all_docs()

        # -------------------------------------------------------------------
        # PREVIOUS APPROACH — sequential ingestion, one doc at a time:
        #
        # chunks = get_chunks(
        #     docs,
        #     strategy=request.strategy,
        #     chunk_size=request.chunk_size,
        #     chunk_overlap=request.chunk_overlap,
        # )
        #
        # Problem: 10 documents took 10x longer than 1 document.
        # Each document waited for the previous one to finish before starting.
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # [UPGRADE: parallel ingestion] — ThreadPoolExecutor processes
        # multiple documents simultaneously.
        #
        # Why ThreadPoolExecutor and not asyncio?
        # Chunking and embedding are CPU-bound and I/O-bound mixed operations.
        # ThreadPoolExecutor lets Python's GIL release during I/O waits
        # (embedding model inference, file reads) so threads run concurrently.
        # For pure CPU-bound work, ProcessPoolExecutor would be better —
        # but for this workload, threads give sufficient parallelism.
        #
        # The thread_lock ensures the shared all_chunks list is updated
        # safely — without it, two threads appending simultaneously could
        # interleave writes and corrupt the list (race condition).
        # -------------------------------------------------------------------

        all_chunks = []

        # Prepare args tuples for each document
        doc_args = [
            (doc, request.strategy, request.chunk_size, request.chunk_overlap)
            for doc in docs
        ]

        # [UPGRADE: parallel ingestion] — process all documents in parallel
        # and returns a list of list - 
        # results = [
        #   [chunk1, chunk2, chunk3],    ← chunks from doc_1.pdf
        #   [chunk4, chunk5],            ← chunks from doc_2.pdf
        #   [chunk6, chunk7, chunk8],    ← chunks from doc_3.pdf
        # ]
        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            results = list(executor.map(_process_single_document, doc_args))

        # Safely merge results from all threads, basically merging the list of list above
        #using extend not append : as append adds it as a single ele eg. [1,2,3,[4,5]]
        with thread_lock:
            for doc_chunks in results:
                all_chunks.extend(doc_chunks)

        # Ingest the combined chunks into Qdrant
        ingest_documents(all_chunks, force_recreate=request.force_recreate)

    except Exception as e:
        # [UPGRADE: observability] — log failure with structured logger
        logger.error_event("/ingest", str(e))
        active_ingestion_gauge.dec()
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    # [UPGRADE: observability] — record ingestion metrics in Prometheus
    ingestion_counter.inc()
    ingestion_latency_histogram.observe(latency / 1000)
    active_ingestion_gauge.dec()

    # [UPGRADE: structured logging] — emit structured log for CloudWatch
    logger.ingest_event(
        docs_loaded=len(docs),
        chunks_created=len(all_chunks),
        strategy=request.strategy,
        latency_ms=latency,
        workers=request.max_workers
    )
 
    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(all_chunks),
        strategy_used=request.strategy,
        latency_ms=latency,
        # [UPGRADE: parallel ingestion] — surface worker count in response
        workers_used=request.max_workers,
    )