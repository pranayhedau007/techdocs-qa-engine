"""
Author: Pranay Hedau
Purpose: Document chunking module for TechDocs QA Engine
        We implement 3 strategies so we can compare them with RAGAS later:
        - Fixed size:    Simple, fast, ignores document structure
        - Recursive:     Smarter, respects natural text boundaries
        - Semantic:      Experimental, splits by meaning not just size
"""

from typing import List, Tuple
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

"""
    Purpose: Strategy 1 — Fixed Size Chunking
    
    Splits text every N characters regardless of content.
    Like cutting a book every 500 characters — mid-sentence, mid-word, anywhere.
    
    Pros: Simple, predictable, fast
    Cons: Can break sentences/paragraphs mid-thought, hurting retrieval quality
    
    When to use: Quick prototypes, highly structured data (CSV rows, logs)
    """
def chunk_fixed(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    
    splitter = CharacterTextSplitter(
        separator="\n",          # Try to split on newlines first
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[chunker] Fixed size: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


"""
    Purpose: Strategy 2 — Recursive Character Splitting
    
    Tries a hierarchy of separators in order:
    ["\\n\\n", "\\n", " ", ""]
    
    First tries to split on double newlines (paragraphs).
    If chunks are still too big, tries single newlines (sentences).
    If still too big, tries spaces (words).
    Last resort: splits anywhere.
    
    Pros: Respects natural language structure, much better retrieval quality
    Cons: Slightly slower than fixed, chunks vary in size
    
    When to use: Almost always as this is considered the industry standard
    """
def chunk_recursive(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Hierarchy of split points
    )
    chunks = splitter.split_documents(documents)
    print(f"[chunker] Recursive: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


"""
    Purpose: Strategy 3 — Semantic Chunking (approximation without sentence transformers)
    
    Groups text by paragraph boundaries — keeps related sentences together.
    True semantic chunking would use an embedding model to detect topic shifts,
    but that's expensive. This is a lightweight approximation.
    
    Pros: Keeps related ideas together, good for narrative/explanatory text
    Cons: Chunks can vary wildly in size
    
    When to use: Documentation, articles, long-form explanatory content
    """
def chunk_semantic(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],  # Extra paragraph break priority
    )
    chunks = splitter.split_documents(documents)

    # Add chunking strategy to metadata — useful for evaluation later
    for chunk in chunks:
        chunk.metadata["chunking_strategy"] = "semantic"

    print(f"[chunker] Semantic: {len(documents)} docs → {len(chunks)} chunks")
    return chunks

"""
    Purpose: Main entry point
    Args:
        documents:     Output of loader.load_all_docs()
        strategy:      "fixed", "recursive", or "semantic"
        chunk_size:    Max characters per chunk (default 500)
        chunk_overlap: Overlap between chunks (default 50)
    
    Returns:
        List of chunked Document objects ready for embedding
    """
def get_chunks(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    
    strategies = {
        "fixed": chunk_fixed,
        "recursive": chunk_recursive,
        "semantic": chunk_semantic,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}"
        )

    chunks = strategies[strategy](documents, chunk_size, chunk_overlap)

    # Tag every chunk with its strategy for evaluation comparisons
    for chunk in chunks:
        chunk.metadata["chunking_strategy"] = strategy
        chunk.metadata["chunk_size_config"] = chunk_size
        chunk.metadata["chunk_overlap_config"] = chunk_overlap

    return chunks


"""
    Purpose: Run all 3 strategies and return comparison stats.
    Used in RAGAS evaluation to find the best strategy for our dataset.
    """
def compare_strategies(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    
    results = {}

    for strategy in ["fixed", "recursive", "semantic"]:
        chunks = get_chunks(documents, strategy, chunk_size, chunk_overlap)
        chunk_lengths = [len(c.page_content) for c in chunks]

        results[strategy] = {
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_lengths) // len(chunk_lengths),
            "min_chunk_size": min(chunk_lengths),
            "max_chunk_size": max(chunk_lengths),
        }

    return results