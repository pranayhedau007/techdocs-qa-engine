"""
Author: Pranay Hedau
Purpose: Document loading module for TechDocs QA Engine
            Responsibilities:
            - Load PDF and text files from the data/docs directory
            - Convert raw files into LangChain Document objects
            - Each Document has .page_content (text) and .metadata (source, page number etc.)
Created Date: 03/02/2026
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.schema import Document


# Resolve the data/docs path relative to project root
DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"


"""
   Purpose: Load a single PDF file.
            PyPDFLoader splits by page automatically — each page becomes one Document.
    """
def load_pdf(file_path: str) -> List[Document]:
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[loader] Loaded {len(documents)} pages from {Path(file_path).name}")
    return documents

"""
    Purpose: Load a single .txt file.
             The entire file becomes one Document.
    """
def load_text(file_path: str) -> List[Document]:
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"[loader] Loaded {len(documents)} document(s) from {Path(file_path).name}")
    return documents


"""
    Purpose: Load ALL supported documents from the docs directory.
             Supports: .pdf, .txt
             Returns a flat list of all Document objects across all files.
    """
def load_all_docs(docs_dir: str = str(DOCS_DIR)) -> List[Document]:
    
    all_documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    # Walk through all files in the directory
    for file_path in sorted(docs_path.rglob("*")):
        if file_path.suffix.lower() == ".pdf":
            docs = load_pdf(str(file_path))
            all_documents.extend(docs)

        elif file_path.suffix.lower() == ".txt":
            docs = load_text(str(file_path))
            all_documents.extend(docs)

    if not all_documents:
        print(f"[loader] ⚠️  No documents found in {docs_path}")
        print("[loader] Add .pdf or .txt files to data/docs/ and try again")
    else:
        print(f"[loader] ✅ Total documents loaded: {len(all_documents)}")

    return all_documents

"""
    Purpose: Return basic stats about loaded documents.
             Useful for debugging and the README benchmarks section.
    """
def get_doc_stats(documents: List[Document]) -> dict:
    
    if not documents:
        return {"total_docs": 0}

    total_chars = sum(len(doc.page_content) for doc in documents)
    sources = list(set(
        Path(doc.metadata.get("source", "unknown")).name
        for doc in documents
    ))

    return {
        "total_docs": len(documents),
        "total_characters": total_chars,
        "avg_chars_per_doc": total_chars // len(documents),
        "unique_sources": sources,
        "num_sources": len(sources)
    }