"""
Author: Pranay Hedau
Purpose: LLM configuration module

Wraps Ollama's llama3.2 for answer generation.
"""

import os
from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


"""
    Purpose: Returns a ChatOllama instance.

    temperature=0.0 means deterministic output — same question
    always gets the same answer. Critical for RAG because I
    want factual retrieval, not creative generation.

    For creative tasks I'd set temperature=0.7-1.0.
    """
def get_llm(
    model: str = LLM_MODEL,
    temperature: float = 0.0,
) -> ChatOllama:
    
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
    return llm