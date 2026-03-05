"""
Author: Pranay Hedau
Purpose: RAG chain module

Combines retrieval + generation into a single pipeline:
  User question
       │
       ▼
  Retrieve top-k chunks from Qdrant
       │
       ▼
  Build prompt: [system] + [context chunks] + [question]
       │
       ▼
  Generate answer with llama3.2
       │
       ▼
  Return answer + source chunks (for citations)
"""

from typing import Any
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.retriever import retrieve
from src.generation.llm import get_llm


# System prompt — instructs the LLM how to behave
# "only use the context" is critical — prevents hallucination
SYSTEM_PROMPT = """You are a helpful technical documentation assistant.
Answer the user's question using ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Be concise and precise. Use bullet points for lists.

Context:
{context}"""

"""
    Purpose: Joins retrieved chunks into a single context string.
    Each chunk separated by a divider for clarity.
    """
def format_context(docs: list[Document]) -> str:
    
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


"""
    Purpose: Main RAG function — ask a question, get an answer.

    Args:
        question:        User's question
        k:               Number of chunks to retrieve
        collection_name: Qdrant collection to search

    Returns:
        {
          "question":  original question,
          "answer":    generated answer string,
          "sources":   list of Document chunks used
        }
    """
def ask(
    question: str,
    k: int = 5,
    method: str = "similarity",
    collection_name: str = "techdocs",
) -> dict[str, Any]:
    
    # Step 1: Retrieve relevant chunks
    source_docs = retrieve(question, k=k, method=method, collection_name=collection_name)

    # Step 2: Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Step 3: Get LLM
    llm = get_llm()

    # Step 4: Build chain using LangChain Expression Language (LCEL)
    # | is the pipe operator — chains steps together
    chain = prompt | llm | StrOutputParser()

    # Step 5: Generate answer
    context = format_context(source_docs)
    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "sources": source_docs,
    }