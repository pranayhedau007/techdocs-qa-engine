"""
Author: Pranay Hedau
Purpose: RAGAS evaluation for TechDocs QA Engine
"""

import asyncio
import math
from ragas.run_config import RunConfig
from typing import List, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.outputs import LLMResult, Generation
from langchain_ollama import ChatOllama, OllamaEmbeddings
import ollama

from src.ingestion.loader import load_all_docs
from src.ingestion.chunker import get_chunks
from src.retrieval.vector_store import ingest_documents
from src.generation.chain import ask


EVAL_DATASET = [
    {
        "question": "How do I install LangChain?",
        "ground_truth": "You can install LangChain by running pip install langchain in your terminal."
    },
    {
        "question": "What is LangChain used for?",
        "ground_truth": "LangChain is a framework for building agents and LLM-powered applications by chaining together interoperable components."
    },
    {
        "question": "What is LangGraph?",
        "ground_truth": "LangGraph is a framework for building controllable agent workflows with more advanced customization and orchestration."
    },
    {
        "question": "Where can I find the LangChain API reference?",
        "ground_truth": "The LangChain API reference is available at reference.langchain.com/python."
    },
    {
        "question": "How can I contribute to LangChain?",
        "ground_truth": "You can contribute to LangChain by following the Contributing Guide at docs.langchain.com and finding good first issues."
    },
]

"""
    Purpose: Custom RAGAS LLM wrapper that calls Ollama correctly.
    Bypasses the temperature kwarg issue in newer ollama async client.
    """
class OllamaRagasLLM(BaseRagasLLM):
    
    def _to_str(self, prompt) -> str:
        """RAGAS passes StringPromptValue objects — extract plain text."""
        if hasattr(prompt, "text"):
            return prompt.text
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    def generate_text(self, prompt, **kwargs) -> LLMResult:
        text_prompt = self._to_str(prompt)
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": text_prompt}],
            options={"temperature": 0},
        )
        text = response["message"]["content"]
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(self, prompt, **kwargs) -> LLMResult:
        text_prompt = self._to_str(prompt)
        client = ollama.AsyncClient()
        response = await client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": text_prompt}],
            options={"temperature": 0},
        )
        text = response["message"]["content"]
        return LLMResult(generations=[[Generation(text=text)]])

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def get_llm_output(self, response: LLMResult) -> Optional[dict]:
        return None


"""Purpose: Run full RAG pipeline for each eval question."""
def build_ragas_dataset(strategy: str, k: int = 5) -> Dataset:
    
    print(f"\n[ragas] Building eval dataset with strategy='{strategy}', k={k}")

    docs = load_all_docs()
    chunks = get_chunks(docs, strategy=strategy)
    ingest_documents(chunks, force_recreate=True)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_DATASET):
        print(f"[ragas] Running query {i+1}/{len(EVAL_DATASET)}: {item['question']}")
        result = ask(item["question"], k=k)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["sources"]])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


"""Purpose: Run RAGAS evaluation for one chunking strategy."""
def run_evaluation(strategy: str, k: int = 5) -> dict:
    
    dataset = build_ragas_dataset(strategy, k)

    llm = OllamaRagasLLM()
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )

    print(f"[ragas] Running RAGAS evaluation for strategy='{strategy}'...")

    result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False,
    run_config=RunConfig(timeout=300, max_workers=1, max_retries=1),
)

    def safe_mean(val):
        if isinstance(val, list):
            valid = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return round(sum(valid) / len(valid), 4) if valid else 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else round(f, 4)
    except:
        return 0.0

    scores = {
        "strategy": strategy,
        "faithfulness": safe_mean(result["faithfulness"]),
        "answer_relevancy": safe_mean(result["answer_relevancy"]),
        "context_precision": safe_mean(result["context_precision"]),
        "context_recall": safe_mean(result["context_recall"]),
    }

    print(f"[ragas] ✅ Results for '{strategy}':")
    for metric, score in scores.items():
        if metric != "strategy":
            print(f"  {metric}: {score}")

    return scores


"""Purpose: Run evaluation across all 3 chunking strategies."""
def compare_all_strategies() -> list[dict]:
    
    results = []
    for strategy in ["fixed", "recursive", "semantic"]:
        scores = run_evaluation(strategy)
        results.append(scores)

    print("\n\n=== FINAL BENCHMARK RESULTS ===")
    print(f"{'Strategy':<12} {'Faithfulness':<14} {'Ans Relevancy':<15} {'Ctx Precision':<15} {'Ctx Recall'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['strategy']:<12} "
            f"{r['faithfulness']:<14} "
            f"{r['answer_relevancy']:<15} "
            f"{r['context_precision']:<15} "
            f"{r['context_recall']}"
        )

    return results