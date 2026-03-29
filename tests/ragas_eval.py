"""
Author: Pranay Hedau
Purpose: RAGAS evaluation — compare similarity vs hybrid retrieval

Evaluates the RAG pipeline on 5 questions across two retrieval methods:
  - similarity: pure dense vector search (baseline — v1)
  - hybrid:     BM25 + dense + RRF (upgrade — v2)

Metrics measured (RAGAS framework):
  - Faithfulness:       are answer claims grounded in retrieved context?
  - Answer Relevancy:   does the answer address the question asked?
  - Context Precision:  are retrieved chunks relevant to the question?
  - Context Recall:     did retrieval find all needed chunks?

Run this after ingesting documents:
  python tests/ragas_eval.py

Expected output: side-by-side comparison table + improvement delta
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import json
from datetime import datetime

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.generation.chain import ask


# ---------------------------------------------------------------------------
# Test questions — written against FastAPI devdocs + LangChain README
# Each question is answerable from one of the two ingested documents.
# Chosen to stress-test both semantic understanding (dense) and
# keyword matching (Bm25) — hybrid should outperform on Q3 and Q4
# where exact technical terms matter more than semantic similarity.
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    # FastAPI questions — tests semantic understanding
    "How do I create a basic FastAPI application?",
    "What is automatic data validation in FastAPI and how does it work?",

    # LangChain questions — tests keyword matching (hybrid advantage)
    "How do I install LangChain?",
    "What is LCEL and how do you use the pipe operator?",

    # Cross-doc question — tests retrieval across both documents
    "How can I integrate LangChain with a FastAPI backend?",
]

# Ground truth answers — manually written based on document content
# RAGAS uses these to measure context recall and answer relevancy
# These should reflect what's actually in your documents
GROUND_TRUTHS = [
    # FastAPI basic app
    "You can create a basic FastAPI application by importing FastAPI, "
    "creating an app instance with app = FastAPI(), and defining route "
    "functions using decorators like @app.get() or @app.post().",

    # Automatic validation
    "FastAPI uses Pydantic models for automatic data validation. "
    "When you define request body parameters using Pydantic BaseModel classes, "
    "FastAPI automatically validates incoming data against the schema and "
    "returns clear error messages if validation fails.",

    # LangChain install
    "LangChain can be installed using pip with the command: "
    "pip install langchain. Additional integrations can be installed "
    "separately such as langchain-openai or langchain-community.",

    # LCEL
    "LCEL stands for LangChain Expression Language. It uses the pipe "
    "operator | to chain components together. For example: "
    "chain = prompt | llm | output_parser. Each component receives "
    "the output of the previous one as input.",

    # Integration
    "LangChain can be integrated with FastAPI by creating async route "
    "functions that invoke LangChain chains. You can use ainvoke() for "
    "async execution and return the chain output as a FastAPI response.",
]


def run_evaluation_for_method(method: str) -> dict:
    """
    Run all test questions through the RAG pipeline with one retrieval method.
    Returns a RAGAS-formatted dataset dict ready for evaluation.

    Args:
        method: "similarity" or "hybrid"

    Returns:
        dict with questions, answers, contexts, ground_truths
    """
    print(f"\n{'='*50}")
    print(f"Running evaluation: method = '{method}'")
    print(f"{'='*50}")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, (question, ground_truth) in enumerate(
        zip(TEST_QUESTIONS, GROUND_TRUTHS)
    ):
        print(f"\nQ{i+1}: {question}")

        try:
            result = ask(
                question=question,
                k=5,
                method=method,
            )

            answer = result["answer"]
            source_docs = result["sources"]

            # RAGAS expects contexts as list of strings — one per chunk
            context_texts = [doc.page_content for doc in source_docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(context_texts)
            ground_truths.append(ground_truth)

            print(f"  Answer: {answer[:100]}...")
            print(f"  Chunks retrieved: {len(context_texts)}")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Add empty entries so dataset stays aligned
            questions.append(question)
            answers.append("ERROR")
            contexts.append([""])
            ground_truths.append(ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def score_dataset(dataset_dict: dict, method_name: str) -> dict:
    """
    Run RAGAS evaluation on the dataset and return scores.

    Args:
        dataset_dict: output of run_evaluation_for_method()
        method_name:  label for logging

    Returns:
        dict of metric_name -> score
    """
    print(f"\nScoring {method_name} results with RAGAS...")

    dataset = Dataset.from_dict(dataset_dict)

    # RAGAS uses OpenAI by default as the judge model
    # Set OPENAI_API_KEY in your .env before running
    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "context_recall": round(scores["context_recall"], 4),
    }


def print_comparison_table(
    similarity_scores: dict,
    hybrid_scores: dict,
) -> None:
    """
    Print a formatted side-by-side comparison table with delta values.
    """
    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
    ]

    print("\n")
    print("=" * 65)
    print("  RAGAS EVALUATION — Similarity vs Hybrid Retrieval")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Similarity':>12} {'Hybrid':>10} {'Delta':>10}")
    print("-" * 65)

    for label, key in metrics:
        sim_score = similarity_scores[key]
        hyb_score = hybrid_scores[key]
        delta = hyb_score - sim_score
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")

        print(
            f"  {label:<22} {sim_score:>12.4f} {hyb_score:>10.4f} "
            f"{delta_str:>8} {arrow}"
        )

    print("=" * 65)

    # Summary
    avg_sim = sum(similarity_scores.values()) / len(similarity_scores)
    avg_hyb = sum(hybrid_scores.values()) / len(hybrid_scores)
    avg_delta = avg_hyb - avg_sim
    delta_str = f"+{avg_delta:.4f}" if avg_delta >= 0 else f"{avg_delta:.4f}"

    print(
        f"  {'Average':<22} {avg_sim:>12.4f} {avg_hyb:>10.4f} "
        f"{delta_str:>8}"
    )
    print("=" * 65)

    # Interpretation
    print("\n  Key finding:")
    recall_delta = hybrid_scores["context_recall"] - similarity_scores["context_recall"]
    if recall_delta > 0:
        print(
            f"  Context recall improved by {recall_delta:+.4f} with hybrid search.")
        print(
            f"  BM25 keyword matching caught chunks that dense search missed.")
    else:
        print(
            f"  Context recall unchanged — documents may be too short for")
        print(
            f"  BM25 to show significant advantage over dense search.")
    print()


def save_results(similarity_scores: dict, hybrid_scores: dict) -> None:
    """Save results to JSON for README update and future reference."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_questions": TEST_QUESTIONS,
        "similarity": similarity_scores,
        "hybrid": hybrid_scores,
        "delta": {
            k: round(hybrid_scores[k] - similarity_scores[k], 4)
            for k in similarity_scores
        }
    }

    output_path = os.path.join(
        os.path.dirname(__file__), "ragas_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    print("\nTechDocs QA Engine — RAGAS Evaluation")
    print("Testing: FastAPI devdocs + LangChain README")
    print("Methods: similarity (v1 baseline) vs hybrid (v2 upgrade)")
    print("\nMake sure:")
    print("  1. Qdrant is running: docker start qdrant")
    print("  2. Ollama is running with llama3.2 and nomic-embed-text")
    print("  3. Documents are ingested: POST /ingest")
    print("  4. OPENAI_API_KEY is set in .env (RAGAS judge model)")
    print()

    # Step 1 — run both methods
    similarity_data = run_evaluation_for_method("similarity")
    hybrid_data = run_evaluation_for_method("hybrid")

    # Step 2 — score both with RAGAS
    print("\nScoring with RAGAS (this calls OpenAI — takes ~60 seconds)...")
    similarity_scores = score_dataset(similarity_data, "similarity")
    hybrid_scores = score_dataset(hybrid_data, "hybrid")

    # Step 3 — print comparison table
    print_comparison_table(similarity_scores, hybrid_scores)

    # Step 4 — save results
    save_results(similarity_scores, hybrid_scores)

    print("  Done. Update your README and presentation with new scores.")
    print()