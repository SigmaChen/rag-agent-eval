import json
import time
from pathlib import Path

from ..config import Settings
from ..retrieval.store import get_collection, search
from ..generation.llm import generate_answer, raw_generate, GenerationError
from ..tracing.schema import (
    TraceRecord, RetrievalRecord, ChunkRecord, GenerationRecord, EvalRecord,
)
from ..tracing.logger import save_trace
from .correctness import score_correctness
from .hallucination import score_hallucination
from .retrieval_relevance import score_retrieval_relevance


def load_ground_truth(path: str = "data/ground_truth/qa_pairs.json") -> list[dict]:
    """Load Q&A pairs from JSON file."""
    filepath = Path(path)
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def make_generate_fn(settings: Settings) -> callable:
    """Create a callable that sends a prompt to the eval LLM and returns text.

    Uses raw_generate (no RAG system prompt) so eval judge prompts
    are sent as-is without being wrapped in context templates.
    """
    api_key = ""
    if settings.llm_provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key

    def generate_fn(prompt: str) -> str:
        return raw_generate(
            prompt=prompt,
            provider=settings.llm_provider,
            model=settings.eval_model,
            max_tokens=1024,
            api_key=api_key or None,
        )

    return generate_fn


def run_evaluation(settings: Settings) -> dict:
    """Run full evaluation: for each Q&A pair, retrieve → generate → score.

    Returns:
        {
            "results": [per-question results],
            "summary": {avg scores across all questions},
        }
    """
    qa_pairs = load_ground_truth()
    if not qa_pairs:
        return {"results": [], "summary": {}}

    collection = get_collection(settings.vectordb_path, settings.embedding_model)
    if collection.count() == 0:
        raise RuntimeError("Knowledge base is empty. Run 'rag ingest' first.")

    generate_fn = make_generate_fn(settings)
    results = []

    for qa in qa_pairs:
        result = _evaluate_single(qa, collection, settings, generate_fn)
        results.append(result)

    summary = _compute_summary(results)
    return {"results": results, "summary": summary}


def _evaluate_single(
    qa: dict, collection, settings: Settings, generate_fn: callable
) -> dict:
    """Run retrieval → generation → scoring for one Q&A pair."""
    question = qa["question"]
    expected = qa["expected_answer"]

    # Retrieval
    t0 = time.perf_counter()
    hits = search(question, collection, settings.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Generation
    api_key = ""
    if settings.llm_provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key

    t0 = time.perf_counter()
    gen_result = generate_answer(
        question, hits,
        provider=settings.llm_provider,
        model=settings.generation_model,
        max_tokens=settings.max_tokens,
        api_key=api_key or None,
    )
    gen_ms = (time.perf_counter() - t0) * 1000

    # Scoring
    correctness = score_correctness(question, gen_result["answer"], expected, generate_fn)
    hallucination = score_hallucination(gen_result["answer"], hits, generate_fn)
    relevance = score_retrieval_relevance(question, hits, generate_fn)

    # Build trace
    trace = TraceRecord(
        question=question,
        retrieval=RetrievalRecord(
            query=question, top_k=settings.top_k,
            chunks=[ChunkRecord(chunk_id=h["chunk_id"], source=h["source"],
                                text=h["text"], score=h["score"]) for h in hits],
            duration_ms=retrieval_ms,
        ),
        generation=GenerationRecord(
            model=gen_result["model"], prompt=question,
            answer=gen_result["answer"],
            input_tokens=gen_result["input_tokens"],
            output_tokens=gen_result["output_tokens"],
            duration_ms=gen_ms,
        ),
        eval=EvalRecord(
            answer_correctness=correctness["score"],
            hallucination_score=hallucination["score"],
            retrieval_relevance=relevance["score"],
            eval_model=settings.eval_model,
            details={
                "correctness_reasoning": correctness["reasoning"],
                "hallucination_reasoning": hallucination["reasoning"],
                "relevance_per_chunk": relevance["per_chunk"],
            },
        ),
        metadata={"qa_id": qa.get("id", ""), "tags": qa.get("tags", [])},
    )
    save_trace(trace, settings.trace_output_dir)

    return {
        "qa_id": qa.get("id", ""),
        "question": question,
        "expected": expected,
        "actual": gen_result["answer"],
        "correctness": correctness["score"],
        "hallucination": hallucination["score"],
        "relevance": relevance["score"],
    }


def _compute_summary(results: list[dict]) -> dict:
    """Average scores across all evaluated Q&A pairs."""
    if not results:
        return {}
    n = len(results)
    return {
        "total_questions": n,
        "avg_correctness": round(sum(r["correctness"] for r in results) / n, 3),
        "avg_hallucination": round(sum(r["hallucination"] for r in results) / n, 3),
        "avg_relevance": round(sum(r["relevance"] for r in results) / n, 3),
    }
