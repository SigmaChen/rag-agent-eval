from .parsing import parse_judge_response

_JUDGE_PROMPT = """You are an impartial judge evaluating search result quality.

Given a user's question and a retrieved document chunk, determine how relevant
the chunk is to answering the question.

Scoring criteria:
- 1.0: Directly answers the question or contains the exact information needed.
- 0.7-0.9: Highly relevant — contains closely related information.
- 0.4-0.6: Somewhat relevant — related topic but doesn't directly answer.
- 0.1-0.3: Barely relevant — tangentially related at best.
- 0.0: Completely irrelevant to the question.

Respond in this exact JSON format (no other text):
{{"reasoning": "<explain relevance in 1-2 sentences>",
"score": <float between 0.0 and 1.0>}}

Question: {question}

Retrieved chunk:
---
{chunk_text}
---"""


def score_retrieval_relevance(
    question: str,
    chunks: list[dict],
    generate_fn: callable,
) -> dict:
    """Score how relevant each retrieved chunk is to the question.

    Calls the eval LLM once per chunk, then averages.

    Returns:
        {"score": float, "per_chunk": [{"chunk_id": str, "score": float, "reasoning": str}]}
    """
    if not chunks:
        return {"score": 0.0, "per_chunk": [], "status": "success"}

    per_chunk = []
    for chunk in chunks:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            chunk_text=chunk.get("text", ""),
        )
        raw = generate_fn(prompt)
        parsed = _parse_judge_response(raw)
        per_chunk.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "score": parsed["score"],
            "reasoning": parsed["reasoning"],
            "status": parsed["status"],
            "raw_response": parsed.get("raw_response"),
        })

    valid_scores = [c["score"] for c in per_chunk if c["score"] is not None]
    if not valid_scores:
        return {"score": None, "per_chunk": per_chunk, "status": "parse_error"}

    avg_score = sum(valid_scores) / len(valid_scores)
    status = "success" if len(valid_scores) == len(per_chunk) else "partial_error"
    return {"score": round(avg_score, 3), "per_chunk": per_chunk, "status": status}


def _parse_judge_response(raw: str) -> dict:
    """Extract score and reasoning from judge LLM response."""
    return parse_judge_response(raw)
