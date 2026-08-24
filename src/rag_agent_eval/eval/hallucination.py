from .parsing import parse_judge_response

_JUDGE_PROMPT = """You are an impartial judge checking whether an AI answer is
grounded in the provided context.

Your task: determine if the answer contains claims that are NOT supported by the context.
This is NOT about whether the answer is correct — it's about whether the answer stays
within the scope of the provided documents.

Scoring criteria:
- 0.0: Fully grounded — every claim in the answer is supported by the context.
- 0.1-0.3: Minor hallucination — mostly grounded but adds a small unsupported detail.
- 0.4-0.6: Moderate hallucination — mixes grounded claims with fabricated information.
- 0.7-0.9: Severe hallucination — most claims are not supported by the context.
- 1.0: Fully hallucinated — the answer has no basis in the context.

An answer that says "I don't have enough information" should score 0.0 (no hallucination).

Respond in this exact JSON format (no other text):
{{"reasoning": "<identify grounded and ungrounded claims in 2-3 sentences>",
"score": <float between 0.0 and 1.0>}}

Context (retrieved documents):
---
{context}
---

Answer to evaluate:
{answer}"""


def score_hallucination(
    answer: str,
    chunks: list[dict],
    generate_fn: callable,
) -> dict:
    """Score how much the answer hallucinates beyond the retrieved chunks.

    Returns:
        {"score": float, "reasoning": str}
        0.0 = fully grounded, 1.0 = fully hallucinated.
    """
    context = "\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
        for c in chunks
    )
    prompt = _JUDGE_PROMPT.format(context=context, answer=answer)
    raw = generate_fn(prompt)
    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> dict:
    """Extract score and reasoning from judge LLM response."""
    return parse_judge_response(raw)
