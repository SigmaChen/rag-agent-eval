from .parsing import parse_judge_response

_JUDGE_PROMPT = """You are an impartial judge evaluating the quality of an AI assistant's answer.

Compare the **actual answer** against the **expected answer** for the given question.

Scoring criteria:
- 1.0: The actual answer fully covers the expected answer's key points, even if worded differently.
- 0.7-0.9: Mostly correct, but missing minor details or slightly imprecise.
- 0.4-0.6: Partially correct — captures some key points but misses important ones.
- 0.1-0.3: Mostly wrong or misleading, but has a small grain of truth.
- 0.0: Completely wrong, irrelevant, or contradicts the expected answer.

Respond in this exact JSON format (no other text):
{{"reasoning": "<your analysis in 2-3 sentences>", "score": <float between 0.0 and 1.0>}}

Question: {question}

Expected answer: {expected_answer}

Actual answer: {actual_answer}"""


def score_correctness(
    question: str,
    actual_answer: str,
    expected_answer: str,
    generate_fn: callable,
) -> dict:
    """Score answer correctness using LLM-as-judge.

    Args:
        generate_fn: A callable(prompt) -> str that calls the eval LLM.
            Decoupled from provider so the scorer is testable without API calls.

    Returns:
        {"score": float, "reasoning": str}
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )
    raw = generate_fn(prompt)
    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> dict:
    """Extract score and reasoning from judge LLM response."""
    return parse_judge_response(raw)
