import json
import math
import re

_CODE_FENCE_PATTERN = re.compile(
    r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.IGNORECASE | re.DOTALL
)


def parse_judge_response(raw: str) -> dict:
    """Parse a judge response without turning malformed output into a score."""
    candidate = raw.strip()
    fenced = _CODE_FENCE_PATTERN.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1)

    try:
        parsed = json.loads(candidate)
        score_value = parsed["score"]
        if isinstance(score_value, bool):
            raise ValueError("score must be numeric")
        score = float(score_value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "score": None,
            "reasoning": "Judge response could not be parsed as a valid score.",
            "status": "parse_error",
            "raw_response": raw[:1000],
        }

    return {
        "score": score,
        "reasoning": str(parsed.get("reasoning", "")),
        "status": "success",
    }
