import logging
import time

import anthropic
from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 10


_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Only use information from the context below. If the context does not contain "
    "enough information to answer, say so clearly — do not make things up."
)

_USER_TEMPLATE = """Context (retrieved documents):
---
{context}
---

Question: {question}"""


class GenerationError(Exception):
    """Raised when LLM generation fails in a way the caller should handle."""
    pass


def _call_with_retry(fn, max_retries=_MAX_RETRIES, initial_backoff=_INITIAL_BACKOFF):
    """Retry fn() with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except GenerationError as e:
            if "rate limit" not in str(e).lower() or attempt == max_retries:
                raise
            wait = initial_backoff * (2 ** attempt)
            logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
            time.sleep(wait)


def raw_generate(
    prompt: str,
    provider: str = "gemini",
    model: str = "gemini-3.6-flash",
    max_tokens: int = 1024,
    api_key: str | None = None,
) -> str:
    """Send a prompt directly to the LLM without RAG context or system prompt.

    Used by eval scorers where the prompt IS the full instruction.
    Retries automatically on rate-limit errors with exponential backoff.
    """
    def _call():
        return _raw_generate_once(prompt, provider, model, max_tokens, api_key)
    return _call_with_retry(_call)


def _raw_generate_once(prompt, provider, model, max_tokens, api_key):
    if provider == "anthropic":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            client = anthropic.Anthropic(**kwargs)
        except TypeError as e:
            if "authentication" in str(e).lower():
                raise GenerationError("No Anthropic API key found.") from e
            raise
        try:
            message = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as e:
            raise GenerationError("Anthropic rate limit hit.") from e
        except anthropic.AuthenticationError as e:
            raise GenerationError("Anthropic API key invalid.") from e
        except anthropic.APIConnectionError as e:
            raise GenerationError(f"Cannot connect to Anthropic API: {e}") from e
        except anthropic.APIStatusError as e:
            raise GenerationError(f"Anthropic API error ({e.status_code}): {e}") from e
        return message.content[0].text

    # Gemini
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = genai.Client(**kwargs)
    try:
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise GenerationError("Gemini rate limit hit.") from e
        if e.code == 403:
            raise GenerationError("Gemini API key invalid.") from e
        raise GenerationError(f"Gemini API error: {e}") from e
    except genai_errors.ServerError as e:
        raise GenerationError(f"Gemini server error: {e}") from e
    return response.text


def generate_answer(
    question: str,
    chunks: list[dict],
    provider: str = "gemini",
    model: str = "gemini-3.6-flash",
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> dict:
    """Generate an answer using the configured LLM provider.

    Retries automatically on rate-limit errors with exponential backoff.
    """
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    user_message = _USER_TEMPLATE.format(context=context, question=question)

    def _call():
        if provider == "anthropic":
            return _generate_anthropic(user_message, model, max_tokens, api_key)
        return _generate_gemini(user_message, model, max_tokens, api_key)

    return _call_with_retry(_call)


def _generate_gemini(
    user_message: str, model: str, max_tokens: int, api_key: str | None
) -> dict:
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = genai.Client(**kwargs)
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
            ),
        )
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise GenerationError(
                "Gemini rate limit hit. Free tier allows ~15 requests/min. Wait and retry."
            ) from e
        if e.code == 403:
            raise GenerationError(
                "Gemini API key invalid or not authorized. Check GEMINI_API_KEY in .env"
            ) from e
        raise GenerationError(f"Gemini API error: {e}") from e
    except genai_errors.ServerError as e:
        raise GenerationError(f"Gemini server error (try again): {e}") from e

    return {
        "answer": response.text,
        "model": model,
        "input_tokens": response.usage_metadata.prompt_token_count or 0,
        "output_tokens": response.usage_metadata.candidates_token_count or 0,
    }


def _generate_anthropic(
    user_message: str, model: str, max_tokens: int, api_key: str | None
) -> dict:
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    try:
        client = anthropic.Anthropic(**kwargs)
    except TypeError as e:
        if "authentication" in str(e).lower():
            raise GenerationError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env"
            ) from e
        raise

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.RateLimitError as e:
        raise GenerationError("Anthropic rate limit hit. Wait and retry.") from e
    except anthropic.AuthenticationError as e:
        raise GenerationError(
            "Anthropic API key invalid. Check ANTHROPIC_API_KEY in .env"
        ) from e
    except anthropic.APIConnectionError as e:
        raise GenerationError(f"Cannot connect to Anthropic API: {e}") from e
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Anthropic API error ({e.status_code}): {e}") from e

    return {
        "answer": message.content[0].text,
        "model": model,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
