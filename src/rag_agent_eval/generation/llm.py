import anthropic
from google import genai


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


def generate_answer(
    question: str,
    chunks: list[dict],
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> dict:
    """Generate an answer using the configured LLM provider."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    user_message = _USER_TEMPLATE.format(context=context, question=question)

    if provider == "anthropic":
        return _generate_anthropic(user_message, model, max_tokens, api_key)
    return _generate_gemini(user_message, model, max_tokens, api_key)


def _generate_gemini(
    user_message: str, model: str, max_tokens: int, api_key: str | None
) -> dict:
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = genai.Client(**kwargs)
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
        ),
    )
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
    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return {
        "answer": message.content[0].text,
        "model": model,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
