# Claude API Basics

## Messages API

The Messages API is the primary way to interact with Claude. You send a list of messages and receive a response.

### Endpoint

`POST /v1/messages`

### Required Parameters

- `model`: The model to use (e.g., `claude-sonnet-5`, `claude-haiku-4-5`)
- `max_tokens`: The maximum number of tokens to generate. This is a required parameter and must be explicitly set in every request.
- `messages`: An array of message objects with `role` and `content` fields.

### Example Request

```python
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content[0].text)
```

### Response Format

The response includes:
- `content`: Array of content blocks (text, tool_use, etc.)
- `model`: The model that handled the request
- `stop_reason`: Why the model stopped (`end_turn`, `max_tokens`, `tool_use`)
- `usage`: Token counts (`input_tokens`, `output_tokens`)

## Authentication

Set your API key via the `ANTHROPIC_API_KEY` environment variable or pass it directly:

```python
client = anthropic.Anthropic(api_key="your-key")
```

The SDK resolves credentials in this order:
1. `ANTHROPIC_API_KEY` environment variable
2. `ANTHROPIC_AUTH_TOKEN` environment variable
3. OAuth profile from `ant auth login`

## Tool Use

Tool use (function calling) lets Claude call functions you define.

### Defining Tools

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"}
            },
            "required": ["location"]
        }
    }
]
```

### Using Tools

When Claude decides to use a tool, the response will have `stop_reason: "tool_use"` and a `tool_use` content block with the tool name and input. You execute the function and send the result back.

## Streaming

Use streaming for real-time response display:

```python
with client.messages.stream(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a poem"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Thinking

Claude can show its reasoning process using extended thinking:

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "Solve this step by step..."}]
)
```

Use `thinking: {"type": "adaptive"}` on all current models. The model dynamically decides when and how much to think.

## Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Sonnet 5 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |
| Claude Opus 5 | $5.00 | $25.00 |
