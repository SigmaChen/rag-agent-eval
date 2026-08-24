# RAG Agent with Eval & Observability

A small, end-to-end RAG Q&A project built to demonstrate practical evaluation and observability—not just answer generation.

The core idea: building a RAG system is the easy part. Knowing whether it actually works — detecting hallucinations, measuring retrieval quality, and tracing every step — is the hard part. This project tackles that head-on.

## What This Project Demonstrates

1. **Automated Answer Evaluation** — Run a ground-truth Q&A test suite with an LLM-as-judge correctness score
2. **Hallucination Detection** — Check whether the generated answer stays within the scope of retrieved documents
3. **Retrieval Quality Assessment** — Measure whether the chunks pulled from the vector store are actually relevant to the question
4. **End-to-End Tracing** — Every query produces a structured trace: retrieval → prompt construction → generation → eval results, stored as append-only JSONL
5. **Provider Routing** — Use Gemini for low-cost development or Anthropic Claude for generation and evaluation

## Architecture

```mermaid
flowchart TD
    subgraph Offline["Offline (one-time)"]
        Docs["📄 Source Docs (.md)"]
        Ingest["Ingestion\n(load → chunk → embed)"]
        VDB[("ChromaDB\n(vector store)")]
        Docs --> Ingest --> VDB
    end

    subgraph Online["Online (per query)"]
        Q["❓ User Question"]
        Ret["Retrieval\n(top-k similarity search)"]
        Gen["Generation\n(LLM: Gemini / Claude)"]
        Q --> Ret
        VDB --> Ret
        Ret -- "relevant chunks" --> Gen
    end

    subgraph Eval["Evaluation & Observability"]
        GT["Ground Truth\n(Q&A pairs)"]
        Ev["Evaluation\n(correctness · hallucination · relevance)"]
        Trace["Tracing\n(JSONL logs)"]
        GT --> Ev
        Gen -- "answer" --> Ev
        Ret -- "chunks" --> Ev
        Gen --> Trace
        Ev -- "scores" --> Trace
    end
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM (generation) | Gemini 3.7 Flash / Claude Sonnet 5 | Selectable provider and generation model |
| LLM (eval judge) | Gemini 3.5 Flash Lite / Claude Haiku 4.5 | Separate, lower-cost judge model |
| Embeddings | `all-MiniLM-L6-v2` (local) | No API dependency, good enough for demo |
| Vector DB | ChromaDB | Simple, embedded, no infra needed |
| Document processing | LangChain text splitters | Battle-tested chunking |
| Config | Pydantic Settings | Type-safe, .env support |
| CLI | Typer + Rich | Clean developer UX |

## Project Structure

```
src/rag_agent_eval/
├── ingestion/       # Document loading, chunking, embedding, indexing
├── retrieval/       # ChromaDB vector search
├── generation/      # LLM prompt construction & response generation
├── eval/            # Answer correctness, hallucination, retrieval quality
├── tracing/         # Structured trace records (JSONL), trace viewer
├── config.py        # Centralized settings (Pydantic)
└── cli.py           # CLI entry point (Typer)

data/
├── raw/             # Source documents
├── vectordb/        # ChromaDB persistent storage
├── ground_truth/    # Q&A test pairs for evaluation
├── traces/          # Query trace logs (JSONL)
```

## Quick Start

```bash
# Clone & setup
git clone https://github.com/YOUR_USERNAME/rag-agent-eval.git
cd rag-agent-eval
cp .env.example .env   # Add your GEMINI_API_KEY (free at https://aistudio.google.com/apikey)

# Install
pip install -e ".[dev]"

# Ingest documents
rag ingest ./data/raw/

# Ask a question
rag ask "How do I use tool use with the Claude API?"

# Run evaluation
rag evaluate

# View traces
rag traces --limit 5
```

## Testing

```bash
pytest
ruff check .
```

Unit tests use injected fake judge functions and do not call external LLM APIs. Running
`rag ask` and `rag evaluate` requires a valid provider API key; evaluation results are
recorded in the JSONL trace log.

## Knowledge Base

The included example knowledge base contains a small Markdown introduction to the Claude API.
Replace or extend the files under `data/raw/`, then run `rag ingest` again.

## Limitations and Production Path

This repository is intentionally a local CLI demo. It does not implement a web service,
authentication, deployment, concurrent-safe trace storage, or comprehensive prompt-injection
defenses. Source replacement prevents stale chunks when a document is re-ingested, but deleting
an entire source file does not yet remove its indexed chunks. A production system would use
versioned vector indexes with atomic cutover, durable observability storage, a larger versioned
evaluation dataset, and explicit security and privacy controls.

## License

MIT
