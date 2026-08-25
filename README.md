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
    subgraph PREP["1 · Knowledge Preparation"]
        direction TB
        DOCS["Source documents<br/>Markdown"]
        CHUNK["Load & chunk<br/>overlapping passages"]
        EMBED["Embed locally<br/>all-MiniLM-L6-v2"]
        VDB[("ChromaDB<br/>persistent vector index")]

        DOCS --> CHUNK --> EMBED --> VDB
    end

    subgraph RAG["2 · RAG Answering"]
        direction TB
        QUESTION["User question"]
        RETRIEVE["Top-k retrieval"]
        GENERATE["Grounded generation<br/>Gemini or Claude"]
        ANSWER["Answer"]

        QUESTION --> RETRIEVE
        GENERATE --> ANSWER
    end

    subgraph QUALITY["3 · Evaluation & Observability"]
        direction TB
        GROUND_TRUTH["Ground-truth Q&A"]
        JUDGES["LLM judges<br/>correctness · hallucination · relevance"]
        TRACE[("JSONL traces<br/>chunks · timing · tokens · scores")]

        GROUND_TRUTH --> JUDGES --> TRACE
    end

    VDB --> RETRIEVE
    RETRIEVE -. "retrieved chunks" .-> GENERATE
    RETRIEVE -. "chunks" .-> JUDGES
    ANSWER --> JUDGES
    RETRIEVE --> TRACE
    ANSWER --> TRACE

    classDef source fill:#EEF2FF,stroke:#4F46E5,color:#1E1B4B,stroke-width:1.5px;
    classDef process fill:#ECFDF5,stroke:#059669,color:#064E3B,stroke-width:1.5px;
    classDef model fill:#FFF7ED,stroke:#EA580C,color:#7C2D12,stroke-width:1.5px;
    classDef store fill:#F5F3FF,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef output fill:#FDF2F8,stroke:#DB2777,color:#831843,stroke-width:1.5px;

    class DOCS,QUESTION,GROUND_TRUTH source;
    class CHUNK,EMBED,RETRIEVE,JUDGES process;
    class GENERATE model;
    class VDB,TRACE store;
    class ANSWER output;
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
git clone https://github.com/SigmaChen/rag-agent-eval.git
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
