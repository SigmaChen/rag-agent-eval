import time

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .ingestion.loader import load_documents
from .ingestion.chunker import chunk_documents
from .retrieval.store import get_collection, index_chunks, search
from .generation.llm import generate_answer, GenerationError
from .eval.runner import run_evaluation
from .tracing.schema import (
    TraceRecord, RetrievalRecord, ChunkRecord, GenerationRecord,
)
from .tracing.logger import save_trace, load_traces

app = typer.Typer(help="RAG Agent with Eval & Observability")
console = Console()


@app.command()
def ingest(source: str = typer.Argument(help="Directory containing .md files")):
    """Ingest markdown documents into the knowledge base."""
    settings = get_settings()
    with console.status("Loading documents..."):
        docs = load_documents(source)
    if not docs:
        console.print(f"[red]No supported files found in {source}[/red]")
        raise typer.Exit(1)
    console.print(f"Loaded {len(docs)} document(s)")

    with console.status("Chunking..."):
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    console.print(f"Created {len(chunks)} chunk(s)")

    with console.status("Indexing into vector DB..."):
        collection = get_collection(settings.vectordb_path, settings.embedding_model)
        count = index_chunks(chunks, collection)
    console.print(f"[green]Indexed {count} chunks into ChromaDB[/green]")


@app.command()
def ask(question: str = typer.Argument(help="Question to ask the RAG agent")):
    """Ask a question — retrieves context and generates an answer."""
    settings = get_settings()
    trace = TraceRecord(question=question)

    collection = get_collection(settings.vectordb_path, settings.embedding_model)
    if collection.count() == 0:
        console.print("[red]Knowledge base is empty. Run 'rag ingest' first.[/red]")
        raise typer.Exit(1)

    t0 = time.perf_counter()
    hits = search(question, collection, settings.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    trace.retrieval = RetrievalRecord(
        query=question,
        top_k=settings.top_k,
        chunks=[ChunkRecord(chunk_id=h["chunk_id"], source=h["source"],
                            text=h["text"], score=h["score"]) for h in hits],
        duration_ms=retrieval_ms,
    )

    api_key = ""
    if settings.llm_provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key

    console.print(f"[dim]Using {settings.llm_provider} / {settings.generation_model}[/dim]")

    with console.status("Generating answer..."):
        t0 = time.perf_counter()
        try:
            result = generate_answer(
                question, hits,
                provider=settings.llm_provider,
                model=settings.generation_model,
                max_tokens=settings.max_tokens,
                api_key=api_key or None,
            )
        except GenerationError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        gen_ms = (time.perf_counter() - t0) * 1000

    trace.generation = GenerationRecord(
        model=result["model"],
        prompt=question,
        answer=result["answer"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        duration_ms=gen_ms,
    )

    trace_id = save_trace(trace, settings.trace_output_dir)

    console.print()
    console.print(result["answer"])
    console.print()
    console.print(
        f"[dim]trace={trace_id[:8]}  "
        f"retrieval={retrieval_ms:.0f}ms  "
        f"generation={gen_ms:.0f}ms  "
        f"tokens={result['input_tokens']}+{result['output_tokens']}[/dim]"
    )


@app.command()
def evaluate():
    """Run evaluation suite against ground truth Q&A pairs."""
    settings = get_settings()
    console.print(f"[dim]Eval model: {settings.eval_model}[/dim]")

    with console.status("Running evaluation..."):
        try:
            output = run_evaluation(settings)
        except GenerationError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    results = output["results"]
    summary = output["summary"]

    if not results:
        console.print("[yellow]No ground truth Q&A pairs found in data/ground_truth/qa_pairs.json[/yellow]")
        return

    table = Table(title="Evaluation Results")
    table.add_column("ID", style="dim", width=14)
    table.add_column("Question", max_width=30)
    table.add_column("Correct", justify="right")
    table.add_column("Halluc.", justify="right")
    table.add_column("Relevance", justify="right")

    for r in results:
        table.add_row(
            r["qa_id"],
            r["question"][:30],
            f"{r['correctness']:.2f}",
            f"{r['hallucination']:.2f}",
            f"{r['relevance']:.2f}",
        )
    console.print(table)

    console.print()
    console.print(
        f"[bold]Summary ({summary['total_questions']} questions):[/bold]  "
        f"correctness={summary['avg_correctness']:.2f}  "
        f"hallucination={summary['avg_hallucination']:.2f}  "
        f"relevance={summary['avg_relevance']:.2f}"
    )


@app.command()
def traces(limit: int = typer.Option(10, help="Number of recent traces to show")):
    """Show recent query traces."""
    settings = get_settings()
    records = load_traces(settings.trace_output_dir, limit)
    if not records:
        console.print("[yellow]No traces found.[/yellow]")
        return

    table = Table(title="Recent Traces")
    table.add_column("Trace ID", style="dim", width=10)
    table.add_column("Question", max_width=40)
    table.add_column("Model")
    table.add_column("Retrieval", justify="right")
    table.add_column("Generation", justify="right")
    table.add_column("Tokens", justify="right")

    for t in records:
        table.add_row(
            t.trace_id[:8],
            t.question[:40],
            t.generation.model,
            f"{t.retrieval.duration_ms:.0f}ms",
            f"{t.generation.duration_ms:.0f}ms",
            f"{t.generation.input_tokens}+{t.generation.output_tokens}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
