import json
from pathlib import Path

from .schema import TraceRecord


def save_trace(trace: TraceRecord, output_dir: str = "data/traces") -> str:
    """Append a trace record to JSONL file. Returns the trace_id."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "traces.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(trace.model_dump_json() + "\n")
    return trace.trace_id


def load_traces(output_dir: str = "data/traces", limit: int = 10) -> list[TraceRecord]:
    """Load the most recent traces from JSONL."""
    filepath = Path(output_dir) / "traces.jsonl"
    if not filepath.exists():
        return []
    lines = filepath.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-limit:] if limit else lines
    return [TraceRecord(**json.loads(line)) for line in reversed(recent)]
