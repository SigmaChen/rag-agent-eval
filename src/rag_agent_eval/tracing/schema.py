from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class RetrievalRecord(BaseModel):
    query: str = ""
    top_k: int = 0
    chunks: list[ChunkRecord] = Field(default_factory=list)
    duration_ms: float = 0.0


class ChunkRecord(BaseModel):
    chunk_id: str = ""
    source: str = ""
    text: str = ""
    score: float = 0.0


class GenerationRecord(BaseModel):
    model: str = ""
    prompt: str = ""
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0


class EvalRecord(BaseModel):
    answer_correctness: float | None = None
    hallucination_score: float | None = None
    retrieval_relevance: float | None = None
    eval_model: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    question: str = ""
    retrieval: RetrievalRecord = Field(default_factory=RetrievalRecord)
    generation: GenerationRecord = Field(default_factory=GenerationRecord)
    eval: EvalRecord = Field(default_factory=EvalRecord)
    metadata: dict[str, Any] = Field(default_factory=dict)
