from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM provider: "gemini" (free) or "anthropic"
    llm_provider: Literal["gemini", "anthropic"] = "gemini"

    # Gemini (free tier)
    gemini_api_key: str = ""
    generation_model_gemini: str = "gemini-3.7-flash"
    eval_model_gemini: str = "gemini-3.5-flash-lite"

    # Anthropic
    anthropic_api_key: str = ""
    generation_model_anthropic: str = "claude-sonnet-5"
    eval_model_anthropic: str = "claude-haiku-4-5"

    max_tokens: int = 4096

    # RAG
    chunk_size: int = Field(default=512, gt=0, validation_alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(
        default=50, ge=0, validation_alias="RAG_CHUNK_OVERLAP"
    )
    top_k: int = Field(default=5, gt=0, validation_alias="RAG_TOP_K")
    embedding_model: str = "all-MiniLM-L6-v2"
    vectordb_path: str = "data/vectordb"

    # Tracing
    trace_output_dir: str = "data/traces"

    @property
    def generation_model(self) -> str:
        if self.llm_provider == "anthropic":
            return self.generation_model_anthropic
        return self.generation_model_gemini

    @property
    def eval_model(self) -> str:
        if self.llm_provider == "anthropic":
            return self.eval_model_anthropic
        return self.eval_model_gemini


def get_settings() -> Settings:
    return Settings()
