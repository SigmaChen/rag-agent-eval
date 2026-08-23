from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM provider: "gemini" (free) or "anthropic"
    llm_provider: str = "gemini"

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
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    vectordb_path: str = "data/vectordb"

    # Knowledge base
    knowledge_base_source: str = "anthropic-docs"

    # Tracing
    trace_output_dir: str = "data/traces"
    eval_output_dir: str = "data/eval_results"

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
