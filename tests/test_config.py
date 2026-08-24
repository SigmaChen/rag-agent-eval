import pytest
from pydantic import ValidationError

from rag_agent_eval.config import Settings


def test_rag_environment_variables_are_applied(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "256")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "25")
    monkeypatch.setenv("RAG_TOP_K", "3")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 256
    assert settings.chunk_overlap == 25
    assert settings.top_k == 3


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(llm_provider="anthorpic", _env_file=None)


@pytest.mark.parametrize(
    ("name", "value"),
    [("RAG_CHUNK_SIZE", "0"), ("RAG_CHUNK_OVERLAP", "-1"), ("RAG_TOP_K", "0")],
)
def test_invalid_rag_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
