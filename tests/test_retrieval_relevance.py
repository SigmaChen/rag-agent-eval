from rag_agent_eval.eval.retrieval_relevance import score_retrieval_relevance


SAMPLE_CHUNKS = [
    {"chunk_id": "doc::0", "text": "Use streaming for real-time response display."},
    {"chunk_id": "doc::1", "text": "Claude pricing varies by model tier."},
    {"chunk_id": "doc::2", "text": "Tool use lets Claude call functions you define."},
]


class TestScoreRetrievalRelevance:

    def test_all_relevant(self):
        def mock_llm(prompt: str) -> str:
            return '{"reasoning": "Directly relevant.", "score": 0.9}'

        result = score_retrieval_relevance(
            question="How does streaming work?",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.9
        assert len(result["per_chunk"]) == 3

    def test_mixed_relevance(self):
        """Different chunks get different scores."""
        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            scores = [0.9, 0.2, 0.1]
            score = scores[call_count % len(scores)]
            call_count += 1
            return f'{{"reasoning": "test", "score": {score}}}'

        result = score_retrieval_relevance(
            question="How does streaming work?",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert result["score"] == round((0.9 + 0.2 + 0.1) / 3, 3)
        assert result["per_chunk"][0]["score"] == 0.9
        assert result["per_chunk"][1]["score"] == 0.2
        assert result["per_chunk"][2]["score"] == 0.1

    def test_empty_chunks(self):
        result = score_retrieval_relevance(
            question="Anything?",
            chunks=[],
            generate_fn=lambda p: "",
        )
        assert result["score"] == 0.0
        assert result["per_chunk"] == []

    def test_calls_llm_once_per_chunk(self):
        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return '{"reasoning": "ok", "score": 0.5}'

        score_retrieval_relevance(
            question="test",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert call_count == len(SAMPLE_CHUNKS)

    def test_per_chunk_has_chunk_id(self):
        def mock_llm(prompt: str) -> str:
            return '{"reasoning": "ok", "score": 0.5}'

        result = score_retrieval_relevance(
            question="test",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        ids = [c["chunk_id"] for c in result["per_chunk"]]
        assert ids == ["doc::0", "doc::1", "doc::2"]
