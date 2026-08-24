from rag_agent_eval.eval.hallucination import _parse_judge_response, score_hallucination

SAMPLE_CHUNKS = [
    {"source": "api-docs.md", "text": "max_tokens is a required parameter."},
    {"source": "api-docs.md", "text": "Use streaming for real-time display."},
]


class TestParseJudgeResponse:

    def test_valid_json(self):
        raw = '{"reasoning": "All claims are grounded.", "score": 0.0}'
        result = _parse_judge_response(raw)
        assert result["score"] == 0.0

    def test_invalid_score_is_not_reported_as_grounded(self):
        raw = '{"reasoning": "Bad.", "score": 1.5}'
        result = _parse_judge_response(raw)
        assert result["score"] is None
        assert result["status"] == "parse_error"

    def test_unparseable(self):
        raw = "No score here."
        result = _parse_judge_response(raw)
        assert result["score"] is None
        assert result["status"] == "parse_error"


class TestScoreHallucination:

    def test_grounded_answer_scores_low(self):
        def mock_llm(prompt: str) -> str:
            return (
                '{"reasoning": "The answer is supported by context.", '
                '"score": 0.0}'
            )

        result = score_hallucination(
            answer="max_tokens is required in every request.",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.0

    def test_hallucinated_answer_scores_high(self):
        def mock_llm(prompt: str) -> str:
            return (
                '{"reasoning": "The claimed default is not in context.", '
                '"score": 0.8}'
            )

        result = score_hallucination(
            answer="max_tokens defaults to 4096 if not set.",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.8

    def test_prompt_contains_chunks_and_answer(self):
        captured = []

        def mock_llm(prompt: str) -> str:
            captured.append(prompt)
            return '{"reasoning": "ok", "score": 0.0}'

        score_hallucination(
            answer="Streaming is great.",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        prompt = captured[0]
        assert "max_tokens is a required parameter." in prompt
        assert "Use streaming for real-time display." in prompt
        assert "Streaming is great." in prompt

    def test_idk_answer_scores_zero(self):
        """An answer admitting insufficient info should not be penalized."""
        def mock_llm(prompt: str) -> str:
            return (
                '{"reasoning": "The answer admits insufficient information.", '
                '"score": 0.0}'
            )

        result = score_hallucination(
            answer="I don't have enough information to answer this.",
            chunks=SAMPLE_CHUNKS,
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.0
