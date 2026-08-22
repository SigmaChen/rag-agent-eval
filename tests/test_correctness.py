from rag_agent_eval.eval.correctness import score_correctness, _parse_judge_response


class TestParseJudgeResponse:
    """Test the JSON parsing logic without any LLM calls."""

    def test_valid_json(self):
        raw = '{"reasoning": "Answers match well.", "score": 0.85}'
        result = _parse_judge_response(raw)
        assert result["score"] == 0.85
        assert result["reasoning"] == "Answers match well."

    def test_score_clamped_above_1(self):
        raw = '{"reasoning": "Great.", "score": 1.5}'
        result = _parse_judge_response(raw)
        assert result["score"] == 1.0

    def test_score_clamped_below_0(self):
        raw = '{"reasoning": "Bad.", "score": -0.3}'
        result = _parse_judge_response(raw)
        assert result["score"] == 0.0

    def test_fallback_extracts_number(self):
        raw = "The score is 0.7 because the answer is mostly correct."
        result = _parse_judge_response(raw)
        assert result["score"] == 0.7

    def test_fallback_normalizes_1_to_10_scale(self):
        raw = "I'd give this a 7 out of 10."
        result = _parse_judge_response(raw)
        assert result["score"] == 0.7

    def test_unparseable_returns_zero(self):
        raw = "I cannot evaluate this."
        result = _parse_judge_response(raw)
        assert result["score"] == 0.0
        assert "(failed to parse)" in result["reasoning"]


class TestScoreCorrectness:
    """Test the full scorer with a mock LLM."""

    def test_high_score_when_answers_match(self):
        def mock_llm(prompt: str) -> str:
            return '{"reasoning": "Both answers say max_tokens is required.", "score": 0.95}'

        result = score_correctness(
            question="Is max_tokens required?",
            actual_answer="Yes, max_tokens must be set.",
            expected_answer="max_tokens is a required parameter.",
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.95

    def test_low_score_when_answers_conflict(self):
        def mock_llm(prompt: str) -> str:
            return '{"reasoning": "Actual answer says optional, expected says required.", "score": 0.1}'

        result = score_correctness(
            question="Is max_tokens required?",
            actual_answer="No, max_tokens is optional.",
            expected_answer="max_tokens is a required parameter.",
            generate_fn=mock_llm,
        )
        assert result["score"] == 0.1

    def test_prompt_contains_all_inputs(self):
        """Verify the prompt sent to the LLM includes question, expected, and actual."""
        captured_prompt = []

        def mock_llm(prompt: str) -> str:
            captured_prompt.append(prompt)
            return '{"reasoning": "ok", "score": 0.5}'

        score_correctness(
            question="What is streaming?",
            actual_answer="Streaming shows tokens in real time.",
            expected_answer="Streaming displays response incrementally.",
            generate_fn=mock_llm,
        )
        prompt = captured_prompt[0]
        assert "What is streaming?" in prompt
        assert "Streaming shows tokens in real time." in prompt
        assert "Streaming displays response incrementally." in prompt
