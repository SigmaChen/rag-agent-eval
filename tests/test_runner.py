from unittest.mock import patch, MagicMock

from rag_agent_eval.eval.runner import (
    load_ground_truth,
    _compute_summary,
    make_generate_fn,
)


class TestLoadGroundTruth:

    def test_loads_existing_file(self):
        pairs = load_ground_truth("data/ground_truth/qa_pairs.json")
        assert len(pairs) >= 1
        assert "question" in pairs[0]
        assert "expected_answer" in pairs[0]

    def test_missing_file_returns_empty(self):
        pairs = load_ground_truth("data/ground_truth/nonexistent.json")
        assert pairs == []


class TestComputeSummary:

    def test_averages_scores(self):
        results = [
            {"correctness": 0.8, "hallucination": 0.1, "relevance": 0.9},
            {"correctness": 0.6, "hallucination": 0.3, "relevance": 0.7},
        ]
        summary = _compute_summary(results)
        assert summary["total_questions"] == 2
        assert summary["avg_correctness"] == 0.7
        assert summary["avg_hallucination"] == 0.2
        assert summary["avg_relevance"] == 0.8

    def test_single_result(self):
        results = [
            {"correctness": 0.5, "hallucination": 0.5, "relevance": 0.5},
        ]
        summary = _compute_summary(results)
        assert summary["total_questions"] == 1
        assert summary["avg_correctness"] == 0.5

    def test_empty_results(self):
        summary = _compute_summary([])
        assert summary == {}
