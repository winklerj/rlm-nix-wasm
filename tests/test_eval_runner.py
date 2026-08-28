"""Tests for OOLONG benchmark eval runner (checkpointing and serialization)."""

import json
from pathlib import Path

from rlm.eval.runner import EvalResult, EvalRunner
from rlm.types import RLMConfig


def _make_result(**kwargs: object) -> EvalResult:
    """Create an EvalResult with sensible defaults."""
    defaults: dict[str, object] = {
        "id": 0,
        "context_window_id": 1,
        "question": "test question",
        "predicted": "pred",
        "gold": "gold",
        "score": 0.5,
        "answer_type": "label",
        "task_group": "counting",
        "cost_usd": 0.01,
        "input_tokens": 100,
        "output_tokens": 50,
        "elapsed_s": 1.5,
        "error": None,
    }
    defaults.update(kwargs)
    return EvalResult(**defaults)  # type: ignore[arg-type]


class TestEvalResultSerialization:
    def test_round_trip(self):
        result = _make_result(id=42, score=0.75, predicted="hello", gold="hello")
        json_str = result.model_dump_json()
        restored = EvalResult.model_validate_json(json_str)
        assert restored.id == 42
        assert restored.score == 0.75
        assert restored.predicted == "hello"
        assert restored.gold == "hello"

    def test_round_trip_with_error(self):
        result = _make_result(id=1, error="timeout", score=0.0)
        json_str = result.model_dump_json()
        restored = EvalResult.model_validate_json(json_str)
        assert restored.error == "timeout"
        assert restored.score == 0.0

    def test_json_contains_all_fields(self):
        result = _make_result()
        data = json.loads(result.model_dump_json())
        expected_fields = {
            "id", "context_window_id", "question", "predicted", "gold",
            "score", "answer_type", "task_group", "cost_usd",
            "input_tokens", "output_tokens", "elapsed_s", "error",
        }
        assert set(data.keys()) == expected_fields


class TestCheckpointLoading:
    def test_load_empty_file(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        output.write_text("")
        runner = EvalRunner(RLMConfig(), output)
        assert runner.load_completed() == set()

    def test_load_nonexistent_file(self, tmp_path: Path):
        output = tmp_path / "does_not_exist.jsonl"
        runner = EvalRunner(RLMConfig(), output)
        assert runner.load_completed() == set()

    def test_load_completed_ids(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        r1 = _make_result(id=10)
        r2 = _make_result(id=20)
        r3 = _make_result(id=30)
        output.write_text(
            r1.model_dump_json() + "\n"
            + r2.model_dump_json() + "\n"
            + r3.model_dump_json() + "\n"
        )
        runner = EvalRunner(RLMConfig(), output)
        completed = runner.load_completed()
        assert completed == {10, 20, 30}

    def test_load_ignores_blank_lines(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        r1 = _make_result(id=5)
        output.write_text(r1.model_dump_json() + "\n\n\n")
        runner = EvalRunner(RLMConfig(), output)
        assert runner.load_completed() == {5}

    def test_load_ignores_malformed_lines(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        r1 = _make_result(id=7)
        output.write_text(
            r1.model_dump_json() + "\n"
            + "not valid json\n"
            + '{"missing_id_field": true}\n'
        )
        runner = EvalRunner(RLMConfig(), output)
        assert runner.load_completed() == {7}


class TestLoadAllResults:
    def test_load_all(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        r1 = _make_result(id=1, score=0.5)
        r2 = _make_result(id=2, score=1.0)
        output.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")
        runner = EvalRunner(RLMConfig(), output)
        results = runner._load_all_results()
        assert len(results) == 2
        assert results[0].id == 1
        assert results[1].id == 2

    def test_load_all_empty(self, tmp_path: Path):
        output = tmp_path / "results.jsonl"
        output.write_text("")
        runner = EvalRunner(RLMConfig(), output)
        assert runner._load_all_results() == []


class TestTraceOutput:
    def _task(self):
        from rlm.eval.datasets import EvalTask
        return EvalTask(id=7, context_window_id=1, context_text="a\nb", question="q?",
                        answer="Answer: 1", answer_type="NUMERIC", task_group="counting",
                        task="t", dataset="trec_coarse", context_len=1024)

    def test_trace_written_per_task_when_enabled(self, tmp_path: Path):
        from unittest.mock import patch

        from rlm.trace import ExecutionTrace
        runner = EvalRunner(RLMConfig(), tmp_path / "out.jsonl", trace_dir=tmp_path / "traces")
        with patch("rlm.orchestrator.RLMOrchestrator.run", return_value="Answer: 1"):
            result = runner.run_task(self._task())
        assert result.error is None
        trace_file = tmp_path / "traces" / "7.json"
        assert trace_file.exists()
        trace = ExecutionTrace.model_validate_json(trace_file.read_text())
        assert trace.root.depth == 0

    def test_trace_written_even_when_task_errors(self, tmp_path: Path):
        from unittest.mock import patch
        runner = EvalRunner(RLMConfig(), tmp_path / "out.jsonl", trace_dir=tmp_path / "traces")
        with patch("rlm.orchestrator.RLMOrchestrator.run", side_effect=RuntimeError("boom")):
            result = runner.run_task(self._task())
        assert result.error == "boom"
        assert (tmp_path / "traces" / "7.json").exists()

    def test_no_trace_dir_writes_nothing(self, tmp_path: Path):
        from unittest.mock import patch
        runner = EvalRunner(RLMConfig(), tmp_path / "out.jsonl")
        with patch("rlm.orchestrator.RLMOrchestrator.run", return_value="Answer: 1"):
            runner.run_task(self._task())
        assert not list(tmp_path.glob("**/*.json"))
