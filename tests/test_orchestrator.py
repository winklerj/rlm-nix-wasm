"""Tests for the RLM orchestrator with mocked LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rlm.orchestrator import RLMOrchestrator
from rlm.types import RLMConfig


@pytest.fixture
def config():
    return RLMConfig(
        model="test-model",
        max_explore_steps=5,
        max_commit_cycles=3,
        max_recursion_depth=1,
        verbose=False,
    )


def _make_orchestrator_with_responses(config, responses):
    """Create an orchestrator with a mocked LLM that returns canned responses."""
    orchestrator = RLMOrchestrator(config)
    response_iter = iter(responses)
    orchestrator.llm.send = MagicMock(side_effect=lambda _: next(response_iter))
    orchestrator.llm.set_system_prompt = MagicMock()
    return orchestrator


class TestFinalAnswer:
    def test_immediate_final(self, config):
        responses = [
            json.dumps({"mode": "final", "answer": "42"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("What is the answer?", "some context")
        assert result == "42"

    def test_final_after_explore(self, config):
        responses = [
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "slice",
                    "args": {"input": "context", "start": 0, "end": 10},
                    "bind": "peek",
                },
            }),
            json.dumps({"mode": "final", "answer": "The context starts with 'some conte'"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("What does the context start with?", "some context text")
        assert "some conte" in result


class TestExploreMode:
    def test_explore_binds_variable(self, config):
        responses = [
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "grep",
                    "args": {"input": "context", "pattern": "hello"},
                    "bind": "matches",
                },
            }),
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "count",
                    "args": {"input": "matches"},
                    "bind": "n",
                },
            }),
            json.dumps({"mode": "final", "answer": "1"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines contain hello?", "hello world\ngoodbye world")
        assert result == "1"

    def test_max_explore_steps(self, config):
        # Generate more explore steps than allowed (config.max_explore_steps = 5)
        explore = json.dumps({
            "mode": "explore",
            "operation": {
                "op": "count",
                "args": {"input": "context"},
            },
        })
        responses = [explore] * 6 + [
            json.dumps({"mode": "final", "answer": "forced"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "line1\nline2")
        assert result == "forced"
        # Should have been told about the limit
        calls = [str(c) for c in orch.llm.send.call_args_list]
        assert any("maximum" in c for c in calls)


class TestCommitMode:
    def test_commit_plan_execution(self, config):
        responses = [
            json.dumps({
                "mode": "commit",
                "operations": [
                    {"op": "grep", "args": {"input": "context", "pattern": "a"}, "bind": "filtered"},
                    {"op": "count", "args": {"input": "filtered"}, "bind": "total"},
                ],
                "output": "total",
            }),
            json.dumps({"mode": "final", "answer": "2"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines contain 'a'?", "apple\nbanana\ncherry")
        assert result == "2"

    def test_max_commit_cycles(self, config):
        commit = json.dumps({
            "mode": "commit",
            "operations": [
                {"op": "count", "args": {"input": "context"}, "bind": "n"},
            ],
            "output": "n",
        })
        responses = [commit] * 4 + [
            json.dumps({"mode": "final", "answer": "forced"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "data")
        assert result == "forced"
        calls = [str(c) for c in orch.llm.send.call_args_list]
        assert any("maximum" in c for c in calls)


class TestRecursion:
    def test_depth_limiting(self, config):
        """At max depth, should use direct call instead of explore/commit."""
        config.max_recursion_depth = 0
        orch = RLMOrchestrator(config)

        # Mock the direct LLM call
        with patch.object(orch, '_direct_call', return_value="direct answer") as mock_direct:
            result = orch.run("test", "context", depth=1)
            assert result == "direct answer"
            mock_direct.assert_called_once()


class TestParseErrorRecovery:
    def test_recovers_from_bad_json(self, config):
        responses = [
            "This is not JSON at all",
            json.dumps({"mode": "final", "answer": "recovered"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "context")
        assert result == "recovered"

    def test_recovers_from_invalid_mode(self, config):
        responses = [
            json.dumps({"mode": "unknown_mode"}),
            json.dumps({"mode": "final", "answer": "recovered"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "context")
        assert result == "recovered"


class TestExploreCommitFlow:
    def test_explore_then_commit_then_final(self, config):
        """Full flow: explore, commit, final."""
        responses = [
            # Explore: peek at context
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "slice",
                    "args": {"input": "context", "start": 0, "end": 20},
                    "bind": "peek",
                },
            }),
            # Commit: grep and count
            json.dumps({
                "mode": "commit",
                "operations": [
                    {"op": "grep", "args": {"input": "context", "pattern": "line"}, "bind": "lines"},
                    {"op": "count", "args": {"input": "lines"}, "bind": "total"},
                ],
                "output": "total",
            }),
            # Final
            json.dumps({"mode": "final", "answer": "3"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines?", "line 1\nline 2\nline 3")
        assert result == "3"
