"""Tests for LLM output parser."""

import json

import pytest

from rlm.llm.parser import ParseError, parse_llm_output
from rlm.types import CommitPlan, ExploreAction, FinalAnswer, OpType


class TestExploreAction:
    def test_basic(self):
        raw = json.dumps({
            "mode": "explore",
            "operation": {
                "op": "slice",
                "args": {"input": "context", "start": 0, "end": 100},
                "bind": "peek",
            },
        })
        action = parse_llm_output(raw)
        assert isinstance(action, ExploreAction)
        assert action.operation.op == OpType.SLICE
        assert action.operation.args["start"] == 0
        assert action.operation.bind == "peek"

    def test_no_bind(self):
        raw = json.dumps({
            "mode": "explore",
            "operation": {"op": "count", "args": {"input": "context"}},
        })
        action = parse_llm_output(raw)
        assert isinstance(action, ExploreAction)
        assert action.operation.bind is None

    def test_grep(self):
        raw = json.dumps({
            "mode": "explore",
            "operation": {
                "op": "grep",
                "args": {"input": "context", "pattern": "User: 123"},
                "bind": "matches",
            },
        })
        action = parse_llm_output(raw)
        assert isinstance(action, ExploreAction)
        assert action.operation.op == OpType.GREP


class TestCommitPlan:
    def test_basic(self):
        raw = json.dumps({
            "mode": "commit",
            "operations": [
                {"op": "grep", "args": {"input": "context", "pattern": "test"}, "bind": "filtered"},
                {"op": "count", "args": {"input": "filtered"}, "bind": "total"},
            ],
            "output": "total",
        })
        action = parse_llm_output(raw)
        assert isinstance(action, CommitPlan)
        assert len(action.operations) == 2
        assert action.output == "total"
        assert action.operations[0].op == OpType.GREP
        assert action.operations[1].op == OpType.COUNT

    def test_with_recursive_ops(self):
        raw = json.dumps({
            "mode": "commit",
            "operations": [
                {"op": "chunk", "args": {"input": "context", "n": 4}, "bind": "chunks"},
                {"op": "map", "args": {"prompt": "Summarize", "input": "chunks"}, "bind": "results"},
                {"op": "combine", "args": {"inputs": "results", "strategy": "concat"}, "bind": "final"},
            ],
            "output": "final",
        })
        action = parse_llm_output(raw)
        assert isinstance(action, CommitPlan)
        assert len(action.operations) == 3
        assert action.operations[1].op == OpType.MAP


class TestFinalAnswer:
    def test_basic(self):
        raw = json.dumps({"mode": "final", "answer": "42"})
        action = parse_llm_output(raw)
        assert isinstance(action, FinalAnswer)
        assert action.answer == "42"

    def test_long_answer(self):
        answer = "This is a long answer with multiple sentences. It explains the result."
        raw = json.dumps({"mode": "final", "answer": answer})
        action = parse_llm_output(raw)
        assert isinstance(action, FinalAnswer)
        assert action.answer == answer


class TestCodeFences:
    def test_json_fence(self):
        raw = '```json\n{"mode": "final", "answer": "hello"}\n```'
        action = parse_llm_output(raw)
        assert isinstance(action, FinalAnswer)
        assert action.answer == "hello"

    def test_plain_fence(self):
        raw = '```\n{"mode": "final", "answer": "world"}\n```'
        action = parse_llm_output(raw)
        assert isinstance(action, FinalAnswer)
        assert action.answer == "world"

    def test_fence_with_surrounding_text(self):
        raw = 'Here is my response:\n```json\n{"mode": "final", "answer": "42"}\n```\n'
        action = parse_llm_output(raw)
        assert isinstance(action, FinalAnswer)
        assert action.answer == "42"


class TestErrorHandling:
    def test_invalid_json(self):
        with pytest.raises(ParseError, match="Invalid JSON"):
            parse_llm_output("not json at all")

    def test_unknown_mode(self):
        with pytest.raises(ParseError, match="Unknown mode"):
            parse_llm_output(json.dumps({"mode": "invalid"}))

    def test_missing_mode(self):
        with pytest.raises(ParseError, match="Unknown mode"):
            parse_llm_output(json.dumps({"answer": "42"}))

    def test_missing_operation_field(self):
        with pytest.raises(KeyError):
            parse_llm_output(json.dumps({"mode": "explore"}))

    def test_invalid_op_type(self):
        raw = json.dumps({
            "mode": "explore",
            "operation": {"op": "nonexistent", "args": {}},
        })
        with pytest.raises(ValueError):
            parse_llm_output(raw)

    def test_missing_output_in_commit(self):
        raw = json.dumps({
            "mode": "commit",
            "operations": [{"op": "count", "args": {"input": "context"}}],
        })
        with pytest.raises(KeyError):
            parse_llm_output(raw)
