"""Tests for the lightweight evaluator."""

import json

import pytest

from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.types import OpType, Operation


@pytest.fixture
def evaluator():
    return LightweightEvaluator()


@pytest.fixture
def bindings():
    return {"context": "line one\nline two\nline three\nline four\nline five"}


class TestLightweightEvaluator:
    def test_slice(self, evaluator, bindings):
        op = Operation(op=OpType.SLICE, args={"input": "context", "start": 0, "end": 8})
        result = evaluator.execute(op, bindings)
        assert result.value == "line one"
        assert result.op == OpType.SLICE

    def test_grep(self, evaluator, bindings):
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "two"})
        result = evaluator.execute(op, bindings)
        assert result.value == "line two"

    def test_count(self, evaluator, bindings):
        op = Operation(op=OpType.COUNT, args={"input": "context", "mode": "lines"})
        result = evaluator.execute(op, bindings)
        assert result.value == "5"

    def test_split(self, evaluator, bindings):
        op = Operation(op=OpType.SPLIT, args={"input": "context", "delimiter": "\n"})
        result = evaluator.execute(op, bindings)
        parts = json.loads(result.value)
        assert len(parts) == 5

    def test_chunk(self, evaluator, bindings):
        op = Operation(op=OpType.CHUNK, args={"input": "context", "n": 2})
        result = evaluator.execute(op, bindings)
        chunks = json.loads(result.value)
        assert len(chunks) == 2

    def test_combine(self, evaluator):
        bindings = {"a": "hello", "b": "world"}
        op = Operation(op=OpType.COMBINE, args={"inputs": ["a", "b"], "strategy": "concat"})
        result = evaluator.execute(op, bindings)
        assert result.value == "hello\nworld"

    def test_binding_storage(self, evaluator, bindings):
        op = Operation(
            op=OpType.GREP,
            args={"input": "context", "pattern": "two"},
            bind="filtered",
        )
        result = evaluator.execute(op, bindings)
        bindings["filtered"] = result.value

        op2 = Operation(op=OpType.COUNT, args={"input": "filtered"})
        result2 = evaluator.execute(op2, bindings)
        assert result2.value == "1"

    def test_unsupported_op(self, evaluator, bindings):
        op = Operation(op=OpType.RLM_CALL, args={"query": "test", "context": "context"})
        with pytest.raises(ValueError, match="not available in explore mode"):
            evaluator.execute(op, bindings)

    def test_map_not_available(self, evaluator, bindings):
        op = Operation(op=OpType.MAP, args={"prompt": "test", "input": "context"})
        with pytest.raises(ValueError, match="not available in explore mode"):
            evaluator.execute(op, bindings)

    def test_cache_key_generated(self, evaluator, bindings):
        op = Operation(op=OpType.SLICE, args={"input": "context", "start": 0, "end": 5})
        result = evaluator.execute(op, bindings)
        assert len(result.cache_key) == 64  # sha256 hex
