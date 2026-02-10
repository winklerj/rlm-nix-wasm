"""Tests for the eval operation dispatch through the evaluator."""

import os
import tempfile
from pathlib import Path

import pytest

from rlm.cache.store import CacheStore
from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.types import OpType, Operation


needs_wasm = pytest.mark.skipif(
    not os.environ.get("RLM_WASM_PYTHON_PATH"),
    reason="RLM_WASM_PYTHON_PATH not set",
)


class TestEvalWithoutSandbox:
    def test_raises_clear_error(self):
        evaluator = LightweightEvaluator()
        op = Operation(op=OpType.EVAL, args={"code": "result = '1'"})
        with pytest.raises(RuntimeError, match="Wasm sandbox"):
            evaluator.execute(op, {"context": "test"})


@needs_wasm
class TestEvalWithSandbox:
    def test_correct_result(self, wasm_sandbox):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir) / "cache")
            evaluator = LightweightEvaluator(cache=cache, wasm_sandbox=wasm_sandbox)
            op = Operation(
                op=OpType.EVAL,
                args={"code": "result = str(2 + 3)", "inputs": []},
                bind="answer",
            )
            result = evaluator.execute(op, {"context": "test"})
            assert result.value == "5"
            assert result.op == OpType.EVAL
            assert not result.cached

    def test_cached_on_second_call(self, wasm_sandbox):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir) / "cache")
            evaluator = LightweightEvaluator(cache=cache, wasm_sandbox=wasm_sandbox)
            op = Operation(
                op=OpType.EVAL,
                args={"code": "result = str(10 * 10)", "inputs": []},
            )
            bindings = {"context": "test"}
            result1 = evaluator.execute(op, bindings)
            assert not result1.cached
            result2 = evaluator.execute(op, bindings)
            assert result2.cached
            assert result2.value == result1.value

    def test_different_code_cache_miss(self, wasm_sandbox):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir) / "cache")
            evaluator = LightweightEvaluator(cache=cache, wasm_sandbox=wasm_sandbox)
            bindings = {"context": "test"}
            op1 = Operation(
                op=OpType.EVAL, args={"code": "result = '1'", "inputs": []}
            )
            op2 = Operation(
                op=OpType.EVAL, args={"code": "result = '2'", "inputs": []}
            )
            r1 = evaluator.execute(op1, bindings)
            r2 = evaluator.execute(op2, bindings)
            assert r1.value == "1"
            assert r2.value == "2"
            assert not r2.cached  # Different code = different cache key

    def test_different_inputs_cache_miss(self, wasm_sandbox):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir) / "cache")
            evaluator = LightweightEvaluator(cache=cache, wasm_sandbox=wasm_sandbox)
            op = Operation(
                op=OpType.EVAL,
                args={"code": "result = str(len(context))", "inputs": ["context"]},
            )
            r1 = evaluator.execute(op, {"context": "short"})
            r2 = evaluator.execute(op, {"context": "a longer string"})
            assert r1.value == "5"
            assert r2.value == "15"
            assert not r2.cached

    def test_binding_works(self, wasm_sandbox):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir) / "cache")
            evaluator = LightweightEvaluator(cache=cache, wasm_sandbox=wasm_sandbox)
            op = Operation(
                op=OpType.EVAL,
                args={"code": "result = str(len(context))", "inputs": ["context"]},
                bind="ctx_len",
            )
            result = evaluator.execute(op, {"context": "hello"})
            assert result.value == "5"
            assert result.op == OpType.EVAL
