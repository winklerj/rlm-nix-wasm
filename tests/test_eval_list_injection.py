"""List-valued bindings are injected into the eval sandbox as Python lists."""

from __future__ import annotations

import json

from rlm.evaluator.lightweight import LightweightEvaluator, _decode_list
from rlm.types import Operation, OpType


class _StubSandbox:
    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def run(self, code: str, variables: dict[str, object]) -> str:
        self.seen = dict(variables)
        return "ok\n"


def test_decode_list_parses_json_arrays_only() -> None:
    assert _decode_list(json.dumps(["a", "b"])) == ["a", "b"]
    assert _decode_list("plain text") == "plain text"
    assert _decode_list("[not json") == "[not json"
    assert _decode_list("{\"k\": 1}") == "{\"k\": 1}"


def test_eval_receives_lists_for_json_array_bindings() -> None:
    sandbox = _StubSandbox()
    evaluator = LightweightEvaluator(cache=None, wasm_sandbox=sandbox)  # type: ignore[arg-type]
    bindings = {
        "context": "line1\nline2",
        "chunks": json.dumps(["line1", "line2"]),
        "counts": json.dumps(["3", "4"]),
    }
    op = Operation(op=OpType.EVAL, args={"code": "result = 1", "inputs": ["chunks", "counts", "context"]})
    result = evaluator.execute(op, bindings)
    assert result.value == "ok"
    assert sandbox.seen["chunks"] == ["line1", "line2"]
    assert sandbox.seen["counts"] == ["3", "4"]
    assert sandbox.seen["context"] == "line1\nline2"
