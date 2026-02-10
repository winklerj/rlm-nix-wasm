"""Eval operation — runs Python code in a Wasm sandbox.

Unlike other ops in this directory, eval is not dispatched through
EXPLORE_OPS. Instead, the LightweightEvaluator handles it directly
via _execute_eval(), which delegates to a WasmSandbox instance.

This file exists for consistency with the ops/ directory structure.
"""

from __future__ import annotations


def op_eval(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Not called directly — eval is dispatched via LightweightEvaluator._execute_eval()."""
    raise RuntimeError(
        "op_eval should not be called directly. "
        "Eval operations are handled by the LightweightEvaluator."
    )
