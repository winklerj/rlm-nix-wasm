"""Compile DSL operations into Nix expressions."""

from __future__ import annotations

import hashlib
import json

from rlm.nix.templates import (
    CHUNK_TEMPLATE,
    COMBINE_CONCAT_TEMPLATE,
    COMBINE_SUM_TEMPLATE,
    COUNT_TEMPLATE,
    GREP_TEMPLATE,
    SLICE_TEMPLATE,
)
from rlm.types import OpType, Operation


class NixCompileError(Exception):
    """Error compiling an operation to Nix."""
    pass


def compile_operation(op: Operation, store_paths: dict[str, str]) -> str:
    """Compile a single DSL operation to a Nix expression.

    Args:
        op: The operation to compile.
        store_paths: Map of variable names to Nix store paths.

    Returns:
        A Nix expression string.
    """
    op_hash = _op_hash(op)

    if op.op == OpType.GREP:
        input_path = store_paths[op.args["input"]]
        pattern = _escape_nix_string(op.args["pattern"])
        return GREP_TEMPLATE.format(
            hash=op_hash[:12],
            input_path=input_path,
            pattern=pattern,
        )

    elif op.op == OpType.SLICE:
        input_path = store_paths[op.args["input"]]
        start = op.args.get("start", 0)
        end = op.args.get("end", 0)
        length = end - start if end > start else 0
        return SLICE_TEMPLATE.format(
            hash=op_hash[:12],
            input_path=input_path,
            start=start + 1,  # tail is 1-indexed
            length=length,
        )

    elif op.op == OpType.COUNT:
        input_path = store_paths[op.args["input"]]
        mode = op.args.get("mode", "lines")
        mode_flag = "l" if mode == "lines" else "c"
        return COUNT_TEMPLATE.format(
            hash=op_hash[:12],
            input_path=input_path,
            mode_flag=mode_flag,
        )

    elif op.op == OpType.CHUNK:
        input_path = store_paths[op.args["input"]]
        n = op.args["n"]
        return CHUNK_TEMPLATE.format(
            hash=op_hash[:12],
            input_path=input_path,
            n=n,
        )

    elif op.op == OpType.COMBINE:
        strategy = op.args.get("strategy", "concat")
        inputs_ref = op.args["inputs"]

        if isinstance(inputs_ref, list):
            paths = [store_paths[ref] for ref in inputs_ref]
        else:
            paths = [store_paths[inputs_ref]]

        input_paths = " ".join(paths)

        if strategy == "sum":
            return COMBINE_SUM_TEMPLATE.format(
                hash=op_hash[:12],
                input_paths=input_paths,
            )
        else:
            return COMBINE_CONCAT_TEMPLATE.format(
                hash=op_hash[:12],
                input_paths=input_paths,
            )

    elif op.op in (OpType.RLM_CALL, OpType.MAP):
        raise NixCompileError(
            f"Operation {op.op} requires orchestrator-level handling. "
            f"The orchestrator makes the API call and imports the result."
        )

    else:
        raise NixCompileError(f"Cannot compile operation {op.op} to Nix")


def _op_hash(op: Operation) -> str:
    """Generate a short hash for an operation."""
    data = json.dumps({"op": op.op.value, "args": op.args}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _escape_nix_string(s: str) -> str:
    """Escape a string for safe inclusion in a Nix expression."""
    return s.replace("\\", "\\\\").replace("'", "'\\''").replace("$", "\\$")
