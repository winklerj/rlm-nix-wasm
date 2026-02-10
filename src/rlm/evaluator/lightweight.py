"""Lightweight evaluator for explore-mode operations."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rlm.cache.store import CacheStore, make_cache_key
from rlm.ops.text import op_chunk, op_count, op_grep, op_slice, op_split
from rlm.ops.recursive import op_combine
from rlm.timing import TimingProfile
from rlm.types import OpResult, OpType, Operation

if TYPE_CHECKING:
    from rlm.evaluator.wasm_sandbox import WasmSandbox

# Registry of explore-safe operations
EXPLORE_OPS = {
    OpType.SLICE: op_slice,
    OpType.GREP: op_grep,
    OpType.COUNT: op_count,
    OpType.SPLIT: op_split,
    OpType.CHUNK: op_chunk,
    OpType.COMBINE: op_combine,
}


class LightweightEvaluator:
    """Executes explore-mode operations in-process."""

    def __init__(
        self,
        cache: CacheStore | None = None,
        profile: TimingProfile | None = None,
        wasm_sandbox: WasmSandbox | None = None,
    ):
        self.cache = cache
        self.profile = profile or TimingProfile()
        self.wasm_sandbox = wasm_sandbox

    def execute(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        """Execute a single operation with current variable bindings."""
        # Eval gets special dispatch — it runs in the Wasm sandbox
        if op.op == OpType.EVAL:
            return self._execute_eval(op, bindings)

        executor = EXPLORE_OPS.get(op.op)
        if executor is None:
            raise ValueError(
                f"Operation {op.op} is not available in explore mode. "
                f"Use COMMIT for {op.op}."
            )

        # Compute content-addressed hashes for bindings used by this op
        with self.profile.measure("hash", "binding_hash"):
            input_hashes = {
                k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in bindings.items()
            }
        cache_key = make_cache_key(op.op, op.args, input_hashes)

        # Check cache
        if self.cache is not None:
            with self.profile.measure("cache", "lookup"):
                cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                self.profile.record_cache_hit()
                return OpResult(
                    op=op.op,
                    cache_key=cache_key,
                    value=cached_value,
                    cached=True,
                )
            self.profile.record_cache_miss()

        with self.profile.measure("evaluator", op.op.value):
            result_value = executor(op.args, bindings)

        # Store in cache
        if self.cache is not None:
            with self.profile.measure("cache", "store"):
                self.cache.put(cache_key, result_value)

        return OpResult(
            op=op.op,
            cache_key=cache_key,
            value=result_value,
        )

    def _execute_eval(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        """Execute an eval operation via the Wasm sandbox."""
        if self.wasm_sandbox is None:
            raise RuntimeError(
                "Eval operations require a Wasm sandbox. "
                "Set RLM_WASM_PYTHON_PATH to the path of a python.wasm binary, "
                "or pass --wasm-python to the CLI."
            )

        code: str = op.args["code"]
        input_names: list[str] | None = op.args.get("inputs")

        # Resolve variables: only those listed in inputs, or all bindings
        if input_names is not None:
            variables = {name: bindings[name] for name in input_names if name in bindings}
        else:
            variables = dict(bindings)

        # Compute cache key — include variable content hashes so different
        # inputs produce different keys even with the same code
        with self.profile.measure("hash", "binding_hash"):
            input_hashes = {
                k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in variables.items()
            }
        # Build augmented args that include resolved variable hashes
        eval_args = dict(op.args)
        eval_args["_input_hashes"] = input_hashes
        cache_key = make_cache_key(op.op, eval_args, {})

        # Check cache
        if self.cache is not None:
            with self.profile.measure("cache", "lookup"):
                cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                self.profile.record_cache_hit()
                return OpResult(
                    op=op.op,
                    cache_key=cache_key,
                    value=cached_value,
                    cached=True,
                )
            self.profile.record_cache_miss()

        # Run in sandbox
        with self.profile.measure("evaluator", "eval"):
            result_value = self.wasm_sandbox.run(code, variables)
            # Strip trailing newline from stdout capture
            result_value = result_value.rstrip("\n")

        # Store in cache
        if self.cache is not None:
            with self.profile.measure("cache", "store"):
                self.cache.put(cache_key, result_value)

        return OpResult(
            op=op.op,
            cache_key=cache_key,
            value=result_value,
        )
