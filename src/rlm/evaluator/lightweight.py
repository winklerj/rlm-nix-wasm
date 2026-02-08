"""Lightweight evaluator for explore-mode operations."""

from __future__ import annotations

import hashlib

from rlm.cache.store import CacheStore, make_cache_key
from rlm.ops.text import op_chunk, op_count, op_grep, op_slice, op_split
from rlm.ops.recursive import op_combine
from rlm.types import OpResult, OpType, Operation

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

    def __init__(self, cache: CacheStore | None = None):
        self.cache = cache

    def execute(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        """Execute a single operation with current variable bindings."""
        executor = EXPLORE_OPS.get(op.op)
        if executor is None:
            raise ValueError(
                f"Operation {op.op} is not available in explore mode. "
                f"Use COMMIT for {op.op}."
            )

        # Compute content-addressed hashes for bindings used by this op
        input_hashes = {
            k: hashlib.sha256(v.encode()).hexdigest()
            for k, v in bindings.items()
        }
        cache_key = make_cache_key(op.op, op.args, input_hashes)

        # Check cache
        if self.cache is not None:
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                return OpResult(
                    op=op.op,
                    cache_key=cache_key,
                    value=cached_value,
                    cached=True,
                )

        result_value = executor(op.args, bindings)

        # Store in cache
        if self.cache is not None:
            self.cache.put(cache_key, result_value)

        return OpResult(
            op=op.op,
            cache_key=cache_key,
            value=result_value,
        )
