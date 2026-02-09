"""Lightweight evaluator for explore-mode operations."""

from __future__ import annotations

import hashlib

from rlm.cache.store import CacheStore, make_cache_key
from rlm.ops.text import op_chunk, op_count, op_grep, op_slice, op_split
from rlm.ops.recursive import op_combine
from rlm.timing import TimingProfile
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

    def __init__(self, cache: CacheStore | None = None, profile: TimingProfile | None = None):
        self.cache = cache
        self.profile = profile or TimingProfile()

    def execute(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        """Execute a single operation with current variable bindings."""
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
