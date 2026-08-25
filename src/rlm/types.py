"""Core types for the RLM system."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Mode(str, Enum):
    """The current mode of the LLM in the explore/commit protocol."""
    EXPLORE = "explore"
    COMMIT = "commit"
    FINAL = "final"


class OpType(str, Enum):
    """Available DSL operations."""
    SLICE = "slice"
    GREP = "grep"
    COUNT = "count"
    CHUNK = "chunk"
    SPLIT = "split"
    RLM_CALL = "rlm_call"
    MAP = "map"
    COMBINE = "combine"
    EVAL = "eval"


class Operation(BaseModel):
    """A single DSL operation emitted by the LLM."""
    op: OpType
    args: dict  # type: ignore[type-arg]
    bind: str | None = None

    def cache_key(self, input_hashes: dict[str, str]) -> str:
        """Compute content-addressed cache key from op + resolved input hashes."""
        key_data = {
            "op": self.op.value,
            "args": {k: input_hashes.get(v, v) if isinstance(v, str) else v
                     for k, v in self.args.items()},
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class ExploreAction(BaseModel):
    """An explore-mode action: a single operation to execute immediately."""
    mode: Literal["explore"] = "explore"
    operation: Operation


class CommitPlan(BaseModel):
    """A commit-mode plan: a DAG of operations to execute."""
    mode: Literal["commit"] = "commit"
    operations: list[Operation]
    output: str


class FinalAnswer(BaseModel):
    """The LLM's final answer."""
    mode: Literal["final"] = "final"
    answer: str


# Union type for LLM actions
LLMAction = ExploreAction | CommitPlan | FinalAnswer


class OpResult(BaseModel):
    """Result of executing an operation."""
    op: OpType
    cache_key: str
    value: str
    cached: bool = False


class Context(BaseModel):
    """A context object stored in the cache."""
    content: str
    hash: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()


class RLMConfig(BaseModel):
    """Configuration for an RLM run."""
    model: str = "claude-opus-4-5"
    child_model: str | None = None
    max_explore_steps: int = 20
    max_commit_cycles: int = 5
    max_recursion_depth: int = 1
    # Child contexts smaller than this (in chars) are answered with a single
    # direct LLM call instead of a full explore/commit loop, regardless of
    # depth. Recursion only pays off when the context is too big to just read.
    min_recursive_chars: int = 4000
    # `map` pieces are answered with a single direct LLM call regardless of
    # size. A map is "apply this prompt to each piece"; a piece large enough
    # to need its own explore/commit loop is a chunking mistake, and letting
    # every piece recurse turns one map into hundreds of child loops.
    map_direct: bool = True
    # Passed to the server as chat_template_kwargs["reasoning_strength"]
    # (e.g. "low"/"high") for chat templates that support it; None = omit.
    reasoning_strength: str | None = None
    # Hard cap on generated tokens per LLM call (max_tokens). Without it a
    # model that never emits EOS runs to the server's context limit, and a
    # client timeout does not stop server-side generation.
    max_output_tokens: int | None = None
    max_parallel_jobs: int = 4
    temperature: float = 1.0
    max_result_chars: int = 8000
    cache_dir: Path = Path.home() / ".cache" / "rlm-nix-wasm"
    use_nix: bool = True
    verbose: bool = False
    benchmark_eval_prompt: bool = False  # Use benchmark-friendly eval prompt
    # Wasm sandbox settings for eval operations
    wasm_python_path: Path | None = None  # Path to python.wasm binary
    wasm_fuel: int = 10_000_000_000  # CPU fuel limit (CPython WASI needs ~2B for startup)
    wasm_memory_mb: int = 256  # Memory limit in MB
