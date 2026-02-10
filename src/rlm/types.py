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
    max_parallel_jobs: int = 4
    temperature: float = 1.0
    cache_dir: Path = Path.home() / ".cache" / "rlm-secure"
    use_nix: bool = False
    verbose: bool = False
    # Wasm sandbox settings for eval operations
    wasm_python_path: Path | None = None  # Path to python.wasm binary
    wasm_fuel: int = 10_000_000_000  # CPU fuel limit (CPython WASI needs ~2B for startup)
    wasm_memory_mb: int = 256  # Memory limit in MB
