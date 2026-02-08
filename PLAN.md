# RLM-Secure Implementation Plan

## Overview

Implement the sandboxed recursive language model system described in RESEARCH.md as a Python CLI tool. The system enables LLMs to break large-context problems into smaller recursive sub-calls, with each call sandboxed via Nix derivations for security, content-addressed for caching, and schedulable in parallel.

The implementation uses a phased approach: first get the core RLM loop working with Python-native execution, then layer on Nix integration for sandboxing, caching, and parallelism.

## Current State Analysis

- **Repository**: Greenfield. Only RESEARCH.md (design document) and .claude/ configuration exist.
- **No code, no dependencies, no build system.**
- **Reference implementation**: The official `alexzhang13/rlm` package (`pip install rlms`) exists but runs unsandboxed Python subprocesses with no caching or parallelism — exactly the limitations RESEARCH.md addresses.

### Key Discoveries:
- The official RLM package uses a REPL-based approach where the LLM writes arbitrary Python. Our system replaces this with a structured DSL + explore/commit protocol.
- `litellm` provides a unified LLM client API across OpenAI, Anthropic, and 100+ providers.
- `instructor` + Pydantic enables type-safe structured output parsing from LLMs.
- Direct `bwrap` subprocess calls are more reliable than the `sandboxlib` Python wrapper.
- Content-addressed caching can be implemented with `hashlib` + filesystem, matching Nix store semantics.

## Desired End State

A working `rlm` CLI that:

1. Accepts a query and a context file (or stdin)
2. Runs the explore/commit protocol with a configurable LLM backend
3. Executes DSL operations in sandboxed environments
4. Caches all deterministic operations content-addressably
5. Parallelizes independent branches in commit plans
6. Optionally compiles commit plans to Nix derivations for stronger guarantees
7. Returns the final answer to stdout

### Verification:
```bash
# Basic usage
rlm run --query "How many entity questions from User 59219?" --context data.txt

# With model selection
rlm run --query "Summarize this document" --context doc.txt --model claude-sonnet-4-20250514

# Cache statistics
rlm cache stats

# Clear cache
rlm cache clear
```

## What We're NOT Doing

- **Custom VM environments** (eval with Wasm/QuickJS) — deferred to future work
- **Distributed Nix caches** (Cachix integration) — single-machine only for now
- **Multi-model coordination** (different models for different sub-tasks) — single model per run
- **Fine-tuning or RL training** for the explore/commit protocol
- **Web UI or API server** — CLI only
- **Multimodal contexts** (images, audio) — text only

## Implementation Approach

Build bottom-up in 7 phases, each independently testable:

1. Project scaffolding and core types
2. DSL operations and lightweight evaluator
3. LLM integration and output parsing
4. Orchestrator and explore/commit loop
5. Content-addressed caching
6. Nix integration (sandboxing + parallel execution)
7. CLI polish and documentation

---

## Phase 1: Project Scaffolding & Core Types

### Overview
Set up the Python project structure, dependencies, and core data types that everything else builds on.

### Changes Required:

#### 1. Project configuration
**File**: `pyproject.toml`

```toml
[project]
name = "rlm-secure"
version = "0.1.0"
description = "Sandboxed Recursive Language Models with Nix"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "litellm>=1.50",
    "instructor>=1.7",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
]

[project.scripts]
rlm = "rlm.cli:main"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

#### 2. Directory structure
Create the following directory tree:

```
src/rlm/
├── __init__.py
├── cli.py
├── config.py
├── types.py
├── orchestrator.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   ├── parser.py
│   └── prompts.py
├── ops/
│   ├── __init__.py
│   ├── base.py
│   ├── text.py
│   ├── recursive.py
│   └── eval_op.py
├── evaluator/
│   ├── __init__.py
│   ├── lightweight.py
│   └── sandbox.py
├── cache/
│   ├── __init__.py
│   └── store.py
└── nix/
    ├── __init__.py
    ├── compiler.py
    ├── store.py
    └── builder.py
tests/
├── __init__.py
├── conftest.py
├── test_ops.py
├── test_evaluator.py
├── test_parser.py
├── test_orchestrator.py
├── test_cache.py
└── test_cli.py
```

#### 3. Core types
**File**: `src/rlm/types.py`

Define the core data model using Pydantic:

```python
"""Core types for the RLM system."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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
    args: dict  # operation-specific arguments
    bind: str | None = None  # variable name to bind result to

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
    output: str  # variable name of the final result


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
    value: str  # all results serialized as strings
    cached: bool = False


class Context(BaseModel):
    """A context object stored in the cache."""
    content: str
    hash: str = ""

    def model_post_init(self, __context) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()


class RLMConfig(BaseModel):
    """Configuration for an RLM run."""
    model: str = "gpt-4o-mini"
    max_explore_steps: int = 20
    max_commit_cycles: int = 5
    max_recursion_depth: int = 1
    max_parallel_jobs: int = 4
    temperature: float = 0.0
    cache_dir: Path = Path.home() / ".cache" / "rlm-secure"
    use_nix: bool = False
    verbose: bool = False
```

#### 4. Configuration loader
**File**: `src/rlm/config.py`

```python
"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

from rlm.types import RLMConfig


def load_config(**overrides) -> RLMConfig:
    """Load configuration from environment variables and overrides."""
    env_mappings = {
        "RLM_MODEL": "model",
        "RLM_MAX_EXPLORE_STEPS": ("max_explore_steps", int),
        "RLM_MAX_COMMIT_CYCLES": ("max_commit_cycles", int),
        "RLM_MAX_RECURSION_DEPTH": ("max_recursion_depth", int),
        "RLM_MAX_PARALLEL_JOBS": ("max_parallel_jobs", int),
        "RLM_CACHE_DIR": ("cache_dir", Path),
        "RLM_USE_NIX": ("use_nix", lambda x: x.lower() in ("1", "true", "yes")),
        "RLM_VERBOSE": ("verbose", lambda x: x.lower() in ("1", "true", "yes")),
    }

    config_data = {}
    for env_var, mapping in env_mappings.items():
        val = os.environ.get(env_var)
        if val is not None:
            if isinstance(mapping, str):
                config_data[mapping] = val
            else:
                field_name, converter = mapping
                config_data[field_name] = converter(val)

    config_data.update({k: v for k, v in overrides.items() if v is not None})
    return RLMConfig(**config_data)
```

#### 5. CLI skeleton
**File**: `src/rlm/cli.py`

```python
"""CLI entry point for rlm-secure."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from rlm.config import load_config

console = Console(stderr=True)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Sandboxed Recursive Language Models with Nix."""
    pass


@main.command()
@click.option("--query", "-q", required=True, help="The query to answer.")
@click.option("--context", "-c", type=click.Path(exists=True), help="Path to context file.")
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--use-nix", is_flag=True, default=False, help="Use Nix for sandboxing.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def run(query, context, model, max_explore, max_depth, use_nix, verbose):
    """Run an RLM query against a context."""
    config = load_config(
        model=model,
        max_explore_steps=max_explore,
        max_recursion_depth=max_depth,
        use_nix=use_nix,
        verbose=verbose,
    )

    # Read context from file or stdin
    if context:
        context_text = Path(context).read_text()
    elif not sys.stdin.isatty():
        context_text = sys.stdin.read()
    else:
        console.print("[red]Error: provide --context or pipe context via stdin[/red]")
        raise SystemExit(1)

    if verbose:
        console.print(f"[dim]Model: {config.model}[/dim]")
        console.print(f"[dim]Context: {len(context_text):,} chars[/dim]")

    # TODO: Wire up orchestrator in Phase 4
    click.echo(f"[placeholder] Would process query with {len(context_text)} chars of context")


@main.group()
def cache():
    """Cache management commands."""
    pass


@cache.command()
def stats():
    """Show cache statistics."""
    config = load_config()
    # TODO: Implement in Phase 5
    click.echo(f"Cache dir: {config.cache_dir}")


@cache.command()
def clear():
    """Clear the operation cache."""
    config = load_config()
    # TODO: Implement in Phase 5
    click.echo(f"Would clear cache at {config.cache_dir}")
```

### Success Criteria:

#### Automated Verification:
- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `rlm --version` prints `0.1.0`
- [ ] `rlm run --help` shows all options
- [ ] `python -m pytest tests/` runs (even if no tests yet)
- [ ] `ruff check src/` passes
- [ ] `mypy src/rlm/types.py src/rlm/config.py` passes

#### Manual Verification:
- [ ] `rlm run -q "test" -c RESEARCH.md` runs without error (prints placeholder)

---

## Phase 2: DSL Operations & Lightweight Evaluator

### Overview
Implement all DSL operations from RESEARCH.md and a lightweight in-process evaluator for explore mode. This is the computational foundation everything else builds on.

### Changes Required:

#### 1. Operation base
**File**: `src/rlm/ops/base.py`

Define the operation interface:

```python
"""Base types for DSL operations."""

from __future__ import annotations

from typing import Protocol


class OpExecutor(Protocol):
    """Protocol for operation executors."""
    def execute(self, args: dict, bindings: dict[str, str]) -> str:
        """Execute an operation with the given arguments and variable bindings.

        Args:
            args: Operation-specific arguments.
            bindings: Map of variable names to their string values.

        Returns:
            The operation result as a string.
        """
        ...
```

#### 2. Text operations
**File**: `src/rlm/ops/text.py`

Implement `slice`, `grep`, `count`, `split`, `chunk`:

```python
"""Text manipulation operations."""

from __future__ import annotations

import math
import re


def op_slice(args: dict, bindings: dict[str, str]) -> str:
    """Return a substring of the input."""
    input_text = bindings[args["input"]]
    start = args.get("start", 0)
    end = args.get("end", len(input_text))
    return input_text[start:end]


def op_grep(args: dict, bindings: dict[str, str]) -> str:
    """Return all lines matching a pattern."""
    input_text = bindings[args["input"]]
    pattern = args["pattern"]
    lines = input_text.split("\n")
    matched = [line for line in lines if re.search(pattern, line)]
    return "\n".join(matched)


def op_count(args: dict, bindings: dict[str, str]) -> str:
    """Count lines in the input."""
    input_text = bindings[args["input"]]
    mode = args.get("mode", "lines")
    if mode == "lines":
        return str(len(input_text.strip().split("\n"))) if input_text.strip() else "0"
    elif mode == "chars":
        return str(len(input_text))
    else:
        return str(len(input_text.strip().split("\n")))


def op_split(args: dict, bindings: dict[str, str]) -> str:
    """Split input on a delimiter. Returns JSON array."""
    import json
    input_text = bindings[args["input"]]
    delimiter = args.get("delimiter", "\n")
    parts = input_text.split(delimiter)
    return json.dumps(parts)


def op_chunk(args: dict, bindings: dict[str, str]) -> str:
    """Split input into n roughly equal pieces. Returns JSON array."""
    import json
    input_text = bindings[args["input"]]
    n = args["n"]
    lines = input_text.split("\n")
    chunk_size = math.ceil(len(lines) / n)
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunks.append("\n".join(lines[i:i + chunk_size]))
    return json.dumps(chunks)
```

#### 3. Recursive operations (stubs)
**File**: `src/rlm/ops/recursive.py`

Stubs for `rlm_call`, `map`, `combine` — these need the orchestrator (Phase 4):

```python
"""Recursive operations — require orchestrator integration."""

from __future__ import annotations

import json


def op_combine(args: dict, bindings: dict[str, str]) -> str:
    """Combine multiple results using a strategy."""
    inputs_ref = args["inputs"]
    strategy = args.get("strategy", "concat")

    # inputs can be a JSON array or a binding name
    if isinstance(inputs_ref, list):
        values = [bindings[ref] for ref in inputs_ref]
    else:
        raw = bindings[inputs_ref]
        values = json.loads(raw) if raw.startswith("[") else [raw]

    if strategy == "concat":
        return "\n".join(values)
    elif strategy == "sum":
        total = sum(int(v.strip()) for v in values if v.strip().isdigit())
        return str(total)
    elif strategy == "vote":
        from collections import Counter
        counts = Counter(v.strip() for v in values)
        return counts.most_common(1)[0][0]
    else:
        # Custom strategy = prompt for an LLM call (handled by orchestrator)
        return "\n".join(values)
```

#### 4. Lightweight evaluator
**File**: `src/rlm/evaluator/lightweight.py`

The in-process evaluator for explore mode:

```python
"""Lightweight evaluator for explore-mode operations."""

from __future__ import annotations

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

    def execute(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        """Execute a single operation with current variable bindings."""
        executor = EXPLORE_OPS.get(op.op)
        if executor is None:
            raise ValueError(
                f"Operation {op.op} is not available in explore mode. "
                f"Use COMMIT for {op.op}."
            )

        result_value = executor(op.args, bindings)

        return OpResult(
            op=op.op,
            cache_key=op.cache_key(
                {k: hash(v) for k, v in bindings.items()}  # lightweight hash for explore
            ),
            value=result_value,
        )
```

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_ops.py` — all text operations produce correct results
- [ ] `pytest tests/test_evaluator.py` — lightweight evaluator handles all explore ops
- [ ] `ruff check src/rlm/ops/ src/rlm/evaluator/` passes
- [ ] `mypy src/rlm/ops/ src/rlm/evaluator/` passes

Tests should cover:
- `slice` with various ranges including edge cases (empty, out of bounds)
- `grep` with literal strings and regex patterns
- `count` in lines and chars mode
- `chunk` with even and uneven splits
- `split` with various delimiters
- `combine` with concat, sum, and vote strategies

---

## Phase 3: LLM Integration & Output Parsing

### Overview
Connect to LLM APIs and parse structured output from the explore/commit protocol. The LLM emits JSON-structured actions that get parsed into our type system.

### Changes Required:

#### 1. LLM client
**File**: `src/rlm/llm/client.py`

Thin wrapper around litellm:

```python
"""LLM client abstraction using litellm."""

from __future__ import annotations

from litellm import completion

from rlm.types import RLMConfig


class LLMClient:
    """Manages LLM conversations for the explore/commit protocol."""

    def __init__(self, config: RLMConfig):
        self.config = config
        self.messages: list[dict[str, str]] = []

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for this conversation."""
        self.messages = [{"role": "system", "content": prompt}]

    def send(self, user_message: str) -> str:
        """Send a message and get the assistant's response."""
        self.messages.append({"role": "user", "content": user_message})

        response = completion(
            model=self.config.model,
            messages=self.messages,
            temperature=self.config.temperature,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)
```

#### 2. System prompts
**File**: `src/rlm/llm/prompts.py`

The system prompt that teaches the LLM the explore/commit protocol:

```python
"""System prompts for the explore/commit protocol."""

SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) agent. You solve problems by examining
large contexts through structured operations.

You have a context variable containing {context_chars} characters of text. You cannot see it
directly. Instead, you use operations to examine and process it.

## Protocol

You operate in two modes. Every response must be a valid JSON object with a "mode" field.

### EXPLORE mode
Emit one operation at a time. You see the result before deciding what to do next.

```json
{{
  "mode": "explore",
  "operation": {{
    "op": "<operation>",
    "args": {{ ... }},
    "bind": "<variable_name>"
  }}
}}
```

### COMMIT mode
Emit a computation plan — multiple operations with dependencies.

```json
{{
  "mode": "commit",
  "operations": [
    {{"op": "grep", "args": {{"input": "context", "pattern": "..."}}, "bind": "filtered"}},
    {{"op": "chunk", "args": {{"input": "filtered", "n": 4}}, "bind": "chunks"}},
    {{"op": "map", "args": {{"prompt": "...", "input": "chunks"}}, "bind": "results"}},
    {{"op": "combine", "args": {{"inputs": "results", "strategy": "sum"}}, "bind": "total"}}
  ],
  "output": "total"
}}
```

### FINAL mode
Return your answer.

```json
{{
  "mode": "final",
  "answer": "your answer here"
}}
```

## Available Operations

- `slice(input, start, end)` — substring of input
- `grep(input, pattern)` — lines matching pattern
- `count(input, mode="lines"|"chars")` — count lines or characters
- `chunk(input, n)` — split into n equal pieces
- `split(input, delimiter)` — split on delimiter
- `rlm_call(query, context)` — recursive call with fresh context (COMMIT only)
- `map(prompt, input)` — apply rlm_call to each element in parallel (COMMIT only)
- `combine(inputs, strategy)` — merge results ("concat", "sum", "vote", or a prompt)

## Variables

- `context` is always available and refers to the full context.
- Each operation with a `bind` field stores its result as a variable.
- Later operations can reference earlier variables by name in their `args`.

## Strategy

1. Start in EXPLORE mode. Peek at the context to understand its structure.
2. Use grep/count to understand the data.
3. When you have a strategy, switch to COMMIT mode with a computation plan.
4. After receiving commit results, either COMMIT again or emit FINAL with your answer.

## Rules

- Always respond with valid JSON. No prose outside the JSON.
- In EXPLORE mode, emit exactly one operation per response.
- In COMMIT mode, list operations in dependency order.
- `rlm_call` and `map` are only available in COMMIT mode.
- Keep explore steps minimal — gather just enough info to form a plan.

Query: {query}
'''
```

#### 3. Output parser
**File**: `src/rlm/llm/parser.py`

Parse LLM JSON output into typed actions:

```python
"""Parse LLM output into typed actions."""

from __future__ import annotations

import json
import re

from rlm.types import (
    CommitPlan,
    ExploreAction,
    FinalAnswer,
    LLMAction,
    Operation,
    OpType,
)


class ParseError(Exception):
    """Failed to parse LLM output."""
    pass


def parse_llm_output(raw: str) -> LLMAction:
    """Parse raw LLM output into a typed action.

    Handles JSON possibly wrapped in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON from LLM: {e}\nRaw output:\n{raw}")

    mode = data.get("mode")
    if mode == "explore":
        op_data = data["operation"]
        return ExploreAction(
            operation=Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind"),
            )
        )
    elif mode == "commit":
        operations = [
            Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind"),
            )
            for op_data in data["operations"]
        ]
        return CommitPlan(operations=operations, output=data["output"])
    elif mode == "final":
        return FinalAnswer(answer=data["answer"])
    else:
        raise ParseError(f"Unknown mode: {mode}")
```

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_parser.py` — parses valid explore, commit, and final JSON
- [ ] Parser handles JSON wrapped in ```json code fences
- [ ] Parser raises `ParseError` on invalid input
- [ ] `ruff check src/rlm/llm/` passes
- [ ] `mypy src/rlm/llm/` passes

Tests should cover:
- Valid explore action parsing
- Valid commit plan with multiple operations
- Final answer parsing
- JSON wrapped in markdown code fences
- Invalid JSON handling
- Unknown mode handling
- Missing required fields

---

## Phase 4: Orchestrator & Explore/Commit Loop

### Overview
Wire everything together into the core orchestrator that manages the explore/commit protocol loop. This is the central component of the system.

### Changes Required:

#### 1. Orchestrator
**File**: `src/rlm/orchestrator.py`

```python
"""RLM Orchestrator — manages the explore/commit protocol loop."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.llm.client import LLMClient
from rlm.llm.parser import ParseError, parse_llm_output
from rlm.llm.prompts import SYSTEM_PROMPT
from rlm.types import (
    CommitPlan,
    Context,
    ExploreAction,
    FinalAnswer,
    OpType,
    RLMConfig,
)

logger = logging.getLogger(__name__)


class RLMOrchestrator:
    """Orchestrates the explore/commit protocol between the LLM and evaluators."""

    def __init__(self, config: RLMConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.evaluator = LightweightEvaluator()
        self.console = Console(stderr=True)

    def run(self, query: str, context_text: str, depth: int = 0) -> str:
        """Execute an RLM query against a context.

        Args:
            query: The question to answer.
            context_text: The full context text.
            depth: Current recursion depth.

        Returns:
            The final answer as a string.
        """
        if depth > self.config.max_recursion_depth:
            # At max depth, just do a direct LLM call with truncated context
            return self._direct_call(query, context_text)

        ctx = Context(content=context_text)
        bindings: dict[str, str] = {"context": ctx.content}

        # Initialize LLM conversation
        system = SYSTEM_PROMPT.format(
            context_chars=f"{len(ctx.content):,}",
            query=query,
        )
        self.llm.set_system_prompt(system)

        explore_steps = 0
        commit_cycles = 0

        # Initial message to start the conversation
        response = self.llm.send("Begin. The context variable is available.")

        while True:
            try:
                action = parse_llm_output(response)
            except ParseError as e:
                logger.warning(f"Parse error: {e}")
                response = self.llm.send(
                    f"Your response was not valid JSON. Please respond with a valid JSON "
                    f"object with a 'mode' field. Error: {e}"
                )
                continue

            if isinstance(action, FinalAnswer):
                if self.config.verbose:
                    self.console.print(
                        f"[green]Final answer after {explore_steps} explore steps, "
                        f"{commit_cycles} commit cycles[/green]"
                    )
                return action.answer

            elif isinstance(action, ExploreAction):
                explore_steps += 1
                if explore_steps > self.config.max_explore_steps:
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_explore_steps} "
                        f"explore steps. Please COMMIT a plan or provide a FINAL answer."
                    )
                    continue

                if self.config.verbose:
                    self.console.print(
                        f"[dim]EXPLORE [{explore_steps}]: "
                        f"{action.operation.op}({action.operation.args})[/dim]"
                    )

                try:
                    result = self.evaluator.execute(action.operation, bindings)
                    if action.operation.bind:
                        bindings[action.operation.bind] = result.value

                    # Truncate result for LLM display if very long
                    display_value = result.value
                    if len(display_value) > 4000:
                        display_value = display_value[:4000] + f"\n... ({len(result.value)} chars total)"

                    response = self.llm.send(
                        f"Result of {action.operation.op}:\n{display_value}"
                    )
                except Exception as e:
                    response = self.llm.send(f"Error executing {action.operation.op}: {e}")

            elif isinstance(action, CommitPlan):
                commit_cycles += 1
                if commit_cycles > self.config.max_commit_cycles:
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_commit_cycles} "
                        f"commit cycles. Please provide a FINAL answer."
                    )
                    continue

                if self.config.verbose:
                    self.console.print(
                        f"[blue]COMMIT [{commit_cycles}]: "
                        f"{len(action.operations)} operations[/blue]"
                    )

                try:
                    result = self._execute_commit_plan(action, bindings, depth)
                    bindings[action.output] = result

                    display_result = result
                    if len(display_result) > 4000:
                        display_result = display_result[:4000] + f"\n... ({len(result)} chars total)"

                    response = self.llm.send(
                        f"Commit plan executed. Result ({action.output}):\n{display_result}"
                    )
                except Exception as e:
                    response = self.llm.send(f"Error executing commit plan: {e}")

    def _execute_commit_plan(
        self, plan: CommitPlan, bindings: dict[str, str], depth: int
    ) -> str:
        """Execute a commit plan, handling recursive calls and parallelism."""
        local_bindings = dict(bindings)

        for op in plan.operations:
            if op.op == OpType.RLM_CALL:
                query = op.args["query"]
                ctx_ref = op.args["context"]
                ctx_text = local_bindings[ctx_ref]
                result_value = self._recursive_call(query, ctx_text, depth)

            elif op.op == OpType.MAP:
                prompt = op.args["prompt"]
                input_ref = op.args["input"]
                raw = local_bindings[input_ref]
                items = json.loads(raw) if raw.startswith("[") else [raw]
                result_value = self._parallel_map(prompt, items, depth)

            else:
                # Use lightweight evaluator for non-recursive ops
                result = self.evaluator.execute(op, local_bindings)
                result_value = result.value

            if op.bind:
                local_bindings[op.bind] = result_value

        return local_bindings[plan.output]

    def _recursive_call(self, query: str, context_text: str, depth: int) -> str:
        """Spawn a recursive RLM call."""
        sub_orchestrator = RLMOrchestrator(self.config)
        return sub_orchestrator.run(query, context_text, depth=depth + 1)

    def _parallel_map(self, prompt: str, items: list[str], depth: int) -> str:
        """Execute map operation with parallel recursive calls."""
        results = [""] * len(items)

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
            futures = {
                executor.submit(self._recursive_call, prompt, item, depth): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return json.dumps(results)

    def _direct_call(self, query: str, context_text: str) -> str:
        """Direct LLM call at max recursion depth (no explore/commit)."""
        # Truncate context to fit in a single call
        max_chars = 100_000  # ~25k tokens, safe for most models
        truncated = context_text[:max_chars]

        client = LLMClient(self.config)
        client.set_system_prompt(
            "Answer the following query based on the provided context. "
            "Be precise and concise."
        )
        return client.send(f"Query: {query}\n\nContext:\n{truncated}")
```

#### 2. Wire CLI to orchestrator
**File**: `src/rlm/cli.py`

Update the `run` command to use the orchestrator:

```python
# In the run() function, replace the placeholder with:
from rlm.orchestrator import RLMOrchestrator

orchestrator = RLMOrchestrator(config)
answer = orchestrator.run(query, context_text)
click.echo(answer)
```

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_orchestrator.py` — orchestrator handles explore/commit/final loop
- [ ] Tests mock the LLM client to verify protocol flow
- [ ] Tests verify max explore steps enforcement
- [ ] Tests verify max commit cycles enforcement
- [ ] Tests verify recursion depth limiting
- [ ] `ruff check src/` passes
- [ ] `mypy src/rlm/orchestrator.py` passes

#### Manual Verification:
- [ ] `rlm run -q "What is the first line?" -c RESEARCH.md -v` produces a reasonable answer
- [ ] Verbose output shows explore/commit steps
- [ ] The system recovers from LLM parse errors

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the end-to-end flow works before proceeding.

---

## Phase 5: Content-Addressed Caching

### Overview
Implement a content-addressed cache that mirrors Nix store semantics. Operations with identical inputs return cached results instantly.

### Changes Required:

#### 1. Cache store
**File**: `src/rlm/cache/store.py`

```python
"""Content-addressed cache store."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from rlm.types import OpResult, OpType


class CacheStore:
    """Content-addressed file-system cache for operation results.

    Each result is stored at: {cache_dir}/{hash[:2]}/{hash[2:4]}/{hash}
    This mirrors Nix store path structure for familiarity.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        """Get the filesystem path for a cache key."""
        return self.cache_dir / key[:2] / key[2:4] / key

    def get(self, key: str) -> str | None:
        """Look up a cached result. Returns None on miss."""
        path = self._path_for_key(key)
        if path.exists():
            return path.read_text()
        return None

    def put(self, key: str, value: str) -> None:
        """Store a result in the cache."""
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self._path_for_key(key).exists()

    def stats(self) -> dict:
        """Return cache statistics."""
        total_files = 0
        total_bytes = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                total_files += 1
                total_bytes += p.stat().st_size
        return {
            "entries": total_files,
            "size_bytes": total_bytes,
            "size_human": _human_size(total_bytes),
            "cache_dir": str(self.cache_dir),
        }

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        stats = self.stats()
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return stats["entries"]


def make_cache_key(op: OpType, args: dict, input_hashes: dict[str, str]) -> str:
    """Compute a deterministic cache key for an operation."""
    resolved_args = {}
    for k, v in args.items():
        if isinstance(v, str) and v in input_hashes:
            resolved_args[k] = input_hashes[v]
        else:
            resolved_args[k] = v
    key_data = {"op": op.value, "args": resolved_args}
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
```

#### 2. Integrate cache into evaluator
**File**: `src/rlm/evaluator/lightweight.py`

Add cache lookup/store around operation execution. Modify `LightweightEvaluator.__init__` to accept an optional `CacheStore`, and wrap `execute()` to check cache before computing.

#### 3. Integrate cache into orchestrator
**File**: `src/rlm/orchestrator.py`

- Create a `CacheStore` in `__init__` using `config.cache_dir`
- Pass it to the `LightweightEvaluator`
- Track input hashes for bindings (hash of context, hash of operation results)
- Cache commit plan results too

#### 4. Wire cache CLI commands
**File**: `src/rlm/cli.py`

Update `cache stats` and `cache clear` to use the real `CacheStore`.

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_cache.py` — cache store get/put/has/stats/clear work correctly
- [ ] Identical operations produce cache hits
- [ ] Different operations produce cache misses
- [ ] Cache key is deterministic (same inputs = same key)
- [ ] `rlm cache stats` shows real statistics
- [ ] `rlm cache clear` removes entries
- [ ] `ruff check src/` passes

#### Manual Verification:
- [ ] Run the same query twice; second run is noticeably faster
- [ ] Verbose output shows cache hits on second run

---

## Phase 6: Nix Integration

### Overview
Add optional Nix integration for commit-mode operations. When `--use-nix` is set, commit plans compile to Nix derivations for stronger sandboxing, automatic caching via the Nix store, and parallel execution via `nix-build --max-jobs`.

### Changes Required:

#### 1. Nix derivation compiler
**File**: `src/rlm/nix/compiler.py`

Compile DSL operations into Nix expressions. Each operation becomes a derivation with content-addressed inputs.

Key behaviors:
- `grep`, `slice`, `count`, `chunk`, `split` compile to simple shell derivations
- `rlm_call` compiles to a fixed-output derivation (the orchestrator makes the API call and imports the result)
- `map` compiles to multiple independent derivations that Nix schedules in parallel
- `combine` compiles to a derivation that depends on all its inputs

#### 2. Nix store interaction
**File**: `src/rlm/nix/store.py`

Wrapper around `nix-store` commands:
- `nix-store --add` to import files into the store
- `nix-store --query --hash` to get store path hashes
- `nix-store --gc` for garbage collection

#### 3. Nix builder
**File**: `src/rlm/nix/builder.py`

Wrapper around `nix-build`:
- Build derivations with `--max-jobs` for parallelism
- Check if output already exists in store (cache hit)
- Import build results back into bindings

#### 4. Nix expression templates
**File**: `src/rlm/nix/templates.py`

String templates for generating `.nix` files:

```python
GREP_TEMPLATE = '''
{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-grep-{hash}" {{
  input = {input_path};
}} ''
  grep -F '{pattern}' "$input" > $out || true
''
'''

# Similar templates for other operations
```

#### 5. Conditional Nix backend in orchestrator
**File**: `src/rlm/orchestrator.py`

When `config.use_nix` is True, `_execute_commit_plan` routes through the Nix compiler/builder instead of the lightweight evaluator.

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_nix_compiler.py` — generates valid Nix expressions (can be tested without Nix installed by checking string output)
- [ ] `ruff check src/rlm/nix/` passes
- [ ] When Nix is not installed, `--use-nix` raises a clear error

#### Manual Verification (requires Nix):
- [ ] `rlm run -q "..." -c data.txt --use-nix -v` shows Nix derivation builds
- [ ] Second run with same query shows Nix cache hits
- [ ] Independent map operations build in parallel (visible in verbose output)

**Implementation Note**: This phase is optional for initial functionality. The Python-native path (Phase 4) provides a fully working system without Nix. Nix adds stronger sandboxing and caching guarantees.

---

## Phase 7: Sandboxing, CLI Polish & Documentation

### Overview
Add bubblewrap sandboxing for explore-mode operations, polish the CLI output, and write comprehensive setup/usage documentation.

### Changes Required:

#### 1. Bubblewrap sandbox
**File**: `src/rlm/evaluator/sandbox.py`

```python
"""Bubblewrap-based sandboxing for explore operations."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


class BwrapSandbox:
    """Execute operations inside a bubblewrap sandbox."""

    def __init__(self):
        self.bwrap_path = shutil.which("bwrap")

    @property
    def available(self) -> bool:
        return self.bwrap_path is not None

    def run(self, command: list[str], input_data: str | None = None,
            timeout: int = 30) -> str:
        """Run a command in a sandboxed environment.

        The sandbox has:
        - Read-only access to /usr, /lib, /bin
        - No network access
        - A tmpfs for /tmp
        - Input data available as /sandbox/input if provided
        """
        if not self.available:
            raise RuntimeError("bubblewrap (bwrap) is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox_dir = Path(tmpdir)
            if input_data is not None:
                (sandbox_dir / "input").write_text(input_data)

            bwrap_args = [
                self.bwrap_path,
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/lib", "/lib",
                "--ro-bind", "/bin", "/bin",
                "--symlink", "/usr/lib64", "/lib64",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
                "--ro-bind", str(sandbox_dir), "/sandbox",
                "--unshare-net",
                "--new-session",
                "--die-with-parent",
                "--", *command,
            ]

            result = subprocess.run(
                bwrap_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Sandbox command failed (exit {result.returncode}): {result.stderr}"
                )
            return result.stdout
```

#### 2. CLI output improvements
**File**: `src/rlm/cli.py`

- Use `rich` for structured verbose output (tables for explore steps, panels for commit plans)
- Progress spinner during LLM API calls
- Summary statistics at the end (explore steps, commit cycles, cache hits, wall time)

#### 3. README documentation
**File**: `README.md`

Write comprehensive documentation covering:

```markdown
# rlm-secure

Sandboxed Recursive Language Models with Nix.

## What is this?

rlm-secure lets LLMs process contexts far larger than their context window by
recursively breaking problems into smaller pieces. Each piece runs in an
isolated sandbox with content-addressed caching and parallel execution.

Based on the Recursive Language Models paper (Zhang & Khattab, 2025).

## Requirements

- Python 3.11+
- An LLM API key (OpenAI, Anthropic, or any litellm-supported provider)

### Optional
- Nix (for sandboxed execution and Nix-store caching)
- bubblewrap (for sandboxed explore operations on Linux)

## Installation

### From source

    git clone <repo-url>
    cd rlm-secure
    pip install -e .

### With development dependencies

    pip install -e ".[dev]"

## Quick Start

### Set your API key

    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

### Run a query

    rlm run --query "How many entity questions are there?" --context data.txt

### Pipe context from stdin

    cat large_dataset.txt | rlm run --query "Summarize the key themes"

## Configuration

### CLI Options

    rlm run --help

| Option | Default | Description |
|--------|---------|-------------|
| --query, -q | (required) | The query to answer |
| --context, -c | stdin | Path to context file |
| --model, -m | gpt-4o-mini | LLM model identifier |
| --max-explore | 20 | Maximum explore steps |
| --max-depth | 1 | Maximum recursion depth |
| --use-nix | false | Enable Nix sandboxing |
| --verbose, -v | false | Show detailed execution trace |

### Environment Variables

| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | OpenAI API key |
| ANTHROPIC_API_KEY | Anthropic API key |
| RLM_MODEL | Default model |
| RLM_MAX_EXPLORE_STEPS | Default max explore steps |
| RLM_MAX_COMMIT_CYCLES | Default max commit cycles |
| RLM_MAX_RECURSION_DEPTH | Default max recursion depth |
| RLM_MAX_PARALLEL_JOBS | Parallel job count for map operations |
| RLM_CACHE_DIR | Cache directory path |
| RLM_USE_NIX | Enable Nix by default |
| RLM_VERBOSE | Enable verbose by default |

## Cache Management

View cache statistics:

    rlm cache stats

Clear the cache:

    rlm cache clear

## How It Works

### The Explore/Commit Protocol

1. The LLM receives a query but NOT the full context.
2. In **explore mode**, it issues operations one at a time (slice, grep, count)
   to understand the context structure.
3. When ready, it switches to **commit mode** and emits a computation plan.
4. The plan is executed — operations run in parallel where possible, recursive
   sub-calls spawn fresh LLM instances for smaller context slices.
5. Results flow back and the LLM can explore again or return a final answer.

### Caching

Every operation is content-addressed: the cache key is a hash of the operation
type and all its inputs. Running the same query twice on the same data reuses
all cached results automatically.

### Sandboxing (with --use-nix)

When Nix mode is enabled, commit-plan operations compile to Nix derivations.
Each derivation runs in a sandbox with no network, no filesystem access outside
its build directory, and no access to other processes. This prevents any
LLM-generated code from affecting the host system.

## Architecture

    User
     │
     ▼
    CLI (click)
     │
     ▼
    Orchestrator ──── LLM Client (litellm)
     │       │
     │       ▼
     │    Lightweight Evaluator ── explore ops (slice, grep, count...)
     │
     ▼
    Commit Plan Executor
     ├── Python-native path (default)
     │   └── ThreadPoolExecutor for parallel map
     └── Nix path (--use-nix)
         └── nix-build with --max-jobs

## Development

### Run tests

    pytest

### Lint

    ruff check src/ tests/

### Type check

    mypy src/

## References

- Zhang, A. & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601v1.
- Nix: https://nixos.org/
- bubblewrap: https://github.com/containers/bubblewrap
```

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/` — all tests pass
- [ ] `ruff check src/ tests/` — no lint errors
- [ ] `mypy src/` — no type errors
- [ ] `rlm --help` — shows all commands
- [ ] `rlm run --help` — shows all options
- [ ] `rlm cache stats` — shows stats
- [ ] `rlm cache clear` — clears cache

#### Manual Verification:
- [ ] README renders correctly on GitHub
- [ ] Installation instructions work on a clean environment
- [ ] `rlm run` with a real API key produces a correct answer
- [ ] Verbose output is clear and useful
- [ ] bubblewrap sandboxing works on Linux (when bwrap is installed)

---

## Testing Strategy

### Unit Tests:
- All text operations (slice, grep, count, chunk, split, combine)
- LLM output parser (valid/invalid JSON, all three modes)
- Cache store (get/put/has/stats/clear, key generation)
- Nix compiler (string output validation, no Nix required)

### Integration Tests:
- Orchestrator with mocked LLM client (verify protocol flow)
- End-to-end with a mock LLM (scripted responses through explore/commit/final)
- Cache integration (verify hits on repeated operations)

### Manual Testing Steps:
1. Run `rlm run -q "What is the first heading?" -c RESEARCH.md -v` — should explore, find the answer, return it
2. Run the same command again — should show cache hits in verbose output
3. Run with a large dataset (generate one with many entries) to test chunking and map
4. Run with `--use-nix` to verify Nix integration (requires Nix)

## Performance Considerations

- **Explore operations** should complete in <100ms (in-process string operations)
- **LLM API calls** dominate latency — parallelism via `map` is critical
- **Cache hits** should be <10ms (filesystem lookup)
- **Nix derivation overhead** is 200-500ms per derivation — only used for commit mode
- **ThreadPoolExecutor** for map parallelism avoids GIL issues since work is I/O-bound (API calls)
- Context truncation at max depth prevents unbounded token usage

## References

- RESEARCH.md — Full design document
- alexzhang13/rlm — Reference implementation (unsandboxed)
- litellm docs — LLM client API
- instructor docs — Structured output parsing
- Nix manual — Derivation and store concepts
