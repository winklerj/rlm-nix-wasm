# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rlm-nix-wasm is a recursive language model system that lets LLMs break down large-context problems into smaller recursive sub-calls. Each call operates through a structured DSL (not arbitrary code execution), with content-addressed caching and optional Nix sandboxing.

## Development Commands

```bash
# Setup (uses uv)
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_ops.py

# Run a single test
pytest tests/test_ops.py::test_slice_basic -v

# Lint
ruff check src/ tests/

# Type check (strict mode)
mypy src/

# Run the CLI
rlm run -q "How many unique users?" -c server.log
rlm run -q "Question" -c context.txt -v --trace
```

## Architecture

### Explore/Commit Protocol

The core execution model is a three-mode protocol between the LLM and the system:

- **EXPLORE**: LLM executes single DSL operations one at a time, immediately seeing results. Used for iterative investigation.
- **COMMIT**: LLM submits a multi-operation DAG to execute as a batch. More efficient for known workflows.
- **FINAL**: LLM returns its answer.

The orchestrator (`orchestrator.py`) manages this loop: prompt LLM → parse response → execute operation(s) → feed results back → repeat.

### Component Layers

**CLI** (`cli.py`) → **RLMOrchestrator** (`orchestrator.py`) → three subsystems:
- **LLM layer** (`llm/`): `LLMClient` wraps litellm, `parse_llm_output()` converts raw LLM JSON into typed `ExploreAction | CommitPlan | FinalAnswer`, `prompts.py` holds the system prompt.
- **Evaluator** (`evaluator/lightweight.py`): Executes operations in-process. Operations are registered in the `EXPLORE_OPS` dict — add new ops by implementing the `OpExecutor` protocol (`ops/base.py`) and registering them.
- **Cache** (`cache/store.py`): Content-addressed filesystem cache keyed by SHA256 of (op_type + args + input_hashes). Identical operations with identical inputs reuse results across runs.

**Optional Nix sandboxing** (`nix/`): Compiles DSL operations to Nix derivations for isolated execution. Only imported when `--use-nix` is passed (lazy loading to avoid hard dependency).

### DSL Operations

Defined in `types.py` as `OpType` enum, implemented in `ops/`:
- **Text ops** (`ops/text.py`): `slice`, `grep`, `count`, `chunk`, `split`
- **Eval op** (`ops/eval_op.py`, `evaluator/wasm_sandbox.py`): `eval` — runs Python code in a Wasm (WASI) sandbox. Requires `--wasm-python`. Dispatched directly by `LightweightEvaluator._execute_eval()`, not through `EXPLORE_OPS`.
- **Recursive ops** (`ops/recursive.py`): `combine` (and `rlm_call`, `map` handled by orchestrator)

The orchestrator handles `rlm_call` and `map` specially — `rlm_call` creates a child `RLMOrchestrator`, and `map` uses `ThreadPoolExecutor` for parallel recursive calls.

### Variable Binding System

Operations can `bind` their results to named variables (e.g., `{"op": "slice", "args": {...}, "bind": "peek"}`). The orchestrator maintains a `bindings: dict[str, str]` that subsequent operations reference. This lets the LLM build up intermediate results without re-executing operations.

### Configuration

Three-layer precedence: defaults in `RLMConfig` (types.py) → environment variables (`RLM_MODEL`, `RLM_MAX_EXPLORE_STEPS`, etc.) → CLI flags. Merged in `config.py:load_config()`.

### Type System

All data models are Pydantic (`types.py`). `LLMAction` is the union type `ExploreAction | CommitPlan | FinalAnswer`. `RLMConfig` holds all runtime configuration. Strict mypy is enforced.

### Tracing and Timing

`TraceCollector` (`trace.py`) and `TimingProfile` (`timing.py`) are thread-safe (use locks) because child orchestrators may run in parallel threads during `map` operations. Trace output is a JSON tree of `OrchestratorTrace` nodes.

## Git Conventions

- Do not include `Co-Authored-By` or any Claude attribution in commit messages.
