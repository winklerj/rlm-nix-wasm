# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rlm-nix-wasm is a recursive language model system that lets LLMs break down large-context problems into smaller recursive sub-calls. Each call operates through a structured DSL (not arbitrary code execution), with content-addressed caching and optional Nix sandboxing.

## Environment

**Always use `nix-shell`** for running commands. The `shell.nix` sets `LD_LIBRARY_PATH` for native dependencies (numpy, tokenizers, etc.) that fail without `libstdc++`.

**Wasm sandbox**: Always pass `--wasm-python .wasm/python-3.12.0.wasm` when running benchmarks or eval. The Wasm sandbox enables the `eval` operation for Python code execution in a sandboxed environment.

**API key**: litellm auto-loads `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`. The OAuth token in `~/.claude/.credentials.json` is a Claude Code session token and cannot be used as an API key.

## Development Commands

```bash
# Setup (uses uv)
uv pip install -e ".[dev]"

# Run all tests (always inside nix-shell)
nix-shell --run "uv run pytest"

# Run a single test file
nix-shell --run "uv run pytest tests/test_ops.py"

# Run a single test
nix-shell --run "uv run pytest tests/test_ops.py::test_slice_basic -v"

# Lint
nix-shell --run "uv run ruff check src/ tests/"

# Type check (strict mode)
nix-shell --run "uv run mypy src/"

# Run the CLI (with Wasm sandbox)
nix-shell --run "uv run rlm run -q 'How many unique users?' -c server.log --wasm-python .wasm/python-3.12.0.wasm"

# Run benchmark eval
nix-shell --run "uv run rlm eval run --dataset trec_coarse --context-len 65536 --model claude-opus-4-5 --wasm-python .wasm/python-3.12.0.wasm --temperature 0.3 --output results/oolong-trec-coarse-65k.jsonl -v"
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

## Running OOLONG Benchmarks

Benchmarks are long-running (hours) and expensive. Run them in tmux sessions so they survive session disconnects.

### Starting benchmark runs

Use `--max-depth 0` for single-level (root orchestrator only, children are direct LLM calls). This matches the paper's depth-1 setup and keeps cost reasonable (~$1-3/task). Without this, `--max-depth 1` (the default) allows children to run their own full explore/commit loops, which is ~$15-25/task.

```bash
# Opus 4.5 benchmark
tmux new-session -d -s opus45 'nix-shell --run "uv run rlm eval run --dataset trec_coarse --context-len 65536 --model claude-opus-4-5 --max-depth 0 --temperature 0.3 --wasm-python .wasm/python-3.12.0.wasm --output results/oolong-trec-coarse-65k-opus45.jsonl -v" 2>&1 | tee /tmp/opus45.log'

# Opus 4.6 benchmark
tmux new-session -d -s opus46 'nix-shell --run "uv run rlm eval run --dataset trec_coarse --context-len 65536 --model claude-opus-4-6 --max-depth 0 --temperature 0.3 --wasm-python .wasm/python-3.12.0.wasm --output results/oolong-trec-coarse-65k-opus46.jsonl -v" 2>&1 | tee /tmp/opus46.log'
```

### Checking on running benchmarks

```bash
# List active tmux sessions
tmux list-sessions

# Watch live output
tmux attach -t opus45   # (Ctrl-B D to detach)
tmux attach -t opus46

# Check recent log output
tail -20 /tmp/opus45.log
tail -20 /tmp/opus46.log

# Check progress summary (scores, costs, tasks completed)
nix-shell --run "python /tmp/check_progress.py"
```

### Resuming interrupted runs

Results are written incrementally to JSONL. If a run is interrupted, use `--resume` to skip already-completed tasks:

```bash
tmux new-session -d -s opus45 'nix-shell --run "uv run rlm eval run --dataset trec_coarse --context-len 65536 --model claude-opus-4-5 --max-depth 0 --temperature 0.3 --wasm-python .wasm/python-3.12.0.wasm --output results/oolong-trec-coarse-65k-opus45.jsonl --resume -v" 2>&1 | tee /tmp/opus45.log'
```

### Key flags

| Flag | Purpose |
|------|---------|
| `--max-depth 0` | Single-level: root does explore/commit, children are direct LLM calls |
| `--max-depth 1` | Two-level: root AND children do explore/commit (default, expensive) |
| `--temperature 0.3` | Lower temperature for deterministic benchmark answers |
| `--wasm-python .wasm/python-3.12.0.wasm` | Enable Wasm sandbox for `eval` operation |
| `--resume` | Skip tasks already in the output JSONL |
| `-v` | Verbose output (shows LLM calls, timings, token counts) |

### Progress check script

The script at `/tmp/check_progress.py` reads the result JSONL files and shows per-run summaries. If it doesn't exist, create it:

```python
import json, sys

files = [
    "results/oolong-trec-coarse-65k-opus45.jsonl",
    "results/oolong-trec-coarse-65k-opus46.jsonl",
]

for f in files:
    try:
        with open(f) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
    except FileNotFoundError:
        print(f"\n{f}: not started yet")
        continue

    scores = [r["score"] for r in lines]
    avg = sum(scores) / len(scores) if scores else 0
    cost = sum(r["cost_usd"] for r in lines)
    types = {}
    for r in lines:
        t = r["answer_type"]
        types.setdefault(t, []).append(r["score"])

    print(f"\n{f}:")
    print(f"  Tasks completed: {len(lines)}/50")
    print(f"  Running avg score: {avg:.3f} ({avg*100:.1f}%)")
    print(f"  Total cost: ${cost:.4f}")
    for t, s in sorted(types.items()):
        tavg = sum(s) / len(s)
        print(f"    {t}: {tavg:.3f} ({len(s)} tasks)")
    if lines:
        print(f"  Last task: id={lines[-1]['id']}, predicted={lines[-1]['predicted']!r}, gold={lines[-1]['gold']!r}")
```

## Git Conventions

- Do not include `Co-Authored-By` or any Claude attribution in commit messages.
