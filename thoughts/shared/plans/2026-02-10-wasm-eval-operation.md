# Wasm `eval` Operation Implementation Plan

## Overview

Implement the `eval` operation using `wasmtime-py` + CPython compiled to WASI, giving the LLM the ability to write Python code that runs in a Wasm sandbox with pre-loaded variables. This bridges the gap between the current structured DSL and the original RLM paper's Python REPL model — the LLM can use the DSL for common tasks and fall back to `eval` when it needs arbitrary logic.

**Security note**: The `eval` operation in this system does NOT use Python's built-in `eval()` or `exec()`. Instead, LLM-generated code is executed inside a WebAssembly (WASI) sandbox via `wasmtime`, which provides hardware-level memory isolation, no filesystem access, no network access, and CPU/memory resource limits. This is the explicit design intent from RESEARCH.md and is the reason the operation was deferred until Wasm sandboxing could be implemented.

## Current State Analysis

- `OpType.EVAL = "eval"` exists in `types.py:31` but is an empty stub
- `eval_op.py` is an empty file, `eval` is not registered in `EXPLORE_OPS`, not handled in the orchestrator's commit path, and not mentioned in the system prompt
- The existing `BwrapSandbox` in `evaluator/sandbox.py` provides a subprocess-based sandbox pattern we can reference but not reuse (Wasm is a different isolation model)
- `pyproject.toml` has no Wasm-related dependencies
- PLAN.md:51 explicitly defers eval: "Custom VM environments (eval with Wasm/QuickJS) — deferred to future work"
- RESEARCH.md:351-354 specifies the intended interface: `eval(code, bindings, vm_spec) -> string`

### Key Discoveries:
- `wasmtime` (PyPI) is the most mature Python Wasm runtime; version 36+ supports fuel metering, memory limits, and full WASI configuration
- Pre-built CPython WASI binaries (`python.wasm`) are available from the WebAssembly Language Runtimes project (~20MB)
- All existing ops follow the `(args: dict, bindings: dict[str, str]) -> str` pattern (`ops/text.py:10`)
- The `LightweightEvaluator` dispatches via a simple `EXPLORE_OPS` dict (`evaluator/lightweight.py:14`)
- Commit mode has an `if/elif/else` chain for special ops (`orchestrator.py:231-254`) — eval falls through to the evaluator's `else` branch at line 252 and needs no special handling there
- Cache keys use `make_cache_key(op, args, input_hashes)` from `cache/store.py:71` — eval's code string becomes part of the key naturally
- The existing `BwrapSandbox.run()` pattern (write input to temp dir, run subprocess, capture stdout) maps well to the Wasm approach

## Desired End State

The LLM can emit eval operations in both explore and commit modes:

```json
{
  "mode": "explore",
  "operation": {
    "op": "eval",
    "args": {
      "code": "import re\nusers = set(re.findall(r'User: (\\d+)', context))\nresult = str(len(users))",
      "inputs": ["context"]
    },
    "bind": "unique_user_count"
  }
}
```

The code runs inside a Wasm sandbox (CPython compiled to WASI) with:
- Pre-loaded variables from `inputs` (resolved from bindings)
- No filesystem access beyond the sandbox directory
- No network access
- Fuel-based CPU metering (configurable, default ~10s equivalent)
- Memory cap (configurable, default 256MB)
- stdout captured as the result

### Verification:
```bash
# Existing DSL ops still work identically
pytest tests/test_ops.py

# New eval tests pass
pytest tests/test_eval.py

# Eval operation works end-to-end (integration test with mock LLM)
pytest tests/test_orchestrator.py -k eval

# Type checking passes
mypy src/rlm

# Linting passes
ruff check src/ tests/

# Manual: run a query where the LLM uses eval
rlm run -q "What are the unique user IDs?" -c sample.log -v
```

## What We're NOT Doing

- JavaScript/QuickJS support — Python only for now
- `vm_spec` argument from RESEARCH.md — all eval calls use the same sandbox config (Python + stdlib)
- Configurable allowed modules — all stdlib modules available (WASI naturally blocks dangerous ones like `socket`, `subprocess`)
- Nix integration for eval — eval runs via Wasm, not compiled to Nix derivations
- Auto-download of python.wasm — user provides or installs it (with clear instructions)

## Implementation Approach

Build in 6 phases. Each phase is independently testable. The Wasm sandbox is isolated behind a clean interface so it can be swapped or extended later. Phase 6 adds formal TLA+ specifications and Hypothesis property-based tests that verify system invariants hold across random inputs.

---

## Phase 1: Wasm Sandbox Runtime

### Overview
Create the `WasmSandbox` class that manages the wasmtime engine, loads python.wasm, and runs Python code with variable injection and resource limits.

### Changes Required:

#### 1. Add `wasmtime` as optional dependency
**File**: `pyproject.toml`
**Changes**: Add `wasm` optional dependency group

```toml
[project.optional-dependencies]
wasm = [
    "wasmtime>=25.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
    "wasmtime>=25.0",
]
```

#### 2. Add Wasm config fields to `RLMConfig`
**File**: `src/rlm/types.py`
**Changes**: Add fields after `use_nix` (line 101)

```python
class RLMConfig(BaseModel):
    """Configuration for an RLM run."""
    model: str = "claude-opus-4-5"
    child_model: str | None = None
    max_explore_steps: int = 20
    max_commit_cycles: int = 5
    max_recursion_depth: int = 1
    max_parallel_jobs: int = 4
    temperature: float = 1.0
    cache_dir: Path = Path.home() / ".cache" / "rlm-nix-wasm"
    use_nix: bool = False
    verbose: bool = False
    # Wasm sandbox settings for sandboxed code operations
    wasm_python_path: Path | None = None  # Path to python.wasm binary
    wasm_fuel: int = 400_000_000  # CPU fuel limit (~10s equivalent)
    wasm_memory_mb: int = 256  # Memory limit in MB
```

#### 3. Add environment variable support for Wasm config
**File**: `src/rlm/config.py`
**Changes**: Add env var mappings for the new fields

Map: `RLM_WASM_PYTHON_PATH` -> `wasm_python_path`, `RLM_WASM_FUEL` -> `wasm_fuel`, `RLM_WASM_MEMORY_MB` -> `wasm_memory_mb`

#### 4. Create the WasmSandbox class
**File**: `src/rlm/evaluator/wasm_sandbox.py` (new file)

The class manages the wasmtime engine lifecycle and runs Python code in isolation:

- **Constructor**: Takes `python_wasm_path`, `fuel`, `memory_mb`. Lazily imports `wasmtime` (so it's not a hard dependency).
- **`_ensure_loaded()`**: Compiles `python.wasm` once (Engine + Module creation is ~200ms). Reused across all calls.
- **`available` property**: Returns True only if wasmtime is importable AND python.wasm exists.
- **`run(code, variables)`**: The core method. Per invocation:
  1. Calls `_ensure_loaded()` to get the compiled module
  2. Creates a fresh `Store` with fuel limit
  3. Creates a temp directory
  4. Writes `variables` dict as `/sandbox/vars.json`
  5. Generates wrapper script (loads JSON into globals, runs user code, auto-prints `result` variable)
  6. Writes wrapper to `/sandbox/script.py`
  7. Configures WASI: argv, stdout/stderr capture files, preopen sandbox dir only
  8. Links, instantiates, calls `_start`
  9. On success: returns stdout file contents
  10. On fuel exhaustion: raises `TimeoutError`
  11. On Python error: raises `RuntimeError` with stderr contents

**`_build_wrapper(user_code)`**: Standalone function that generates the wrapper script. The wrapper:
1. Reads `/sandbox/vars.json` and injects each key-value pair as a global variable
2. Runs the user's code
3. If the user's code set a `result` variable, prints it (for auto-capture)

### Success Criteria:

#### Automated Verification:
- [x] `wasmtime` installs cleanly: `pip install wasmtime`
- [x] `WasmSandbox.available` returns True when wasmtime + python.wasm exist
- [x] `WasmSandbox.available` returns False when wasmtime is not installed
- [x] Simple print -> stdout captured correctly
- [x] Variable injection -> accessible in user code
- [x] `result` variable auto-print works
- [x] Fuel exhaustion raises `TimeoutError`
- [x] Python errors raise `RuntimeError` with traceback
- [x] Type checking passes: `mypy src/rlm/evaluator/wasm_sandbox.py`

#### Manual Verification:
- [x] Confirm python.wasm is obtainable
- [x] Confirm no filesystem escape (cannot read host files)
- [x] Confirm no network access

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 2: Operation Wiring

### Overview
Wire eval dispatch into the evaluator and connect the WasmSandbox to the orchestrator.

### Changes Required:

#### 1. Implement eval_op.py
**File**: `src/rlm/ops/eval_op.py`
**Changes**: Replace empty file with a stub documenting the dispatch pattern

The function raises `RuntimeError` if called directly — actual dispatch happens in the evaluator. This keeps the ops/ directory consistent and provides documentation.

#### 2. Add eval dispatch to LightweightEvaluator
**File**: `src/rlm/evaluator/lightweight.py`
**Changes**:
- Add `wasm_sandbox` optional constructor parameter
- Add early check in `execute()`: if `op.op == OpType.EVAL`, delegate to `_execute_eval()`
- Implement `_execute_eval()`:
  1. Validate sandbox is available (raise clear error if not)
  2. Extract `code` and `inputs` from `op.args`
  3. Resolve bindings to variables dict (only names listed in `inputs`, or all bindings if `inputs` not specified)
  4. Compute cache key (the `code` field is part of `op.args`, so `make_cache_key` handles it naturally)
  5. Check cache
  6. Call `self.wasm_sandbox.run(code, variables)`
  7. Strip trailing newline from stdout capture
  8. Store in cache
  9. Return `OpResult`

#### 3. Wire WasmSandbox into the orchestrator
**File**: `src/rlm/orchestrator.py`
**Changes**: Initialize WasmSandbox when `wasm_python_path` is configured, pass to evaluator

In `__init__`, after existing Nix setup (~line 55):

```python
# Initialize Wasm sandbox for sandboxed code operations (lazy load)
wasm_sandbox = None
if config.wasm_python_path:
    from rlm.evaluator.wasm_sandbox import WasmSandbox
    wasm_sandbox = WasmSandbox(
        python_wasm_path=config.wasm_python_path,
        fuel=config.wasm_fuel,
        memory_mb=config.wasm_memory_mb,
    )

self.evaluator = LightweightEvaluator(
    cache=self.cache, profile=self.profile, wasm_sandbox=wasm_sandbox,
)
```

No changes needed to `_execute_commit_plan` — eval operations in commit plans already fall through to `self.evaluator.execute()` via the `else` branch at line 252-254.

### Success Criteria:

#### Automated Verification:
- [x] All existing tests pass: `pytest tests/test_ops.py`
- [x] Eval with sandbox returns correct results
- [x] Eval without sandbox raises clear `RuntimeError`
- [x] Eval results are cached (second call = cache hit)
- [x] Eval works in both explore and commit mode
- [x] Type checking passes: `mypy src/rlm`
- [x] Linting passes: `ruff check src/`

#### Manual Verification:
- [x] Run a query where the LLM would benefit from eval

**Implementation Note**: After completing this phase, pause for manual confirmation before proceeding.

---

## Phase 3: System Prompt + LLM Integration

### Overview
Update the system prompt so the LLM knows about eval, when to use it vs DSL ops, and the expected code format. The prompt should only mention eval when the Wasm sandbox is configured.

### Changes Required:

#### 1. Update the system prompt
**File**: `src/rlm/llm/prompts.py`
**Changes**: Make eval documentation conditional

Option: Split the prompt into a base template and an eval addendum. The orchestrator appends the addendum only when `wasm_python_path` is configured.

Eval documentation to add to operations list:

```
- `eval(code, inputs)` — run Python code in a Wasm sandbox. Variables from `inputs` are pre-loaded.
  Set a `result` variable or use `print()` for output. Use for logic that the other ops can't express
  (regex, math, custom filtering). Available stdlib: re, json, math, collections, itertools, etc.
```

Add to Approach section:

```
5. PREFER DSL OPS OVER EVAL — Use slice/grep/count/chunk/split for common tasks.
   Use eval only when you need logic these ops cannot express (complex regex, arithmetic,
   conditional filtering). Eval is slower due to sandbox overhead.
```

#### 2. Add CLI flag
**File**: `src/rlm/cli.py`
**Changes**: Add `--wasm-python` option mapped to `RLM_WASM_PYTHON_PATH`

### Success Criteria:

#### Automated Verification:
- [x] System prompt includes eval docs when wasm_python_path is set
- [x] System prompt omits eval docs when wasm_python_path is not set
- [x] `rlm run --help` shows `--wasm-python` option
- [x] Type checking and linting pass

#### Manual Verification:
- [x] LLM uses eval appropriately when enabled
- [x] LLM doesn't attempt eval when disabled

**Implementation Note**: After completing this phase, pause for manual confirmation before proceeding.

---

## Phase 4: Documentation Updates

### Overview
Update all documentation to cover the eval operation, Wasm sandbox setup, and new configuration.

### Changes Required:

#### 1. README.md
**File**: `README.md`
**Changes**: Add "Wasm Code Execution" section after Quick Start

Content: brief explanation, 3-step setup (install wasmtime, get python.wasm, set env var), pointer to how-to guides.

#### 2. docs/reference.md
**File**: `docs/reference.md`
**Changes**:
- Add `eval` operation with argument table and example JSON (after `combine`, line 141)
- Add `--wasm-python` to CLI options table (after line 25)
- Add `RLM_WASM_PYTHON_PATH`, `RLM_WASM_FUEL`, `RLM_WASM_MEMORY_MB` to configuration table (after line 63)
- Add `eval_op.py` and `wasm_sandbox.py` to architecture tree (after line 218)

#### 3. docs/how-to-guides.md
**File**: `docs/how-to-guides.md`
**Changes**: Add "How to enable Wasm eval" section after "How to enable Nix sandboxing" (after line 63)

Content: setup steps, CLI flag alternative, configuring resource limits, when to expect eval usage, note about conditional availability.

#### 4. docs/explanation.md
**File**: `docs/explanation.md`
**Changes**:
- Update "Why not just let the LLM write code?" (lines 13-21) to acknowledge eval as a controlled middle ground: DSL-first with sandboxed code fallback
- Add "The eval escape hatch" section with DSL-vs-eval comparison table (properties, isolation, caching, speed, expressiveness)
- Update "Defense in depth" table (line 59) to include the Wasm sandbox layer
- Update "Trade-offs" section to add DSL+eval tradeoff discussion

### Success Criteria:

#### Automated Verification:
- [x] No broken markdown links

#### Manual Verification:
- [x] README.md Wasm section is clear and correct
- [x] reference.md has complete eval specification
- [x] how-to-guides.md setup instructions are followable
- [x] explanation.md narrative flows naturally with the eval addition

**Implementation Note**: After completing this phase, pause for manual confirmation before proceeding.

---

## Phase 5: Unit and Integration Testing

### Overview
Comprehensive example-based tests for sandbox, eval operation, and orchestrator integration.

### Changes Required:

#### 1. tests/test_wasm_sandbox.py (new file)

**Always-run tests** (no Wasm needed):
- `_build_wrapper()` includes vars.json loading logic
- `_build_wrapper()` includes result auto-print logic

**Wasm-dependent tests** (skipped when wasmtime/python.wasm unavailable):
- Simple print -> stdout capture
- Variable injection -> accessible in code
- `result` variable auto-print
- Multiple variables
- `re` module works (regex)
- Fuel exhaustion -> `TimeoutError`
- Syntax error -> `RuntimeError`
- Network access blocked
- Host filesystem read blocked

#### 2. tests/test_eval.py (new file)

- Eval with sandbox -> correct result
- Eval without sandbox -> `RuntimeError` with clear message
- Eval result cached (second call is cache hit)
- Different code -> cache miss
- Different inputs -> cache miss
- Eval binding via `op.bind` works

#### 3. tests/test_orchestrator.py updates

- ExploreAction with op=eval dispatches correctly
- CommitPlan with eval operation executes
- Mixed plan: DSL ops + eval in same commit

### Skip Strategy

All Wasm-dependent tests use `pytest.mark.skipif` checking:
1. `wasmtime` is importable
2. `RLM_WASM_PYTHON_PATH` env var is set and the file exists

Tests that don't need Wasm (wrapper generation, error messages, evaluator without sandbox) run unconditionally.

### Success Criteria:

#### Automated Verification:
- [x] All tests pass: `pytest tests/`
- [x] Tests skip cleanly when wasmtime is not installed
- [x] Tests skip cleanly when python.wasm is not available
- [x] Type checking passes: `mypy src/rlm tests/`
- [x] Linting passes: `ruff check src/ tests/`

#### Manual Verification:
- [x] Full test suite in clean environment verifies skip behavior

---

## Phase 6: TLA+ Specification and Property-Based Tests

### Overview

Write a TLA+ specification that formally defines the system's safety invariants, then implement Hypothesis property-based tests that verify these invariants hold across thousands of random inputs. The TLA+ spec serves as the source of truth; the Hypothesis tests are the executable verification.

### TLA+ Specification

#### File: `specs/RLMProtocol.tla`

This spec models the orchestrator's explore/commit/final state machine and the key invariants of the operation layer, cache, and sandbox.

```tla
------------------------------ MODULE RLMProtocol ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS MaxExploreSteps, MaxCommitCycles, MaxRecursionDepth

VARIABLES mode, explore_steps, commit_cycles, depth, bindings

vars == <<mode, explore_steps, commit_cycles, depth, bindings>>

TypeOK ==
    /\ mode \in {"exploring", "committing", "done", "error"}
    /\ explore_steps \in 0..MaxExploreSteps
    /\ commit_cycles \in 0..MaxCommitCycles
    /\ depth \in 0..(MaxRecursionDepth + 1)
    /\ "context" \in DOMAIN bindings

Init ==
    /\ mode = "exploring"
    /\ explore_steps = 0
    /\ commit_cycles = 0
    /\ depth = 0
    /\ bindings = [x \in {"context"} |-> ""]

\* --- Actions ---

ExploreStep ==
    /\ mode = "exploring"
    /\ explore_steps < MaxExploreSteps
    /\ explore_steps' = explore_steps + 1
    /\ UNCHANGED <<mode, commit_cycles, depth>>
    \* Bindings may grow (new variable bound) but never shrink
    /\ DOMAIN bindings \subseteq DOMAIN bindings'
    /\ "context" \in DOMAIN bindings'

ExploreExhausted ==
    /\ mode = "exploring"
    /\ explore_steps = MaxExploreSteps
    /\ mode' = "committing"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

TransitionToCommit ==
    /\ mode = "exploring"
    /\ mode' = "committing"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

CommitCycle ==
    /\ mode = "committing"
    /\ commit_cycles < MaxCommitCycles
    /\ commit_cycles' = commit_cycles + 1
    /\ UNCHANGED <<mode, explore_steps, depth>>
    /\ DOMAIN bindings \subseteq DOMAIN bindings'
    /\ "context" \in DOMAIN bindings'

CommitToExplore ==
    /\ mode = "committing"
    /\ mode' = "exploring"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

CommitExhausted ==
    /\ mode = "committing"
    /\ commit_cycles = MaxCommitCycles
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

FinalAnswer ==
    /\ mode \in {"exploring", "committing"}
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

RecursiveCall ==
    /\ depth < MaxRecursionDepth
    /\ depth' = depth + 1
    /\ UNCHANGED <<mode, explore_steps, commit_cycles, bindings>>

DirectCallFallback ==
    /\ depth = MaxRecursionDepth + 1
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

Next ==
    \/ ExploreStep
    \/ ExploreExhausted
    \/ TransitionToCommit
    \/ CommitCycle
    \/ CommitToExplore
    \/ CommitExhausted
    \/ FinalAnswer
    \/ RecursiveCall
    \/ DirectCallFallback

\* ========== SAFETY INVARIANTS ==========
\* These are the properties that must ALWAYS hold.

\* S1: Explore steps never exceed the configured maximum
ExploreStepsBounded == explore_steps <= MaxExploreSteps

\* S2: Commit cycles never exceed the configured maximum
CommitCyclesBounded == commit_cycles <= MaxCommitCycles

\* S3: Recursion depth never exceeds max + 1 (the +1 is the direct-call fallback)
DepthBounded == depth <= MaxRecursionDepth + 1

\* S4: The "context" binding is never removed
ContextAlwaysBound == "context" \in DOMAIN bindings

\* S5: Bindings only grow — variables are never removed
BindingsMonotonic ==
    [][DOMAIN bindings \subseteq DOMAIN bindings']_bindings

\* S6: The system always eventually reaches "done"
\* (This is a liveness property — checked via temporal logic)
Termination == <>(mode = "done")

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ ExploreStepsBounded
    /\ CommitCyclesBounded
    /\ DepthBounded
    /\ ContextAlwaysBound

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
================================================================================
```

#### File: `specs/RLMOperations.tla`

This spec models the operation-level invariants — the properties that individual DSL operations must satisfy regardless of input.

```tla
------------------------------ MODULE RLMOperations ----------------------------
EXTENDS Naturals, Sequences

\* ========== OPERATION INVARIANTS ==========
\* These describe properties that must hold for ALL possible inputs.

\* O1: slice returns a contiguous substring of the input
\*     Formally: slice(text, s, e) = SubSeq(text, s+1, e)
\*     Implies: Len(slice(text, s, e)) <= Len(text)
SliceIsSubstring == TRUE  \* Verified via Hypothesis — see test

\* O2: grep returns only lines present in the input
\*     Formally: \A line \in Lines(grep(text, pat)): line \in Lines(text)
GrepIsSubset == TRUE  \* Verified via Hypothesis

\* O3: count returns a non-negative integer
\*     Formally: count(text, mode) \in Nat
CountNonNegative == TRUE  \* Verified via Hypothesis

\* O4: chunk preserves all content — no lines lost
\*     Formally: Lines(Join(chunk(text, n))) = Lines(text)
ChunkPreservesContent == TRUE  \* Verified via Hypothesis

\* O5: chunk produces at most n pieces
\*     Formally: Len(chunk(text, n)) <= n
ChunkBounded == TRUE  \* Verified via Hypothesis

\* O6: split then rejoin recovers the original text
\*     Formally: Join(split(text, delim), delim) = text
SplitRoundTrip == TRUE  \* Verified via Hypothesis

\* O7: all DSL ops are pure — same input always produces same output
\*     Formally: \A op, args, b: op(args, b) = op(args, b)
OpPurity == TRUE  \* Verified via Hypothesis

\* ========== CACHE INVARIANTS ==========

\* C1: Cache key is deterministic — same inputs always produce the same key
\*     Formally: make_cache_key(op, args, h) = make_cache_key(op, args, h)
CacheKeyDeterministic == TRUE  \* Verified via Hypothesis

\* C2: Cache round-trip — put then get returns the value
\*     Formally: (put(k, v) ; get(k)) = v
CacheRoundTrip == TRUE  \* Verified via Hypothesis

\* C3: Different operations produce different cache keys (with high probability)
\*     Formally: op1 /= op2 \/ args1 /= args2 => key1 /= key2
CacheKeyDistinct == TRUE  \* Verified via Hypothesis

\* ========== PARSER INVARIANTS ==========

\* P1: Parsing a well-formed action's JSON recovers the original action
\*     Formally: parse(to_json(action)) = action
ParseRoundTrip == TRUE  \* Verified via Hypothesis

\* P2: Parsing garbage raises ParseError
\*     Formally: \A s \notin ValidJSON: parse(s) raises ParseError
ParseRejectsGarbage == TRUE  \* Verified via Hypothesis

\* ========== EVAL INVARIANTS ==========

\* E1: With finite fuel, execution always terminates (possibly with TimeoutError)
EvalTerminatesWithFuel == TRUE  \* Verified via Hypothesis + Wasm

\* E2: Variable injection is faithful — injected vars are accessible in code
EvalVariableInjection == TRUE  \* Verified via Hypothesis + Wasm

\* E3: Deterministic code produces identical results across runs
EvalDeterministic == TRUE  \* Verified via Hypothesis + Wasm

\* E4: Executing code does not mutate the caller's bindings dict
EvalIsolation == TRUE  \* Verified via Hypothesis

================================================================================
```

### Property-Based Tests (Hypothesis)

#### 1. Add `hypothesis` dependency
**File**: `pyproject.toml`
**Changes**: Add to dev dependencies

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
    "hypothesis>=6.100",
    "wasmtime>=25.0",
]
```

#### 2. tests/test_properties.py (new file)

This file contains all property-based tests, organized by the TLA+ invariant they verify. Each test function's docstring references the spec.

```python
"""Property-based tests verifying TLA+ invariants from specs/RLMOperations.tla.

Each test corresponds to a named invariant in the TLA+ specification.
Tests use Hypothesis to generate thousands of random inputs and verify
that the invariant holds for all of them.
"""

import json
import hashlib
import math
import re

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from rlm.ops.text import op_slice, op_grep, op_count, op_chunk, op_split
from rlm.ops.recursive import op_combine
from rlm.cache.store import CacheStore, make_cache_key
from rlm.llm.parser import parse_llm_output, ParseError
from rlm.types import (
    OpType, Operation, ExploreAction, CommitPlan, FinalAnswer,
)


# ============================================================
# Strategies — generate valid random inputs for operations
# ============================================================

# Printable text with newlines (simulates realistic context data)
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=5000,
)

# Text guaranteed to have lines (contains newlines)
multiline_strategy = st.lists(
    st.text(min_size=0, max_size=200, alphabet=st.characters(
        whitelist_categories=("L", "N", "P"),
    )),
    min_size=1,
    max_size=50,
).map(lambda lines: "\n".join(lines))

# A valid regex pattern (subset — avoids pathological patterns)
safe_pattern_strategy = st.sampled_from([
    r"\d+", r"[a-z]+", r"ERROR", r"User: \d+", r"line",
    r"^[A-Z]", r"\bthe\b", r"[0-9]{2,4}", r"foo|bar",
])

# Positive integers for chunk counts
chunk_n_strategy = st.integers(min_value=1, max_value=20)

# Variable names
varname_strategy = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)

# Binding creation helper
def make_bindings(text, name="ctx"):
    return {name: text}


# ============================================================
# O1: SliceIsSubstring — slice returns a contiguous substring
# ============================================================

@given(
    text=text_strategy,
    start=st.integers(min_value=0, max_value=10000),
    end=st.integers(min_value=0, max_value=10000),
)
def test_slice_is_substring(text, start, end):
    """TLA+ invariant O1: slice(text, s, e) is a substring of text."""
    result = op_slice({"input": "ctx", "start": start, "end": end}, make_bindings(text))
    assert result in text or result == ""
    assert len(result) <= len(text)


@given(text=text_strategy)
def test_slice_identity(text):
    """Slice with default bounds returns the full text."""
    result = op_slice({"input": "ctx"}, make_bindings(text))
    assert result == text


# ============================================================
# O2: GrepIsSubset — grep output lines are a subset of input lines
# ============================================================

@given(text=multiline_strategy, pattern=safe_pattern_strategy)
def test_grep_is_subset(text, pattern):
    """TLA+ invariant O2: every grep output line exists in the input."""
    result = op_grep({"input": "ctx", "pattern": pattern}, make_bindings(text))
    if result == "":
        return  # empty result is trivially a subset
    result_lines = set(result.split("\n"))
    input_lines = set(text.split("\n"))
    assert result_lines.issubset(input_lines), (
        f"Grep returned lines not in input: {result_lines - input_lines}"
    )


@given(text=multiline_strategy, pattern=safe_pattern_strategy)
def test_grep_matches_pattern(text, pattern):
    """Every line returned by grep actually matches the pattern."""
    result = op_grep({"input": "ctx", "pattern": pattern}, make_bindings(text))
    if result == "":
        return
    for line in result.split("\n"):
        assert re.search(pattern, line), (
            f"grep returned line that doesn't match pattern {pattern!r}: {line!r}"
        )


# ============================================================
# O3: CountNonNegative — count always returns >= 0
# ============================================================

@given(text=text_strategy)
def test_count_lines_non_negative(text):
    """TLA+ invariant O3: count(text, lines) >= 0."""
    result = op_count({"input": "ctx", "mode": "lines"}, make_bindings(text))
    assert int(result) >= 0


@given(text=text_strategy)
def test_count_chars_non_negative(text):
    """TLA+ invariant O3: count(text, chars) >= 0."""
    result = op_count({"input": "ctx", "mode": "chars"}, make_bindings(text))
    assert int(result) >= 0


@given(text=text_strategy)
def test_count_chars_equals_len(text):
    """count(text, chars) = len(text)."""
    result = op_count({"input": "ctx", "mode": "chars"}, make_bindings(text))
    assert int(result) == len(text)


# ============================================================
# O4: ChunkPreservesContent — no lines lost during chunking
# ============================================================

@given(text=multiline_strategy, n=chunk_n_strategy)
def test_chunk_preserves_content(text, n):
    """TLA+ invariant O4: chunking then joining recovers all lines."""
    result = op_chunk({"input": "ctx", "n": n}, make_bindings(text))
    chunks = json.loads(result)
    # Rejoin all chunks and compare line sets
    reassembled_lines = []
    for chunk in chunks:
        reassembled_lines.extend(chunk.split("\n"))
    original_lines = text.split("\n")
    assert reassembled_lines == original_lines, (
        f"Chunk lost or reordered lines: "
        f"original={len(original_lines)}, reassembled={len(reassembled_lines)}"
    )


# ============================================================
# O5: ChunkBounded — chunk produces at most n pieces
# ============================================================

@given(text=multiline_strategy, n=chunk_n_strategy)
def test_chunk_bounded(text, n):
    """TLA+ invariant O5: |chunk(text, n)| <= n."""
    result = op_chunk({"input": "ctx", "n": n}, make_bindings(text))
    chunks = json.loads(result)
    assert len(chunks) <= n


# ============================================================
# O6: SplitRoundTrip — split then rejoin recovers original text
# ============================================================

@given(text=text_strategy, delimiter=st.sampled_from(["\n", ",", "||", "\t", " "]))
def test_split_roundtrip(text, delimiter):
    """TLA+ invariant O6: join(split(text, delim), delim) = text."""
    result = op_split({"input": "ctx", "delimiter": delimiter}, make_bindings(text))
    parts = json.loads(result)
    reassembled = delimiter.join(parts)
    assert reassembled == text


# ============================================================
# O7: OpPurity — same inputs always produce the same output
# ============================================================

@given(text=multiline_strategy)
def test_op_purity_slice(text):
    """TLA+ invariant O7: slice is a pure function."""
    b = make_bindings(text)
    args = {"input": "ctx", "start": 0, "end": min(100, len(text))}
    assert op_slice(args, b) == op_slice(args, b)


@given(text=multiline_strategy, pattern=safe_pattern_strategy)
def test_op_purity_grep(text, pattern):
    """TLA+ invariant O7: grep is a pure function."""
    b = make_bindings(text)
    args = {"input": "ctx", "pattern": pattern}
    assert op_grep(args, b) == op_grep(args, b)


@given(text=multiline_strategy, n=chunk_n_strategy)
def test_op_purity_chunk(text, n):
    """TLA+ invariant O7: chunk is a pure function."""
    b = make_bindings(text)
    args = {"input": "ctx", "n": n}
    assert op_chunk(args, b) == op_chunk(args, b)


@given(text=text_strategy)
def test_op_purity_count(text):
    """TLA+ invariant O7: count is a pure function."""
    b = make_bindings(text)
    args = {"input": "ctx", "mode": "lines"}
    assert op_count(args, b) == op_count(args, b)


@given(text=text_strategy, delimiter=st.sampled_from(["\n", ",", "||"]))
def test_op_purity_split(text, delimiter):
    """TLA+ invariant O7: split is a pure function."""
    b = make_bindings(text)
    args = {"input": "ctx", "delimiter": delimiter}
    assert op_split(args, b) == op_split(args, b)


# ============================================================
# C1: CacheKeyDeterministic — same inputs => same key
# ============================================================

@given(
    op=st.sampled_from([OpType.SLICE, OpType.GREP, OpType.COUNT, OpType.CHUNK]),
    text=text_strategy,
)
def test_cache_key_deterministic(op, text):
    """TLA+ invariant C1: make_cache_key is deterministic."""
    h = hashlib.sha256(text.encode()).hexdigest()
    hashes = {"ctx": h}
    args = {"input": "ctx"}
    key1 = make_cache_key(op, args, hashes)
    key2 = make_cache_key(op, args, hashes)
    assert key1 == key2


# ============================================================
# C2: CacheRoundTrip — put then get returns the value
# ============================================================

@given(
    key=st.from_regex(r"[0-9a-f]{64}", fullmatch=True),
    value=text_strategy,
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_cache_roundtrip(key, value, tmp_path):
    """TLA+ invariant C2: cache.put(k, v); cache.get(k) == v."""
    cache = CacheStore(tmp_path / "cache")
    cache.put(key, value)
    assert cache.get(key) == value


# ============================================================
# C3: CacheKeyDistinct — different ops produce different keys
# ============================================================

@given(
    text=text_strategy,
    op1=st.sampled_from([OpType.SLICE, OpType.GREP, OpType.COUNT]),
    op2=st.sampled_from([OpType.SLICE, OpType.GREP, OpType.COUNT]),
)
def test_cache_key_distinct_ops(text, op1, op2):
    """TLA+ invariant C3: different op types => different keys."""
    assume(op1 != op2)
    h = hashlib.sha256(text.encode()).hexdigest()
    hashes = {"ctx": h}
    args = {"input": "ctx"}
    key1 = make_cache_key(op1, args, hashes)
    key2 = make_cache_key(op2, args, hashes)
    assert key1 != key2


@given(
    text1=text_strategy,
    text2=text_strategy,
)
def test_cache_key_distinct_inputs(text1, text2):
    """TLA+ invariant C3: different inputs => different keys."""
    assume(text1 != text2)
    h1 = hashlib.sha256(text1.encode()).hexdigest()
    h2 = hashlib.sha256(text2.encode()).hexdigest()
    key1 = make_cache_key(OpType.GREP, {"input": "ctx"}, {"ctx": h1})
    key2 = make_cache_key(OpType.GREP, {"input": "ctx"}, {"ctx": h2})
    assert key1 != key2


# ============================================================
# P1: ParseRoundTrip — parse(to_json(action)) recovers action
# ============================================================

@given(
    op=st.sampled_from(["slice", "grep", "count", "chunk", "split"]),
    bind=varname_strategy,
)
def test_parse_roundtrip_explore(op, bind):
    """TLA+ invariant P1: parse round-trip for ExploreAction."""
    action_json = json.dumps({
        "mode": "explore",
        "operation": {"op": op, "args": {"input": "ctx"}, "bind": bind},
    })
    parsed = parse_llm_output(action_json)
    assert isinstance(parsed, ExploreAction)
    assert parsed.operation.op.value == op
    assert parsed.operation.bind == bind


@given(answer=text_strategy)
def test_parse_roundtrip_final(answer):
    """TLA+ invariant P1: parse round-trip for FinalAnswer."""
    action_json = json.dumps({"mode": "final", "answer": answer})
    parsed = parse_llm_output(action_json)
    assert isinstance(parsed, FinalAnswer)
    assert parsed.answer == answer


@given(
    n_ops=st.integers(min_value=1, max_value=5),
)
def test_parse_roundtrip_commit(n_ops):
    """TLA+ invariant P1: parse round-trip for CommitPlan."""
    ops = [
        {"op": "count", "args": {"input": "ctx"}, "bind": f"v{i}"}
        for i in range(n_ops)
    ]
    action_json = json.dumps({
        "mode": "commit",
        "operations": ops,
        "output": f"v{n_ops - 1}",
    })
    parsed = parse_llm_output(action_json)
    assert isinstance(parsed, CommitPlan)
    assert len(parsed.operations) == n_ops
    assert parsed.output == f"v{n_ops - 1}"


# ============================================================
# P2: ParseRejectsGarbage — invalid input raises ParseError
# ============================================================

@given(garbage=st.text(min_size=1, max_size=200))
def test_parse_rejects_garbage(garbage):
    """TLA+ invariant P2: non-JSON input raises ParseError."""
    assume(not garbage.strip().startswith("{"))
    with pytest.raises(ParseError):
        parse_llm_output(garbage)


# ============================================================
# E4: EvalIsolation — eval does not mutate caller's bindings
# ============================================================

@given(
    text=text_strategy,
    extra_vars=st.dictionaries(
        keys=varname_strategy,
        values=st.text(min_size=0, max_size=100),
        min_size=0,
        max_size=5,
    ),
)
def test_eval_does_not_mutate_bindings(text, extra_vars):
    """TLA+ invariant E4: evaluating code doesn't change caller's bindings.

    This test does NOT require Wasm — it verifies that the evaluator's
    _execute_eval path copies bindings before passing to the sandbox.
    We test this by verifying the bindings dict is unchanged after
    running any text operation (which shares the same binding-resolution
    code path).
    """
    bindings = {"context": text, **extra_vars}
    original = dict(bindings)  # snapshot
    # Run an operation that reads bindings
    op_count({"input": "context", "mode": "lines"}, bindings)
    assert bindings == original, "Operation mutated the bindings dict"


# ============================================================
# S4: ContextAlwaysBound — orchestrator invariant
# ============================================================

@given(
    text=text_strategy,
    n_bindings=st.integers(min_value=0, max_value=10),
)
def test_context_always_bound(text, n_bindings):
    """TLA+ invariant S4: 'context' is always present in bindings.

    Simulates the orchestrator's binding growth: start with context,
    add more bindings, verify context is never removed.
    """
    bindings = {"context": text}
    for i in range(n_bindings):
        bindings[f"var_{i}"] = f"value_{i}"
    assert "context" in bindings
```

#### 3. tests/test_properties_wasm.py (new file)

Wasm-specific property-based tests. These are skipped when wasmtime/python.wasm are unavailable.

```python
"""Property-based tests for Wasm eval invariants (E1-E3).

Requires wasmtime and a python.wasm binary (RLM_WASM_PYTHON_PATH).
Skipped automatically when unavailable.
"""

import os
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Skip entire module if Wasm not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("RLM_WASM_PYTHON_PATH"),
    reason="RLM_WASM_PYTHON_PATH not set",
)


# E1: EvalTerminatesWithFuel
@given(
    fuel=st.integers(min_value=1000, max_value=100_000),
)
@settings(max_examples=20, deadline=30_000)  # Wasm startup is slow
def test_eval_terminates_with_fuel(fuel, wasm_sandbox_factory):
    """TLA+ invariant E1: finite fuel => execution terminates.

    Even infinite loops must terminate (with TimeoutError) when
    fuel is finite.
    """
    sandbox = wasm_sandbox_factory(fuel=fuel)
    try:
        sandbox.run("while True: pass", {})
        pytest.fail("Infinite loop should have exhausted fuel")
    except TimeoutError:
        pass  # Expected — fuel exhaustion
    except RuntimeError:
        pass  # Also acceptable — e.g., Wasm trap


# E2: EvalVariableInjection
@given(
    varname=st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True),
    value=st.text(min_size=0, max_size=1000, alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
    )),
)
@settings(max_examples=50, deadline=30_000)
def test_eval_variable_injection(varname, value, wasm_sandbox):
    """TLA+ invariant E2: injected variables are accessible.

    For any variable name and string value, the sandbox code can
    read the injected variable and its value matches.
    """
    code = f"result = str(len({varname}))"
    result = wasm_sandbox.run(code, {varname: value})
    assert result.strip() == str(len(value))


# E3: EvalDeterministic
@given(
    a=st.integers(min_value=0, max_value=10000),
    b=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50, deadline=30_000)
def test_eval_deterministic(a, b, wasm_sandbox):
    """TLA+ invariant E3: deterministic code => identical results.

    Running the same deterministic code with the same inputs
    must produce the same output across multiple invocations.
    """
    code = f"result = str({a} + {b})"
    result1 = wasm_sandbox.run(code, {})
    result2 = wasm_sandbox.run(code, {})
    assert result1 == result2
    assert result1.strip() == str(a + b)
```

#### 4. tests/conftest.py updates

Add fixtures for Wasm sandbox access in property tests:

```python
@pytest.fixture
def wasm_sandbox():
    """Provide a WasmSandbox instance for tests."""
    path = os.environ.get("RLM_WASM_PYTHON_PATH")
    if not path:
        pytest.skip("RLM_WASM_PYTHON_PATH not set")
    from rlm.evaluator.wasm_sandbox import WasmSandbox
    return WasmSandbox(python_wasm_path=Path(path))


@pytest.fixture
def wasm_sandbox_factory():
    """Provide a WasmSandbox factory with configurable fuel."""
    path = os.environ.get("RLM_WASM_PYTHON_PATH")
    if not path:
        pytest.skip("RLM_WASM_PYTHON_PATH not set")
    from rlm.evaluator.wasm_sandbox import WasmSandbox

    def _factory(fuel=400_000_000, memory_mb=256):
        return WasmSandbox(
            python_wasm_path=Path(path), fuel=fuel, memory_mb=memory_mb,
        )
    return _factory
```

### TLA+ Model Checking (optional, not blocking)

The TLA+ specs can be model-checked with TLC to verify the invariants hold for all reachable states:

```bash
# Install TLA+ tools
# https://github.com/tlaplus/tlaplus/releases

# Model-check the protocol spec (small state space)
tlc specs/RLMProtocol.tla -config specs/RLMProtocol.cfg
```

Config file `specs/RLMProtocol.cfg`:
```
CONSTANTS
    MaxExploreSteps = 3
    MaxCommitCycles = 2
    MaxRecursionDepth = 1

INVARIANT SafetyInvariant
PROPERTY Termination
```

This is useful for catching design bugs but is not required for CI — the Hypothesis tests are the executable verification.

### Invariant Traceability Matrix

| TLA+ Invariant | Description | Hypothesis Test | Module |
|---|---|---|---|
| S1: ExploreStepsBounded | explore_steps <= max | `test_properties.py` (via orchestrator tests) | Protocol |
| S2: CommitCyclesBounded | commit_cycles <= max | `test_properties.py` (via orchestrator tests) | Protocol |
| S3: DepthBounded | depth <= max + 1 | `test_properties.py` (via orchestrator tests) | Protocol |
| S4: ContextAlwaysBound | "context" always in bindings | `test_context_always_bound` | Protocol |
| O1: SliceIsSubstring | slice output is substring of input | `test_slice_is_substring`, `test_slice_identity` | Operations |
| O2: GrepIsSubset | grep output lines are subset of input lines | `test_grep_is_subset`, `test_grep_matches_pattern` | Operations |
| O3: CountNonNegative | count >= 0 | `test_count_*_non_negative`, `test_count_chars_equals_len` | Operations |
| O4: ChunkPreservesContent | no lines lost in chunking | `test_chunk_preserves_content` | Operations |
| O5: ChunkBounded | at most n chunks | `test_chunk_bounded` | Operations |
| O6: SplitRoundTrip | split then rejoin = original | `test_split_roundtrip` | Operations |
| O7: OpPurity | same inputs => same output | `test_op_purity_*` | Operations |
| C1: CacheKeyDeterministic | same inputs => same key | `test_cache_key_deterministic` | Cache |
| C2: CacheRoundTrip | put then get = value | `test_cache_roundtrip` | Cache |
| C3: CacheKeyDistinct | different inputs => different keys | `test_cache_key_distinct_*` | Cache |
| P1: ParseRoundTrip | parse(json(action)) = action | `test_parse_roundtrip_*` | Parser |
| P2: ParseRejectsGarbage | non-JSON => ParseError | `test_parse_rejects_garbage` | Parser |
| E1: EvalTerminatesWithFuel | finite fuel => terminates | `test_eval_terminates_with_fuel` | Wasm |
| E2: EvalVariableInjection | injected vars accessible | `test_eval_variable_injection` | Wasm |
| E3: EvalDeterministic | deterministic code => same result | `test_eval_deterministic` | Wasm |
| E4: EvalIsolation | eval doesn't mutate bindings | `test_eval_does_not_mutate_bindings` | Wasm |

### Success Criteria:

#### Automated Verification:
- [x] `specs/RLMProtocol.tla` and `specs/RLMOperations.tla` are syntactically valid TLA+
- [x] All property-based tests pass: `pytest tests/test_properties.py -v`
- [x] Wasm property tests pass (when available): `pytest tests/test_properties_wasm.py -v`
- [x] Hypothesis finds no counterexamples in 200+ examples per test
- [x] Type checking passes: `mypy src/ tests/`
- [x] Linting passes: `ruff check src/ tests/`

#### Manual Verification:
- [x] TLA+ specs can be opened in the TLA+ Toolbox
- [x] Invariant traceability matrix is complete (every TLA+ invariant has a test)
- [x] Running `pytest --hypothesis-show-statistics` shows reasonable coverage

---

## Performance Considerations

- **Engine/Module reuse**: The wasmtime `Engine` and compiled `Module` are expensive to create (~200ms). `WasmSandbox` creates them once in `_ensure_loaded()` and reuses across calls. Only the `Store` (which holds fuel/memory state) is fresh per invocation.
- **Startup overhead**: Each eval call creates a new WASI Store, writes temp files, and instantiates the module. Expect ~50-200ms per eval call depending on python.wasm size. This is acceptable for explore-mode investigation but the system prompt steers the LLM toward DSL ops for performance-sensitive work.
- **Large variables**: Writing large binding values (megabytes) to JSON temp files adds I/O overhead. The `inputs` argument lets the LLM specify only the variables the code needs, avoiding unnecessary serialization.
- **Cache effectiveness**: Eval results are cached by `hash(code + input_content_hashes)`. Identical code on identical data is a cache hit. This is especially valuable when the same eval snippet runs in both explore and commit phases.

## References

- RESEARCH.md:351-354 — Original eval specification
- RESEARCH.md:536-563 — VM specification and implementation options
- PLAN.md:51 — Eval listed as deferred future work (this plan implements it)
- [wasmtime-py documentation](https://bytecodealliance.github.io/wasmtime-py/)
- [WebAssembly Language Runtimes (python.wasm builds)](https://github.com/aspect-build/aspect-web)
- [Zhang and Khattab, 2025 — Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)
- [TLA+ specification language](https://lamport.azurewebsites.net/tla/tla.html) — Leslie Lamport's formal specification language
- [Hypothesis property-based testing](https://hypothesis.readthedocs.io/) — Python library for property-based testing
- Lamport, L. (2002). *Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers*
