# Execution Trace Implementation Plan

## Overview

Add a `--trace PATH` CLI option that writes a detailed JSON trace of the entire RLM execution. The trace captures every LLM message in/out, every explore step with operation and result, every commit cycle with all operations, and the full recursion tree when child orchestrators are spawned. This gives the user a complete post-run record of how the query was approached.

## Current State

- No trace/logging infrastructure exists. All output is ephemeral Rich console prints.
- `TimingProfile` in `src/rlm/timing.py` is the closest pattern: a shared collector passed to components, thread-safe, no-op when disabled.
- `LLMClient.send()` already tracks per-call token counts and elapsed time (from the recent verbose change).
- Child orchestrators are already tracked in `self.child_orchestrators` and merged via `get_total_profile()` / `get_total_token_usage()`.
- All types are Pydantic v2 BaseModel with `model_dump_json()`.

## Desired End State

Running `rlm run -q "..." -c context.txt --trace trace.json` produces a JSON file containing:
- A tree of `OrchestratorTrace` nodes mirroring the recursion structure
- Each node contains ordered lists of LLM calls, explore steps, commit cycles, and the final answer
- LLM calls include the full user message and assistant response, plus tokens and timing
- Explore steps include the operation, full result value, and cache status
- Commit cycles include each operation with its result and links to child traces
- The `--trace` flag works independently of `--verbose`

Verify by: `rlm run -q "..." -c data/needle_context.txt --trace /tmp/trace.json && python3 -m json.tool /tmp/trace.json`

## What We're NOT Doing

- No streaming/live trace output (the trace file is written once at the end)
- No truncation of values in the trace (user explicitly wants full data)
- No changes to the existing verbose console output
- No OpenTelemetry or structured logging integration
- No trace viewer UI

## Implementation Approach

One new file (`src/rlm/trace.py`) with Pydantic data models and a `TraceCollector` class. Surgical changes to 3 existing files to wire it in. The `TraceCollector` follows the same shared-collector pattern as `TimingProfile`.

Each orchestrator creates its own `OrchestratorTrace` node. The `TraceCollector` is shared across the entire tree (passed to children). When a child orchestrator finishes, its trace node is appended to the parent's `children` list. The LLM client records call events directly onto the owning orchestrator's trace node.

---

## Phase 1: Create `src/rlm/trace.py`

### Pydantic Data Models

```python
"""Execution trace recording for RLM orchestrator runs."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class LLMCallTrace(BaseModel):
    """One LLM send/response round-trip."""
    call_number: int
    timestamp: float           # time.time() wall-clock
    elapsed_s: float
    model: str
    input_tokens: int
    output_tokens: int
    user_message: str
    assistant_message: str


class ExploreStepTrace(BaseModel):
    """One explore-mode step."""
    step_number: int
    timestamp: float
    elapsed_s: float
    operation_op: str          # OpType.value
    operation_args: dict
    operation_bind: str | None
    result_value: str
    cached: bool
    error: str | None = None


class CommitOperationTrace(BaseModel):
    """One operation within a commit plan."""
    index: int
    operation_op: str
    operation_args: dict
    operation_bind: str | None
    elapsed_s: float
    result_value: str
    error: str | None = None
    child_trace_ids: list[int] = []


class CommitCycleTrace(BaseModel):
    """One full commit cycle."""
    cycle_number: int
    timestamp: float
    output_variable: str
    operations: list[CommitOperationTrace]
    result_value: str


class FinalAnswerTrace(BaseModel):
    """The final answer event."""
    timestamp: float
    answer: str
    total_explore_steps: int
    total_commit_cycles: int


class OrchestratorTrace(BaseModel):
    """Trace for one orchestrator run() invocation."""
    trace_id: int
    depth: int
    query: str
    context_length: int
    model: str
    elapsed_s: float = 0.0
    llm_calls: list[LLMCallTrace] = []
    explore_steps: list[ExploreStepTrace] = []
    commit_cycles: list[CommitCycleTrace] = []
    final_answer: FinalAnswerTrace | None = None
    children: list[OrchestratorTrace] = []


class ExecutionTrace(BaseModel):
    """Top-level trace document."""
    version: str = "1.0"
    timestamp: str
    root: OrchestratorTrace
```

### TraceCollector Class

```python
class TraceCollector:
    """Thread-safe trace event collector. No-op when disabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._next_id = 0

    def next_trace_id(self) -> int:
        if not self.enabled:
            return -1
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            return tid

    def record_llm_call(self, node: OrchestratorTrace, *, call_number: int,
                         elapsed_s: float, model: str, input_tokens: int,
                         output_tokens: int, user_message: str,
                         assistant_message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            node.llm_calls.append(LLMCallTrace(
                call_number=call_number, timestamp=time.time(),
                elapsed_s=elapsed_s, model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                user_message=user_message, assistant_message=assistant_message,
            ))

    def record_explore_step(self, node: OrchestratorTrace, *, step_number: int,
                             elapsed_s: float, op_type: str, op_args: dict,
                             op_bind: str | None, result_value: str,
                             cached: bool, error: str | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            node.explore_steps.append(ExploreStepTrace(
                step_number=step_number, timestamp=time.time(),
                elapsed_s=elapsed_s, operation_op=op_type,
                operation_args=op_args, operation_bind=op_bind,
                result_value=result_value, cached=cached, error=error,
            ))

    def record_commit_cycle(self, node: OrchestratorTrace, *, cycle_number: int,
                             output_variable: str,
                             operations: list[CommitOperationTrace],
                             result_value: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            node.commit_cycles.append(CommitCycleTrace(
                cycle_number=cycle_number, timestamp=time.time(),
                output_variable=output_variable,
                operations=operations, result_value=result_value,
            ))

    def record_final_answer(self, node: OrchestratorTrace, *, answer: str,
                             explore_steps: int, commit_cycles: int) -> None:
        if not self.enabled:
            return
        node.final_answer = FinalAnswerTrace(
            timestamp=time.time(), answer=answer,
            total_explore_steps=explore_steps,
            total_commit_cycles=commit_cycles,
        )

    @staticmethod
    def write_trace(trace: ExecutionTrace, path: Path) -> None:
        path.write_text(trace.model_dump_json(indent=2))
```

### Success Criteria
- [x] `uv run python -c "from rlm.trace import TraceCollector, ExecutionTrace"` succeeds
- [x] `uv run mypy src/rlm/trace.py` passes

---

## Phase 2: Wire trace into `LLMClient`

**File:** `src/rlm/llm/client.py`

### Changes

1. Add `trace_collector` and `trace_node` params to `__init__`:
   ```python
   def __init__(self, config: RLMConfig, profile: TimingProfile | None = None,
                verbose: bool = False, console: Console | None = None,
                trace: TraceCollector | None = None,
                trace_node: OrchestratorTrace | None = None):
       # ... existing ...
       self._trace = trace or TraceCollector()
       self._trace_node = trace_node
   ```

2. At end of `send()`, after `assistant_message` is set but before `return`, add:
   ```python
   if self._trace_node is not None:
       self._trace.record_llm_call(
           self._trace_node,
           call_number=call_num, elapsed_s=elapsed, model=self.config.model,
           input_tokens=call_in, output_tokens=call_out,
           user_message=user_message, assistant_message=assistant_message,
       )
   ```

### Success Criteria
- [x] `uv run mypy src/rlm/llm/client.py` passes
- [x] Existing tests still pass: `uv run pytest tests/ --ignore=tests/test_orchestrator.py -q`

---

## Phase 3: Wire trace into orchestrator

**File:** `src/rlm/orchestrator.py`

### 3a. Constructor changes

Add `trace_collector` param. Create `trace_node`. Pass both to LLMClient.

```python
def __init__(self, config: RLMConfig, parent: "RLMOrchestrator | None" = None,
             trace_collector: TraceCollector | None = None):
    self.config = config
    self.trace_collector = trace_collector or TraceCollector()
    self.trace_node = OrchestratorTrace(
        trace_id=self.trace_collector.next_trace_id(),
        depth=0, query="", context_length=0, model=config.model,
    )
    self.profile = TimingProfile(enabled=config.verbose)
    self.console = Console(stderr=True)
    self.llm = LLMClient(config, profile=self.profile,
                          verbose=config.verbose, console=self.console,
                          trace=self.trace_collector,
                          trace_node=self.trace_node)
    # ... rest unchanged ...
```

### 3b. `run()` changes

At top of run(), set trace metadata and capture start time:
```python
self.trace_node.depth = depth
self.trace_node.query = query
self.trace_node.context_length = len(context_text)
run_start = time.monotonic()
```

**FinalAnswer branch** — before `return action.answer`:
```python
self.trace_collector.record_final_answer(
    self.trace_node, answer=action.answer,
    explore_steps=explore_steps, commit_cycles=commit_cycles,
)
self.trace_node.elapsed_s = time.monotonic() - run_start
```

**ExploreAction branch** — after evaluator execute succeeds (after verbose output):
```python
self.trace_collector.record_explore_step(
    self.trace_node, step_number=explore_steps,
    elapsed_s=step_elapsed, op_type=op.op.value,
    op_args=op.args, op_bind=op.bind,
    result_value=result.value, cached=result.cached,
)
```

In the `except Exception` block:
```python
self.trace_collector.record_explore_step(
    self.trace_node, step_number=explore_steps,
    elapsed_s=time.monotonic() - step_start,
    op_type=op.op.value, op_args=op.args, op_bind=op.bind,
    result_value="", cached=False, error=str(e),
)
```

**CommitPlan branch** — `_execute_commit_plan` returns `tuple[str, list[CommitOperationTrace]]`. After the call:
```python
commit_result, op_traces = self._execute_commit_plan(action, bindings, depth)
bindings[action.output] = commit_result
self.trace_collector.record_commit_cycle(
    self.trace_node, cycle_number=commit_cycles,
    output_variable=action.output,
    operations=op_traces, result_value=commit_result,
)
```

### 3c. `_execute_commit_plan()` changes

Change return type to `tuple[str, list[CommitOperationTrace]]`. Accumulate `op_traces` list:

- For each op, after execution, append a `CommitOperationTrace`.
- For `RLM_CALL` ops, capture the last child's `trace_id` in `child_trace_ids`.
- For `MAP` ops, capture all new children's `trace_id`s.
- Return `(local_bindings[plan.output], op_traces)`.

### 3d. `_recursive_call()` changes

Pass `trace_collector` to child orchestrator. After child finishes, embed its trace:
```python
sub_orchestrator = RLMOrchestrator(self.config, parent=self,
                                    trace_collector=self.trace_collector)
self.child_orchestrators.append(sub_orchestrator)
result = sub_orchestrator.run(query, context_text, depth=depth + 1)
if self.trace_collector.enabled:
    self.trace_node.children.append(sub_orchestrator.trace_node)
return result
```

### 3e. `_direct_call()` changes

Pass trace collector and trace node to the LLMClient created here:
```python
client = LLMClient(self.config, verbose=self.config.verbose,
                    console=self.console,
                    trace=self.trace_collector,
                    trace_node=self.trace_node)
```

### 3f. New method: `get_trace()`

```python
def get_trace(self) -> ExecutionTrace:
    from datetime import datetime, timezone
    return ExecutionTrace(
        timestamp=datetime.now(timezone.utc).isoformat(),
        root=self.trace_node,
    )
```

### Success Criteria
- [x] `uv run mypy src/rlm/orchestrator.py` passes
- [x] Existing tests: `uv run pytest tests/ --ignore=tests/test_orchestrator.py -q`

---

## Phase 4: Add `--trace` CLI option

**File:** `src/rlm/cli.py`

### Changes

1. Add Click option after `--verbose`:
   ```python
   @click.option("--trace", "trace_path", type=click.Path(), default=None,
                 help="Write execution trace JSON to PATH.")
   ```

2. Add `trace_path: str | None` to `run()` signature.

3. Before orchestrator construction:
   ```python
   from rlm.trace import TraceCollector
   trace_collector = TraceCollector(enabled=trace_path is not None)
   ```

4. Pass to orchestrator:
   ```python
   orchestrator = RLMOrchestrator(config, trace_collector=trace_collector)
   ```

5. After answer is echoed, write trace:
   ```python
   if trace_path is not None:
       trace = orchestrator.get_trace()
       trace_file = Path(trace_path)
       TraceCollector.write_trace(trace, trace_file)
       console.print(f"[dim]Trace written to {trace_file}[/dim]")
   ```

### Success Criteria
- [x] `uv run rlm run --help` shows `--trace` option
- [x] `uv run pytest tests/ --ignore=tests/test_orchestrator.py -q` — all pass

---

## Phase 5: Tests

**New file:** `tests/test_trace.py`

Test cases:
1. `TraceCollector(enabled=False)` — all methods are no-ops, `next_trace_id()` returns -1
2. `TraceCollector(enabled=True)` — `next_trace_id()` returns sequential IDs
3. `record_llm_call` appends to node's `llm_calls`
4. `record_explore_step` appends to node's `explore_steps`
5. `record_commit_cycle` appends to node's `commit_cycles`
6. `record_final_answer` sets node's `final_answer`
7. `write_trace` writes valid JSON that round-trips through `ExecutionTrace.model_validate_json()`
8. Thread safety: concurrent `record_llm_call` from multiple threads
9. `OrchestratorTrace` with nested `children` serializes correctly
10. `ExecutionTrace.model_dump_json()` produces valid JSON

### Success Criteria
- [x] `uv run pytest tests/test_trace.py -v` — all pass
- [x] `uv run pytest tests/ --ignore=tests/test_orchestrator.py -q` — full suite passes
- [x] `uv run mypy src/rlm/trace.py src/rlm/llm/client.py src/rlm/orchestrator.py src/rlm/cli.py` — clean

---

## References

- Pattern to follow: `src/rlm/timing.py` (TimingProfile shared collector)
- Existing child orchestrator tree: `orchestrator.py:42-43,219-225,268-283`
- LLM call site with per-call metrics: `client.py:32-69`
- CLI entry point: `cli.py:85-146`
