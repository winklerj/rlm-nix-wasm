# Unified Trace Event List Implementation Plan

## Overview

Replace the three separate event lists (`llm_calls`, `explore_steps`, `commit_cycles`) plus the `final_answer` field on `OrchestratorTrace` with a single chronologically-ordered `events: list[TraceEvent]` using a Pydantic v2 discriminated union. This preserves the execution timeline that the current structure discards.

## Current State Analysis

`OrchestratorTrace` stores events in four separate fields (`trace.py:66-78`):
```python
llm_calls: list[LLMCallTrace] = []
explore_steps: list[ExploreStepTrace] = []
commit_cycles: list[CommitCycleTrace] = []
final_answer: FinalAnswerTrace | None = None
```

The real execution interleaves these (LLM call → explore step → LLM call → commit cycle → ...), but the JSON output groups all LLM calls together, then all explore steps, then all commit cycles. The narrative is lost.

### Key Discoveries:
- All four event classes already carry `timestamp` fields (`trace.py:15,27,52,60`), proving the original intent was to support ordering
- `TraceCollector.record_*` methods (`trace.py:105-198`) all append to type-specific lists — changing the append target is the only modification needed
- No code outside `trace.py` and `test_trace.py` accesses these list fields — `orchestrator.py` and `client.py` only call `TraceCollector.record_*()` methods
- `cli.py:130` imports only `TraceCollector`, not the event models

## Desired End State

After this plan is complete:

1. `OrchestratorTrace` has a single `events: list[TraceEvent]` field instead of four separate fields
2. Each event class has a `type` Literal discriminator field
3. `TraceEvent` is a Pydantic v2 discriminated union over the four event types
4. The trace JSON output lists events in chronological execution order
5. `ExecutionTrace.version` is bumped to `"1.1"`
6. All existing tests pass (updated to use the new structure)
7. `record_*` method signatures are unchanged — callers (`orchestrator.py`, `client.py`) need no modifications

### Verification:
```bash
python -m pytest tests/test_trace.py -v
```

## What We're NOT Doing

- **Not renaming classes** (`LLMCallTrace` stays, not `LLMCallEvent`) — minimizes churn, class names don't affect JSON output
- **Not adding convenience accessors** (e.g., `@property def llm_calls`) — YAGNI, only tests consume these
- **Not changing `CommitOperationTrace`** — it's a nested model within commit cycles, unaffected
- **Not changing `orchestrator.py` or `client.py`** — they interact through `TraceCollector` methods only

## Implementation Approach

Single phase. The change is contained to two files with no callers affected.

## Phase 1: Unified Event List

### Overview
Add type discriminators to event classes, create the union type, replace separate lists with `events`, and update the collector methods and tests.

### Changes Required:

#### 1. Add `type` discriminator to each event class
**File**: `src/rlm/trace.py`

Add a `type` field as the first field in each of the four event classes:

```python
# trace.py:1 — add imports
from typing import Annotated, Literal
from pydantic import BaseModel, Field
```

```python
class LLMCallTrace(BaseModel):
    """One LLM send/response round-trip."""
    type: Literal["llm_call"] = "llm_call"
    call_number: int
    # ... rest unchanged
```

```python
class ExploreStepTrace(BaseModel):
    """One explore-mode step."""
    type: Literal["explore_step"] = "explore_step"
    step_number: int
    # ... rest unchanged
```

```python
class CommitCycleTrace(BaseModel):
    """One full commit cycle."""
    type: Literal["commit_cycle"] = "commit_cycle"
    cycle_number: int
    # ... rest unchanged
```

```python
class FinalAnswerTrace(BaseModel):
    """The final answer event."""
    type: Literal["final_answer"] = "final_answer"
    timestamp: float
    # ... rest unchanged
```

#### 2. Create discriminated union type
**File**: `src/rlm/trace.py`

Add after `FinalAnswerTrace` and before `OrchestratorTrace`:

```python
TraceEvent = Annotated[
    LLMCallTrace | ExploreStepTrace | CommitCycleTrace | FinalAnswerTrace,
    Field(discriminator="type"),
]
```

#### 3. Replace separate lists with unified events list
**File**: `src/rlm/trace.py`

Change `OrchestratorTrace` from:
```python
class OrchestratorTrace(BaseModel):
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
    children: list["OrchestratorTrace"] = []
```

To:
```python
class OrchestratorTrace(BaseModel):
    trace_id: int
    depth: int
    query: str
    context_length: int
    model: str
    elapsed_s: float = 0.0
    events: list[TraceEvent] = []
    children: list["OrchestratorTrace"] = []
```

#### 4. Update TraceCollector methods
**File**: `src/rlm/trace.py`

Each `record_*` method changes its append target from a specific list to `node.events`:

- `record_llm_call` (line 121): `node.llm_calls.append(...)` → `node.events.append(...)`
- `record_explore_step` (line 149): `node.explore_steps.append(...)` → `node.events.append(...)`
- `record_commit_cycle` (line 174): `node.commit_cycles.append(...)` → `node.events.append(...)`
- `record_final_answer` (line 193): `node.final_answer = FinalAnswerTrace(...)` → `node.events.append(FinalAnswerTrace(...))`

Note: `record_final_answer` also needs `with self._lock:` wrapping since it's now appending to a shared list (currently it does a direct assignment without the lock).

#### 5. Bump version
**File**: `src/rlm/trace.py`

```python
class ExecutionTrace(BaseModel):
    version: str = "1.1"  # was "1.0"
```

#### 6. Update tests
**File**: `tests/test_trace.py`

Update imports to include `TraceEvent` (not strictly needed but useful for type clarity).

**`test_disabled_collector_noop`**: Change assertions from:
```python
assert node.llm_calls == []
assert node.explore_steps == []
assert node.commit_cycles == []
assert node.final_answer is None
```
To:
```python
assert node.events == []
```

**`test_record_llm_call`**: Change from:
```python
assert len(node.llm_calls) == 1
call = node.llm_calls[0]
```
To:
```python
assert len(node.events) == 1
call = node.events[0]
assert call.type == "llm_call"
```

**`test_record_explore_step`**: Change from:
```python
assert len(node.explore_steps) == 1
step = node.explore_steps[0]
```
To:
```python
assert len(node.events) == 1
step = node.events[0]
assert step.type == "explore_step"
```

**`test_record_explore_step_with_error`**: Change from:
```python
assert node.explore_steps[0].error == "KeyError: x"
```
To:
```python
assert node.events[0].error == "KeyError: x"
```

**`test_record_commit_cycle`**: Change from:
```python
assert len(node.commit_cycles) == 1
cycle = node.commit_cycles[0]
```
To:
```python
assert len(node.events) == 1
cycle = node.events[0]
assert cycle.type == "commit_cycle"
```

**`test_record_final_answer`**: Change from:
```python
assert node.final_answer is not None
assert node.final_answer.answer == "The answer is 42"
assert node.final_answer.total_explore_steps == 5
assert node.final_answer.total_commit_cycles == 2
assert node.final_answer.timestamp > 0
```
To:
```python
assert len(node.events) == 1
fa = node.events[0]
assert fa.type == "final_answer"
assert fa.answer == "The answer is 42"
assert fa.total_explore_steps == 5
assert fa.total_commit_cycles == 2
assert fa.timestamp > 0
```

**`test_execution_trace_json_roundtrip`**: Change from:
```python
node.final_answer = FinalAnswerTrace(...)
node.llm_calls.append(LLMCallTrace(...))
...
assert parsed["version"] == "1.0"
assert len(parsed["root"]["llm_calls"]) == 1
...
assert restored.root.final_answer is not None
assert restored.root.final_answer.answer == "ok"
```
To:
```python
node.events.append(LLMCallTrace(...))
node.events.append(FinalAnswerTrace(...))
...
assert parsed["version"] == "1.1"
assert len(parsed["root"]["events"]) == 2
assert parsed["root"]["events"][0]["type"] == "llm_call"
assert parsed["root"]["events"][1]["type"] == "final_answer"
...
fa = [e for e in restored.root.events if e.type == "final_answer"][0]
assert fa.answer == "ok"
```

**`test_concurrent_record_llm_calls`**: Change from:
```python
assert len(node.llm_calls) == n_threads * calls_per_thread
```
To:
```python
assert len(node.events) == n_threads * calls_per_thread
```

**`test_nested_children_serialize`** and **`test_write_trace`**: No changes needed — they don't reference the event lists.

### Success Criteria:

#### Automated Verification:
- [x] All tests pass: `python -m pytest tests/test_trace.py -v`
- [x] Full test suite still passes: `python -m pytest tests/ -v`
- [x] No import errors: `python -c "from rlm.trace import TraceEvent, OrchestratorTrace"`

#### Manual Verification:
- [ ] Run an actual `--trace` query and verify the output JSON has a single `events` list with interleaved event types in chronological order
- [ ] Verify the `version` field in the output is `"1.1"`

## References

- Research document: `thoughts/shared/research/2026-02-09-trace-event-ordering.md`
- Existing trace plan: `thoughts/shared/plans/2026-02-09-execution-trace.md`
