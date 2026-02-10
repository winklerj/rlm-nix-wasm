---
date: "2026-02-09T20:15:00+00:00"
researcher: claude
git_commit: 48581e6edf9ebc27c7646e4bafdd7b355d7f6403
branch: main
repository: rlm-secure
topic: "Trace event ordering: unified event list vs separate typed lists"
tags: [research, codebase, trace, execution-trace, data-model]
status: complete
last_updated: "2026-02-09"
last_updated_by: claude
---

# Research: Trace event ordering — unified event list vs separate typed lists

**Date**: 2026-02-09T20:15:00+00:00
**Researcher**: claude
**Git Commit**: 48581e6
**Branch**: main
**Repository**: rlm-secure

## Research Question

The `--trace` flag separates LLM calls, explore steps, and commit cycles into three parallel lists on `OrchestratorTrace`. Can we instead list all events in chronological order with a `type` discriminator?

## Summary

**Yes, and it's a clear improvement.** The current structure groups events by type into three separate lists (`llm_calls`, `explore_steps`, `commit_cycles`), which loses the execution order that is the trace's primary purpose. All events already carry a `timestamp` field, proving the original intent was to support ordering, but the structure forces consumers to merge-sort three lists to reconstruct the timeline.

A single `events: list[TraceEvent]` with a `type` discriminator field preserves chronological order naturally and is straightforward to implement with Pydantic v2's discriminated unions.

## Detailed Findings

### Current Structure (`trace.py:66-78`)

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

Three separate lists, each internally ordered but not interleaved with each other.

### What the actual execution looks like (from `trace.json`)

The real execution order for "How many rolls for Laura?" was:

| Order | Type | Timestamp |
|-------|------|-----------|
| 1 | LLM call 1 | 1770663582.79 |
| 2 | Explore step 1 (grep "Laura") | 1770663582.79 |
| 3 | LLM call 2 | 1770663602.04 |
| 4 | Commit cycle 1 (grep + count) | 1770663602.04 |
| 5 | LLM call 3 | 1770663605.52 |
| 6 | Final answer | (implicit) |

But the JSON output groups all 3 LLM calls together (lines 11-42), then all explore steps (lines 43-57), then all commit cycles (lines 59-end). The narrative is lost.

### Where events are recorded (`orchestrator.py:90-200`)

The main loop in `run()` follows a clear sequential pattern:

1. `self.llm.send(...)` → LLM call is recorded inside `client.py:75-85`
2. Parse response into action
3. If ExploreAction → execute op → `record_explore_step()` at line 149
4. If CommitPlan → execute plan → `record_commit_cycle()` at line 198
5. If FinalAnswer → `record_final_answer()` at line 115

These always happen in series: LLM call → action → LLM call → action → ... → final answer. The current separate-list structure discards this ordering.

### Proposed Structure

```python
from typing import Literal, Annotated
from pydantic import Field

class LLMCallEvent(BaseModel):
    type: Literal["llm_call"] = "llm_call"
    call_number: int
    timestamp: float
    elapsed_s: float
    model: str
    input_tokens: int
    output_tokens: int
    user_message: str
    assistant_message: str

class ExploreStepEvent(BaseModel):
    type: Literal["explore_step"] = "explore_step"
    step_number: int
    timestamp: float
    elapsed_s: float
    operation_op: str
    operation_args: dict
    operation_bind: str | None
    result_value: str
    cached: bool
    error: str | None = None

class CommitCycleEvent(BaseModel):
    type: Literal["commit_cycle"] = "commit_cycle"
    cycle_number: int
    timestamp: float
    output_variable: str
    operations: list[CommitOperationTrace]
    result_value: str

class FinalAnswerEvent(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    timestamp: float
    answer: str
    total_explore_steps: int
    total_commit_cycles: int

TraceEvent = Annotated[
    LLMCallEvent | ExploreStepEvent | CommitCycleEvent | FinalAnswerEvent,
    Field(discriminator="type"),
]

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

### Impact on TraceCollector

Each `record_*` method currently appends to a specific list. Change them all to append to `node.events` instead. The method signatures stay the same; only the append target changes. The `final_answer` special field goes away — it becomes just another event in the list.

### Impact on Tests (`tests/test_trace.py`)

All 11 tests reference the separate lists (`node.llm_calls`, `node.explore_steps`, etc.). They'd need updating to check `node.events` and filter by type. The assertions are otherwise the same.

### Impact on Consumers

Currently no code reads the trace except `write_trace` (serialization). The trace JSON is consumed externally by humans reading the file. A unified event list is **much** easier for humans to read — you just scan top-to-bottom.

### Version Bump

Bump `ExecutionTrace.version` from `"1.0"` to `"1.1"` (or `"2.0"` if we want to signal the breaking change clearly). Since the trace format is days old and has no downstream consumers, `"1.1"` is fine.

## Code References

- `src/rlm/trace.py:66-78` — `OrchestratorTrace` with separate lists
- `src/rlm/trace.py:12-63` — The four event model classes
- `src/rlm/trace.py:105-198` — `TraceCollector.record_*` methods
- `src/rlm/orchestrator.py:90-200` — Main loop showing interleaved LLM/explore/commit flow
- `src/rlm/llm/client.py:75-85` — LLM call recording
- `tests/test_trace.py` — 11 tests covering all event types and serialization

## Architecture Insights

- Pydantic v2 discriminated unions (`Field(discriminator="type")`) handle this cleanly with no performance cost — Pydantic uses the discriminator field for O(1) dispatch during validation.
- The `CommitOperationTrace` model (used inside commit cycles) is unaffected — it stays as a nested list within `CommitCycleEvent`.
- The `children` list for recursive orchestrator calls is also unaffected.
- Thread safety in `TraceCollector` remains the same: all methods still `with self._lock:` and append to a list, just one list instead of three.

## Open Questions

- **Naming**: Rename `LLMCallTrace` → `LLMCallEvent` (etc.) or keep the `*Trace` suffix? Renaming to `*Event` is cleaner with the new structure but means more churn.
- **Convenience accessors**: Add helper methods like `node.llm_calls` as a `@property` that filters `events`? Probably not — YAGNI, and the test file is the only current consumer.
