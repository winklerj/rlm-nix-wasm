# Plan: Timing Instrumentation + Verbose Output Improvements

## Context

The rlm-secure CLI's `--verbose` mode currently has two problems:
1. **No per-step timing** — only total wall-clock time is shown at the end. There's no visibility into how long each LLM call, operation, or cache lookup takes.
2. **Unclear verbose output** — Explore steps show `EXPLORE [3]: grep(...)` where `[3]` is an unlabeled step counter. Commit output shows `COMMIT [1]: 3 operations` with no detail about what those operations are.

The goal is twofold: add a timing library that measures per-component latency, AND improve verbose output so you can follow what's happening in real time.

## What We're NOT Doing

- No changes to `web/server.py` (has its own tracing via WebSocket)
- No heavy dependencies (no OpenTelemetry/Datadog)
- No separate `--timing` flag — timing integrates into `--verbose`

## Phase 1: Core Timing Module

### Overview
Create `src/rlm/timing.py` with a lightweight, thread-safe `TimingProfile` collector.

### Changes Required:

#### 1. New file: `src/rlm/timing.py` (~100 lines)

```python
"""Lightweight timing instrumentation for RLM operations."""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Iterator

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table


class TimingEntry(BaseModel):
    """A single timing measurement."""
    category: str       # "llm", "evaluator", "cache", "parse", "recursive", "parallel"
    label: str          # "send", "grep", "lookup", "parse_response", etc.
    elapsed_s: float
    metadata: dict[str, object] = {}


class CategorySummary(BaseModel):
    """Aggregated timing stats for one category."""
    category: str
    count: int
    total_s: float
    mean_s: float
    min_s: float
    max_s: float


class TimingProfile:
    """Thread-safe timing collector. No-op when disabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._entries: list[TimingEntry] = []
        self._lock = threading.Lock()
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @contextmanager
    def measure(self, category: str, label: str, **metadata: object) -> Iterator[None]:
        """Context manager that records elapsed time. No-op when disabled."""
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        entry = TimingEntry(
            category=category, label=label,
            elapsed_s=elapsed, metadata=metadata,
        )
        with self._lock:
            self._entries.append(entry)

    def record_cache_hit(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.cache_misses += 1

    def merge(self, other: TimingProfile) -> None:
        """Merge entries from a child profile."""
        with self._lock:
            self._entries.extend(other.entries)
            self.cache_hits += other.cache_hits
            self.cache_misses += other.cache_misses

    @property
    def entries(self) -> list[TimingEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> dict[str, CategorySummary]:
        """Aggregate entries by category."""
        groups: dict[str, list[float]] = defaultdict(list)
        for e in self.entries:
            groups[e.category].append(e.elapsed_s)
        result = {}
        for cat, times in groups.items():
            result[cat] = CategorySummary(
                category=cat,
                count=len(times),
                total_s=sum(times),
                mean_s=statistics.mean(times),
                min_s=min(times),
                max_s=max(times),
            )
        return result

    @staticmethod
    def print_summary(profile: TimingProfile, wall_s: float, console: Console) -> None:
        """Render timing summary as a Rich table."""
        summaries = profile.summary()
        if not summaries:
            return

        table = Table(title=f"Timing Breakdown ({wall_s:.1f}s wall clock)")
        table.add_column("Category")
        table.add_column("Count", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Wall %", justify="right")

        # Sort by total time descending
        for s in sorted(summaries.values(), key=lambda x: x.total_s, reverse=True):
            pct = (s.total_s / wall_s * 100) if wall_s > 0 else 0
            table.add_row(
                s.category, str(s.count),
                f"{s.total_s:.3f}s", f"{s.mean_s:.3f}s",
                f"{s.min_s:.3f}s", f"{s.max_s:.3f}s",
                f"{pct:.1f}%",
            )

        console.print(table)

        total_lookups = profile.cache_hits + profile.cache_misses
        if total_lookups > 0:
            hit_pct = profile.cache_hits / total_lookups * 100
            console.print(
                f"[dim]Cache: {profile.cache_hits} hits / "
                f"{total_lookups} lookups ({hit_pct:.1f}% hit rate)[/dim]"
            )
```

#### 2. New file: `tests/test_timing.py`

Unit tests for:
- `TimingProfile(enabled=False)` measure is no-op
- `TimingProfile(enabled=True)` collects entries
- `merge()` combines profiles correctly
- `summary()` aggregates by category
- Thread safety with concurrent `measure()` calls
- `cache_hits`/`cache_misses` tracking

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/test_timing.py` passes
- [ ] `mypy src/rlm/timing.py` passes

---

## Phase 2: Wire Timing Into Components

### Overview
Pass `TimingProfile` through the dependency chain and instrument the hot paths. Simultaneously improve verbose output to show live timing and better commit details.

### Changes Required:

#### 1. `src/rlm/types.py` — No config change needed
Timing activates when `verbose=True`. No new flag.

#### 2. `src/rlm/llm/client.py` — Instrument LLM calls

Add `profile` parameter and wrap `completion()`:

```python
class LLMClient:
    def __init__(self, config: RLMConfig, profile: TimingProfile | None = None):
        self.config = config
        self.messages: list[dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.profile = profile or TimingProfile()  # disabled by default

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        with self.profile.measure("llm", "send", model=self.config.model):
            response = completion(
                model=self.config.model,
                messages=self.messages,
                temperature=self.config.temperature,
            )

        # ... rest unchanged
```

#### 3. `src/rlm/evaluator/lightweight.py` — Instrument cache + ops

Add `profile` parameter and wrap hash computation, cache lookup, op execution, cache store:

```python
class LightweightEvaluator:
    def __init__(self, cache: CacheStore | None = None, profile: TimingProfile | None = None):
        self.cache = cache
        self.profile = profile or TimingProfile()

    def execute(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        executor = EXPLORE_OPS.get(op.op)
        if executor is None:
            raise ValueError(...)

        with self.profile.measure("hash", "binding_hash"):
            input_hashes = {
                k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in bindings.items()
            }
        cache_key = make_cache_key(op.op, op.args, input_hashes)

        if self.cache is not None:
            with self.profile.measure("cache", "lookup"):
                cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                self.profile.record_cache_hit()
                return OpResult(op=op.op, cache_key=cache_key, value=cached_value, cached=True)
            self.profile.record_cache_miss()

        with self.profile.measure("evaluator", op.op.value):
            result_value = executor(op.args, bindings)

        if self.cache is not None:
            with self.profile.measure("cache", "store"):
                self.cache.put(cache_key, result_value)

        return OpResult(op=op.op, cache_key=cache_key, value=result_value)
```

#### 4. `src/rlm/orchestrator.py` — Central integration + verbose improvements

This is the biggest change. Create profile, distribute it, instrument parse/recursive/parallel, AND improve verbose output.

**`__init__`**: Create and distribute `TimingProfile`:
```python
def __init__(self, config: RLMConfig, parent: "RLMOrchestrator | None" = None):
    self.config = config
    self.profile = TimingProfile(enabled=config.verbose)
    self.llm = LLMClient(config, profile=self.profile)
    self.cache = CacheStore(config.cache_dir)
    self.evaluator = LightweightEvaluator(cache=self.cache, profile=self.profile)
    # ... rest unchanged
```

**`run()` — Improved verbose output with live timing**:

Current explore output (line 107-111):
```
EXPLORE [3]: grep({"input": "context", "pattern": "error"})
```

New explore output — label the counter, show timing after execution:
```
EXPLORE step 3/20: grep(input=context, pattern="error")  (0.002s, cached)
```

Current commit output (line 140-144):
```
COMMIT [1]: 3 operations
```

New commit output — show each operation with its binding and timing:
```
COMMIT cycle 1/5: 3 operations
  → chunk(input=context, n=4) → chunks  (0.001s)
  → map(prompt="Summarize this section", input=chunks) → map_results  (12.34s)
  → combine(inputs=["map_results"]) → final  (0.001s)
  output: final
```

**Specific code changes in `run()`**:

For explore actions (replace lines 98-129):
```python
elif isinstance(action, ExploreAction):
    explore_steps += 1
    if explore_steps > self.config.max_explore_steps:
        response = self.llm.send(
            f"You have reached the maximum of {self.config.max_explore_steps} "
            f"explore steps. Please COMMIT a plan or provide a FINAL answer."
        )
        continue

    op = action.operation
    op_desc = self._format_op(op)

    try:
        step_start = time.monotonic()
        result = self.evaluator.execute(op, bindings)
        step_elapsed = time.monotonic() - step_start

        if op.bind:
            bindings[op.bind] = result.value

        if self.config.verbose:
            cache_note = ", cached" if result.cached else ""
            bind_note = f" → {op.bind}" if op.bind else ""
            self.console.print(
                f"[dim]EXPLORE step {explore_steps}/{self.config.max_explore_steps}: "
                f"{op_desc}{bind_note}  ({step_elapsed:.3f}s{cache_note})[/dim]"
            )

        display_value = result.value
        if len(display_value) > 4000:
            display_value = (
                display_value[:4000]
                + f"\n... ({len(result.value)} chars total)"
            )

        response = self.llm.send(
            f"Result of {op.op}:\n{display_value}"
        )
    except Exception as e:
        response = self.llm.send(f"Error executing {op.op}: {e}")
```

For commit actions (replace lines 131-161):
```python
elif isinstance(action, CommitPlan):
    commit_cycles += 1
    if commit_cycles > self.config.max_commit_cycles:
        response = self.llm.send(
            f"You have reached the maximum of {self.config.max_commit_cycles} "
            f"commit cycles. Please provide a FINAL answer."
        )
        continue

    if self.config.verbose:
        ops_detail = ", ".join(
            f"{op.op.value}→{op.bind}" if op.bind else op.op.value
            for op in action.operations
        )
        self.console.print(
            f"[blue]COMMIT cycle {commit_cycles}/{self.config.max_commit_cycles}: "
            f"{len(action.operations)} ops [{ops_detail}], "
            f"output={action.output}[/blue]"
        )

    try:
        commit_result = self._execute_commit_plan(action, bindings, depth)
        bindings[action.output] = commit_result

        # ... rest of commit handling unchanged
```

For `_execute_commit_plan` — add per-operation verbose output:
```python
def _execute_commit_plan(
    self, plan: CommitPlan, bindings: dict[str, str], depth: int
) -> str:
    local_bindings = dict(bindings)

    for i, op in enumerate(plan.operations, 1):
        step_start = time.monotonic()

        if op.op == OpType.RLM_CALL:
            query = op.args["query"]
            ctx_ref = op.args["context"]
            ctx_text = local_bindings[ctx_ref]
            result_value = self._recursive_call(query, ctx_text, depth)

        elif op.op == OpType.MAP:
            prompt = op.args["prompt"]
            input_ref = op.args["input"]
            raw = local_bindings[input_ref]
            items: list[str] = json.loads(raw) if raw.startswith("[") else [raw]
            result_value = self._parallel_map(prompt, items, depth)

        else:
            result = self.evaluator.execute(op, local_bindings)
            result_value = result.value

        step_elapsed = time.monotonic() - step_start

        if self.config.verbose:
            op_desc = self._format_op(op)
            bind_note = f" → {op.bind}" if op.bind else ""
            self.console.print(
                f"[dim]  {i}. {op_desc}{bind_note}  ({step_elapsed:.3f}s)[/dim]"
            )

        if op.bind:
            local_bindings[op.bind] = result_value

    return local_bindings[plan.output]
```

**Add helper method `_format_op`** for readable operation display:
```python
def _format_op(self, op: Operation) -> str:
    """Format an operation for human-readable display."""
    parts = []
    for k, v in op.args.items():
        if isinstance(v, str) and len(v) > 40:
            parts.append(f'{k}="{v[:37]}..."')
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return f"{op.op.value}({', '.join(parts)})"
```

**Instrument recursive/parallel with timing**:
```python
def _recursive_call(self, query: str, context_text: str, depth: int) -> str:
    with self.profile.measure("recursive", "recursive_call", depth=depth + 1):
        sub_orchestrator = RLMOrchestrator(self.config, parent=self)
        self.child_orchestrators.append(sub_orchestrator)
        return sub_orchestrator.run(query, context_text, depth=depth + 1)

def _parallel_map(self, prompt: str, items: list[str], depth: int) -> str:
    with self.profile.measure("parallel", "parallel_map", item_count=len(items)):
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
```

**Add `get_total_profile()` method** (mirrors `get_total_token_usage()`):
```python
def get_total_profile(self) -> TimingProfile:
    """Get merged timing profile including all child orchestrators."""
    merged = TimingProfile(enabled=self.profile.enabled)
    merged.merge(self.profile)
    for child in self.child_orchestrators:
        merged.merge(child.get_total_profile())
    return merged
```

**Add `import time`** at the top of the file.

#### 5. `src/rlm/cli.py` — Add timing summary after run

After line 142 (end of verbose block), add:
```python
if verbose:
    # ... existing token/cost output ...

    # Timing breakdown
    from rlm.timing import TimingProfile
    merged_profile = orchestrator.get_total_profile()
    TimingProfile.print_summary(merged_profile, elapsed, console)
```

### Success Criteria:

#### Automated Verification:
- [ ] `pytest tests/` — all existing tests pass unchanged
- [ ] `pytest tests/test_timing.py` — new timing tests pass
- [ ] `mypy src/rlm/` passes

#### Manual Verification:
- [ ] `rlm run -q "..." -c sample.txt -v` shows:
  - Live per-step timing for explore ops
  - Detailed commit operation list with per-op timing
  - Summary timing table at the end
  - Cache hit rate
- [ ] `rlm run -q "..." -c sample.txt` (no -v) — zero output change, no overhead

---

## Example Output (verbose mode)

```
Model: gpt-5-nano
Context: 45,230 chars

EXPLORE step 1/20: count(input="context", mode="lines")  (0.001s, cached)
EXPLORE step 2/20: slice(input="context", start=0, end=5000) → first_part  (0.001s)
EXPLORE step 3/20: grep(input="context", pattern="error") → errors  (0.003s)
EXPLORE step 4/20: count(input="errors", mode="lines")  (0.001s)

COMMIT cycle 1/5: 3 ops [chunk→chunks, map→results, combine→final], output=final
  1. chunk(input="context", n=4) → chunks  (0.001s)
  2. map(prompt="Summarize this section in ...", input="chunks") → results  (12.340s)
  3. combine(inputs=["results"]) → final  (0.001s)

Final answer after 4 explore steps, 1 commit cycles

[answer text]

Completed in 15.2s
Tokens: 8,432 in + 1,201 out = 9,633 total
Estimated cost: $0.0048

     Timing Breakdown (15.2s wall clock)
┌───────────┬───────┬─────────┬─────────┬─────────┬─────────┬───────┐
│ Category  │ Count │   Total │    Mean │     Min │     Max │ Wall% │
├───────────┼───────┼─────────┼─────────┼─────────┼─────────┼───────┤
│ llm       │     6 │ 14.200s │  2.367s │  0.812s │  4.320s │ 93.4% │
│ parallel  │     1 │ 12.340s │ 12.340s │ 12.340s │ 12.340s │ 81.2% │
│ recursive │     4 │ 12.300s │  3.075s │  2.100s │  4.200s │ 80.9% │
│ evaluator │     7 │  0.006s │  0.001s │  0.000s │  0.003s │  0.0% │
│ cache     │    14 │  0.004s │  0.000s │  0.000s │  0.001s │  0.0% │
│ hash      │     7 │  0.002s │  0.000s │  0.000s │  0.001s │  0.0% │
│ parse     │     6 │  0.001s │  0.000s │  0.000s │  0.001s │  0.0% │
└───────────┴───────┴─────────┴─────────┴─────────┴─────────┴───────┘
Cache: 1 hits / 7 lookups (14.3% hit rate)
```

## Files Modified

| File | Change |
|------|--------|
| `src/rlm/timing.py` | **NEW** ~100 lines — TimingEntry, CategorySummary, TimingProfile |
| `src/rlm/llm/client.py` | Accept `profile` param, wrap `completion()` |
| `src/rlm/evaluator/lightweight.py` | Accept `profile` param, wrap hash/cache/execute |
| `src/rlm/orchestrator.py` | Create/distribute profile, improve verbose output, add `_format_op()`, add `get_total_profile()`, instrument recursive/parallel, add `import time` |
| `src/rlm/cli.py` | Add timing summary display after verbose output |
| `tests/test_timing.py` | **NEW** ~80 lines — unit tests for TimingProfile |

## Implementation Sequence

1. Create `src/rlm/timing.py` + `tests/test_timing.py`, run timing tests
2. Instrument `src/rlm/llm/client.py` and `src/rlm/evaluator/lightweight.py`
3. Update `src/rlm/orchestrator.py` — timing + verbose improvements
4. Add summary display in `src/rlm/cli.py`
5. Run full test suite, verify no regressions

## Verification

1. `pytest tests/` — all tests pass
2. `rlm run -q "Find the hidden word" -c data/needle_context.txt -v` — verify live timing + summary
3. `rlm run -q "Find the hidden word" -c data/needle_context.txt` — verify no output change without -v
