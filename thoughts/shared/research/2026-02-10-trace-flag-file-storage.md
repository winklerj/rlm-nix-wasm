---
date: "2026-02-10T00:00:00+00:00"
researcher: claude
git_commit: 9b7e5b179169cf9c31c83e3c70bef1e5ac319cdf
branch: main
repository: rlm-secure
topic: "Trace flag and file storage"
tags: [research, codebase, trace, cache, file-storage, cli]
status: complete
last_updated: "2026-02-10"
last_updated_by: claude
---

# Research: Trace Flag and File Storage

**Date**: 2026-02-10
**Researcher**: claude
**Git Commit**: 9b7e5b179169cf9c31c83e3c70bef1e5ac319cdf
**Branch**: main
**Repository**: rlm-secure

## Research Question
How does the `--trace` flag work and how are trace files and cached operation results stored on disk?

## Summary

The `--trace` CLI flag enables comprehensive execution recording. When provided a file path, a `TraceCollector` captures every LLM call, explore step, commit cycle, and final answer as chronological events in a tree of `OrchestratorTrace` nodes. At completion, the tree is serialized to JSON via Pydantic and written to the specified path. Separately, a content-addressed filesystem cache stores operation results keyed by SHA256 of (op_type + args + input_hashes), using a Nix-like `{hash[:2]}/{hash[2:4]}/{hash}` directory structure under `~/.cache/rlm-secure/`.

## Detailed Findings

### 1. The `--trace` CLI Flag

The flag is defined at `src/rlm/cli.py:97-98`:
```python
@click.option("--trace", "trace_path", type=click.Path(), default=None,
              help="Write execution trace JSON to PATH.")
```

At `cli.py:144`, a `TraceCollector` is created based on whether the flag was provided:
```python
trace_collector = TraceCollector(enabled=trace_path is not None)
```

The collector is passed to the orchestrator at `cli.py:147`. After the run completes, the trace is written at `cli.py:166-170`:
```python
if trace_path is not None:
    trace = orchestrator.get_trace()
    trace_file = Path(trace_path)
    TraceCollector.write_trace(trace, trace_file)
```

The trace flag does **not** flow through `RLMConfig` - it is handled entirely in the CLI layer.

### 2. TraceCollector Architecture

Defined at `src/rlm/trace.py:97-213`.

**Thread safety**: Uses `threading.Lock()` (line 102) because child orchestrators from `map` operations run in parallel via `ThreadPoolExecutor`.

**No-op pattern**: All recording methods check `self.enabled` first and return immediately when disabled, making it zero-cost when `--trace` is not used.

**Shared collector, individual nodes**: A single `TraceCollector` instance is shared across the orchestrator tree. Each orchestrator creates its own `OrchestratorTrace` node with a unique sequential `trace_id` allocated via `next_trace_id()`.

### 3. Trace Data Model

#### ExecutionTrace (top-level, `trace.py:90-94`)
- `version: str` - Format version, currently `"1.1"`
- `timestamp: str` - ISO 8601 UTC timestamp
- `root: OrchestratorTrace` - Root orchestrator node

#### OrchestratorTrace (`trace.py:78-88`)
- `trace_id: int` - Unique sequential ID
- `depth: int` - Recursion depth (0 for root)
- `query: str` - Query string for this orchestrator
- `context_length: int` - Character count of context
- `model: str` - LLM model used
- `elapsed_s: float` - Total execution time
- `events: list[TraceEvent]` - Chronological event list
- `children: list[OrchestratorTrace]` - Child traces from recursive calls

#### TraceEvent (discriminated union, `trace.py:72-75`)
Four event types distinguished by a `type` literal field:
1. **LLMCallTrace** (`trace.py:14-24`): call_number, timestamp, elapsed_s, model, input/output tokens, user/assistant messages
2. **ExploreStepTrace** (`trace.py:27-38`): step_number, operation details, result, cached flag, error
3. **CommitCycleTrace** (`trace.py:53-60`): cycle_number, output_variable, list of `CommitOperationTrace`, result
4. **FinalAnswerTrace** (`trace.py:63-69`): answer, total explore_steps, total commit_cycles

#### CommitOperationTrace (`trace.py:41-50`)
Nested within `CommitCycleTrace.operations`:
- `child_trace_ids: list[int]` links to child orchestrator traces spawned by `rlm_call`/`map`

### 4. Event Recording Points

| Event | Location | Trigger |
|-------|----------|---------|
| LLM call | `llm/client.py:75-85` | After each litellm completion |
| Explore step | `orchestrator.py:169-174` | After explore operation executes |
| Explore error | `orchestrator.py:187-192` | When explore operation raises exception |
| Commit cycle | `orchestrator.py:220-224` | After commit plan executes |
| Final answer | `orchestrator.py:135-138` | When LLM returns mode=final |

### 5. Child Orchestrator Trace Linkage

At `orchestrator.py:309-317`, child orchestrators receive the **same** `trace_collector`. After a child completes, its `trace_node` is appended to the parent's `children` list (line 316), building the tree. For `map` operations using `ThreadPoolExecutor` (lines 319-333), all spawned children's trace IDs are collected into `child_trace_ids` on the commit operation trace.

### 6. Trace File Serialization

`TraceCollector.write_trace()` at `trace.py:210-213`:
```python
path.write_text(trace.model_dump_json(indent=2))
```

Uses Pydantic's `model_dump_json()` for type-safe serialization with 2-space indentation.

### 7. Version History

- **v1.0**: Events grouped by type in separate lists (`llm_calls`, `explore_steps`, `commit_cycles`, `final_answer`). Seen in `trace.json`, `trace2.json`.
- **v1.1** (current): Single chronological `events` list preserving execution order. Seen in `trace3.json`-`trace7.json`. Rationale documented in `thoughts/shared/plans/2026-02-09-trace-unified-event-list.md`.

### 8. Content-Addressed Cache System

#### CacheStore (`cache/store.py:13-68`)

**Directory structure**: Three-level Nix-like hierarchy:
```
{cache_dir}/{key[:2]}/{key[2:4]}/{key}
```
Default cache directory: `~/.cache/rlm-secure/` (configurable via `RLM_CACHE_DIR` env var or config).

**Operations**:
- `get(key)` - Returns `path.read_text()` if exists, else `None` (`store.py:28-33`)
- `put(key, value)` - Creates parent dirs and writes text (`store.py:35-39`)
- `has(key)` - Boolean existence check (`store.py:41-43`)
- `stats()` - Recursive walk counting entries and bytes (`store.py:45-58`)
- `clear()` - Removes entire tree and recreates empty dir (`store.py:60-68`)

#### Cache Key Computation (`cache/store.py:71-80`)

`make_cache_key(op, args, input_hashes)`:
1. For each arg value, replaces variable names with their content hashes (if present in `input_hashes`)
2. Builds `key_data = {"op": op.value, "args": resolved_args}`
3. Returns `SHA256(json.dumps(key_data, sort_keys=True))`

This ensures identical operations with identical inputs always produce the same key, regardless of variable naming.

### 9. Cache Integration with Evaluator

`LightweightEvaluator.execute()` at `evaluator/lightweight.py:41-88`:

1. Computes SHA256 hashes of all input binding values (lines 55-60)
2. Generates cache key via `make_cache_key()` (line 60)
3. If cache exists and has key: returns `OpResult(cached=True)` (lines 62-73)
4. Executes operation via registered executor (lines 76-77)
5. Stores result in cache (lines 79-82)
6. Returns `OpResult(cached=False)` (lines 84-88)

### 10. Cache Configuration

Three-layer precedence:
- Default: `Path.home() / ".cache" / "rlm-secure"` (`types.py:100`)
- Environment: `RLM_CACHE_DIR` (`config.py:21`)
- CLI: `--cache-dir` flag

CLI cache management commands at `cli.py:173-201`:
- `rlm cache stats` - Shows entry count and total size
- `rlm cache clear` - Removes all cached entries

### 11. Other File Storage Mechanisms

| Mechanism | Location | Lifecycle |
|-----------|----------|-----------|
| Trace output | User-specified `--trace PATH` | Written once at end of run |
| Cache entries | `~/.cache/rlm-secure/` | Persistent until `rlm cache clear` |
| Nix store ops | System temp dir then `/nix/store/` | Temp files auto-cleaned |
| Wasm sandbox | `tempfile.TemporaryDirectory()` | Auto-cleaned via context manager |
| Nix builder | `tempfile.NamedTemporaryFile()` | Cleaned in finally blocks |

## Code References
- `src/rlm/trace.py:97-213` - TraceCollector class with thread-safe event recording
- `src/rlm/trace.py:78-88` - OrchestratorTrace data model
- `src/rlm/trace.py:14-75` - All trace event types and discriminated union
- `src/rlm/trace.py:90-94` - ExecutionTrace top-level model
- `src/rlm/trace.py:210-213` - write_trace static method
- `src/rlm/cli.py:97-98` - --trace CLI option definition
- `src/rlm/cli.py:144-147` - TraceCollector creation and orchestrator wiring
- `src/rlm/cli.py:166-170` - Trace file writing after run
- `src/rlm/orchestrator.py:36-49` - Orchestrator trace initialization
- `src/rlm/orchestrator.py:135-138` - Final answer recording
- `src/rlm/orchestrator.py:169-174` - Explore step recording
- `src/rlm/orchestrator.py:187-192` - Explore error recording
- `src/rlm/orchestrator.py:220-224` - Commit cycle recording
- `src/rlm/orchestrator.py:309-317` - Child orchestrator trace linkage
- `src/rlm/orchestrator.py:387-393` - get_trace() method
- `src/rlm/llm/client.py:75-85` - LLM call trace recording
- `src/rlm/cache/store.py:13-68` - CacheStore class
- `src/rlm/cache/store.py:71-80` - make_cache_key function
- `src/rlm/evaluator/lightweight.py:41-88` - Cache integration in evaluator
- `src/rlm/types.py:100` - Default cache directory
- `src/rlm/config.py:21` - RLM_CACHE_DIR env var mapping
- `tests/test_trace.py` - Full trace test suite including thread safety
- `tests/test_cache.py` - Cache key determinism and store tests

## Architecture Insights

1. **Separation of concerns**: Tracing is a CLI-only concern - `RLMConfig` has no trace fields. The cache, by contrast, flows through config with env var and CLI overrides.
2. **Shared collector pattern**: One `TraceCollector` instance spans the entire orchestrator tree, with each node getting its own `OrchestratorTrace`. This enables unified ID allocation and thread-safe recording.
3. **Content addressing**: Cache keys are pure functions of operation type, arguments, and input content hashes, making them deterministic and cross-run reusable.
4. **Nix-inspired storage**: The `{hash[:2]}/{hash[2:4]}/{hash}` directory layout prevents filesystem directory bloat with large caches.
5. **Version evolution**: The trace format evolved from v1.0 (grouped by type) to v1.1 (chronological events list) to preserve execution narrative.

## Historical Context (from thoughts/)
- `thoughts/shared/plans/2026-02-09-execution-trace.md` - Original implementation plan for the --trace flag feature
- `thoughts/shared/plans/2026-02-09-trace-unified-event-list.md` - Plan for migrating from grouped lists to unified chronological events
- `thoughts/shared/research/2026-02-09-trace-event-ordering.md` - Research comparing unified vs separate event list approaches
- `thoughts/shared/plans/2026-02-10-wasm-eval-operation.md` - Wasm plan that extends trace recording

## Open Questions
- No explicit cache eviction policy exists - the cache grows unbounded until manually cleared
- No file locking on cache writes - concurrent orchestrator instances could theoretically race on the same key (though content addressing makes this benign since the value would be identical)
- Trace files are written at the end of a run; if the process crashes, no partial trace is saved
