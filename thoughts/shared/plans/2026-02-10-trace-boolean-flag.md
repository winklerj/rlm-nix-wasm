# Trace Boolean Flag Implementation Plan

## Overview

Change `--trace` from a path-accepting option (`--trace trace.json`) to a boolean flag (`--trace`). When enabled, trace files are automatically written to a `traces/` directory in the current working directory with a millisecond-precision timestamp filename (e.g., `traces/2026-02-10T14-30-45.123.json`). This eliminates the need to specify a unique filename on every invocation.

## Current State Analysis

The `--trace` flag is defined at `cli.py:97-98` as:
```python
@click.option("--trace", "trace_path", type=click.Path(), default=None,
              help="Write execution trace JSON to PATH.")
```

The trace is written at `cli.py:166-170`:
```python
if trace_path is not None:
    trace = orchestrator.get_trace()
    trace_file = Path(trace_path)
    TraceCollector.write_trace(trace, trace_file)
    console.print(f"[dim]Trace written to {trace_file}[/dim]")
```

`TraceCollector.write_trace()` at `trace.py:210-213` is a static method that takes a trace and a path. No changes needed there.

The `run()` function parameter is `trace_path: str | None`.

### Key Discoveries:
- Trace handling is entirely in the CLI layer — no changes to `RLMConfig`, orchestrator, or `TraceCollector` needed
- `TraceCollector` is instantiated with `enabled=trace_path is not None` (`cli.py:144`) — this becomes `enabled=trace`
- `write_trace()` is a simple static method that writes JSON to a given path — unchanged
- Tests in `test_trace.py` test `TraceCollector` and serialization directly, not CLI flag behavior — no test changes needed for the unit tests

## Desired End State

Running `rlm run -q "..." -c file.txt --trace` automatically creates `./traces/2026-02-10T14-30-45.123.json` (or similar timestamp). The user never specifies a filename. Rerunning the same command produces a new file with a different timestamp.

### Verification:
- `rlm run -q "test" -c file.txt --trace` creates a file in `./traces/` with a timestamp filename
- `rlm run -q "test" -c file.txt` (no flag) produces no trace file
- The generated JSON content is identical in structure to previous `--trace path` output

## What We're NOT Doing

- Not changing `TraceCollector`, `OrchestratorTrace`, or any trace data models
- Not changing the orchestrator's trace recording logic
- Not making the `traces/` directory configurable (can be added later if needed)
- Not adding a `--trace-dir` option (keep it simple)
- Not cleaning up old trace files

## Implementation Approach

Minimal change: modify only `cli.py` to change the flag type and add timestamp-based path generation. One small change in the function signature, and the path generation logic replaces the literal path usage.

## Phase 1: Change `--trace` to Boolean Flag

### Changes Required:

#### 1. CLI flag definition and handler
**File**: `src/rlm/cli.py`

**Change 1** — Flag definition (line 97-98):
```python
# Before:
@click.option("--trace", "trace_path", type=click.Path(), default=None,
              help="Write execution trace JSON to PATH.")

# After:
@click.option("--trace", is_flag=True, default=False,
              help="Write execution trace JSON to traces/ directory.")
```

**Change 2** — Function signature (line 109):
```python
# Before:
    trace_path: str | None,

# After:
    trace: bool,
```

**Change 3** — Add `datetime` import at top of file (after existing imports, ~line 7):
```python
from datetime import datetime, timezone
```

**Change 4** — TraceCollector creation (line 144):
```python
# Before:
trace_collector = TraceCollector(enabled=trace_path is not None)

# After:
trace_collector = TraceCollector(enabled=trace)
```

**Change 5** — Trace file writing (lines 166-170):
```python
# Before:
if trace_path is not None:
    trace = orchestrator.get_trace()
    trace_file = Path(trace_path)
    TraceCollector.write_trace(trace, trace_file)
    console.print(f"[dim]Trace written to {trace_file}[/dim]")

# After:
if trace:
    execution_trace = orchestrator.get_trace()
    now = datetime.now(timezone.utc)
    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)
    trace_file = trace_dir / f"{now.strftime('%Y-%m-%dT%H-%M-%S')}.{now.strftime('%f')[:3]}.json"
    TraceCollector.write_trace(execution_trace, trace_file)
    console.print(f"[dim]Trace written to {trace_file}[/dim]")
```

Note: the local variable was renamed from `trace` to `execution_trace` to avoid shadowing the `trace` bool parameter.

### Success Criteria:

#### Automated Verification:
- [x] All existing tests pass: `pytest`
- [x] Type checking passes: `mypy src/`
- [x] Linting passes: `ruff check src/ tests/`
- [ ] Manual smoke test: `rlm run -q "count lines" -c README.md --trace` creates a file in `./traces/` with correct JSON

#### Manual Verification:
- [ ] Running with `--trace` creates `traces/` directory if missing
- [ ] Filename matches pattern `YYYY-MM-DDTHH-MM-SS.mmm.json`
- [ ] Running twice produces two distinct files
- [ ] Running without `--trace` produces no trace output
- [ ] The `--trace` flag no longer accepts a path argument (e.g., `--trace foo.json` should error or treat `foo.json` as a separate positional)

---

## Phase 2: Update Documentation

All documentation that references `--trace <path>` must be updated to reflect the new boolean flag behavior.

### Changes Required:

#### 1. README.md
**File**: `README.md`

**Change** — Quick Start example (line 13):
```bash
# Before:
rlm run -q "How many unique users are in this log?" -c server.log -v --trace trace.json

# After:
rlm run -q "How many unique users are in this log?" -c server.log -v --trace
```

#### 2. CLAUDE.md
**File**: `CLAUDE.md`

**Change** — Development Commands example:
```bash
# Before:
rlm run -q "Question" -c context.txt -v --trace trace.json

# After:
rlm run -q "Question" -c context.txt -v --trace
```

#### 3. docs/how-to-guides.md
**File**: `docs/how-to-guides.md`

**Change** — "How to trace execution" section (lines 161-188). Replace the entire section:

```markdown
## How to trace execution

Use `--trace` to write a detailed JSON record of every step in the run -- LLM messages in and out, explore operations with full results, commit plans with per-operation detail, and the full recursion tree:

```bash
rlm run -q "How many errors?" -c server.log --trace
```

The trace file is written to the `traces/` directory in the current working directory, with a timestamp filename like `traces/2026-02-10T14-30-45.123.json`. The directory is created automatically if it doesn't exist. Each run produces a new file, so you can compare traces across runs without worrying about overwriting.

Inspect the latest trace with:

```bash
python3 -m json.tool traces/*.json | less
```

Or use `jq` to drill into specific parts:

```bash
# Show each LLM round-trip (pick the trace file you want)
jq '[.root.events[] | select(.type == "llm_call")] | .[] | {call_number, elapsed_s, input_tokens, output_tokens}' traces/2026-02-10T14-30-45.123.json

# Show explore steps
jq '[.root.events[] | select(.type == "explore_step")] | .[] | {step_number, operation_op, cached}' traces/2026-02-10T14-30-45.123.json

# Show recursive child calls
jq '.root.children[] | {trace_id, depth, query, elapsed_s}' traces/2026-02-10T14-30-45.123.json
```

`--trace` works independently of `--verbose` -- you can use both together or either alone. See the [Reference](reference.md) for the full trace JSON schema.
```

#### 4. docs/reference.md
**File**: `docs/reference.md`

**Change** — CLI options table (line 26):
```markdown
# Before:
| `--trace` | | path | none | Write a full execution trace (JSON) to the given file path |

# After:
| `--trace` | | flag | `false` | Write execution trace (JSON) to `traces/` directory with auto-generated timestamp filename |
```

#### 5. docs/tutorial.md
**File**: `docs/tutorial.md`

**Change 1** — Trace example command (lines 89-90):
```bash
# Before:
rlm run -q "How many ERROR lines are in this log?" -c sample.log --trace trace.json

# After:
rlm run -q "How many ERROR lines are in this log?" -c sample.log --trace
```

**Change 2** — Description text (line 91):
```markdown
# Before:
This writes a JSON file you can inspect after the run. See the [How-to Guides](how-to-guides.md#how-to-trace-execution) for details.

# After:
This writes a JSON file to the `traces/` directory that you can inspect after the run. See the [How-to Guides](how-to-guides.md#how-to-trace-execution) for details.
```

### Success Criteria:

#### Automated Verification:
- [x] No broken markdown links or formatting issues
- [x] All `--trace` references updated (verify with `grep -r '\-\-trace' docs/ README.md CLAUDE.md`)

#### Manual Verification:
- [ ] Documentation examples are consistent with new CLI behavior
- [ ] No stale references to `--trace <filename>` pattern remain in user-facing docs

---

## Testing Strategy

### Existing Tests:
- `tests/test_trace.py` tests `TraceCollector` and trace models directly — these are unaffected since we're only changing CLI wiring

### No New Tests Needed:
- The change is in CLI flag handling only. The trace recording and serialization logic is unchanged and already well-tested.
- If CLI integration tests exist, they would need updating, but currently trace behavior is tested via unit tests on `TraceCollector`.

## References

- Research: `thoughts/shared/research/2026-02-10-trace-flag-file-storage.md`
- `src/rlm/cli.py:97-170` — all code being modified
- `src/rlm/trace.py:210-213` — `write_trace()` (unchanged)
- `README.md:13` — Quick Start trace example
- `CLAUDE.md` — Development Commands trace example
- `docs/how-to-guides.md:161-188` — "How to trace execution" section
- `docs/reference.md:26` — CLI options table
- `docs/tutorial.md:87-93` — Tutorial trace example
