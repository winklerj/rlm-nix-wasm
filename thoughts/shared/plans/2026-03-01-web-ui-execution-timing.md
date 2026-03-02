# Web UI Execution Timing Implementation Plan

## Overview

Add execution timing to the web UI's execution trace. The trace system already captures `elapsed_s` at every level (LLM calls, explore steps, commit operations, total run) — but the web server doesn't measure or send timing to the frontend, and the frontend doesn't display it.

Note: the web server (`web/server.py`) runs its own orchestration loop (it doesn't use `RLMOrchestrator` or `TraceCollector` directly for the main loop), so we need to add `time.monotonic()` measurements inline. A future refactor could add a callback interface to `RLMOrchestrator` so the web server can reuse it with interactive pause points, but that's out of scope here.

## Current State Analysis

- **`src/rlm/trace.py`**: Full timing infrastructure (`elapsed_s` on every trace event). Not used by `web/server.py`.
- **`web/server.py`**: `run_trace()` runs the orchestration loop. No timing measurements. WebSocket messages (`step`, `result`, `final`, `tokens`, `error`, `pending`) contain no timing data.
- **`web/index.html`**: Status bar shows Status, Steps, Tokens. No time display. `addStep()` and `addResult()` don't render durations.

### Key Discoveries:
- `web/server.py:190-192` — LLM calls via `asyncio.to_thread(client.send, ...)` with no timing wrapper
- `web/server.py:284` — Operation execution via `evaluator.execute()` with no timing wrapper
- `web/server.py:231-237` — `step` message has `type`, `mode`, `summary`, `action`, `raw` — no timing
- `web/server.py:293-296` — `result` message has `type`, `result` — no timing
- `web/index.html:393-403` — Status bar: Status, Steps, Tokens — no Time
- `web/index.html:599-608` — `addStep()` renders step header with mode badge and summary — no duration

## What We're NOT Doing

- Not integrating `TraceCollector` into the web server (it has its own loop)
- Not adding `TimingProfile` category breakdowns (llm vs evaluator vs cache)
- Not adding timeline/Gantt chart visualization
- Not changing the offline trace system

## Implementation Approach

Measure wall-clock time around each LLM call and operation execution in `server.py`, send it with the existing WebSocket messages, and display it in the frontend.

## Phase 1: Add Timing to Server

### Overview
Wrap LLM calls and operation executions with `time.monotonic()` measurements. Add timing fields to WebSocket messages.

### Changes Required:

#### 1. `web/server.py` — Add timing measurements and send them

**a) Import `time` at the top** (not currently imported):
```python
import time
```

**b) Track run start time** — after `run_trace()` sets up `evaluator`, record start:
```python
run_start = time.monotonic()
```

**c) Wrap every LLM call with timing.** There are 6 `asyncio.to_thread(client.send, ...)` calls in `run_trace()`:
1. Line 190: Initial message
2. Line 208: Parse error recovery
3. Line 255: Skip recovery
4. Line 275: Max explore steps warning
5. Line 298: Explore result feedback
6. Line 400: Commit result feedback

Each gets wrapped:
```python
llm_start = time.monotonic()
response = await asyncio.to_thread(client.send, ...)
llm_elapsed = time.monotonic() - llm_start
```

**d) Add `elapsed_s` to `step` messages** — every `ws.send_json({"type": "step", ...})` gets a new `elapsed_s` field carrying the preceding LLM call duration:
```python
await ws.send_json({
    "type": "step",
    "mode": mode,
    "summary": summary,
    "action": action_to_dict(action),
    "raw": response[:1000],
    "elapsed_s": llm_elapsed
})
```

**e) Wrap `evaluator.execute()` calls with timing:**

Explore step (line ~284):
```python
op_start = time.monotonic()
result = evaluator.execute(action.operation, bindings)
op_elapsed = time.monotonic() - op_start
```

Commit operations (line ~383):
```python
op_start = time.monotonic()
result = evaluator.execute(op, bindings)
result_value = result.value
op_elapsed = time.monotonic() - op_start
```

**f) Add `elapsed_s` to `result` messages:**
```python
await ws.send_json({
    "type": "result",
    "result": result_str[:2000],
    "elapsed_s": op_elapsed
})
```

**g) Add `elapsed_s` to `final` message:**
```python
await ws.send_json({
    "type": "final",
    "answer": action.answer,
    "elapsed_s": time.monotonic() - run_start
})
```

**h) Wrap recursive `sub_orch.run()` calls** (lines ~341, ~371) with timing and include in their step messages.

### Success Criteria:

#### Automated Verification:
- [ ] Server starts without errors: `nix-shell --run "cd web && python server.py"` (check no import/syntax errors, then Ctrl-C)
- [ ] Lint passes: `nix-shell --run "uv run ruff check web/server.py"`

#### Manual Verification:
- [ ] WebSocket messages include `elapsed_s` field (check browser DevTools Network > WS tab)

---

## Phase 2: Display Timing in Frontend

### Overview
Show per-step duration in step headers and total elapsed time in the status bar.

### Changes Required:

#### 1. `web/index.html` — Add timing display

**a) Add elapsed time to status bar** — new status item after Tokens:
```html
<div class="status-item">
    Time: <span id="elapsedTime">0.0s</span>
</div>
```

**b) Add `formatTime()` helper function:**
```javascript
function formatTime(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + 's';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return m + 'm ' + s + 's';
}
```

**c) Track a client-side live timer** — use `setInterval` to update elapsed time while running:
```javascript
let timerInterval = null;
let runStartTime = null;
```

In `startTrace()`, inside `ws.onopen`:
```javascript
runStartTime = Date.now();
timerInterval = setInterval(() => {
    const elapsed = (Date.now() - runStartTime) / 1000;
    document.getElementById('elapsedTime').textContent = formatTime(elapsed);
}, 100);
```

In `stopTrace()` and `showFinalAnswer()`:
```javascript
clearInterval(timerInterval);
```

In `clearTrace()`:
```javascript
clearInterval(timerInterval);
runStartTime = null;
document.getElementById('elapsedTime').textContent = '0.0s';
```

**d) Show per-step duration in step header** — update `addStep()`. The existing code builds the step header with `msg.summary`. Add `msg.elapsed_s` as a subtle time badge on the right side, next to the collapse arrow. Use `escapeHtml()` for any user-derived content (already the pattern in the codebase). The time value is a float from the server, not user input, so it's safe to render directly.

**e) Show operation duration on result** — update `addResult()`. Append the timing as a parenthetical after the "Result" heading when `msg.elapsed_s` is present.

**f) Show total time on final answer** — update `showFinalAnswer()`. Use `msg.elapsed_s` from the `final` message to show total server-side elapsed time in the status text (e.g., "Complete (12.3s)").

**g) Add CSS for timing in step headers:**
```css
.step-time {
    color: #8b949e;
    font-size: 12px;
    font-weight: normal;
    margin-left: 8px;
}
```

### Success Criteria:

#### Automated Verification:
- [ ] No JavaScript console errors when loading the page
- [ ] Lint passes: `nix-shell --run "uv run ruff check web/server.py"`

#### Manual Verification:
- [ ] Status bar shows live-updating "Time: X.Xs" while trace is running
- [ ] Each step header shows its LLM call duration (e.g., "2.3s" on the right)
- [ ] Result sections show operation execution duration
- [ ] Final answer shows total elapsed time in status bar
- [ ] Timer stops when trace completes or is stopped
- [ ] Clear resets the timer to 0.0s

---

## Testing Strategy

### Manual Testing Steps:
1. Start the web server: `nix-shell --run "cd web && python server.py"`
2. Load the Needle Test sample
3. Run with auto-approve enabled using a cheap model (gpt-5-nano)
4. Verify timing appears on each step header
5. Verify the status bar timer counts up in real time
6. Verify the final answer shows total time
7. Click Clear and verify timer resets
8. Run again and Stop mid-trace — verify timer stops

## References

- `web/server.py` — WebSocket server with inline orchestration loop
- `web/index.html` — Single-page web UI
- `src/rlm/trace.py` — Trace data structures (for reference on timing patterns)
