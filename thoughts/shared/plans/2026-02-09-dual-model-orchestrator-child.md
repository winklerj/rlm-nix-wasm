# Dual-Model Support (Orchestrator vs. Child Model) Implementation Plan

## Overview

Add dual-model support so the orchestrator uses a reasoning model (defaulting to `claude-opus-4-6`) while child/recursive calls use a smaller, cheaper model (configurable via `--child-model` / `RLM_CHILD_MODEL`). When no child model is specified, children use the same model as the orchestrator (backward-compatible).

## Current State Analysis

- **Single model field**: `RLMConfig.model` (default `"gpt-5-nano"`) in `src/rlm/types.py:93`
- **Config shared by reference**: `_recursive_call` at `orchestrator.py:284` passes `self.config` unchanged to child orchestrators
- **Cost estimation flat**: `cli.py:144` prices all tokens at the parent model's rate via `estimate_cost(config.model, ...)`
- **Trace already per-node**: `OrchestratorTrace.model` and `LLMCallTrace.model` record per-orchestrator models
- **Docs stale**: `reference.md` shows default `gpt-4o-mini`, code has `gpt-5-nano`

### Key Discoveries:
- Each orchestrator creates its own `LLMClient` in `__init__` (`orchestrator.py:45-48`), reading `config.model`. No shared LLM client -- changing the model per-orchestrator requires only giving each a config with the right `model` value.
- `RLMConfig` is a Pydantic `BaseModel`, so `model_copy(update={...})` is available for creating modified copies.
- `get_total_token_usage()` (`orchestrator.py:336-343`) recursively sums tokens but discards model info. Need a new `get_total_cost()` method that prices tokens per-orchestrator.

## Desired End State

After implementation:
1. `rlm run -q "..." -c file.txt -m claude-opus-4-6 --child-model gpt-5-nano` uses Opus for the root orchestrator and gpt-5-nano for all recursive child calls.
2. `rlm run -q "..." -c file.txt` uses `claude-opus-4-6` (new default) for all calls (no child model set, backward-compatible behavior).
3. Cost estimation accurately prices orchestrator tokens at the orchestrator model rate and child tokens at the child model rate.
4. Traces show the correct model at each depth level.
5. All existing tests pass; new tests cover dual-model propagation.

### Verification:
- `pytest tests/` passes with no failures
- `rlm run -q "test" -c README.md -v --child-model gpt-5-nano` shows different models in verbose output and correctly priced cost estimate
- Trace JSON (`--trace trace.json`) shows `claude-opus-4-6` at depth 0 and `gpt-5-nano` at depth 1+

## What We're NOT Doing

- **Per-model temperature**: Not adding `child_temperature` -- follow-up if needed.
- **Per-model max tokens**: Not adding separate token limits per model.
- **Model validation**: Not validating that model strings are valid litellm identifiers.
- **Pricing table updates beyond verification**: Not overhauling the pricing table structure -- just ensuring `claude-opus-4-6` key exists (it doesn't currently; `claude-opus-4.6` with a dot exists).

## Implementation Approach

Use "Option A" from research: create a child config copy where `model` is replaced with `child_model`, so all descendants of the root orchestrator use the child model. This is clean because each orchestrator already creates its own LLMClient from its own config.

---

## Phase 1: Configuration Foundation

### Overview
Add the `child_model` field to `RLMConfig`, update the default model, and wire up the environment variable.

### Changes Required:

#### 1. RLMConfig
**File**: `src/rlm/types.py`
**Lines**: 91-101

Change `model` default from `"gpt-5-nano"` to `"claude-opus-4-6"`. Add `child_model` field.

```python
class RLMConfig(BaseModel):
    """Configuration for an RLM run."""
    model: str = "claude-opus-4-6"
    child_model: str | None = None
    max_explore_steps: int = 20
    max_commit_cycles: int = 5
    max_recursion_depth: int = 1
    max_parallel_jobs: int = 4
    temperature: float = 1.0
    cache_dir: Path = Path.home() / ".cache" / "rlm-nix-wasm"
    use_nix: bool = False
    verbose: bool = False
```

#### 2. Environment Variable Mapping
**File**: `src/rlm/config.py`
**Lines**: 14-22

Add `RLM_CHILD_MODEL` mapping after the `RLM_MODEL` line:

```python
    env_mappings: dict[str, str | tuple[str, Any]] = {
        "RLM_MODEL": "model",
        "RLM_CHILD_MODEL": "child_model",
        "RLM_MAX_EXPLORE_STEPS": ("max_explore_steps", int),
        ...
    }
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from rlm.types import RLMConfig; c = RLMConfig(); assert c.model == 'claude-opus-4-6'; assert c.child_model is None"`
- [x] `pytest tests/` -- existing tests pass (test fixtures explicitly set `model="test-model"` so default change doesn't break them)

---

## Phase 2: CLI Layer

### Overview
Add `--child-model` flag, pass it through to `load_config`, and update verbose output.

### Changes Required:

#### 1. CLI Option
**File**: `src/rlm/cli.py`
**Lines**: 85-94

Add `--child-model` option after `--model`. Update `--model` help text.

```python
@main.command()
@click.option("--query", "-q", required=True, help="The query to answer.")
@click.option("--context", "-c", type=click.Path(exists=True), help="Path to context file.")
@click.option("--model", "-m", default=None, help="LLM model for the orchestrator.")
@click.option("--child-model", default=None, help="LLM model for recursive sub-calls (defaults to --model).")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--use-nix", is_flag=True, default=False, help="Use Nix for sandboxing.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
@click.option("--trace", "trace_path", type=click.Path(), default=None,
              help="Write execution trace JSON to PATH.")
```

#### 2. Function Signature
**File**: `src/rlm/cli.py`
**Lines**: 95-104

Add `child_model` parameter:

```python
def run(
    query: str,
    context: str | None,
    model: str | None,
    child_model: str | None,
    max_explore: int | None,
    max_depth: int | None,
    use_nix: bool,
    verbose: bool,
    trace_path: str | None,
) -> None:
```

#### 3. load_config Call
**File**: `src/rlm/cli.py`
**Lines**: 106-112

Pass `child_model`:

```python
    config = load_config(
        model=model,
        child_model=child_model,
        max_explore_steps=max_explore,
        max_recursion_depth=max_depth,
        use_nix=use_nix,
        verbose=verbose,
    )
```

#### 4. Verbose Output
**File**: `src/rlm/cli.py`
**Lines**: 123-128

Show both models when child_model is set:

```python
    if verbose:
        console.print(f"[dim]Model: {config.model}[/dim]")
        if config.child_model:
            console.print(f"[dim]Child model: {config.child_model}[/dim]")
        console.print(f"[dim]Context: {len(context_text):,} chars[/dim]")
        if config.use_nix:
            console.print("[dim]Nix sandboxing: enabled[/dim]")
```

### Success Criteria:

#### Automated Verification:
- [x] `rlm run --help` shows `--child-model` option
- [x] `pytest tests/` -- all existing tests pass

---

## Phase 3: Orchestrator Child Config

### Overview
When creating child orchestrators, create a modified config copy where `model` is set to the child model.

### Changes Required:

#### 1. Child Config in _recursive_call
**File**: `src/rlm/orchestrator.py`
**Lines**: 280-291

Create a child config copy when `child_model` is set:

```python
    def _recursive_call(self, query: str, context_text: str, depth: int) -> str:
        """Spawn a recursive RLM call."""
        with self.profile.measure("recursive", "recursive_call", depth=depth + 1):
            child_config = self.config
            if self.config.child_model:
                child_config = self.config.model_copy(update={
                    "model": self.config.child_model,
                    "child_model": None,  # children of children also use child_model value via model
                })
            sub_orchestrator = RLMOrchestrator(
                child_config, parent=self,
                trace_collector=self.trace_collector,
            )
            self.child_orchestrators.append(sub_orchestrator)
            result = sub_orchestrator.run(query, context_text, depth=depth + 1)
            if self.trace_collector.enabled:
                self.trace_node.children.append(sub_orchestrator.trace_node)
            return result
```

**Design note**: Setting `child_model=None` in the child config means the child's `model` field IS the child model, and any further recursive children from the child will also use that model (since `child_model` is `None`, no further copy is made, and `model` stays as the child model). This gives us the desired behavior: only the root uses the orchestrator model.

#### 2. No changes needed in _direct_call or _parallel_map
- `_direct_call` (`orchestrator.py:309-322`): Creates its own `LLMClient(self.config, ...)`. If called from a child orchestrator, `self.config.model` is already the child model. Correct behavior.
- `_parallel_map` (`orchestrator.py:293-307`): Calls `_recursive_call`, which now handles the config copy. Correct behavior.

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/` -- all existing tests pass
- [x] New unit tests (Phase 5) verify child model propagation

---

## Phase 4: Per-Orchestrator Cost Estimation

### Overview
Replace flat token-sum cost estimation with per-orchestrator cost calculation that prices each orchestrator's tokens at its own model's rate.

### Changes Required:

#### 1. Add get_total_cost Method
**File**: `src/rlm/orchestrator.py`
**After**: `get_total_token_usage` (line 343)

Add a new method that calculates cost recursively:

```python
    def get_total_cost(self, pricing_fn: "Callable[[str, int, int], float]") -> float:
        """Get total cost including all child orchestrators, priced per-model."""
        input_tokens, output_tokens = self.llm.get_token_usage()
        cost = pricing_fn(self.config.model, input_tokens, output_tokens)
        for child in self.child_orchestrators:
            cost += child.get_total_cost(pricing_fn)
        return cost
```

This takes a pricing function (the existing `estimate_cost` from `cli.py`) and applies it per-orchestrator with the correct model.

#### 2. Add Import
**File**: `src/rlm/orchestrator.py`

Add `Callable` to the `TYPE_CHECKING` imports (or use `from __future__ import annotations`). Check current imports first.

#### 3. Update CLI Cost Display
**File**: `src/rlm/cli.py`
**Lines**: 141-148

Use `get_total_cost` when child_model is set:

```python
    if verbose:
        input_tokens, output_tokens = orchestrator.get_total_token_usage()
        total_tokens = input_tokens + output_tokens
        cost = orchestrator.get_total_cost(estimate_cost)

        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
        console.print(f"[dim]Tokens: {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total[/dim]")
        console.print(f"[dim]Estimated cost: ${cost:.4f}[/dim]")
```

Note: We always use `get_total_cost` now (not just when child_model is set). When there's no child model, all orchestrators use the same model, so the result is identical to the old calculation.

#### 4. Pricing Key for claude-opus-4-6
**File**: `src/rlm/cli.py`
**Lines**: 17-42

The pricing table has `"claude-opus-4.6"` (with dot). The new default model string is `"claude-opus-4-6"` (with hyphen). Add the hyphenated key:

```python
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},    # litellm ID
    "claude-opus-4.6": {"input": 5.00, "output": 25.00},
```

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/` -- all existing tests pass
- [x] New unit test verifies `get_total_cost` returns correct per-model pricing

#### Manual Verification:
- [ ] Run with `--child-model` and `-v`: cost breakdown uses correct rates for each model

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that cost output looks correct before proceeding.

---

## Phase 5: Tests

### Overview
Add tests for dual-model propagation, cost estimation, and config loading.

### Changes Required:

#### 1. Orchestrator Tests
**File**: `tests/test_orchestrator.py`

Add a new test class:

```python
class TestChildModel:
    def test_child_uses_child_model(self, config):
        """Child orchestrator should use child_model when set."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        parent = RLMOrchestrator(config)

        # Trigger _recursive_call and verify child's config
        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        assert len(parent.child_orchestrators) == 1
        child = parent.child_orchestrators[0]
        assert child.config.model == "child-model"
        assert child.config.child_model is None

    def test_child_falls_back_to_parent_model(self, config):
        """Child orchestrator should use parent model when child_model is None."""
        config.model = "orchestrator-model"
        config.child_model = None
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        child = parent.child_orchestrators[0]
        assert child.config.model == "orchestrator-model"

    def test_parallel_map_uses_child_model(self, config):
        """Parallel map should create children with child_model."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        config.max_parallel_jobs = 2
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._parallel_map("prompt", ["a", "b"], depth=0)

        assert len(parent.child_orchestrators) == 2
        for child in parent.child_orchestrators:
            assert child.config.model == "child-model"

    def test_grandchild_uses_child_model(self, config):
        """Children of children should also use the child model."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        parent = RLMOrchestrator(config)

        # Create child manually using _recursive_call logic
        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        child = parent.child_orchestrators[0]
        # Child's config.child_model should be None, model should be child-model
        # If the child spawns its own child, it passes its config unchanged
        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            child._recursive_call("q2", "ctx2", depth=1)

        grandchild = child.child_orchestrators[0]
        assert grandchild.config.model == "child-model"

    def test_get_total_cost_dual_model(self, config):
        """get_total_cost should price each orchestrator at its own model rate."""
        config.model = "expensive-model"
        config.child_model = "cheap-model"
        parent = RLMOrchestrator(config)
        parent.llm.total_input_tokens = 1000
        parent.llm.total_output_tokens = 500

        # Create a child with child model
        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)
        child = parent.child_orchestrators[0]
        child.llm.total_input_tokens = 2000
        child.llm.total_output_tokens = 1000

        # Pricing function that returns different rates per model
        def mock_pricing(model, inp, out):
            if model == "expensive-model":
                return inp * 0.01 + out * 0.05  # expensive
            return inp * 0.001 + out * 0.005     # cheap

        total = parent.get_total_cost(mock_pricing)
        expected_parent = 1000 * 0.01 + 500 * 0.05   # 35.0
        expected_child = 2000 * 0.001 + 1000 * 0.005  # 7.0
        assert total == pytest.approx(expected_parent + expected_child)
```

#### 2. Config Tests
**File**: `tests/test_config.py` (new file)

```python
"""Tests for configuration loading."""
import os
import pytest
from rlm.config import load_config


class TestLoadConfig:
    def test_default_model(self):
        config = load_config()
        assert config.model == "claude-opus-4-6"
        assert config.child_model is None

    def test_model_override(self):
        config = load_config(model="gpt-5-nano")
        assert config.model == "gpt-5-nano"

    def test_child_model_override(self):
        config = load_config(child_model="gpt-5-nano")
        assert config.child_model == "gpt-5-nano"

    def test_child_model_env_var(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "gpt-5-nano")
        config = load_config()
        assert config.child_model == "gpt-5-nano"

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "env-model")
        config = load_config(child_model="cli-model")
        assert config.child_model == "cli-model"

    def test_none_override_uses_env(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "env-model")
        config = load_config(child_model=None)
        assert config.child_model == "env-model"
```

#### 3. Trace Test
**File**: `tests/test_trace.py`

Add test verifying child trace nodes record the child model (at end of file):

```python
def test_child_trace_records_child_model():
    """Child trace nodes should reflect the child model, not the parent model."""
    parent = _make_node(trace_id=0)
    parent.model = "orchestrator-model"

    child = _make_node(trace_id=1)
    child.model = "child-model"
    child.depth = 1
    parent.children.append(child)

    trace = ExecutionTrace(timestamp="2026-02-09T00:00:00Z", root=parent)
    raw = trace.model_dump_json()
    restored = ExecutionTrace.model_validate_json(raw)
    assert restored.root.model == "orchestrator-model"
    assert restored.root.children[0].model == "child-model"
```

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/test_orchestrator.py::TestChildModel -v` -- all 5 tests pass
- [x] `pytest tests/test_config.py -v` -- all 6 tests pass
- [x] `pytest tests/test_trace.py::test_child_trace_records_child_model -v` -- passes
- [x] `pytest tests/` -- full suite green

---

## Phase 6: Documentation

### Overview
Update reference docs, how-to guides, and fix stale defaults.

### Changes Required:

#### 1. CLI Options Table
**File**: `docs/reference.md`
**Lines**: 15-24

Add `--child-model` row and fix `--model` default:

```markdown
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--query` | `-q` | string | required | The question to answer |
| `--context` | `-c` | path | stdin | Path to the context file. If omitted, reads from stdin |
| `--model` | `-m` | string | `claude-opus-4-6` | LLM model for the orchestrator (any litellm-supported model) |
| `--child-model` | | string | same as `--model` | LLM model for recursive sub-calls |
| `--max-explore` | | int | `20` | Maximum explore steps before forcing a commit |
| `--max-depth` | | int | `1` | Maximum recursion depth |
| `--use-nix` | | flag | `false` | Compile operations to Nix derivations |
| `--verbose` | `-v` | flag | `false` | Show model name, context size, operation trace, and timing |
| `--trace` | | path | none | Write a full execution trace (JSON) to the given file path |
```

#### 2. Configuration Table
**File**: `docs/reference.md`
**Lines**: 52-61

Add `RLM_CHILD_MODEL` row and fix model default:

```markdown
| Setting | Environment Variable | CLI Flag | Default |
|---------|---------------------|----------|---------|
| LLM model | `RLM_MODEL` | `--model` | `claude-opus-4-6` |
| Child LLM model | `RLM_CHILD_MODEL` | `--child-model` | same as `--model` |
| Max explore steps | `RLM_MAX_EXPLORE_STEPS` | `--max-explore` | `20` |
| Max commit cycles | `RLM_MAX_COMMIT_CYCLES` | -- | `5` |
| Max recursion depth | `RLM_MAX_RECURSION_DEPTH` | `--max-depth` | `1` |
| Max parallel jobs | `RLM_MAX_PARALLEL_JOBS` | -- | `4` |
| Cache directory | `RLM_CACHE_DIR` | -- | `~/.cache/rlm-nix-wasm` |
| Enable Nix | `RLM_USE_NIX` | `--use-nix` | `false` |
| Verbose output | `RLM_VERBOSE` | `--verbose` | `false` |
```

#### 3. How-To Guide
**File**: `docs/how-to-guides.md`
**After**: "How to use a different LLM model" section (line 22)

Add a new section:

```markdown
## How to use different models for orchestrator and child calls

Use a powerful reasoning model for the orchestrator while keeping sub-calls cheap:

```bash
# Opus for orchestration, nano for recursive sub-calls
rlm run -q "Analyze this" -c data.txt -m claude-opus-4-6 --child-model gpt-5-nano

# Set defaults via environment variables
export RLM_MODEL=claude-opus-4-6
export RLM_CHILD_MODEL=gpt-5-nano
rlm run -q "Analyze this" -c data.txt
```

When `--child-model` is omitted, all recursive calls use the same model as the orchestrator.
```

#### 4. Trace Format Documentation
**File**: `docs/reference.md`
**Lines**: 226-297

Add a note in the trace format section mentioning that `model` in child nodes reflects the child model when dual-model is configured.

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/` -- full suite still green (docs changes don't break anything)

#### Manual Verification:
- [ ] `docs/reference.md` CLI table has `--child-model` row with correct formatting
- [ ] `docs/reference.md` config table has `RLM_CHILD_MODEL` row
- [ ] `docs/how-to-guides.md` has new section with working examples
- [ ] All model defaults consistently say `claude-opus-4-6`

---

## Testing Strategy

### Unit Tests:
- Child model propagation through `_recursive_call`
- Fallback to parent model when `child_model` is `None`
- Parallel map uses child model
- Grandchild inherits child model (not orchestrator model)
- `get_total_cost` prices each orchestrator at its own rate
- Config loading: env vars, CLI overrides, precedence

### Integration Tests:
- Trace JSON shows correct model per depth level

### Manual Testing Steps:
1. `rlm run -q "test" -c README.md -v` -- verify default model is now `claude-opus-4-6`
2. `rlm run -q "test" -c README.md -v --child-model gpt-5-nano --max-depth 2` -- verify child model appears in verbose output
3. `rlm run -q "test" -c README.md --trace /tmp/trace.json --child-model gpt-5-nano --max-depth 2` -- verify trace shows different models at different depths
4. `rlm run --help` -- verify `--child-model` appears with correct description

## Performance Considerations

- `model_copy()` is called once per child orchestrator spawn. This is negligible compared to LLM call latency.
- `get_total_cost()` traverses the orchestrator tree once, same as `get_total_token_usage()`. No performance concern.

## References

- Research document: `thoughts/shared/research/2026-02-09-dual-model-orchestrator-child.md`
- `src/rlm/types.py:91-101` -- RLMConfig class
- `src/rlm/config.py:12-36` -- load_config function
- `src/rlm/cli.py:85-152` -- run command and cost display
- `src/rlm/orchestrator.py:280-343` -- recursive call and token aggregation
- `src/rlm/llm/client.py:48-54` -- model usage in completion calls
