---
date: 2026-02-09T00:00:00-05:00
researcher: claude
git_commit: a9d5b348b577b986bf31415e2e958a67e2a33e1c
branch: main
repository: rlm-nix-wasm
topic: "Dual-model support: reasoning model for orchestrator, smaller model for child calls"
tags: [research, codebase, model-configuration, orchestrator, llm-client, cli]
status: complete
last_updated: 2026-02-09
last_updated_by: claude
---

# Research: Dual-Model Support (Orchestrator vs. Child Model)

**Date**: 2026-02-09
**Researcher**: claude
**Git Commit**: a9d5b348b577b986bf31415e2e958a67e2a33e1c
**Branch**: main
**Repository**: rlm-nix-wasm

## Research Question

How to implement dual-model support so the orchestrator uses a reasoning model (e.g., Opus 4.6) by default, while child/recursive calls use a smaller, cheaper model (e.g., gpt-5-nano). Both models should be configurable via CLI flags.

## Summary

The current architecture uses a **single model** throughout the entire execution tree. The `RLMConfig` has one `model` field, and when child orchestrators are spawned (via `rlm_call` or `map`), they receive the **same config object by reference**. Implementing dual-model support requires changes across 7 files spanning CLI, config, types, orchestrator, LLM client, trace, and documentation.

The key design insight is that child orchestrators are already created as separate instances with their own `LLMClient` -- the only reason they use the same model is that they inherit `self.config` unchanged. The change is architecturally clean: add a second model field, and have child orchestrators use it.

## Detailed Findings

### 1. Configuration Layer (`types.py`, `config.py`)

**Current state** -- single model field:

- `src/rlm/types.py:93`: `model: str = "gpt-5-nano"` in `RLMConfig` (Pydantic BaseModel)
- `src/rlm/config.py:15`: `RLM_MODEL` env var maps to `model` field

**Changes needed**:

- **`src/rlm/types.py:91-101`**: Add `child_model: str | None = None` field to `RLMConfig`. When `None`, children use the same model as the orchestrator. This preserves backward compatibility.
  - Change the default `model` from `"gpt-5-nano"` to `"claude-opus-4-6"` (the new orchestrator default).

- **`src/rlm/config.py:14-23`**: Add `RLM_CHILD_MODEL` env var mapping:
  ```python
  "RLM_CHILD_MODEL": "child_model",
  ```

### 2. CLI Layer (`cli.py`)

**Current state** -- single `--model` flag:

- `src/rlm/cli.py:88`: `@click.option("--model", "-m", default=None, help="LLM model to use.")`
- `src/rlm/cli.py:107`: passed to `load_config(model=model, ...)`

**Changes needed**:

- **`src/rlm/cli.py:88`**: Update help text for `--model` to clarify it controls the orchestrator model.
- **`src/rlm/cli.py` (near line 88)**: Add new CLI option:
  ```python
  @click.option("--child-model", default=None, help="LLM model for recursive sub-calls (defaults to --model).")
  ```
- **`src/rlm/cli.py:107`**: Pass `child_model=child_model` to `load_config()`.
- **`src/rlm/cli.py:124`** (verbose output): Display both models in verbose mode.
- **`src/rlm/cli.py:144`** (cost estimation): Cost estimation currently uses `config.model` only. The `estimate_cost` function would need to be aware that child calls may use a different model. However, since the actual token counts come from the LLMClient, and each client already reports its own usage separately, this may mostly work -- but the summary cost display at the end uses the single `config.model` for pricing. This needs to be updated to price orchestrator tokens at the orchestrator model rate and child tokens at the child model rate.
- **`src/rlm/cli.py:17-42`** (MODEL_PRICING): Ensure `claude-opus-4-6` is in the pricing table (it already is as `"claude-opus-4.6"` -- verify the exact string litellm expects).

### 3. Orchestrator Layer (`orchestrator.py`)

**Current state** -- passes `self.config` to children unchanged:

- `src/rlm/orchestrator.py:45-48`: Creates `LLMClient(config, ...)` using full config
- `src/rlm/orchestrator.py:283-286`: Creates child orchestrator with `self.config` (same object)
- `src/rlm/orchestrator.py:314-317`: Creates direct-call LLMClient with `self.config`

**Changes needed**:

- **`src/rlm/orchestrator.py:280-291`** (`_recursive_call`): When creating child `RLMOrchestrator`, create a modified config where `model` is set to `self.config.child_model or self.config.model`. Two approaches:

  **Option A** (recommended): Create a child config copy:
  ```python
  child_config = self.config.model_copy(update={
      "model": self.config.child_model or self.config.model
  })
  sub_orchestrator = RLMOrchestrator(child_config, parent=self, ...)
  ```
  This means child orchestrators use the child model for their own LLM calls AND for any further recursive children.

  **Option B**: Have the orchestrator select which model to use based on whether it has a parent. This is more complex and less clean.

- **`src/rlm/orchestrator.py:309-322`** (`_direct_call`): The max-depth fallback also creates a new `LLMClient(self.config, ...)`. If this is called from a child orchestrator, `self.config.model` will already be the child model (with Option A), so no extra change needed. But consider: should the direct-call fallback use the child model or the orchestrator model? Likely the child model since it's a child's fallback.

### 4. LLM Client Layer (`llm/client.py`)

**Current state** -- uses `self.config.model` for all calls:

- `src/rlm/llm/client.py:51`: `model=self.config.model`
- `src/rlm/llm/client.py:53`: `temperature=self.config.temperature`

**No changes needed** if using Option A above. The LLMClient already reads `config.model` -- if the child config has the child model set as `model`, the client will use it automatically.

### 5. Trace Layer (`trace.py`, `orchestrator.py`)

**Current state** -- records `model` per orchestrator node:

- `src/rlm/trace.py:84`: `OrchestratorTrace` has `model: str` field
- `src/rlm/trace.py:20`: `LLMCallTrace` has `model: str` field
- `src/rlm/orchestrator.py:41`: Sets `model=config.model` in trace node

**No changes strictly needed** -- with Option A, each orchestrator's trace node already records its own `config.model`, which will naturally be the child model for child orchestrators. The trace will accurately reflect which model was used at each level.

**Optional enhancement**: Add `child_model` field to trace metadata so the root trace shows what child model was configured.

### 6. Documentation Files

**Files that need updating**:

- **`docs/reference.md:15-24`** (CLI options table): Add `--child-model` row. Also update `--model` default from `gpt-4o-mini` to `claude-opus-4-6` (it's already wrong -- code says `gpt-5-nano`).
- **`docs/reference.md:48-61`** (Configuration table): Add `RLM_CHILD_MODEL` row with `--child-model` flag and default `None` (falls back to `--model`).
- **`docs/how-to-guides.md:5-22`** (model section): Add a new section "How to use different models for orchestrator and child calls" with examples:
  ```bash
  # Use a reasoning model for orchestration, nano for sub-calls
  rlm run -q "Analyze this" -c data.txt -m claude-opus-4-6 --child-model gpt-5-nano

  # Set defaults via environment variables
  export RLM_MODEL=claude-opus-4-6
  export RLM_CHILD_MODEL=gpt-5-nano
  rlm run -q "Analyze this" -c data.txt
  ```
- **`docs/reference.md:226-297`** (trace format): Mention that `model` in child nodes reflects the child model.
- **`README.md`**: If model configuration is mentioned, update accordingly.

### 7. Tests (`tests/`)

**Files that need updating**:

- **`tests/test_orchestrator.py:14-16`** (config fixture): Add `child_model` to test configs. Add new tests for:
  - Child orchestrator uses child_model when set
  - Child orchestrator falls back to orchestrator model when child_model is None
  - Verify model propagation through recursive calls
  - Verify parallel map uses child model

- **`tests/test_trace.py`**: Add test verifying child trace nodes record the child model.

- **`tests/test_config.py`** (if exists) or add new test: Test `RLM_CHILD_MODEL` env var, CLI `--child-model` flag, and precedence.

## Code References

| File | Lines | What's There | Change Needed |
|------|-------|-------------|---------------|
| `src/rlm/types.py` | 91-101 | `RLMConfig` dataclass | Add `child_model` field, change `model` default |
| `src/rlm/config.py` | 14-23 | Env var mappings | Add `RLM_CHILD_MODEL` mapping |
| `src/rlm/cli.py` | 85-98 | `run` command options | Add `--child-model` option |
| `src/rlm/cli.py` | 107 | `load_config()` call | Pass `child_model` |
| `src/rlm/cli.py` | 124 | Verbose output | Show both models |
| `src/rlm/cli.py` | 144 | Cost estimation | Handle dual-model pricing |
| `src/rlm/cli.py` | 17-42 | `MODEL_PRICING` dict | Verify `claude-opus-4-6` key |
| `src/rlm/orchestrator.py` | 280-291 | `_recursive_call` | Create child config with child model |
| `src/rlm/orchestrator.py` | 293-307 | `_parallel_map` | Uses `_recursive_call` (inherits fix) |
| `src/rlm/orchestrator.py` | 309-322 | `_direct_call` | Already uses config.model (inherits fix) |
| `src/rlm/llm/client.py` | 50-54 | `completion()` call | No change needed (reads config.model) |
| `src/rlm/trace.py` | 78-87 | `OrchestratorTrace` | Optional: add child_model field |
| `docs/reference.md` | 15-24 | CLI options table | Add --child-model row, fix default |
| `docs/reference.md` | 48-61 | Config table | Add RLM_CHILD_MODEL row |
| `docs/how-to-guides.md` | 5-22 | Model how-to | Add dual-model section |
| `tests/test_orchestrator.py` | 14-16 | Config fixture | Add child_model tests |

## Architecture Insights

### Why This Change is Clean

1. **Separate LLMClient instances**: Each orchestrator (parent or child) creates its own `LLMClient` in `__init__`. There's no shared LLM client. This means changing the model per-orchestrator is just a matter of giving each orchestrator the right config.

2. **Config is a Pydantic model**: `RLMConfig` is a Pydantic `BaseModel`, so `model_copy(update={...})` is available for creating modified copies without mutating the original.

3. **Child orchestrators are already independent**: They have separate conversation histories, separate token counters, and separate trace nodes. The only shared thing is the config reference.

4. **Trace already records per-orchestrator model**: `OrchestratorTrace.model` and `LLMCallTrace.model` already exist, so the trace will naturally show different models at different depths.

### Design Decision: Should Children of Children Also Use the Child Model?

With Option A (recommended), yes. Once you set `child_config.model = child_model`, and child orchestrators pass their own config to their children, all descendants use the child model. This is the expected behavior -- only the top-level orchestrator uses the reasoning model.

### Cost Estimation Consideration

The current cost estimation in `cli.py:144-150` sums all tokens and prices them at one model's rate. With dual models, you'd need to either:
- Track orchestrator vs. child tokens separately and price each at their respective rates
- Or (simpler) sum costs from each LLMClient's token usage, priced at their individual model rates

The orchestrator already tracks `child_orchestrators` as a list. You could recursively sum costs from the tree.

## Open Questions

1. **Model name format**: litellm uses model names like `claude-opus-4-6` (with hyphens). The pricing table uses `claude-opus-4.6` (with dot). Need to verify which format litellm expects and ensure consistency.

2. **Should `--child-model` also accept a short flag?** Perhaps `-M` (uppercase)? This is a UX decision.

3. **Temperature per model**: Should child models have a separate temperature setting? Currently `temperature` is a single field. Reasoning models sometimes work better with `temperature=1.0` while smaller models might want `temperature=0.0`. This could be a follow-up.

4. **Direct call model at max depth**: When the orchestrator hits max recursion depth and falls back to `_direct_call`, should it use the orchestrator model (for better quality on the final answer) or the child model (since it's a child)? With Option A it would use the child model, which seems correct.
