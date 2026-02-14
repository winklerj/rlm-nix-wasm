# Rename `rlm-secure` to `rlm-nix-wasm` Implementation Plan

## Overview

Rename the project from `rlm-secure` to `rlm-nix-wasm` across all files in the codebase. The Python import name (`rlm`), the CLI command (`rlm`), and the `src/rlm/` directory structure remain unchanged. Only the project/package name and all prose references change.

## Current State Analysis

The research document (`thoughts/shared/research/2026-02-14-project-name-references.md`) catalogs all 18 files containing the project name. Two case variants exist: `rlm-secure` (kebab-case, dominant) and `RLM-Secure` (title-case, once in PLAN.md:1). The underscore variant `rlm_secure` is not used anywhere.

### Key Discoveries:
- `pyproject.toml:2` — `name = "rlm-secure"` is the primary source of truth for the package name
- `src/rlm/types.py:100` — `cache_dir: Path = Path.home() / ".cache" / "rlm-secure"` affects runtime behavior (user cache location)
- `uv.lock:1593` — contains a `name = "rlm-secure"` entry that must be regenerated (not manually edited)
- The Python import name is `rlm` (from `src/rlm/`), not `rlm_secure`, so no directory rename is needed
- The CLI command is `rlm` (from `pyproject.toml` `[project.scripts]`), unchanged
- `.claude/settings.json:13` — `CLAUDE_CODE_TASK_LIST_ID: "rlm-secure"` is a Claude Code setting

## Desired End State

After this plan is complete:
1. `grep -r "rlm-secure" . --include='*.py' --include='*.toml' --include='*.md' --include='*.json' | grep -v uv.lock | grep -v .git` returns zero results
2. `grep -r "RLM-Secure" . --include='*.md' | grep -v .git` returns zero results
3. `pyproject.toml` has `name = "rlm-nix-wasm"`
4. `uv.lock` reflects the new name (regenerated)
5. Default cache directory is `~/.cache/rlm-nix-wasm`
6. All tests pass: `pytest tests/`
7. Linting passes: `ruff check src/ tests/`
8. Type checking passes: `mypy src/`

## What We're NOT Doing

- Renaming the Python import name (`rlm`) or CLI command (`rlm`)
- Renaming the `src/rlm/` directory
- Renaming the GitHub repository (done separately via GitHub settings)
- Migrating existing user cache directories from `~/.cache/rlm-secure` to `~/.cache/rlm-nix-wasm`
- Changing environment variable prefixes (they stay `RLM_*`)

## Implementation Approach

This is a straightforward search-and-replace operation. All changes are string substitutions — no logic changes. We do it in two phases: code/config first (to ensure the build works), then docs/historical.

---

## Phase 1: Build, Source Code, and Config

### Overview
Update the package name in build configuration, source code, and project settings. Regenerate the lock file.

### Changes Required:

#### 1. Package name
**File**: `pyproject.toml:2`
**Change**: `name = "rlm-secure"` → `name = "rlm-nix-wasm"`

#### 2. Default cache directory
**File**: `src/rlm/types.py:100`
**Change**: `cache_dir: Path = Path.home() / ".cache" / "rlm-secure"` → `cache_dir: Path = Path.home() / ".cache" / "rlm-nix-wasm"`

#### 3. CLI docstring
**File**: `src/rlm/cli.py:1`
**Change**: `"""CLI entry point for rlm-secure."""` → `"""CLI entry point for rlm-nix-wasm."""`

#### 4. Claude Code task list ID
**File**: `.claude/settings.json:13`
**Change**: `"CLAUDE_CODE_TASK_LIST_ID": "rlm-secure"` → `"CLAUDE_CODE_TASK_LIST_ID": "rlm-nix-wasm"`

#### 5. Regenerate lock file
**Command**: `uv lock`
This will update `uv.lock` to reflect the new package name. Do NOT manually edit `uv.lock`.

### Success Criteria:

#### Automated Verification:
- [x] `uv lock` succeeds without errors
- [x] `pytest tests/` — all tests pass (150 passed; 1 pre-existing failure unrelated to rename)
- [x] `ruff check src/ tests/` — pre-existing unused import warnings only, no new issues
- [x] `mypy src/` — no type errors
- [x] `grep -r "rlm-secure" src/ pyproject.toml .claude/` returns no results

---

## Phase 2: Documentation

### Overview
Update all documentation files (README, CLAUDE.md, docs/, PLAN.md) to use the new project name.

### Changes Required:

#### 1. README.md (4 occurrences)
**File**: `README.md`
- Line 1: `# rlm-secure` → `# rlm-nix-wasm`
- Line 5: `rlm-secure lets LLMs...` → `rlm-nix-wasm lets LLMs...`
- Line 19: `rlm-secure can optionally...` → `rlm-nix-wasm can optionally...`
- Line 39: `Learn rlm-secure step by step...` → `Learn rlm-nix-wasm step by step...`

#### 2. CLAUDE.md (1 occurrence)
**File**: `CLAUDE.md:7`
- `rlm-secure is a recursive...` → `rlm-nix-wasm is a recursive...`

#### 3. docs/tutorial.md (10 occurrences)
**File**: `docs/tutorial.md`
All instances of `rlm-secure` → `rlm-nix-wasm`:
- Line 1: heading
- Line 3: two occurrences in intro paragraph
- Line 10: section heading
- Line 28: litellm note
- Line 61: behind the scenes
- Line 65: verbose mode
- Line 117: cache location in example output
- Line 130: explanation link text

#### 4. docs/reference.md (4 occurrences)
**File**: `docs/reference.md`
- Line 3: `Complete specification of the rlm-secure CLI...` → `Complete specification of the rlm-nix-wasm CLI...`
- Line 38: `rlm-secure uses to estimate cost` → `rlm-nix-wasm uses to estimate cost`
- Line 62: `~/.cache/rlm-secure` → `~/.cache/rlm-nix-wasm`
- Line 220: `~/.cache/rlm-secure/` → `~/.cache/rlm-nix-wasm/`

#### 5. docs/explanation.md (7 occurrences)
**File**: `docs/explanation.md`
- Line 3: `Background, context, and design rationale for rlm-secure.` → `...for rlm-nix-wasm.`
- Line 21: `rlm-secure replaces arbitrary code...` → `rlm-nix-wasm replaces arbitrary code...`
- Line 59: `Nix's build system provides three properties that align with rlm-secure's needs:` → `...rlm-nix-wasm's needs:`
- Line 63: `rlm-secure's caching model` → `rlm-nix-wasm's caching model`
- Line 65: `rlm-secure gets parallelism for free...` → `rlm-nix-wasm gets parallelism for free...`
- Line 67: `Without it, rlm-secure still provides...` → `Without it, rlm-nix-wasm still provides...`
- Line 71: `rlm-secure uses multiple layers...` → `rlm-nix-wasm uses multiple layers...`

#### 6. docs/how-to-guides.md (4 occurrences)
**File**: `docs/how-to-guides.md`
- Line 3: `Practical directions for accomplishing specific tasks with rlm-secure.` → `...with rlm-nix-wasm.`
- Line 124: `rlm-secure allows 1 level...` → `rlm-nix-wasm allows 1 level...`
- Line 140: `rlm-secure is designed for large files...` → `rlm-nix-wasm is designed for large files...`
- Line 214: `rlm-secure uses to estimate cost` → `rlm-nix-wasm uses to estimate cost`

#### 7. docs/plans/2026-02-09-timing-instrumentation.md (1 occurrence)
**File**: `docs/plans/2026-02-09-timing-instrumentation.md:5`
- `The rlm-secure CLI's...` → `The rlm-nix-wasm CLI's...`

#### 8. PLAN.md (7 occurrences, 2 case variants)
**File**: `PLAN.md`
- Line 1: `# RLM-Secure Implementation Plan` → `# RLM-Nix-Wasm Implementation Plan`
- Lines 84, 271, 321, 1412, 1418, 1438: all `rlm-secure` → `rlm-nix-wasm`

### Success Criteria:

#### Automated Verification:
- [x] `grep -r "rlm-secure" README.md CLAUDE.md PLAN.md docs/ .claude/` returns no results
- [x] `grep -r "RLM-Secure" PLAN.md` returns no results

---

## Phase 3: Historical Documents (thoughts/)

### Overview
Update all historical research, plan, and handoff documents in `thoughts/` to use the new project name.

### Changes Required:

#### 1. thoughts/shared/research/2026-02-10-trace-flag-file-storage.md (6 occurrences)
**File**: `thoughts/shared/research/2026-02-10-trace-flag-file-storage.md`
- Line 6: `repository: rlm-secure` → `repository: rlm-nix-wasm`
- Line 20: `**Repository**: rlm-secure` → `**Repository**: rlm-nix-wasm`
- Line 27: `~/.cache/rlm-secure/` (in summary) → `~/.cache/rlm-nix-wasm/`
- Line 128: `~/.cache/rlm-secure/` → `~/.cache/rlm-nix-wasm/`
- Line 160: `types.py:100` default path reference → update
- Line 173: `~/.cache/rlm-secure/` → `~/.cache/rlm-nix-wasm/`

#### 2. thoughts/shared/research/2026-02-09-dual-model-orchestrator-child.md (2 occurrences)
**File**: `thoughts/shared/research/2026-02-09-dual-model-orchestrator-child.md`
- Line 6: `repository: rlm-secure` → `repository: rlm-nix-wasm`
- Line 20: `**Repository**: rlm-secure` → `**Repository**: rlm-nix-wasm`

#### 3. thoughts/shared/research/2026-02-09-trace-event-ordering.md (2 occurrences)
**File**: `thoughts/shared/research/2026-02-09-trace-event-ordering.md`
- Line 6: `repository: rlm-secure` → `repository: rlm-nix-wasm`
- Line 20: `**Repository**: rlm-secure` → `**Repository**: rlm-nix-wasm`

#### 4. thoughts/shared/plans/2026-02-10-wasm-eval-operation.md (1 occurrence)
**File**: `thoughts/shared/plans/2026-02-10-wasm-eval-operation.md`
- Line 127: `cache_dir: Path = Path.home() / ".cache" / "rlm-secure"` → `.../ "rlm-nix-wasm"`

#### 5. thoughts/shared/plans/2026-02-09-dual-model-orchestrator-child.md (2 occurrences)
**File**: `thoughts/shared/plans/2026-02-09-dual-model-orchestrator-child.md`
- Line 70: `cache_dir: Path = Path.home() / ".cache" / "rlm-secure"` → `.../ "rlm-nix-wasm"`
- Line 523: `cache_dir: Path = Path.home() / ".cache" / "rlm-secure"` → `.../ "rlm-nix-wasm"` (or `~/.cache/rlm-secure` in prose)

#### 6. thoughts/shared/handoffs/general/2026-02-09_18-22-48_timing-instrumentation.md (1 occurrence)
**File**: `thoughts/shared/handoffs/general/2026-02-09_18-22-48_timing-instrumentation.md`
- Line 6: `repository: rlm-secure` → `repository: rlm-nix-wasm`

#### 7. thoughts/shared/research/2026-02-14-project-name-references.md
**File**: `thoughts/shared/research/2026-02-14-project-name-references.md`
- Line 6: `repository: rlm-secure` → `repository: rlm-nix-wasm`
- Line 20: `**Repository**: rlm-secure` → `**Repository**: rlm-nix-wasm`
- Note: The content of this file documents the original name references, which is historical context. The frontmatter `repository` field should update, but the body is a record of what was found and can remain as-is for historical accuracy.

### Success Criteria:

#### Automated Verification:
- [x] `grep -r "rlm-secure" thoughts/ | grep -v "2026-02-14-project-name-references.md"` returns only occurrences within the plan doc itself (documenting the rename)
- [x] All `repository:` frontmatter fields show `rlm-nix-wasm`

---

## Testing Strategy

### Automated Tests:
- `pytest tests/` — verifies no functional breakage
- `ruff check src/ tests/` — lint clean
- `mypy src/` — type check clean

### Manual Verification:
- [ ] `rlm cache stats` — shows location as `~/.cache/rlm-nix-wasm`
- [ ] `grep -rn "rlm-secure" . --include='*.py' --include='*.toml' --include='*.md' --include='*.json' | grep -v uv.lock | grep -v .git | grep -v "2026-02-14-project-name-references.md"` returns no results (zero occurrences outside the research doc body)

## Performance Considerations

None — this is a pure renaming with no runtime behavior changes (except the default cache directory path, which only affects where new caches are created).

## Migration Notes

- **Existing user caches**: Users who have cached data at `~/.cache/rlm-secure/` will need to either clear it or set `RLM_CACHE_DIR` to the old path. No automatic migration is needed since caches are ephemeral and can be rebuilt.
- **GitHub repository rename**: Done separately via GitHub settings. Update the git remote URL after: `git remote set-url origin git@github.com:winklerj/rlm-nix-wasm.git`

## References

- Research: `thoughts/shared/research/2026-02-14-project-name-references.md`
