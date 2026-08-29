# Context File/Folder Support - Plan Review

Review of `thoughts/shared/plans/2026-02-14-context-file-folder-support.md` against RESEARCH.md and original RLM paper intent.

---

## Issue 1: Breaking change eliminates the core RLM mechanism

**Severity**: Design concern

The plan removes the `context` string binding entirely, forcing the LLM to always `read_file` before doing anything. RESEARCH.md (lines 92-98) describes the fundamental pattern: "The context is stored as a variable in a programming environment." The current `bindings = {"context": ctx.content}` faithfully implements this.

Removing it adds an unnecessary extra LLM turn for text contexts and breaks the "context as opaque variable" abstraction that the RLM paper relies on.

**Recommendation**: Keep `bindings["context"]` for single-file text contexts. Auto-populate it with file contents. Only skip it for directories and binary files where it doesn't make sense. This preserves backward compatibility and the original RLM design.

**Decision**:

---

## Issue 2: Filesystem ops bypass the explore/commit speed guarantee

**Severity**: Design concern

RESEARCH.md (lines 249-254) says explore ops should be "fast (milliseconds per operation)." The plan registers `grep_file` (walks entire file tree, reads every text file) and `list_files` (computes recursive subdirectory sizes) in `EXPLORE_OPS`. These can be very slow on large directories.

The plan acknowledges this at line 863 but doesn't address it.

**Recommendation**: Add limits. `list_files` should not compute recursive sizes by default. `grep_file` on directories should have a max file count or total size cap, returning a truncation notice if exceeded.

**Decision**:

---

## Issue 3: `read_file` adds overhead for single-file text contexts

**Severity**: Design concern

The LLM must always do a two-step dance: `read_file` then `grep`/`slice`/etc. For single-file text contexts, this is pure overhead compared to the current direct `grep(input="context", ...)`. Each extra round-trip costs an LLM API call during explore mode.

The RLM paper's efficiency comes from minimizing LLM turns.

**Recommendation**: Same as Issue 1 -- auto-populate `context` binding for single-file text contexts.

**Decision**:

---

## Issue 4: Directory hashing is expensive at init time

**Severity**: Performance

`Context._compute_hash()` for directories walks the entire tree, stats every file, and builds a manifest. This runs at context creation time before the LLM starts. For large codebases (e.g., `rlm run -c src/`), this adds significant startup latency.

Top-level directory-context cache hits are rare anyway since the LLM's exploration path is non-deterministic.

**Recommendation**: Consider lazy hashing (compute on first cache check), a lighter hash (path + count + total size), or make it opt-in.

**Decision**:

---

## Issue 5: `_resolve_path` has fragile path traversal check

**Severity**: Bug

The string prefix check `str(resolved).startswith(str(root_resolved) + "/")` is fragile. Example: `root_resolved = "/tmp/ctx"` and `resolved = "/tmp/ctx_evil"` passes the check.

Also has a TOCTOU race between `resolve()` and file access (low risk for single-user CLI).

**Recommendation**: Use `Path.is_relative_to()` (Python 3.9+) instead of string prefix comparison:
```python
if not resolved.is_relative_to(root_resolved):
    raise ValueError(...)
```

**Decision**:

---

## Issue 6: Wasm context mounting has asymmetric behavior

**Severity**: Minor design issue

Single-file contexts get copied into a subdirectory (`shutil.copy2`). Directory contexts get preopened as a second mount. This asymmetry means single-file contexts see stale data if the file changes, while directory contexts don't.

**Recommendation**: Use consistent behavior -- always preopen as a second mount point. For single files, preopen the parent directory or create a temp directory with a symlink.

**Decision**:

---

## Issue 7: `_recursive_call` leaks temp files

**Severity**: Bug

`tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)` is never cleaned up. Over many recursive calls (especially with `map`), this leaks temp files on disk.

**Recommendation**: Track temp files and clean them up when the orchestrator finishes, or use a shared temp directory that's cleaned up as a batch.

**Decision**:

---

## Issue 8: System prompt `.format()` has injection risk

**Severity**: Bug

`SYSTEM_PROMPT.format(context_description=context_desc, ...)` where `context_desc` includes `context_path.name`. If a filename contains `{` or `}` characters, this raises `KeyError` or produces garbled output. Same class of vulnerability fixed in commit `f320a3a`.

**Recommendation**: Use `.format_map()` with a `defaultdict`, sanitize the filename, or build the prompt through string concatenation.

**Decision**:

---

## Issue 9: `grep_file` vulnerable to ReDoS

**Severity**: Bug

`re.search(pattern, line)` where `pattern` comes from the LLM. A malicious or hallucinated regex could cause catastrophic backtracking. `grep_file` amplifies this by running the regex against every line in every file in a directory.

**Recommendation**: Add `re.compile()` with a try/except for `re.error` at minimum. Consider a regex timeout or complexity limit.

**Decision**:

---

## Issue 10: Binary file base64 return is dead code

**Severity**: Design concern

`read_file` returns base64 for binary files, but no DSL operations can work with base64 data and the LLM can't reason about binary content in text form. The Wasm sandbox path (`open("/sandbox/context/file.bin", "rb")`) already handles binary processing.

**Recommendation**: Return an error/descriptor for binary files from `read_file` (e.g., "binary file, use eval to process via Wasm sandbox") rather than base64.

**Decision**:

---

## Issue 11: Cache key computation not addressed

**Severity**: Gap in plan

The cache system keys on `(op_type, args, input_hashes)`. Filesystem ops derive behavior from `_context_path` in bindings, but the plan doesn't discuss how cache keys incorporate the context path. Two different directories with the same relative file structure would produce colliding cache keys.

**Recommendation**: Add a section to the plan addressing how `_context_path` and `_context_type` factor into cache key computation.

**Decision**:

---

## Issue 12: No migration path for breaking API change

**Severity**: Minor design issue

`orchestrator.run(query, context_text: str)` changes to `orchestrator.run(query, context_path: Path)`. Any external consumers of the orchestrator API break instantly with no transition path.

**Recommendation**: Consider a transitional period where both signatures work (union type `str | Path`), or document as a breaking change with a version bump.

**Decision**:

---

## Alignment Summary

**Aligns well with**:
- RLM paper's goal of handling non-text contexts
- RESEARCH.md vision of LLM exploring context structure before committing
- DSL approach (new ops are pure functions with defined semantics)

**Misaligns with**:
- "Context as opaque variable" paradigm -- paper's LLM operates on a pre-loaded variable, not a filesystem
- Explore phase speed guarantee -- filesystem ops can be slow
- "Every operation is a pure function" (RESEARCH.md line 178) -- `read_file` depends on external filesystem state
