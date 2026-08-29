# Context File/Folder Support Implementation Plan

## Overview

Extend the RLM context system from text-only strings to supporting any file type or directory as context. The LLM explores files through new DSL operations and the Wasm sandbox's mounted filesystem, never seeing raw file contents directly.

**Design choice: Mount only.** All contexts (files and directories) are mounted into the sandbox filesystem. There is no `context` string binding -- the LLM uses `read_file` to load specific files into bindings, then operates on those bindings with existing text ops. This is a breaking change from the current string-based approach but produces a cleaner, uniform design.

## Current State Analysis

- `cli.py:124-125` -- always does `Path(context).read_text()`, passes text string
- `types.py:81-88` -- `Context(content: str, hash: str)` -- string-only
- `orchestrator.py:79,98-99` -- `run(query, context_text: str)`, creates `bindings = {"context": ctx.content}`
- `wasm_sandbox.py:151` -- only preopens the sandbox temp dir, no context mounting
- `prompts.py:7-8` -- hardcoded "You have a context variable containing text"
- `lightweight.py:18-25` -- `EXPLORE_OPS` only has text/recursive ops
- All ops in `ops/text.py` resolve `bindings[args["input"]]` as strings

### Key Discoveries:
- The Wasm sandbox (`wasm_sandbox.py:151`) already uses `wasi_config.preopen_dir()` -- adding a second preopen for context is straightforward
- The evaluator (`lightweight.py:41-88`) dispatches sandboxed-code ops specially via `_execute_eval()` -- filesystem ops will use the standard `EXPLORE_OPS` registry
- Internal bindings prefixed with `_` can be used for metadata without conflicting with user-facing bindings
- The `OpExecutor` protocol (`ops/base.py:8-12`) takes `(args, bindings)` -- filesystem ops will read `_context_path` and `_context_type` from bindings

## Desired End State

After implementation:
1. `rlm run -c file.txt -q "..."` mounts the file and exposes it through filesystem ops
2. `rlm run -c data_dir/ -q "..."` mounts the directory and lets the LLM explore all files
3. `echo "text" | rlm run -q "..."` materializes stdin to a temp file and mounts it
4. The LLM uses `list_files`, `read_file`, `file_info`, `grep_file` to explore context
5. Existing text ops (`slice`, `grep`, `count`, etc.) work on bindings populated by `read_file`
6. The Wasm sandbox can access files at `/sandbox/context/` for arbitrary Python processing
7. Binary files can be processed through the Wasm sandbox (`open("/sandbox/context/photo.jpg", "rb")`)

### Verification:
- `pytest` passes (all new and existing tests)
- `mypy src/` passes
- `ruff check src/ tests/` passes
- Manual test: `rlm run -c tests/ -q "How many test files are there?" --wasm-python ./python.wasm`

## What We're NOT Doing

- Image/audio/video processing in DSL ops (handled by Wasm sandbox code)
- Recursive directory watching or live file system monitoring
- Compression/archive extraction (zip, tar, etc.)
- Network-based context sources (URLs, S3, etc.)
- Changes to Nix sandboxing (filesystem ops run in-process only, like existing text ops)

## Implementation Approach

The implementation flows bottom-up: types first, then operations, then integration layers. This ensures each phase produces testable, self-contained changes.

Internal bindings (prefixed with `_`) carry metadata through the pipeline:
- `_context_path`: real filesystem path to context root (file or directory)
- `_context_type`: `"file"` or `"directory"`

Filesystem ops read these from bindings to resolve paths. The Wasm sandbox receives `context_path` separately for mounting. The LLM never sees `_`-prefixed bindings.

---

## Phase 1: Types

### Overview
Add context type enum, extend Context model, add new OpType values.

### Changes Required:

#### 1. Context types and new operations
**File**: `src/rlm/types.py`
**Changes**: Add `ContextType` enum, extend `Context`, add filesystem `OpType` values

```python
class ContextType(str, Enum):
    """Type of context input."""
    FILE = "file"
    DIRECTORY = "directory"
```

Add to `OpType` enum:
```python
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    FILE_INFO = "file_info"
    GREP_FILE = "grep_file"
```

Extend `Context`:
```python
class Context(BaseModel):
    """A context object for an RLM run."""
    context_type: ContextType
    path: Path                           # Real filesystem path
    content: str | None = None           # Only for backward compat / stdin temp files
    hash: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash based on context type."""
        if self.context_type == ContextType.FILE:
            # Hash the file content
            return hashlib.sha256(self.path.read_bytes()).hexdigest()
        else:
            # Hash manifest: sorted (relative_path, size, mtime) tuples
            entries = []
            for p in sorted(self.path.rglob("*")):
                if p.is_file():
                    stat = p.stat()
                    rel = str(p.relative_to(self.path))
                    entries.append(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}")
            manifest = "\n".join(entries)
            return hashlib.sha256(manifest.encode()).hexdigest()
```

### Success Criteria:

#### Automated Verification:
- [ ] Type checking passes: `mypy src/rlm/types.py`
- [ ] Linting passes: `ruff check src/rlm/types.py`
- [ ] Existing tests still pass: `pytest tests/test_ops.py tests/test_evaluator.py`

---

## Phase 2: Filesystem Operations

### Overview
New `ops/filesystem.py` with four operations: `list_files`, `read_file`, `file_info`, `grep_file`. All resolve paths relative to the context root stored in `_context_path` binding.

### Changes Required:

#### 1. Filesystem operations module
**File**: `src/rlm/ops/filesystem.py` (new file)
**Changes**: Implement four filesystem operations

```python
"""Filesystem operations for exploring context files and directories."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path


def _resolve_path(args: dict, bindings: dict[str, str]) -> tuple[Path, Path]:
    """Resolve a path argument relative to the context root.

    Returns (context_root, resolved_path).
    Raises ValueError on path traversal attempts.
    """
    context_root = Path(bindings["_context_path"])
    relative = args.get("path", "")

    if relative:
        resolved = (context_root / relative).resolve()
    else:
        resolved = context_root.resolve()

    # Security: ensure resolved path is within context root
    root_resolved = context_root.resolve()
    if resolved != root_resolved and not str(resolved).startswith(str(root_resolved) + "/"):
        raise ValueError(
            f"Path traversal detected: '{relative}' resolves outside context root"
        )

    return root_resolved, resolved


def op_list_files(args: dict, bindings: dict[str, str]) -> str:
    """List files at a path within the context.

    Args (via args dict):
        path: relative path within context (optional, defaults to root)

    Returns: JSON array of {name, type, size} objects.
    """
    root, target = _resolve_path(args, bindings)

    if target.is_file():
        # Single file context or pointing at a file
        stat = target.stat()
        return json.dumps([{
            "name": target.name,
            "type": "file",
            "size": stat.st_size,
        }])

    entries = []
    for child in sorted(target.iterdir()):
        entry = {"name": child.name}
        if child.is_dir():
            entry["type"] = "directory"
            entry["size"] = sum(
                f.stat().st_size for f in child.rglob("*") if f.is_file()
            )
        else:
            entry["type"] = "file"
            entry["size"] = child.stat().st_size
        entries.append(entry)

    return json.dumps(entries)


def op_read_file(args: dict, bindings: dict[str, str]) -> str:
    """Read a file from the context into a binding.

    Args (via args dict):
        path: relative path to file within context (optional for single-file context)
        offset: byte offset to start reading from (optional, default 0)
        limit: max bytes to read (optional, default all)

    Returns: file contents as string (text files) or base64 (binary files).
    """
    _, target = _resolve_path(args, bindings)

    if target.is_dir():
        raise ValueError(f"Cannot read_file on a directory: '{args.get('path', '')}'")

    offset = args.get("offset", 0)
    limit = args.get("limit")

    # Try text first, fall back to binary
    try:
        with open(target, "r") as f:
            if offset:
                f.seek(offset)
            if limit:
                return f.read(limit)
            return f.read()
    except UnicodeDecodeError:
        import base64
        with open(target, "rb") as f:
            if offset:
                f.seek(offset)
            data = f.read(limit) if limit else f.read()
            return base64.b64encode(data).decode("ascii")


def op_file_info(args: dict, bindings: dict[str, str]) -> str:
    """Get metadata about a file or directory.

    Args (via args dict):
        path: relative path within context (optional, defaults to root)

    Returns: JSON object with metadata.
    """
    root, target = _resolve_path(args, bindings)

    stat = target.stat()
    mime_type, _ = mimetypes.guess_type(str(target))

    info: dict[str, object] = {
        "name": target.name,
        "size": stat.st_size,
        "type": "directory" if target.is_dir() else "file",
        "mime": mime_type,
    }

    if target.is_file():
        # Count lines for text files
        try:
            content = target.read_text()
            info["lines"] = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            info["chars"] = len(content)
        except UnicodeDecodeError:
            info["binary"] = True

    elif target.is_dir():
        file_count = sum(1 for f in target.rglob("*") if f.is_file())
        total_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        info["file_count"] = file_count
        info["total_size"] = total_size

    return json.dumps(info)


def op_grep_file(args: dict, bindings: dict[str, str]) -> str:
    """Grep for a pattern within context files.

    Args (via args dict):
        pattern: regex pattern to search for
        path: relative path within context (optional, defaults to root)

    For a single file: returns matching lines.
    For a directory: returns file:line_number:line matches across all text files.
    """
    root, target = _resolve_path(args, bindings)
    pattern = args["pattern"]

    if target.is_file():
        try:
            content = target.read_text()
        except UnicodeDecodeError:
            return ""
        lines = content.split("\n")
        matched = [line for line in lines if re.search(pattern, line)]
        return "\n".join(matched)

    # Directory: grep across all files
    results = []
    for f in sorted(target.rglob("*")):
        if not f.is_file():
            continue
        try:
            content = f.read_text()
        except UnicodeDecodeError:
            continue
        rel = str(f.relative_to(root))
        for i, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line):
                results.append(f"{rel}:{i}:{line}")

    return "\n".join(results)
```

#### 2. Register in evaluator
**File**: `src/rlm/evaluator/lightweight.py`
**Changes**: Import and register new ops in `EXPLORE_OPS`

Add imports:
```python
from rlm.ops.filesystem import op_list_files, op_read_file, op_file_info, op_grep_file
```

Add to `EXPLORE_OPS`:
```python
EXPLORE_OPS = {
    OpType.SLICE: op_slice,
    OpType.GREP: op_grep,
    OpType.COUNT: op_count,
    OpType.SPLIT: op_split,
    OpType.CHUNK: op_chunk,
    OpType.COMBINE: op_combine,
    OpType.LIST_FILES: op_list_files,
    OpType.READ_FILE: op_read_file,
    OpType.FILE_INFO: op_file_info,
    OpType.GREP_FILE: op_grep_file,
}
```

### Success Criteria:

#### Automated Verification:
- [ ] Type checking passes: `mypy src/rlm/ops/filesystem.py`
- [ ] Linting passes: `ruff check src/rlm/ops/filesystem.py`
- [ ] New unit tests pass: `pytest tests/test_filesystem_ops.py`
- [ ] Existing tests pass: `pytest tests/test_ops.py tests/test_evaluator.py`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the filesystem ops behavior is correct before proceeding.

---

## Phase 3: Wasm Sandbox Mounting

### Overview
Extend the Wasm sandbox to accept an optional context path and mount it read-only into the sandbox filesystem at `/sandbox/context`.

### Changes Required:

#### 1. WasmSandbox context mounting
**File**: `src/rlm/evaluator/wasm_sandbox.py`
**Changes**: Add `context_path` parameter to `run()`

Update `run()` signature:
```python
def run(self, code: str, variables: dict[str, str],
        context_path: Path | None = None) -> str:
```

After the existing `wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")` line (line 151), add context mounting:

```python
            wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")

            # Mount context into sandbox
            if context_path is not None:
                context_resolved = context_path.resolve()
                if context_resolved.is_file():
                    # Create a context subdirectory with just this file
                    context_mount = sandbox_dir / "context"
                    context_mount.mkdir()
                    import shutil
                    shutil.copy2(str(context_resolved), str(context_mount / context_resolved.name))
                    # Already under sandbox_dir, so accessible via /sandbox/context/
                else:
                    wasi_config.preopen_dir(str(context_resolved), "/sandbox/context")
```

This way:
- **Single file**: copied into `/sandbox/context/<filename>` within the sandbox temp dir (already preopened as `/sandbox`)
- **Directory**: mounted directly at `/sandbox/context/`

#### 2. Evaluator passes context_path to sandbox
**File**: `src/rlm/evaluator/lightweight.py`
**Changes**: Update `_execute_eval_op()` to pass context_path to the sandbox

In `_execute_eval()`, extract and pass context_path:

```python
    def _execute_eval(self, op: Operation, bindings: dict[str, str]) -> OpResult:
        ...
        # Resolve variables: only those listed in inputs, or all bindings
        if input_names is not None:
            variables = {name: bindings[name] for name in input_names if name in bindings}
        else:
            variables = dict(bindings)

        # Extract context path for sandbox mounting (not a user-visible variable)
        context_path = None
        raw_path = bindings.get("_context_path")
        if raw_path is not None:
            context_path = Path(raw_path)

        # Strip internal bindings from variables passed to sandbox
        variables = {k: v for k, v in variables.items() if not k.startswith("_")}

        ...

        # Run in sandbox (pass context_path for mounting)
        with self.profile.measure("evaluator", "wasm_exec"):
            result_value = self.wasm_sandbox.run(code, variables, context_path=context_path)
            result_value = result_value.rstrip("\n")
```

### Success Criteria:

#### Automated Verification:
- [ ] Type checking passes: `mypy src/rlm/evaluator/`
- [ ] Linting passes: `ruff check src/rlm/evaluator/`
- [ ] Existing Wasm tests still pass: `pytest tests/test_wasm_sandbox.py tests/test_eval.py` (when `RLM_WASM_PYTHON_PATH` is set)

---

## Phase 4: Orchestrator, CLI, and Prompts

### Overview
Wire everything together: the orchestrator accepts a context path instead of text, sets up internal bindings, uses context-type-aware prompts, and the CLI detects file vs directory.

### Changes Required:

#### 1. Orchestrator
**File**: `src/rlm/orchestrator.py`
**Changes**: Accept `context_path` instead of `context_text`, set up filesystem bindings

Change `run()` signature and body:

```python
    def run(self, query: str, context_path: Path, depth: int = 0) -> str:
        """Run an RLM query against a context.

        Args:
            query: The question to answer.
            context_path: Path to the context file or directory.
            depth: Current recursion depth.

        Returns:
            The final answer as a string.
        """
        self.trace_node.depth = depth
        self.trace_node.query = query
        run_start = time.monotonic()

        context_path = context_path.resolve()

        if depth > self.config.max_recursion_depth:
            return self._direct_call(query, context_path)

        ctx = Context(
            context_type=ContextType.DIRECTORY if context_path.is_dir() else ContextType.FILE,
            path=context_path,
        )
        self.trace_node.context_length = self._context_size(context_path)

        # Set up bindings with internal metadata
        bindings: dict[str, str] = {
            "_context_path": str(context_path),
            "_context_type": ctx.context_type.value,
        }
```

Build context-aware system prompt:

```python
        import mimetypes

        # Build context description for the system prompt
        if ctx.context_type == ContextType.DIRECTORY:
            file_count = sum(1 for f in context_path.rglob("*") if f.is_file())
            total_size = sum(f.stat().st_size for f in context_path.rglob("*") if f.is_file())
            context_desc = (
                f"You have a context directory containing {file_count} files "
                f"({self._human_size(total_size)}). "
                f"Use list_files to explore its structure. "
                f"Use read_file to load specific files into variables. "
                f"Use grep_file to search across files."
            )
        else:
            size = context_path.stat().st_size
            mime, _ = mimetypes.guess_type(str(context_path))
            context_desc = (
                f"You have a context file ({context_path.name}, "
                f"{self._human_size(size)}, {mime or 'unknown type'}). "
                f"Use read_file to load it into a variable. "
                f"Use file_info for metadata. "
                f"Use grep_file to search within it."
            )
```

Format the system prompt (requires updating the template -- see prompt changes below):

```python
        system = SYSTEM_PROMPT.format(
            context_description=context_desc,
            query=query,
            wasm_ops=wasm_ops,
            wasm_approach=wasm_approach,
            filesystem_ops=FILESYSTEM_OPS_ADDENDUM,
        )
```

Update `_direct_call`:

```python
    def _direct_call(self, query: str, context_path: Path) -> str:
        """Direct LLM call at max recursion depth (no explore/commit)."""
        if context_path.is_dir():
            entries = []
            for f in sorted(context_path.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(context_path)
                    entries.append(f"  {rel} ({f.stat().st_size} bytes)")
            manifest = "\n".join(entries[:100])  # Cap at 100 files
            context_summary = f"Directory with files:\n{manifest}"
        else:
            max_chars = 100_000
            try:
                content = context_path.read_text()
                context_summary = content[:max_chars]
            except UnicodeDecodeError:
                context_summary = (
                    f"[Binary file: {context_path.name}, "
                    f"{context_path.stat().st_size} bytes]"
                )

        client = LLMClient(self.config, verbose=self.config.verbose,
                            console=self.console,
                            trace=self.trace_collector,
                            trace_node=self.trace_node)
        client.set_system_prompt(
            "Answer the following query based on the provided context. "
            "Be precise and concise."
        )
        return client.send(f"Query: {query}\n\nContext:\n{context_summary}")
```

Update `_recursive_call` -- the `rlm_call` op in commit plans provides a binding name as its `context` argument. That binding is a string (from `read_file`). For recursive calls, we materialize it to a temp file:

```python
    def _recursive_call(self, query: str, context_text_or_path: str | Path, depth: int) -> str:
        """Spawn a recursive RLM call."""
        with self.profile.measure("recursive", "recursive_call", depth=depth + 1):
            child_config = self.config
            if self.config.child_model:
                child_config = self.config.model_copy(update={
                    "model": self.config.child_model,
                    "child_model": None,
                })
            sub_orchestrator = RLMOrchestrator(
                child_config, parent=self,
                trace_collector=self.trace_collector,
            )
            self.child_orchestrators.append(sub_orchestrator)

            # If given a string, materialize to temp file
            if isinstance(context_text_or_path, str):
                import tempfile
                tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                tmp.write(context_text_or_path)
                tmp.close()
                context_path = Path(tmp.name)
            else:
                context_path = context_text_or_path

            result = sub_orchestrator.run(query, context_path, depth=depth + 1)
            if self.trace_collector.enabled:
                self.trace_node.children.append(sub_orchestrator.trace_node)
            return result
```

In `_execute_commit_plan`, update `rlm_call` and `map` handling:

```python
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
                before_count = len(self.child_orchestrators)
                result_value = self._parallel_map(prompt, items, depth)
```

And `_parallel_map` similarly passes strings that get materialized to temp files:
```python
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

Add helper methods:

```python
    @staticmethod
    def _context_size(path: Path) -> int:
        """Get total size of context in bytes."""
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    @staticmethod
    def _human_size(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        size = float(num_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
```

#### 2. CLI
**File**: `src/rlm/cli.py`
**Changes**: Pass path to orchestrator instead of text. Handle stdin as temp file.

Replace the context reading block (lines 123-130):

```python
    # Resolve context from file, directory, or stdin
    if context:
        context_path = Path(context)
    elif not sys.stdin.isatty():
        import tempfile
        stdin_content = sys.stdin.read()
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write(stdin_content)
        tmp.close()
        context_path = Path(tmp.name)
    else:
        console.print("[red]Error: provide --context or pipe context via stdin[/red]")
        raise SystemExit(1)
```

Update verbose output (lines 132-140):

```python
    if verbose:
        console.print(f"[dim]Model: {config.model}[/dim]")
        if config.child_model:
            console.print(f"[dim]Child model: {config.child_model}[/dim]")
        if context_path.is_dir():
            file_count = sum(1 for f in context_path.rglob("*") if f.is_file())
            console.print(f"[dim]Context: directory with {file_count} files[/dim]")
        else:
            console.print(f"[dim]Context: {context_path.name} ({context_path.stat().st_size:,} bytes)[/dim]")
        if config.use_nix:
            console.print("[dim]Nix sandboxing: enabled[/dim]")
        if config.wasm_python_path:
            console.print(f"[dim]Wasm sandbox: {config.wasm_python_path}[/dim]")
```

Update orchestrator call (line 149):

```python
    answer = orchestrator.run(query, context_path)
```

#### 3. System prompt
**File**: `src/rlm/llm/prompts.py`
**Changes**: Replace hardcoded text context description with a format variable, add filesystem ops documentation

Replace the context description line (line 7-8):
```python
# Old:
# You have a context variable containing text. You cannot see it directly.

# New:
# {context_description} You cannot see the context directly.
```

Add filesystem ops addendum:

```python
FILESYSTEM_OPS_ADDENDUM = (
    '- `list_files(path?)` -- list files/directories at a path within the context. '
    'Returns JSON array of {{name, type, size}}.\n'
    '- `read_file(path?, offset?, limit?)` -- read a file into a variable. '
    'Text files return content as string. Binary files return base64.\n'
    '- `file_info(path?)` -- get metadata about a file (size, mime type, line count). '
    'Returns JSON.\n'
    '- `grep_file(pattern, path?)` -- search for regex pattern within files. '
    'For directories, searches across all text files and returns file:line:match format.\n'
)
```

Update the operations section of `SYSTEM_PROMPT` to include `{filesystem_ops}`:

```python
## Available Operations

{context_description} You cannot see the context directly. Instead, you use operations to examine and process it.

{filesystem_ops}
- `slice(input, start, end)` -- extract a substring from a variable
- `grep(input, pattern)` -- find lines matching a regex in a variable
...
```

Add filesystem approach guidance:

```python
FILESYSTEM_APPROACH_ADDENDUM = (
    '6. EXPLORE FILES FIRST -- Use list_files to understand the context structure. '
    'Use file_info to check file sizes before reading. '
    'Use grep_file to find relevant content across files. '
    'Use read_file to load specific files into variables for detailed processing.\n'
)
```

### Success Criteria:

#### Automated Verification:
- [ ] Type checking passes: `mypy src/`
- [ ] Linting passes: `ruff check src/`
- [ ] Existing orchestrator tests updated and passing: `pytest tests/test_orchestrator.py`

#### Manual Verification:
- [ ] `rlm run -c <text_file> -q "..." -v` works and shows file info
- [ ] `rlm run -c <directory> -q "..." -v` works and shows directory info
- [ ] `echo "text" | rlm run -q "..." -v` works via stdin

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the end-to-end flow works correctly before proceeding to tests.

---

## Phase 5: Tests

### Overview
Comprehensive tests for all new functionality.

### Changes Required:

#### 1. Filesystem operations unit tests
**File**: `tests/test_filesystem_ops.py` (new file)

Tests for all four filesystem ops covering:
- `list_files`: directory listing, single file, subdirectory navigation
- `read_file`: text files, directory files by path, offset/limit, binary files (base64), error on directory
- `file_info`: text file metadata (lines, chars), directory metadata (file_count, total_size), binary detection
- `grep_file`: single file matching, directory-wide grep with file:line:match format, regex patterns, no-match case
- Path traversal: parent directory references blocked, symlinks outside context blocked

#### 2. Updated orchestrator tests
**File**: `tests/test_orchestrator.py`
**Changes**: Update all tests to use `context_path: Path` instead of `context_text: str`

All tests currently call `orch.run("query", "some context text")`. Update to:
```python
@pytest.fixture
def context_file(tmp_path):
    f = tmp_path / "context.txt"
    f.write_text("some context")
    return f

# Update calls from:
#   result = orch.run("query", "some context text")
# To:
#   result = orch.run("query", context_file)
```

The LLM mock responses also need updating -- the LLM would first `read_file` before using text ops on the content. For test simplicity, adjust the mock sequence to include an initial `read_file` step.

#### 3. Evaluator test updates
**File**: `tests/test_evaluator.py`
**Changes**: Add tests for filesystem ops through the evaluator

```python
class TestFilesystemOps:
    def test_list_files(self, evaluator, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        bindings = {"_context_path": str(tmp_path), "_context_type": "directory"}
        op = Operation(op=OpType.LIST_FILES, args={})
        result = evaluator.execute(op, bindings)
        entries = json.loads(result.value)
        assert len(entries) == 2

    def test_read_file(self, evaluator, tmp_path):
        (tmp_path / "data.txt").write_text("hello world")
        bindings = {"_context_path": str(tmp_path), "_context_type": "directory"}
        op = Operation(op=OpType.READ_FILE, args={"path": "data.txt"}, bind="data")
        result = evaluator.execute(op, bindings)
        assert result.value == "hello world"
```

### Success Criteria:

#### Automated Verification:
- [ ] All tests pass: `pytest`
- [ ] Type checking passes: `mypy src/`
- [ ] Linting passes: `ruff check src/ tests/`
- [ ] Test coverage for filesystem ops: `pytest tests/test_filesystem_ops.py -v`

#### Manual Verification:
- [ ] Full pipeline test with a real directory: `rlm run -c src/ -q "How many Python files?" -v`
- [ ] Full pipeline test with a single file: `rlm run -c README.md -q "Summarize" -v`
- [ ] Wasm sandbox file access: `rlm run -c data.csv -q "Sum the values" --wasm-python ./python.wasm -v`

---

## Testing Strategy

### Unit Tests:
- All four filesystem ops with file and directory contexts
- Path traversal prevention (parent references, symlinks)
- Binary file handling (base64 encoding)
- Offset/limit on read_file
- Empty directories, deeply nested paths
- Context hash computation for both files and directories

### Integration Tests:
- Orchestrator with file context (mock LLM does read_file -> grep -> final)
- Orchestrator with directory context (mock LLM does list_files -> read_file -> final)
- Recursive calls with file context
- `_direct_call` with directory context (sends manifest)

### Edge Cases:
- Empty files
- Binary files in grep_file (should skip without error)
- Very large directories (performance of list_files, manifest generation)
- Unicode filenames
- Context with no files (empty directory)

## Performance Considerations

- `list_files` computes subdirectory sizes which requires walking the tree -- for deeply nested directories this could be slow. Consider making subdirectory size computation optional in a follow-up.
- `grep_file` on large directories reads every text file -- this is O(total_text_size). The LLM should use `list_files` first to narrow down, then `grep_file` on specific subdirectories.
- Directory context hashing walks the tree for metadata -- computed once per `Context` instantiation.

## Migration Notes

This is a **breaking change** for the `orchestrator.run()` API:
- Old: `run(query: str, context_text: str, depth: int = 0)`
- New: `run(query: str, context_path: Path, depth: int = 0)`

Any code calling `orchestrator.run()` directly needs to pass a `Path` instead of a string. The CLI handles this transparently. Stdin input is materialized to a temp file.

## Security Considerations

- **Path traversal**: `_resolve_path()` validates that resolved paths are within the context root
- **Symlink escape**: `.resolve()` follows symlinks, then checks containment -- symlinks pointing outside are rejected
- **Wasm isolation**: Context is mounted read-only in the sandbox. The sandbox cannot modify context files.
- **No shell execution**: `grep_file` uses Python's `re` module, not shell commands -- no injection risk

## References

- Research: `thoughts/shared/research/2026-02-14-context-file-folder-support.md`
- Wasm sandbox: `src/rlm/evaluator/wasm_sandbox.py`
- Current context flow: `src/rlm/orchestrator.py:79-99`
- OpExecutor protocol: `src/rlm/ops/base.py:8-12`
