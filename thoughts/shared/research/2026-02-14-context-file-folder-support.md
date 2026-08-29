---
date: 2026-02-14T12:00:00-05:00
researcher: claude
git_commit: 9558a4bf1f005d7686dfe1ca63f5243d53c2174f
branch: main
repository: rlm-secure
topic: "Context input: supporting any file type or folder exploration"
tags: [research, codebase, context, cli, orchestrator, file-input, folder-support, wasm, dsl]
status: complete
last_updated: 2026-02-14
last_updated_by: claude
---

# Research: Context Input — Supporting Any File Type or Folder Exploration

**Date**: 2026-02-14T12:00:00-05:00
**Researcher**: claude
**Git Commit**: 9558a4bf1f005d7686dfe1ca63f5243d53c2174f
**Branch**: main
**Repository**: rlm-secure

## Research Question

Current context is expected to be a file string. What needs to change so it can be any file type or a folder which can be explored?

## Core Design Principle

From RESEARCH.md — the foundational constraint of the entire system:

> "The 'root' LLM receives only the query. It knows the context exists but can't see it directly."
> "It never sees the full context directly. It never runs code itself. It only decides *what* to compute."

**The LLM never sees the raw context.** It only sees:
1. Metadata about the context (char count, file count, etc.)
2. Results of DSL operations (`slice`, `grep`, `count`, etc.)
3. Results of sandboxed Python code that ran inside the Wasm sandbox

The Wasm sandbox (`wasm_sandbox.py`) is the key enabler: it lets the LLM write Python code that runs *against* the context data inside an isolated sandbox. The context data goes into the sandbox as variables, not into the LLM's context window.

**Any approach that flattens/concatenates file contents and passes them through the pipeline as a single string is wrong** — it would send the context to the model (in bindings that get truncated to 4000 chars for display, and directly at max recursion depth via `_direct_call`). The correct approach extends the DSL and sandbox so the LLM can explore files/folders through operations, keeping the actual data out of the model's context.

## Current Architecture (Text-Only)

### How context flows today

```
CLI (cli.py:124-125)
  Path(context).read_text()  ->  context_text: str
      |
Orchestrator (orchestrator.py:98-99)
  ctx = Context(content=context_text)
  bindings = {"context": ctx.content}    <- entire file in memory as string
      |
  +------------------------------------------+
  | LLM never sees context directly          |
  | Only sees operation results              |
  +------------------------------------------+
      |
  DSL ops (ops/text.py)                  <- slice/grep/count work on string
      |
  Wasm sandbox (wasm_sandbox.py:134)     <- variables dict -> /sandbox/vars.json
                                            LLM's code reads from globals
```

Key files:
- `src/rlm/cli.py:89,124-125` — `--context` as `click.Path(exists=True)`, read via `Path.read_text()`
- `src/rlm/types.py:81-88` — `Context(content: str, hash: str)`
- `src/rlm/orchestrator.py:79,98-99` — `run(query, context_text: str)`, `bindings["context"] = ctx.content`
- `src/rlm/llm/prompts.py:7-8` — "You have a context variable containing text"
- `src/rlm/evaluator/wasm_sandbox.py:134,151` — writes vars as JSON, preopens sandbox dir
- `src/rlm/evaluator/lightweight.py:41-88` — ops resolve `bindings[args["input"]]` as strings

### Wasm sandbox mechanism (the key enabler)

`WasmSandbox.run(code, variables)` (wasm_sandbox.py:112-200):
1. Creates a temp directory as the sandbox root
2. Writes `variables` dict as `/sandbox/vars.json`
3. Writes wrapper script as `/sandbox/script.py`
4. Preopens ONLY the sandbox dir: `wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")`
5. Runs CPython-WASI with fuel/memory limits
6. Returns stdout

The LLM's sandboxed code accesses context through globals injected from vars.json. The context data stays inside the sandbox — it is never sent to the LLM.

## Proposed Architecture (File/Folder Support)

### Design: Filesystem-aware context with sandbox mounting

Instead of loading file contents into a string binding, the context becomes a **path** that is mounted into the sandbox filesystem. The LLM explores it through new DSL operations and through sandboxed code that can read files directly.

```
CLI
  context_path = Path(context)  ->  could be file or directory
      |
Orchestrator
  bindings = {
    "context_path": "/sandbox/context",    <- path reference, not content
    "context_type": "directory",           <- "file" | "directory"
    "context_manifest": "...",             <- JSON listing of files with metadata
  }
      |
  +--------------------------------------------------+
  | LLM sees: context_type, context_manifest         |
  | LLM does NOT see: raw file contents              |
  +--------------------------------------------------+
      |
  New DSL ops:
    list_files(path)     ->  JSON list of {name, size, type}
    read_file(path)      ->  string contents (text files)
    file_info(path)      ->  metadata (size, mime type, etc.)
      |
  Wasm sandbox:
    sandbox_dir/context/ <- context mounted read-only
    LLM's code can: os.listdir("/sandbox/context"),
                    open("/sandbox/context/data.csv"),
                    process binary files, etc.
```

### Layer-by-layer changes

#### 1. CLI (`cli.py`)

Accept files or directories. Don't read content — pass the path through.

```python
# Current
@click.option("--context", "-c", type=click.Path(exists=True))
context_text = Path(context).read_text()

# Proposed
@click.option("--context", "-c", type=click.Path(exists=True))
context_path = Path(context)
# Don't read the content — pass the path to the orchestrator
```

For stdin, we'd still need to materialize it to a temp file since the sandbox needs a filesystem path.

#### 2. Context model (`types.py`)

Extend `Context` to support both string content and path-based context:

```python
class ContextType(str, Enum):
    TEXT = "text"        # Single text string (current behavior)
    FILE = "file"        # Single file on disk
    DIRECTORY = "directory"  # Directory tree

class Context(BaseModel):
    context_type: ContextType
    path: Path | None = None         # For file/directory contexts
    content: str | None = None       # For text contexts (stdin, backward compat)
    hash: str = ""
```

#### 3. New DSL operations (`ops/filesystem.py`)

New operations that work against the context path, not string bindings:

- `list_files(path)` — list files/dirs at a path within the context. Returns JSON: `[{"name": "foo.txt", "type": "file", "size": 1234}, ...]`
- `read_file(path)` — read a file from the context into a binding. Returns the file content as a string (text files). For binary files: returns base64 or metadata.
- `file_info(path)` — get metadata about a file. Returns JSON: `{"size": 1234, "type": "text/csv", "lines": 500, ...}`

These ops do NOT pass file content to the LLM. They:
- `list_files`: returns a directory listing (small metadata)
- `read_file`: reads content into a **binding** that the LLM references by name
- `file_info`: returns only metadata

The LLM uses these to navigate the filesystem, then uses `slice`/`grep`/sandboxed code on the resulting bindings to process the actual data.

#### 4. Wasm sandbox changes (`wasm_sandbox.py`)

Mount the context directory read-only into the sandbox:

```python
# Current (line 151):
wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")

# Proposed — add context mounting:
wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")
if context_path is not None:
    wasi_config.preopen_dir(str(context_path), "/sandbox/context")
```

Now the LLM's sandboxed code can access files at `/sandbox/context/` — reading, parsing, and processing any file type (CSV, JSON, binary) inside the sandbox, without any of that data going to the model.

This is the **most powerful** capability: the LLM can write arbitrary Python to process any file type inside the sandbox.

#### 5. Orchestrator changes (`orchestrator.py`)

- Accept `context_path: Path` in addition to (or instead of) `context_text: str`
- Initialize bindings differently based on context type:
  - Text: `{"context": content}` (current behavior)
  - File: `{"context": content, "context_path": str(path)}` (for backward compat + filesystem access)
  - Directory: `{"context_type": "directory", "context_manifest": json_listing}`
- Pass `context_path` to `WasmSandbox` for mounting
- For `_direct_call` at max depth: send manifest/metadata instead of truncated content

#### 6. Prompt changes (`prompts.py`)

Conditional prompt based on context type:

- **For text (current):** "You have a context variable containing {context_chars} characters of text."
- **For file:** "You have a context file ({filename}, {size} bytes, {mime_type}). Use read_file, slice, grep to examine it. Use the sandbox to process it with Python."
- **For directory:** "You have a context directory containing {file_count} files ({total_size} bytes). Use list_files to explore the structure. Use read_file to load specific files. Use the sandbox to process files with Python."

#### 7. New operations registration

Register new ops in `EXPLORE_OPS` (`lightweight.py`):

```python
EXPLORE_OPS = {
    # Existing
    OpType.SLICE: op_slice,
    OpType.GREP: op_grep,
    OpType.COUNT: op_count,
    OpType.SPLIT: op_split,
    OpType.CHUNK: op_chunk,
    OpType.COMBINE: op_combine,
    # New filesystem ops
    OpType.LIST_FILES: op_list_files,
    OpType.READ_FILE: op_read_file,
    OpType.FILE_INFO: op_file_info,
}
```

### How the LLM would use this

**Example: directory with CSV files**

```
User: rlm run -q "What is the total revenue across all regions?" -c sales_data/

LLM sees: "You have a context directory containing 5 files (2.3 MB)"

LLM -> EXPLORE: list_files(context_path)
SYS -> [{"name": "north.csv", "size": 450000, "type": "file"},
        {"name": "south.csv", "size": 510000, "type": "file"}, ...]

LLM -> EXPLORE: file_info(context_path, "north.csv")
SYS -> {"size": 450000, "mime": "text/csv", "lines": 12500}

LLM -> EXPLORE: read_file(context_path, "north.csv", bind: "north_data")
SYS -> (file loaded into binding "north_data", 450000 chars)
      Result preview: "date,region,revenue,units\n2024-01-01,North,15230,..."

LLM -> EXPLORE: wasm_run(
  code: "import csv; reader = csv.reader(open('/sandbox/context/north.csv')); ..."
  bind: "north_total"
)
SYS -> "15234567.89"

LLM -> COMMIT: [
  list_files -> process each file via wasm_run -> combine(strategy="sum")
]
```

**Example: binary file (image)**

```
User: rlm run -q "What are the dimensions?" -c photo.jpg --wasm-python ./python.wasm

LLM sees: "You have a context file (photo.jpg, 2.1 MB, image/jpeg)"

LLM -> EXPLORE: file_info(context_path)
SYS -> {"size": 2100000, "mime": "image/jpeg"}

LLM -> EXPLORE: wasm_run(
  code: "import struct; f=open('/sandbox/context','rb'); ..."
  bind: "dimensions"
)
SYS -> "1920x1080"
```

## Backward Compatibility

The text path remains the default. Single text files work exactly as they do today — the `context` binding is a string, all existing ops work. The new filesystem ops are additive.

For text files:
- `rlm run -c file.txt` -> reads into string, binds as `context` (unchanged)
- `list_files`, `read_file`, `file_info` are not available (context is a string, not a path)

For directories:
- `rlm run -c data_dir/` -> does NOT read contents, passes path
- `list_files`, `read_file`, `file_info` become available
- Existing text ops (`slice`, `grep`) work on individual file contents after `read_file` loads them into bindings
- Sandboxed code can access `/sandbox/context/` directly in the Wasm sandbox

## Code References

- `src/rlm/cli.py:89` — CLI `--context` option definition
- `src/rlm/cli.py:124-125` — File reading: `Path(context).read_text()`
- `src/rlm/types.py:81-88` — `Context` model with `content: str`
- `src/rlm/orchestrator.py:79` — `run()` signature: `context_text: str`
- `src/rlm/orchestrator.py:98-99` — Context creation and binding
- `src/rlm/orchestrator.py:105-106` — Context chars injected into system prompt
- `src/rlm/orchestrator.py:335-348` — `_direct_call`: truncates context to 100k chars (must change for directories)
- `src/rlm/llm/prompts.py:7-8` — "You have a context variable containing text"
- `src/rlm/llm/prompts.py:12` — `Variables: context (your input data)`
- `src/rlm/evaluator/wasm_sandbox.py:112-200` — `WasmSandbox.run()`: variables as JSON, sandbox mounting
- `src/rlm/evaluator/wasm_sandbox.py:134` — `json.dumps(variables)` -> `/sandbox/vars.json`
- `src/rlm/evaluator/wasm_sandbox.py:151` — `wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")`
- `src/rlm/evaluator/lightweight.py:18-25` — `EXPLORE_OPS` registry
- `src/rlm/evaluator/lightweight.py:90-149` — Sandbox code execution: resolves variables, runs in Wasm
- `src/rlm/ops/text.py:12,20,29,39,47` — All ops resolve `bindings[args["input"]]`

## Historical Context (from thoughts/)

No existing research or plans address context input types. The Wasm plan (`thoughts/shared/plans/2026-02-10-wasm-eval-operation.md`) established the sandbox infrastructure that this feature builds on.

RESEARCH.md explicitly identifies this as future work:
> "Extension to Multimodal Contexts: The RLM paper focuses on text. But the framework generalizes — the 'context' could be images, audio, video, or structured data. The DSL would need new primitives (crop, transcribe, query_table), and the VM environments would need corresponding capabilities."

## Summary of Changes Needed

| Layer | File | What Changes |
|-------|------|-------------|
| CLI | `cli.py` | Accept file or dir, don't `read_text()` for dirs, pass path |
| Types | `types.py` | `ContextType` enum, `Context` supports path-based context |
| New ops | `ops/filesystem.py` | `list_files`, `read_file`, `file_info` |
| Op registry | `evaluator/lightweight.py` | Register new ops in `EXPLORE_OPS` |
| Op types | `types.py` | Add `LIST_FILES`, `READ_FILE`, `FILE_INFO` to `OpType` enum |
| Wasm sandbox | `wasm_sandbox.py` | Mount context dir read-only into sandbox |
| Orchestrator | `orchestrator.py` | Accept `context_path`, conditional binding setup |
| Prompt | `prompts.py` | Context-type-aware descriptions and op documentation |
| Direct call | `orchestrator.py` | Handle directory context at max depth (send manifest, not content) |
| Trace | `trace.py` | Track context type, file count, total size |

## Open Questions

1. **Wasm stdlib availability**: Does the CPython WASI binary include enough stdlib for processing common file types (csv, json, xml)? Need to verify what modules are available.
2. **Large file handling in read_file**: Should `read_file` support offset/limit like `slice`, or should the LLM always `read_file` then `slice`?
3. **Binary file access in non-Wasm mode**: Without Wasm, sandboxed code isn't available. Should filesystem ops support a `mode="binary"` that returns base64, or should binary files only be processable via the Wasm sandbox?
4. **Cache key for directories**: How to efficiently hash a directory tree for caching? `mtime`-based is fast but unreliable; content-hashing all files is correct but slow for large directories.
5. **Recursive calls with directory context**: When `rlm_call` or `map` creates child orchestrators, should they receive the full directory or a subset? A child working on one file shouldn't need the whole directory.
6. **Security**: Mounting user directories into the Wasm sandbox is safe (read-only, memory-isolated), but should we validate paths to prevent traversal (e.g., symlinks pointing outside the context)?
