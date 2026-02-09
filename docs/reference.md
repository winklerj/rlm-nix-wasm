# Reference

Complete specification of the rlm-secure CLI, operations, configuration, and architecture.

## CLI Commands

### `rlm run`

Execute a query against a context.

```
rlm run [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--query` | `-q` | string | required | The question to answer |
| `--context` | `-c` | path | stdin | Path to the context file. If omitted, reads from stdin |
| `--model` | `-m` | string | `gpt-4o-mini` | LLM model identifier (any litellm-supported model) |
| `--max-explore` | | int | `20` | Maximum explore steps before forcing a commit |
| `--max-depth` | | int | `1` | Maximum recursion depth |
| `--use-nix` | | flag | `false` | Compile operations to Nix derivations |
| `--verbose` | `-v` | flag | `false` | Show model name, context size, operation trace, and timing |

### `rlm cache stats`

Display cache statistics: entry count, total size, and cache directory location.

### `rlm cache clear`

Delete all cached operation results.

### `rlm list-model-pricing`

Show the known model prices that `rlm-secure` uses to estimate cost.

```bash
rlm list-model-pricing
```

The command prints a table (if `rich` is installed) or plain text with:

- model name
- input price per 1M tokens
- output price per 1M tokens

## Configuration

All settings can be configured via environment variables. CLI flags take precedence over environment variables. Environment variables take precedence over defaults.

| Setting | Environment Variable | CLI Flag | Default |
|---------|---------------------|----------|---------|
| LLM model | `RLM_MODEL` | `--model` | `gpt-4o-mini` |
| Max explore steps | `RLM_MAX_EXPLORE_STEPS` | `--max-explore` | `20` |
| Max commit cycles | `RLM_MAX_COMMIT_CYCLES` | -- | `5` |
| Max recursion depth | `RLM_MAX_RECURSION_DEPTH` | `--max-depth` | `1` |
| Max parallel jobs | `RLM_MAX_PARALLEL_JOBS` | -- | `4` |
| Cache directory | `RLM_CACHE_DIR` | -- | `~/.cache/rlm-secure` |
| Enable Nix | `RLM_USE_NIX` | `--use-nix` | `false` |
| Verbose output | `RLM_VERBOSE` | `--verbose` | `false` |

LLM provider API keys are set via their standard environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See the [litellm provider documentation](https://docs.litellm.ai/docs/providers) for the full list.

## Operations

### Explore-mode operations

These operations are available during the explore phase. They execute in-process and return results to the LLM immediately.

#### `slice`

Extract a substring from the input.

| Argument | Type | Description |
|----------|------|-------------|
| `input` | string | The input text (or a variable reference) |
| `start` | int | Start position (character offset) |
| `end` | int | End position (character offset) |

Returns the substring from `start` to `end`.

#### `grep`

Filter lines matching a regular expression.

| Argument | Type | Description |
|----------|------|-------------|
| `input` | string | The input text |
| `pattern` | string | A Python regular expression |

Returns all lines where the pattern matches, separated by newlines.

#### `count`

Count lines or characters.

| Argument | Type | Description |
|----------|------|-------------|
| `input` | string | The input text |
| `mode` | string | `"lines"` or `"chars"` |

Returns the count as a string.

#### `chunk`

Split input into N equal-sized pieces.

| Argument | Type | Description |
|----------|------|-------------|
| `input` | string | The input text |
| `n` | int | Number of chunks |

Returns a JSON array of strings.

#### `split`

Split input on a delimiter.

| Argument | Type | Description |
|----------|------|-------------|
| `input` | string | The input text |
| `delimiter` | string | The delimiter string |

Returns a JSON array of strings.

#### `combine`

Merge multiple results using a strategy.

| Argument | Type | Description |
|----------|------|-------------|
| `inputs` | list or string | A JSON array of strings (or a variable reference to one) |
| `strategy` | string | `"concat"`, `"sum"`, or `"vote"` |

- `"concat"`: Joins all inputs with newlines.
- `"sum"`: Parses each input as a number and returns the sum.
- `"vote"`: Returns the most common value (majority vote).

### Commit-mode operations

All explore-mode operations are also available in commit mode. Commit mode adds:

#### `rlm_call`

Spawn a recursive RLM sub-call.

| Argument | Type | Description |
|----------|------|-------------|
| `query` | string | The sub-question to answer |
| `context` | string | The context for the sub-call (or a variable reference) |

Returns the sub-call's final answer as a string.

#### `map`

Apply `rlm_call` to each element of an array in parallel.

| Argument | Type | Description |
|----------|------|-------------|
| `prompt` | string | The question to ask for each element |
| `input` | string | A variable reference to a JSON array |

Returns a JSON array of answers, one per input element. Elements are processed in parallel up to `RLM_MAX_PARALLEL_JOBS` concurrent workers.

### Variable binding

In commit mode, each operation can include a `"bind"` field that names the variable to store its result in. Subsequent operations reference previous results by variable name in their `input` or `context` fields. The special variable `"context"` refers to the original user-provided context.

Example commit plan:

```json
{
  "mode": "commit",
  "operations": [
    {"op": "chunk", "args": {"input": "context", "n": 4}, "bind": "chunks"},
    {"op": "map", "args": {"prompt": "Count errors in this section", "input": "chunks"}, "bind": "counts"},
    {"op": "combine", "args": {"inputs": "counts", "strategy": "sum"}, "bind": "total"}
  ],
  "output": "total"
}
```

## Cache

The cache stores operation results in a content-addressed filesystem layout:

```
~/.cache/rlm-secure/
  ab/
    cd/
      abcd1234...   # full SHA-256 hash as filename
```

Cache keys are computed from the SHA-256 hash of the operation type and its input content. Two calls with identical operation types and identical inputs always produce the same cache key, regardless of when or where they run.

## Architecture

```
src/rlm/
  types.py           # Core data models (Operation, ExploreAction, CommitPlan)
  config.py          # Configuration from env vars / CLI
  cli.py             # Click CLI (rlm run, rlm cache)
  orchestrator.py    # Main explore/commit loop
  llm/
    client.py        # LiteLLM wrapper
    parser.py        # JSON response parsing
    prompts.py       # System prompt for the protocol
  ops/
    text.py          # Text operations (slice, grep, count, split, chunk)
    recursive.py     # Combine operation
  evaluator/
    lightweight.py   # In-process execution with caching
    sandbox.py       # Bubblewrap sandbox (optional)
  cache/
    store.py         # Content-addressed filesystem cache
  nix/
    compiler.py      # DSL -> Nix expression compiler
    builder.py       # nix-build wrapper
    store.py         # nix-store wrapper
    templates.py     # Nix derivation templates
```

### Execution flow

1. The CLI (`cli.py`) parses arguments and reads the context file or stdin.
2. `RLMOrchestrator` initializes an `LLMClient` conversation with the system prompt from `prompts.py`.
3. The orchestrator enters the **explore loop**: sends the query to the LLM, receives a JSON response, parses it with `parser.py`, executes the requested operation via `lightweight.py` (with cache lookup via `store.py`), and appends the result to the conversation.
4. When the LLM switches to commit mode, the orchestrator executes the **commit plan**: a sequence of operations with variable bindings. `rlm_call` operations spawn recursive `RLMOrchestrator` instances. `map` operations use a `ThreadPoolExecutor` for parallelism.
5. When the Nix path is enabled (`--use-nix`), non-recursive operations are compiled to Nix expressions via `compiler.py` and built via `builder.py`.
6. When the LLM switches to final mode, the orchestrator returns the answer.
7. At maximum recursion depth, the orchestrator bypasses the protocol and makes a direct LLM call with the context truncated to 100K characters.
