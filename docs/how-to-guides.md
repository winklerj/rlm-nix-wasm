# How-to Guides

Practical directions for accomplishing specific tasks with rlm-secure.

## How to use a different LLM model

Pass `--model` (or `-m`) with any [litellm-supported model identifier](https://docs.litellm.ai/docs/providers):

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
rlm run -q "Summarize this" -c doc.txt -m claude-sonnet-4-20250514

# OpenAI GPT-4o
rlm run -q "Summarize this" -c doc.txt -m gpt-4o

# Set a default model via environment variable
export RLM_MODEL=claude-sonnet-4-20250514
rlm run -q "Summarize this" -c doc.txt
```

CLI flags take precedence over environment variables.

## How to pipe data from stdin

Omit the `-c` flag and pipe data in:

```bash
cat server.log | rlm run -q "How many 500 errors?"
curl -s https://example.com/data.csv | rlm run -q "What is the average of column 3?"
```

## How to enable Nix sandboxing

Install [Nix](https://nixos.org/download.html), then pass `--use-nix`:

```bash
rlm run -q "Process this data" -c data.txt --use-nix
```

This compiles operations into Nix derivations that run in isolated sandboxes with no network access and no filesystem access outside the build directory. Independent operations build in parallel automatically.

To enable Nix by default:

```bash
export RLM_USE_NIX=true
```

## How to increase recursion depth

By default, rlm-secure allows 1 level of recursion (the root call can spawn sub-calls, but sub-calls cannot spawn further sub-calls). For deeply nested problems, increase the limit:

```bash
rlm run -q "Analyze this large dataset" -c data.csv --max-depth 3
```

Or via environment variable:

```bash
export RLM_MAX_RECURSION_DEPTH=3
```

At maximum depth, sub-calls fall back to a direct LLM call with a truncated context (100K characters) instead of using the explore/commit protocol.

## How to process very large files

rlm-secure is designed for large files. The LLM never sees the full context -- it explores the data incrementally and emits a computation plan. For best results with very large files:

1. **Increase explore steps** so the LLM has more room to understand the data structure:

```bash
rlm run -q "Find all anomalies" -c huge.log --max-explore 40
```

2. **Increase recursion depth** so the LLM can break the problem into more levels:

```bash
rlm run -q "Summarize each section" -c book.txt --max-depth 3
```

3. **Increase parallel jobs** (via environment variable) so map operations run faster:

```bash
export RLM_MAX_PARALLEL_JOBS=8
rlm run -q "Analyze each chapter" -c book.txt
```

## How to trace execution

Use `--trace` to write a detailed JSON record of every step in the run -- LLM messages in and out, explore operations with full results, commit plans with per-operation detail, and the full recursion tree:

```bash
rlm run -q "How many errors?" -c server.log --trace trace.json
```

The trace is written after the run completes. Inspect it with:

```bash
python3 -m json.tool trace.json | less
```

Or use `jq` to drill into specific parts:

```bash
# Show each LLM round-trip
jq '[.root.events[] | select(.type == "llm_call")] | .[] | {call_number, elapsed_s, input_tokens, output_tokens}' trace.json

# Show explore steps
jq '[.root.events[] | select(.type == "explore_step")] | .[] | {step_number, operation_op, cached}' trace.json

# Show recursive child calls
jq '.root.children[] | {trace_id, depth, query, elapsed_s}' trace.json
```

`--trace` works independently of `--verbose` -- you can use both together or either alone. See the [Reference](reference.md) for the full trace JSON schema.

## How to manage the cache

View cache statistics (entry count, total size, location):

```bash
rlm cache stats
```

Clear all cached results:

```bash
rlm cache clear
```

Change the cache directory:

```bash
export RLM_CACHE_DIR=/tmp/rlm-cache
```

## How to see model pricing

You can see the prices that `rlm-secure` uses to estimate cost.

```bash
rlm list-model-pricing
```

This shows each model name, plus input and output price per 1M tokens.

## How to run the development tools

Install development dependencies:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest
```

Run the linter and type checker:

```bash
uv run ruff check src/
uv run mypy src/rlm
```
