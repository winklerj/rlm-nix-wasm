# rlm-nix-wasm

Sandboxed Recursive Language Models with Nix.

rlm-nix-wasm lets LLMs break down large-context problems into smaller recursive sub-calls. Each call operates through a structured DSL instead of arbitrary code execution, with content-addressed caching and Nix sandboxing (enabled by default).

## Quick Start

```bash
# Install dependencies (requires Nix)
nix-shell --run "uv pip install -e ."

# Set your API key (loaded automatically from .env)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run a query (always use nix-shell for native dependency support)
nix-shell --run "uv run rlm run -q 'How many unique users are in this log?' -c server.log --wasm-python .wasm/python-3.12.0.wasm"
nix-shell --run "uv run rlm run -q 'How many unique users are in this log?' -c server.log --wasm-python .wasm/python-3.12.0.wasm -v --trace"
nix-shell --run "uv run rlm list-model-pricing"
```

## Wasm Code Execution

rlm-nix-wasm can optionally run LLM-generated Python code inside a WebAssembly (WASI) sandbox. This bridges the structured DSL with arbitrary logic when needed (regex, math, custom filtering), while maintaining security through hardware-level memory isolation, no filesystem or network access.

```bash
# 1. Install wasmtime
pip install wasmtime

# 2. Download python.wasm (~25MB)
curl -L -o python.wasm \
  "https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python/3.12.0%2B20231211-040d5a6/python-3.12.0.wasm"

# 3. Run with Wasm eval enabled
nix-shell --run "uv run rlm run -q 'What are the unique user IDs?' -c server.log --wasm-python ./python.wasm"
```

See [How-to Guides](docs/how-to-guides.md) for setup details and [Explanation](docs/explanation.md) for design rationale.

## Docker

Run without installing Nix or managing Python environments.

```bash
# 1. Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 2. Build and start the web demo
docker compose up --build
# Open http://localhost:8765
```

To run the CLI against a local file, put it in a `data/` directory:

```bash
cp server.log data/
docker compose run --rm rlm rlm run \
  -q "How many unique users?" \
  -c /app/data/server.log \
  --wasm-python /app/.wasm/python-3.12.0.wasm
```

Or mount an arbitrary file directly:

```bash
docker compose run --rm -v ~/server.log:/app/input.log rlm \
  rlm run -q "How many unique users?" -c /app/input.log \
  --wasm-python /app/.wasm/python-3.12.0.wasm
```

## Documentation

| | |
|---|---|
| [Tutorial](docs/tutorial.md) | Learn rlm-nix-wasm step by step with a hands-on walkthrough |
| [How-to Guides](docs/how-to-guides.md) | Solve specific problems: change models, enable sandboxing, tune performance |
| [Reference](docs/reference.md) | Complete specification of the CLI, operations, configuration, and architecture |
| [Explanation](docs/explanation.md) | Understand the design: why recursive models, why the explore/commit protocol, why Nix |

## Design Documents

- [RESEARCH.md](RESEARCH.md) -- Full design document covering RLMs, security model, caching, and Nix integration
- [PLAN.md](PLAN.md) -- Implementation plan with all seven phases
