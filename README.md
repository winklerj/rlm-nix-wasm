# rlm-nix-wasm

Sandboxed Recursive Language Models with Nix.

rlm-nix-wasm lets LLMs break down large-context problems into smaller recursive sub-calls. Each call operates through a structured DSL instead of arbitrary code execution, with content-addressed caching and optional Nix sandboxing.

## Quick Start

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
rlm run -q "How many unique users are in this log?" -c server.log
rlm run -q "How many unique users are in this log?" -c server.log -v --trace
rlm list-model-pricing
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
rlm run -q "What are the unique user IDs?" -c server.log --wasm-python ./python.wasm
```

See [How-to Guides](docs/how-to-guides.md) for setup details and [Explanation](docs/explanation.md) for design rationale.

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
