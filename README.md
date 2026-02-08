# rlm-secure

Sandboxed Recursive Language Models with Nix.

rlm-secure lets LLMs break down large-context problems into smaller recursive sub-calls. Each call operates through a structured DSL instead of arbitrary code execution, with content-addressed caching and optional Nix sandboxing.

## Quick Start

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
rlm run -q "How many unique users are in this log?" -c server.log
```

## Documentation

| | |
|---|---|
| [Tutorial](docs/tutorial.md) | Learn rlm-secure step by step with a hands-on walkthrough |
| [How-to Guides](docs/how-to-guides.md) | Solve specific problems: change models, enable sandboxing, tune performance |
| [Reference](docs/reference.md) | Complete specification of the CLI, operations, configuration, and architecture |
| [Explanation](docs/explanation.md) | Understand the design: why recursive models, why the explore/commit protocol, why Nix |

## Design Documents

- [RESEARCH.md](RESEARCH.md) -- Full design document covering RLMs, security model, caching, and Nix integration
- [PLAN.md](PLAN.md) -- Implementation plan with all seven phases
