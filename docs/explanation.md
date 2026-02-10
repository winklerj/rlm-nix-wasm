# Explanation

Background, context, and design rationale for rlm-secure.

For the full design document, see [RESEARCH.md](../RESEARCH.md).

## Why recursive language models?

LLMs have a hard limit on how much text they can process at once (the context window). Even when a model technically supports a large window -- say, 200K tokens -- its quality *degrades* as you fill it. This is called context rot: the longer the input, the more likely the model is to miss details or make mistakes.

Recursive Language Models solve this by never showing the full context to any single LLM call. Instead, the LLM breaks the problem into smaller sub-problems, solves each in a clean context, and combines the results. Each sub-call sees only the relevant slice of data, so accuracy stays high regardless of total input size.

## Why not just let the LLM write code?

The original RLM approach (from the [alexzhang13/rlm](https://github.com/alexzhang13/rlm) reference implementation) gives the LLM a Python REPL and lets it write arbitrary code to process the data. This works, but introduces serious problems:

1. **Security**: Arbitrary Python can read files, open network connections, or damage the host system. There's no isolation between the LLM's code and the rest of the machine.
2. **Caching**: There's no way to tell if two code snippets produce the same result without running both. Identical sub-questions are recomputed every time.
3. **Parallelism**: Recursive calls execute sequentially because arbitrary code can have side effects, making it unsafe to run calls in parallel.

rlm-secure replaces arbitrary code with a structured DSL -- a fixed set of pure operations like `slice`, `grep`, `count`, `chunk`, and `combine`. Because these operations are pure functions (output depends only on input, with no side effects), they can be safely cached and parallelized.

### The eval escape hatch

While the DSL covers most common tasks, sometimes the LLM needs arbitrary logic that the fixed operations can't express (complex regex, arithmetic, conditional filtering). The `eval` operation provides a controlled middle ground: the LLM writes Python code, but it runs inside a WebAssembly (WASI) sandbox with hardware-enforced isolation.

| Property | DSL operations | Eval (Wasm sandbox) |
|----------|:-:|:-:|
| Isolation | Pure functions, no side effects | Wasm memory isolation, no FS/network |
| Caching | Deterministic, content-addressed | Deterministic (same code + inputs = same key) |
| Speed | In-process, ~microseconds | Wasm startup + execution, ~100ms-1s |
| Expressiveness | Fixed operation set | Full Python stdlib |
| Security model | No code execution | Hardware-level sandbox |

The system prompt steers the LLM toward DSL operations first, only falling back to eval when necessary. This keeps the common path fast and cacheable while allowing arbitrary logic when needed.

## The explore/commit protocol

The protocol has two modes because they serve different purposes:

**Explore mode** lets the LLM understand the data incrementally. The LLM issues one operation at a time (e.g., "show me the first 500 characters", "how many lines are there?") and sees the result immediately. This is like scanning a document before deciding how to process it. Explore operations are fast and cheap -- they run in-process with no overhead.

**Commit mode** lets the LLM execute a batch computation plan. Once the LLM understands the data structure, it emits a sequence of operations including recursive sub-calls and parallel map operations. The orchestrator executes these operations, resolving variable dependencies and parallelizing where possible.

The separation exists because exploration requires low-latency feedback (the LLM needs to see each result before deciding the next step), while computation benefits from batch planning (the orchestrator can optimize and parallelize a known plan).

## Content-addressed caching

Every operation result is cached using a key derived from the SHA-256 hash of the operation type and its input content. This means:

- **Automatic deduplication**: If two different queries grep the same file for the same pattern, the second one is a cache hit.
- **No manual invalidation**: If the file changes, its content hash changes, so old cache entries are never incorrectly reused.
- **Deterministic keys**: The same operation on the same input always produces the same cache key, regardless of when or where it runs.

The cache layout (`{hash[:2]}/{hash[2:4]}/{hash}`) mirrors Nix store conventions, keeping directory listings manageable even with millions of entries.

## Why Nix?

Nix's build system provides three properties that align with rlm-secure's needs:

1. **Sandboxing**: Nix builds run in isolated environments with no network access and no filesystem access outside the build directory. This is the same isolation mechanism that protects NixOS package builds from interfering with each other.

2. **Content-addressed storage**: Nix already names build outputs by the hash of their inputs. This matches rlm-secure's caching model exactly -- when Nix sees the same derivation twice, it skips the build and returns the cached result.

3. **Automatic parallelism**: `nix-build` automatically schedules independent derivations in parallel, up to a configurable number of jobs. rlm-secure gets parallelism for free by compiling its operations into a dependency graph of Nix derivations.

Nix integration is optional. Without it, rlm-secure still provides caching and parallelism (via Python's `ThreadPoolExecutor`), but without process-level isolation.

## Defense in depth

rlm-secure uses multiple layers of protection:

| Layer | Always active | Description |
|-------|:---:|-------------|
| **Structured DSL** | Yes | The LLM can only emit predefined operations. No arbitrary code execution. |
| **Pydantic validation** | Yes | Every LLM response is parsed and validated against strict type schemas. Malformed output is rejected. |
| **Content-addressed cache** | Yes | Prevents redundant computation. Cache keys are deterministic. |
| **Recursion depth limit** | Yes | Prevents infinite recursion. Configurable, defaults to 1. |
| **Operation timeout** | Yes | Each sandboxed operation times out after 30 seconds. |
| **Wasm sandbox** | Optional | WebAssembly sandbox for eval operations. Hardware-level memory isolation, no FS/network. |
| **Bubblewrap sandbox** | Optional | Lightweight Linux container with read-only filesystem and no network. |
| **Nix derivation sandbox** | Optional | Full build-time isolation via Nix. No network, no host filesystem access. |

The outermost layer (structured DSL) is the most important. Even without any sandboxing, the LLM cannot execute arbitrary code -- it can only request operations from the predefined set.

## Trade-offs

**DSL expressiveness vs. security**: A richer DSL would let the LLM solve more complex problems, but each new operation is a potential attack surface. The current set is deliberately minimal.

**Recursion depth vs. cost**: Deeper recursion lets the LLM handle more complex problems, but each recursive call is an LLM API call. The default depth of 1 balances capability against cost.

**Nix overhead vs. isolation**: Nix provides stronger isolation than in-process execution, but adds latency for derivation compilation and build. For small operations, the overhead may exceed the operation time. Use Nix when security matters more than speed.

**Cache size vs. speed**: The cache grows without bound. For long-running deployments, periodic `rlm cache clear` or a custom `RLM_CACHE_DIR` on a tmpfs may be appropriate.

**DSL simplicity vs. eval flexibility**: The DSL is fast, cacheable, and easy to reason about, but limited in expressiveness. Eval provides full Python but adds ~100ms-1s per invocation due to Wasm startup overhead. The system prompt steers the LLM to prefer DSL operations, using eval only as a fallback for logic the DSL cannot express.
