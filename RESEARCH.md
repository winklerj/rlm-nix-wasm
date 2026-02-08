# Sandboxed Recursive Language Models with Nix

## A Design Document for Secure, Cacheable, Parallel LLM Recursion

**Version:** 0.1 (Draft)\
**Date:** February 2026\
**Status:** Proposal / Early Design

---

## Table of Contents

 1. [Glossary](#glossary)
 2. [What Problem Are We Solving?](#what-problem-are-we-solving)
 3. [Background: Recursive Language Models](#background-recursive-language-models)
 4. [Background: Nix and Sandboxing](#background-nix-and-sandboxing)
 5. [Core Insight: Why These Ideas Fit Together](#core-insight-why-these-ideas-fit-together)
 6. [System Architecture](#system-architecture)
 7. [The Explore/Commit Protocol](#the-explorecommit-protocol)
 8. [The Operation DSL](#the-operation-dsl)
 9. [Nix Integration Layer](#nix-integration-layer)
10. [Security Model](#security-model)
11. [Caching and Performance](#caching-and-performance)
12. [Custom VM Environments](#custom-vm-environments)
13. [Example Walkthrough](#example-walkthrough)
14. [Open Questions](#open-questions)
15. [Areas for Further Research](#areas-for-further-research)

---

## Glossary

These are the key terms used throughout this document. If you're new to any of these topics, read this section first.

| Term | Definition |
| --- | --- |
| **LLM** | Large Language Model. An AI system (like GPT or Claude) that reads text and produces text. Think of it as a very smart autocomplete that can follow instructions. |
| **Context window** | The amount of text an LLM can "see" at once. Like a desk — you can only spread out so many papers before things start falling off the edge. |
| **Context rot** | A known problem where LLMs get worse at their job as you give them more text to read. The longer the input, the more likely the model is to miss details or make mistakes. |
| **RLM** | Recursive Language Model. An approach where an LLM can call *itself* (or another LLM) to handle smaller pieces of a big problem. Like a manager who breaks a task into subtasks and assigns them out. |
| **REPL** | Read-Eval-Print Loop. A programming environment where you type a command, the computer runs it, and shows you the result. Then you type another command. Think of a calculator — you enter something, it shows the answer, and you decide what to do next. |
| **Nix** | A build tool and package manager that treats software like math. Given the same inputs, it always produces the same outputs. It also runs builds in isolated "sandboxes" where programs can't access anything they shouldn't. |
| **Derivation** | A Nix concept. A recipe that says "given these specific inputs, run this script, and produce this output." If the inputs haven't changed, Nix skips the work and reuses the old output. |
| **Nix store** | A folder on disk (usually `/nix/store`) where Nix keeps every output it has ever built. Each item is named by a hash of its inputs, so identical work is never repeated. |
| **Content-addressed** | A way of naming things by *what they contain* rather than *where they are*. Two files with the same content always get the same name. This makes caching automatic — if you've seen this exact input before, you already have the answer. |
| **Sandbox** | An isolated environment where a program can run without affecting (or being affected by) the rest of the system. Like a padded room — whatever happens inside stays inside. |
| **bubblewrap (bwrap)** | A lightweight Linux sandboxing tool. It creates isolated environments quickly, without the overhead of a full virtual machine. Nix uses it under the hood. |
| **DAG** | Directed Acyclic Graph. A chart of tasks where arrows show dependencies. "Task B needs Task A's result" is an arrow from A to B. "Acyclic" means there are no circular dependencies. |
| **DSL** | Domain-Specific Language. A small, focused programming language designed for one particular job. Like how a TV remote has buttons for specific tasks instead of being a full computer. |
| **Pure function** | A function where the output depends *only* on its inputs, with no side effects. `add(2, 3)` always returns `5`. It doesn't read a file, check the clock, or change anything else. These are easy to cache and safe to run in parallel. |
| **Memoization** | Remembering the result of a function so you don't have to compute it again. If someone asks you "what's 37 × 24?" and you already calculated it yesterday, you just look up the answer instead of doing the math again. |
| **Token** | The basic unit LLMs use to measure text. Roughly ¾ of a word. "Hello world" is about 2 tokens. A 1M token context is roughly a 750,000-word document. |
| **Map/Reduce** | A pattern for processing large datasets. "Map" applies an operation to every piece. "Reduce" combines all the results into one answer. Like grading exams: map = grade each exam, reduce = calculate the class average. |
| **CRIU** | Checkpoint/Restore In Userspace. A Linux tool that can freeze a running program, save its entire state to disk, and restart it later exactly where it left off. |

---

## What Problem Are We Solving?

Modern LLMs have a hard limit on how much text they can work with at once. Even when a model technically supports a large context window (say, 200,000 tokens), its quality *degrades* as you fill that window. This is context rot.

Recursive Language Models (RLMs) solve this by letting the LLM break a big problem into smaller pieces, solve each piece in a clean context, and combine the results. But the original RLM implementation has three problems:

1. **No security.** Each recursive call runs arbitrary Python code in an unsandboxed subprocess. A hallucinated or malicious command could damage the host system.

2. **No caching.** If two queries ask similar sub-questions about the same data, the work is repeated from scratch. There's no memory of past computation.

3. **No parallelism.** Recursive calls execute one at a time, sequentially. Independent sub-tasks that *could* run simultaneously are forced to wait in line.

This document proposes a system that solves all three problems by combining RLMs with Nix's build infrastructure. We treat each recursive computation as a Nix derivation — sandboxed, content-addressed, and schedulable in parallel.

---

## Background: Recursive Language Models

### The Basic Idea

A normal LLM call looks like this:

```markdown
answer = llm(query + context)
```

You give the model your question and all the relevant data. The model reads everything and gives you an answer. This breaks down when the context is very large.

An RLM call looks the same from the outside:

```markdown
answer = rlm(query, context)
```

But under the hood, the RLM does something different. It *never* shows the full context to any single LLM call. Instead:

1. The context is stored as a variable in a programming environment (the REPL).
2. The "root" LLM receives only the query. It knows the context exists but can't see it directly.
3. The LLM writes code to peek at parts of the context, search through it, and break it into chunks.
4. For each chunk, the LLM can spawn a *recursive* call — a smaller RLM that handles just that chunk.
5. The sub-calls return their results, and the root LLM combines them into a final answer.

### Why It Works

No single LLM call ever needs to handle the full context. Each call works with a manageable piece. This avoids context rot while still allowing the system to process millions of tokens.

### Key Results from the Paper

The original RLM paper (Zhang & Khattab, 2025) demonstrated that:

- An RLM using GPT-4o-mini **outperformed GPT-4o** on hard long-context benchmarks by more than double the correct answers while being cheaper per query.
- RLMs maintained perfect performance at 1,000 documents (10M+ tokens), while baseline approaches degraded significantly.
- The LLM naturally develops useful strategies: peeking at data, grepping for patterns, chunking and mapping, and summarization.

### What the Paper Leaves Unsolved

The paper's implementation runs each recursive call as a blocking Python subprocess with no sandboxing, no caching, and no parallel execution. The authors explicitly call these out as limitations. This is what we address.

---

## Background: Nix and Sandboxing

### What Is Nix?

Nix is a package manager and build system based on a simple principle: **if you haven't changed the inputs, you don't need to redo the work.**

Nix describes every piece of software as a **derivation** — a recipe with explicit inputs and a build script. Nix hashes all the inputs together to create a unique name for the output. If that name already exists in the Nix store, the build is skipped entirely.

```markdown
# Pseudocode for how Nix thinks:
hash = sha256(all_inputs + build_script)
if exists("/nix/store/{hash}-output"):
    return cached_result
else:
    run build_script in sandbox
    store result at "/nix/store/{hash}-output"
    return result
```

### Nix Sandboxing

When Nix builds a derivation, it runs the build script inside a sandbox:

- **No network access** (by default)
- **Read-only filesystem** (except the designated output directory)
- **No access to other builds** running at the same time
- **Restricted environment variables** (only what's declared as input)

This means a build script can't accidentally (or intentionally) break anything outside its sandbox. It's the same idea as a Docker container, but more strict and with better caching.

### Why Nix Fits the RLM Problem

The mapping between RLM concepts and Nix concepts is remarkably clean:

| RLM Concept | Nix Concept |
| --- | --- |
| A recursive call with specific inputs | A derivation |
| The context slice passed to a sub-call | An input store path |
| The REPL environment | The build sandbox |
| The result of a sub-call | The build output |
| Two identical sub-calls on the same data | Same hash → cached, runs once |
| Independent sub-calls | Independent derivations → run in parallel |

---

## Core Insight: Why These Ideas Fit Together

The key realization is that **iterative exploration** and **mutable state** are not the same thing.

The original RLM uses a Python notebook where the LLM writes code cell by cell, building up variables over time. This feels like it *requires* mutable state — each cell modifies variables that the next cell reads.

But look at what the LLM is actually doing:

```markdown
Cell 1: snippet = context[0:2000]         → takes context, returns a string
Cell 2: lines = context.split('\n')       → takes context, returns a list
Cell 3: filtered = grep(lines, pattern)   → takes a list, returns a list
Cell 4: result = rlm_call(query, chunk)   → takes strings, returns a string
```

Every single operation is a **pure function**. The output depends only on the inputs. The "mutability" is just the LLM choosing which pure function to call next based on what it learned from the last result.

This means we can split the LLM's work into two distinct phases:

- **Exploring:** The LLM calls lightweight pure functions one at a time, sees results, and plans a strategy. This is a conversation loop, not mutable state.
- **Committing:** The LLM describes a computation plan (a DAG of operations) that Nix executes with full sandboxing, caching, and parallelism.

This separation is the foundation of the architecture.

---

## System Architecture

### High-Level Overview

```markdown
┌─────────────────────────────────────────────────────┐
│                    User / Caller                     │
│              rlm(query, context) → answer            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  RLM Orchestrator                     │
│                                                       │
│  - Manages the explore/commit loop                    │
│  - Talks to the LLM (the only impure component)      │
│  - Compiles commit plans into Nix expressions         │
│  - Dispatches explore ops to lightweight evaluator    │
│  - Collects results and feeds them back to the LLM   │
└───────┬───────────────────────┬──────────────────────┘
        │ explore ops           │ commit plans
┌───────▼───────┐       ┌──────▼──────────────────┐
│  Lightweight   │       │  Nix Evaluation Engine   │
│  Evaluator     │       │                          │
│  (bwrap,       │       │  - Compiles plan to .nix │
│   in-process,  │       │  - Checks store for hits │
│   fast)        │       │  - Builds missing derivs │
│                │       │  - Parallelizes branches  │
└────────────────┘       └──────────────────────────┘
```

### Components

**1. The Orchestrator**

The central coordinator. It holds the LLM conversation, manages the explore/commit protocol, and translates between the LLM's outputs and the execution engines. This is the only component that is "impure" — it talks to the LLM, which is inherently non-deterministic.

**2. The Lightweight Evaluator**

Handles explore-mode operations. These are small, fast, pure functions (slice some text, grep for a pattern, count lines). They run in a minimal sandbox (bwrap or even just a restricted subprocess) with low overhead. The goal is fast feedback — the LLM needs to see results in milliseconds, not seconds.

**3. The Nix Evaluation Engine**

Handles commit-mode computation plans. Takes a DAG of operations, compiles it to Nix expressions, checks the Nix store for cached results, and builds what's missing. This is where the heavy sandboxing, caching, and parallelism happen.

**4. The LLM**

The "brain" of the system. It receives the query, explores the context through the orchestrator, formulates a strategy, and emits a computation plan. It never sees the full context directly. It never runs code itself. It only decides *what* to compute.

---

## The Explore/Commit Protocol

This is the core of the system's design. The LLM operates in two modes, and explicitly signals which mode it's in.

### Explore Mode

The LLM emits one operation at a time. Each operation runs immediately. The LLM sees the result before deciding what to do next.

**Purpose:** Let the LLM peek at the data, understand its structure, and formulate a strategy.

**Properties:**

- Fast (milliseconds per operation)
- Sequential (each op depends on the last result)
- Lightweight sandboxing (bwrap or in-process)
- Cacheable but caching is optional (these are cheap to recompute)

**Example:**

```markdown
LLM → EXPLORE: slice(context, 0, 2000)
SYS → "Date: Dec 12, 2022 || User: 63685 || Instance: How many..."

LLM → EXPLORE: grep(context, "User: 59219")
SYS → "Date: ... || User: 59219 || Instance: What war saw... (47 matches)"

LLM → EXPLORE: count(grep_result)
SYS → 47
```

After each operation, the orchestrator appends the `(operation, result)` pair to the LLM's conversation history. The LLM sees the full chain of what it has explored so far.

### Commit Mode

The LLM emits a computation plan — a structured description of multiple operations, their dependencies, and how to combine results.

**Purpose:** Execute the actual recursive computation with full Nix guarantees.

**Properties:**

- May take seconds to minutes (involves LLM API calls for recursive sub-calls)
- Parallel where possible (independent branches run simultaneously)
- Full Nix sandboxing (each derivation isolated)
- Fully cached (identical inputs always produce identical outputs from cache)

**Example:**

```markdown
LLM → COMMIT:
  let
    filtered = grep(context, "User: 59219")    # already cached from explore
    chunks = chunk(filtered, 5)
    labels = map("Classify each question as entity/description/numeric", chunks)
    entity_counts = map("Count entries labeled 'entity'", labels)
    total = combine(entity_counts, "sum")
  in
    total
```

The orchestrator compiles this plan into Nix derivations and executes them. The result flows back to the LLM, which can either return it as the final answer or continue with another explore/commit cycle.

### Transition Rules

- The LLM can switch from explore to commit at any time.
- The LLM can switch from commit results back to explore (for multi-stage strategies).
- The orchestrator enforces a maximum number of explore steps and commit cycles to prevent runaway computation.
- If the LLM emits `FINAL(answer)`, the orchestrator returns the answer to the caller.

---

## The Operation DSL

The LLM doesn't write arbitrary Python. It emits operations from a defined set of primitives. This keeps the system predictable and secure.

### Core Operations

```markdown
slice(input, start, end) → string
    Return a substring of the input from position start to end.
    Use case: Peeking at context to understand its structure.

grep(input, pattern) → string
    Return all lines in input that match the pattern.
    Use case: Filtering context to relevant entries.

count(input) → number
    Count lines, characters, or items in the input.
    Use case: Understanding the size of filtered results.

chunk(input, n) → [string]
    Split input into n roughly equal pieces.
    Use case: Preparing context for parallel recursive calls.

split(input, delimiter) → [string]
    Split input on a delimiter.
    Use case: Breaking structured data into records.

rlm_call(query, context) → string
    Spawn a recursive RLM call with the given query and context.
    This is a full recursive call — the sub-RLM can explore and commit too.
    Use case: Having a fresh LLM instance analyze a smaller context slice.

map(prompt, inputs) → [string]
    Apply an rlm_call with the given prompt to each input in parallel.
    Equivalent to [rlm_call(prompt, x) for x in inputs] but concurrent.
    Use case: Processing many chunks simultaneously.

combine(inputs, strategy) → string
    Merge multiple results into one. Strategy can be "concat", "sum",
    "vote", or a custom prompt.
    Use case: Aggregating sub-results into a final answer.

eval(code, bindings, vm_spec) → string
    Run arbitrary code in a sandboxed VM with the given variable bindings.
    The vm_spec defines what language and libraries are available.
    Use case: Escape hatch for computations that don't fit the DSL.
```

### Why a DSL Instead of Arbitrary Code?

1. **Security.** Each operation has a known, bounded behavior. `slice` can't delete files. `grep` can't make network calls. The `eval` escape hatch exists for complex cases but runs in a separately defined, more restricted sandbox.

2. **Caching.** Each operation is trivially hashable. `grep(ctx_hash, "User: 59219")` has a deterministic cache key. Arbitrary Python code would require hashing the entire script plus its dependencies.

3. **Parallelism.** The orchestrator can analyze the DAG of operations and identify independent branches without parsing Python ASTs.

4. **Compilation to Nix.** Each operation maps directly to a Nix derivation. Arbitrary Python would require serializing interpreter state between derivations.

---

## Nix Integration Layer

### Compiling Operations to Derivations

When the LLM commits a computation plan, the orchestrator compiles each operation into a Nix derivation.

**Example: A** `grep` **operation**

```nix
# Auto-generated by the orchestrator
{ pkgs }:

pkgs.runCommand "rlm-grep-${builtins.hashString "sha256" pattern}" {
  input = /nix/store/abc123-context-slice;
  pattern = "User: 59219";
} ''
  grep -F "$pattern" "$input" > $out
''
```

The derivation's name includes a hash of the pattern, so different patterns produce different derivations. If this exact grep on this exact input has been run before, Nix returns the cached result instantly.

**Example: A** `map` **operation (parallel recursive calls)**

```nix
{ pkgs, rlm-lib }:

let
  chunks = import /nix/store/def456-chunks;  # list of store paths
  results = builtins.map (chunk:
    rlm-lib.rlmCall {
      query = "Classify each question as entity/description/numeric";
      context = chunk;
      model = "gpt-4o-mini";
    }
  ) chunks;
in
  pkgs.runCommand "rlm-combine" {
    inputs = results;  # Nix resolves all dependencies, builds in parallel
  } ''
    cat $inputs > $out
  ''
```

Nix sees that each `rlmCall` in the map is independent. It schedules them in parallel up to the configured job limit (`--max-jobs`).

### The Nix Store as Inference Cache

Every completed operation produces an output in the Nix store, named by a hash of its inputs. This creates a persistent, content-addressed cache of all past computation.

**Scenario: Two queries about the same dataset**

```markdown
Query 1: "How many 'entity' questions from User 59219?"
Query 2: "How many 'numeric' questions from User 59219?"
```

Both queries will likely explore the same way (peek at context, grep for User 59219). The grep result is cached after Query 1. Query 2 gets it for free.

Both queries will chunk the filtered results the same way. The chunking is cached.

Only the recursive LLM calls differ (different classification prompts). These run fresh. But the infrastructure around them — the slicing, grepping, chunking — is all cache hits.

### Handling LLM API Calls in Nix

LLM API calls are inherently impure — the same prompt can produce different responses. This breaks Nix's purity model. We handle this with a **fixed-output derivation** pattern:

1. The orchestrator makes the LLM API call *outside* of Nix.
2. The response is written to a file with a content-based hash.
3. This file is imported into the Nix store as a known input.
4. Downstream derivations reference this store path.

This keeps the Nix DAG pure after the LLM call boundary. The impurity is contained to the orchestrator.

An alternative approach: for recursive `rlm_call` operations where we *want* caching (same query + same context = same answer is acceptable), we can use a **hash of the inputs as a cache key** and accept that the cached answer might differ from what a fresh call would produce. This is a deliberate trade-off — correctness of the cache depends on the LLM being roughly deterministic, which is true at temperature 0.

---

## Security Model

### Threat Model

The primary threat is **the LLM emitting harmful code**. This can happen through:

- Hallucination (the LLM writes code that accidentally does something destructive)
- Prompt injection (the context contains instructions that trick the LLM into emitting harmful code)
- Adversarial inputs (a malicious user crafts context designed to exploit the system)

### Defense Layers

**Layer 1: The DSL restricts the action space**

The LLM emits operations from a fixed set. Most operations (`slice`, `grep`, `chunk`) have no mechanism to cause harm — they're string manipulation functions. The attack surface is limited to `eval` and, indirectly, `rlm_call`.

**Layer 2: Nix sandbox isolation**

Every operation that runs as a Nix derivation is sandboxed:

- No network access (API calls are made by the orchestrator, not inside derivations)
- No filesystem access outside the build directory
- No access to environment variables, secrets, or credentials
- No access to other running derivations
- Resource limits (CPU time, memory) via cgroups

**Layer 3: Per-operation VM confinement**

The `eval` escape hatch runs code inside a restricted VM (see the Custom VM Environments section). The VM specification is part of the derivation's inputs, so it's explicit, auditable, and reproducible.

**Layer 4: Output validation**

The orchestrator can validate derivation outputs before passing them back to the LLM or returning them to the caller. For example, it can check that an output is valid UTF-8 text under a certain size, contains no known injection patterns, and so on.

### What Can Go Wrong

| Attack | Mitigation | Residual Risk |
| --- | --- | --- |
| LLM writes `os.system("rm -rf /")` in eval | Sandbox blocks filesystem writes | None if sandbox is correctly configured |
| LLM tries to exfiltrate data via network | Sandbox blocks all network | None if sandbox is correctly configured |
| LLM spawns infinite recursive calls | Orchestrator enforces max depth and max total operations | Cost/time of calls up to the limit |
| Context contains prompt injection | DSL limits what the LLM can do regardless of what it wants to do | LLM might produce wrong answers but can't cause system harm |
| Malicious user provides harmful context | Each operation runs sandboxed; context is just data | Privacy of context data within the system (see Open Questions) |

---

## Caching and Performance

### What Gets Cached

| Operation | Cacheable? | Cache Key | Notes |
| --- | --- | --- | --- |
| `slice` | Yes | hash(input + start + end) | Very fast to recompute; caching optional |
| `grep` | Yes | hash(input + pattern) | Often reused across queries on same dataset |
| `chunk` | Yes | hash(input + n) | Reused when multiple queries chunk the same data |
| `rlm_call` | Conditionally | hash(query + context + model) | Only valid at temperature 0; see notes below |
| `map` | Yes (per element) | Each element cached independently | Parallel elements share cache |
| `combine` | Depends | hash(inputs + strategy) | "concat" is deterministic; prompt-based is not |
| `eval` | Yes | hash(code + bindings + vm_spec) | Deterministic if the code is deterministic |

### Cache Invalidation

Nix's content-addressing means there is no manual cache invalidation. If the inputs change, the hash changes, and a new derivation is built. Old cached results remain in the store and can be garbage-collected periodically.

For LLM calls cached at temperature 0: the orchestrator can optionally set a TTL (time-to-live) for these cache entries, after which they're rebuilt. This handles cases where you want to benefit from newer model versions.

### Performance Expectations

| Phase | Latency Target | Bottleneck |
| --- | --- | --- |
| Single explore op | &lt; 100ms | String processing + bwrap overhead |
| Nix derivation setup | 200-500ms per derivation | Nix evaluation and sandbox creation |
| Parallel recursive calls | Depends on LLM API latency | API rate limits and model response time |
| Cache hit | &lt; 10ms | Nix store lookup |

The biggest performance concern is Nix derivation overhead for fine-grained operations. This is why explore mode uses the lightweight evaluator — we only pay Nix overhead for commit-mode operations, which are chunkier and less frequent.

---

## Custom VM Environments

### The Idea

Different operations may need different execution environments. A simple grep needs almost nothing. A git-diff processor needs Python with specific libraries. A data analysis step might need pandas.

Instead of one fixed VM for all operations, the LLM can request (and the system can provide) custom VMs tailored to each operation. The VM specification is part of the derivation's inputs, making it explicit and reproducible.

### VM Specification

```json
{
  "language": "python-restricted",
  "allowedModules": ["re", "json", "collections", "itertools"],
  "memoryLimit": "512MB",
  "cpuTimeLimit": "30s",
  "diskLimit": "100MB",
  "networkAccess": false
}
```

This specification is hashed as part of the derivation. Two `eval` calls with the same code and bindings but different VM specs produce different derivations.

### Implementation Options

**Option A: Nix-built Python environments**

Nix already excels at building reproducible Python environments with specific package sets. Each VM spec maps to a Nix expression that builds a Python environment with exactly the allowed modules.

**Option B: Lightweight JavaScript VMs**

For operations that don't need Python-specific libraries, a lightweight JS VM (like QuickJS) embedded in the evaluator provides faster startup and stronger isolation. QuickJS runs in a single process, has no filesystem access by default, and starts in milliseconds.

**Option C: Wasm-based sandboxing**

WebAssembly (Wasm) runtimes like Wasmtime or Wasmer provide hardware-level isolation with near-native performance. The LLM's code is compiled to Wasm and executed in a memory-safe sandbox. This is the most secure option but adds compilation overhead.

The system should support all three, selected based on the operation's requirements. The LLM doesn't need to know which is used — it specifies what it needs (language, libraries), and the orchestrator picks the best runtime.

---

## Example Walkthrough

This section walks through a complete example to show how all the pieces fit together. The task is from the OOLONG benchmark described in the RLM paper.

### The Query

> "Among instances associated with users 59219 and 63685, how many data points should be classified as label 'entity'?"

### The Context

A 4.2 million character text file containing thousands of entries like:

```markdown
Date: Dec 12, 2022 || User: 63685 || Instance: How many years old is Benny Carter?
Date: Dec 30, 2024 || User: 35875 || Instance: What war saw battles at Parrot's Beak?
...
```

### Step-by-Step Execution

**1. The caller invokes the system:**

```markdown
answer = rlm(
    query="Among instances for users 59219 and 63685, classify as 'entity' and count",
    context=load("big_dataset.txt")    # 4.2M chars
)
```

**2. The orchestrator stores the context in the Nix store and starts the LLM conversation:**

```markdown
Orchestrator → LLM:
    "You have a query and a context variable. The context is 4,200,000 characters.
     Use EXPLORE operations to examine it, then COMMIT a computation plan.
     Query: Among instances for users 59219 and 63685, classify as 'entity' and count."
```

**3. Explore phase — the LLM peeks:**

```markdown
LLM → EXPLORE: slice(context, 0, 2000)
Evaluator → "Date: Dec 12, 2022 || User: 63685 || Instance: How many years old..."

LLM → EXPLORE: grep(context, "User: 59219")
Evaluator → "Date: ... || User: 59219 || Instance: What war saw... (47 lines)"

LLM → EXPLORE: grep(context, "User: 63685")
Evaluator → "Date: ... || User: 63685 || Instance: How many years... (52 lines)"
```

**4. The LLM now knows the structure and sizes. It commits a plan:**

```markdown
LLM → COMMIT:
  let
    lines_59219 = grep(context, "User: 59219")    # cached from explore
    lines_63685 = grep(context, "User: 63685")    # cached from explore
    all_lines = combine([lines_59219, lines_63685], "concat")
    chunks = chunk(all_lines, 4)
    classifications = map(
      "For each line, determine if the Instance is asking about an entity
       (a person, place, organization, or thing). Return ONLY a count of
       entity-type questions.",
      chunks
    )
    total = combine(classifications, "sum")
  in
    total
```

**5. The orchestrator compiles this to Nix derivations:**

```markdown
   ┌──────────┐    ┌──────────┐
   │grep 59219│    │grep 63685│     ← both cached from explore!
   └────┬─────┘    └────┬─────┘
        └──────┬────────┘
         ┌─────▼─────┐
         │  combine   │
         └─────┬──────┘
         ┌─────▼──────┐
         │  chunk(4)   │
         └──┬──┬──┬──┬┘
          ┌─▼┐┌▼─┐┌▼┐┌▼─┐
          │C1││C2││C3││C4│     ← 4 parallel rlm_calls via map()
          └─┬┘└┬─┘└┬┘└┬─┘
            └──┴┬──┴──┘
          ┌─────▼──────┐
          │combine(sum)│
          └─────┬──────┘
                │
              answer
```

**6. Nix evaluates the DAG:**

- `grep 59219` → cache hit (already computed in explore).
- `grep 63685` → cache hit (already computed in explore).
- `combine(concat)` → new derivation, runs instantly (string concat).
- `chunk(4)` → new derivation, runs instantly.
- `map(classify, 4 chunks)` → 4 new derivations. Nix runs all 4 in parallel. Each spawns a sandboxed LLM API call.
- `combine(sum)` → waits for all 4 map results, then sums them.

**7. The orchestrator returns the final answer to the caller.**

Total LLM API calls: 4 (one per chunk, in parallel) + the root LLM conversation.\
Cached operations: 2 (both greps from explore phase).\
Wall-clock time: Dominated by the slowest of the 4 parallel LLM calls, not their sum.

---

## Open Questions

These are unresolved design decisions that need further investigation before implementation.

### 1. How should the LLM learn the DSL?

The LLM needs to know what operations are available and how to use them. Options include: a detailed system prompt, few-shot examples in the conversation, or fine-tuning. The system prompt approach is simplest but may be brittle. Fine-tuning is the most robust but requires training data of RLM traces.

### 2. What is the right granularity for Nix derivations?

Too fine-grained (every `slice` is a derivation) adds overhead. Too coarse-grained (the entire commit plan is one derivation) loses caching and parallelism benefits. The current proposal — derivations at the operation level within commit plans — is a middle ground, but the right answer probably depends on the workload.

### 3. How do we handle LLM API rate limits?

A `map` over 100 chunks spawns 100 parallel LLM calls. Most API providers have rate limits. The orchestrator needs to manage concurrency — but Nix's `--max-jobs` only controls build parallelism, not the API calls within those builds. We may need a semaphore or token bucket at the orchestrator level.

### 4. How do we handle non-deterministic LLM responses in the cache?

At temperature 0, most LLMs are *mostly* deterministic, but not perfectly so. Caching `rlm_call` results means accepting that a fresh call might give a slightly different answer. Is this acceptable? Should we cache with a TTL? Should we never cache LLM calls and only cache deterministic operations? Answer: To start this is acceptable. We need to note that this is an issue, but we won't address it until later.

### 5. What is the right policy for the `eval` escape hatch?

The `eval` operation provides maximum flexibility but minimum safety guarantees. Should the orchestrator require explicit user approval for `eval`? Should certain VM specs be pre-approved while others require confirmation? Should `eval` be disabled by default? Answer: Let's keep it enabled by default for now and address it later.

### 6. How do we handle context privacy?

Context data is stored in the Nix store, which is world-readable by default. For sensitive data (medical records, financial data, private communications), we need to either encrypt store paths, use a private Nix store, or never persist context to disk (defeating the caching benefit). Answer: Let's address this later. For now it's single user.

### 7. Can the LLM generate Nix expressions directly?

Instead of emitting DSL operations that the orchestrator compiles to Nix, the LLM could emit Nix expressions directly. This would be more flexible but harder to validate for safety. Current LLMs have limited Nix knowledge compared to Python. This may change as models improve. Answer: In my experience they can, but the safety question is still open.

### 8. How deep should recursion go?

The original paper only tested depth-1 recursion (root LM calls sub-LMs, but sub-LMs can't recurse further). Deeper recursion is theoretically possible and might help with very large contexts, but it multiplies cost and latency exponentially. What's the practical depth limit? Answer: For now let's do the same depth as the paper, but it should be configurable so we can test what works best.

### 9. How do we debug failed computation plans?

When the LLM's committed plan produces a wrong answer, we need to understand why. The Nix DAG provides a trace of what ran, but understanding *why* the LLM chose that strategy requires inspecting the explore phase conversation. What logging and visualization tools do we need? We should have conversation and execution/tool call traces.

### 10. Multi-model coordination

The current design assumes one LLM per RLM call. But different models might be better at different sub-tasks — a small, fast model for classification, a large model for complex reasoning. The DSL's `rlm_call` could accept a `model` parameter, but how does the root LLM decide which model to use for which sub-task? Answer: this is a follow-on assumption to test. For now we just adjust the model manually for the whole stack or a different root model than the children.

---

## Areas for Further Research

### Training LLMs for the Explore/Commit Protocol

Current LLMs aren't trained to emit structured DSL operations or to distinguish between explore and commit modes. Research questions include:

- Can we generate synthetic training data from existing RLM traces?
- What reward signal teaches an LLM to explore efficiently (fewer explore ops before committing)?
- Can RL optimize for total cost (explore ops + commit ops + API calls)?

### Distributed Nix Stores for Team/Org Caching

A single machine's Nix store caches locally. For an organization processing many similar queries against shared datasets, a distributed cache (like Cachix or a self-hosted Nix binary cache) could share computed results across users and machines. Research: what's the hit rate in realistic workloads?

### Adaptive Chunk Sizing

The LLM currently decides how to chunk context. But optimal chunk size depends on factors the LLM can't easily reason about: model context window, API cost per token, available parallelism. The orchestrator could suggest or constrain chunk sizes based on system-level knowledge.

### Formal Verification of Safety Properties

The DSL's operations have well-defined semantics. It may be possible to formally verify that any composition of DSL operations satisfies certain safety properties (no data exfiltration, bounded resource usage, termination). This would provide stronger guarantees than runtime sandboxing alone.

### Hybrid Nix + Container Approaches

Nix provides strong build isolation but has overhead. OCI containers (Docker) provide runtime isolation with different trade-offs. A hybrid approach — Nix for reproducibility and caching, containers for runtime isolation — might offer the best of both worlds. Research: what's the performance profile?

### Extension to Multimodal Contexts

The RLM paper focuses on text. But the framework generalizes — the "context" could be images, audio, video, or structured data. The DSL would need new primitives (`crop`, `transcribe`, `query_table`), and the VM environments would need corresponding capabilities. Research: how does the explore/commit protocol change for non-text modalities?

### Comparison with Other Sandboxing Approaches

This document proposes Nix + bwrap. Alternatives include gVisor (Google's container sandbox), Firecracker (AWS's microVM), and Wasm-based isolation. A systematic comparison of security properties, overhead, and developer experience would inform the best choice for production use.

### Economic Optimization

RLM calls have a measurable cost (LLM API fees + compute). The orchestrator could optimize for cost by choosing cheaper models for simpler sub-tasks, increasing cache TTLs to reduce API calls, or pre-computing common operations for frequently-queried datasets.

---

## References

1. Zhang, A. & Khattab, O. (2025). *Recursive Language Models*. arXiv:2512.24601v1. https://alexzhang13.github.io/blog/2025/rlm/
2. Dolstra, E. (2006). *The Purely Functional Software Deployment Model*. PhD thesis, Utrecht University. (The foundational Nix paper.)
3. The Nix project. https://nixos.org/
4. bubblewrap (bwrap). https://github.com/containers/bubblewrap
5. Wang, X. et al. (2024). *CodeAct: Executable Code Actions Elicit Better LLM Agents*. arXiv:2402.01030.
6. Anthropic. (2025). *Effective Context Engineering for AI Agents*. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents