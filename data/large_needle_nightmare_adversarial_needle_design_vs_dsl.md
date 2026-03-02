# Adversarial Needle Design vs. RLM DSL Operations

## The Solver's Playbook

Before designing hard tests, map out what the orchestrator will *try* to do. It has 9 ops and will compose them into strategies.

### Likely Strategy Compositions

```
Strategy A: "Grep & Extract"
  grep(question keywords) → slice(matched region) → rlm_call(extract answer)

Strategy B: "Chunk & Map"
  chunk(text, N) → map(rlm_call("does this chunk answer Q?")) → combine(results)

Strategy C: "Split & Scan"
  split(by paragraphs) → map(grep(keyword)) → slice(matching paragraph) → rlm_call(extract)

Strategy D: "Eval Compute"
  grep/slice(relevant region) → eval(python code to parse/calculate)

Strategy E: "Recursive Decompose"
  rlm_call(decompose question into sub-questions) → map(sub-queries) → combine
```

---

## Operation-by-Operation Threat Model

### 1. `grep` — Pattern Matching

**What it solves:** Finding paragraphs containing question keywords.

**How to defeat it:**
- **Synonym substitution**: Question says "closed the longest", text says "abandoned in 1971". Zero lexical overlap.
- **Hypernym indirection**: Question says "the animal", text says "Calliphora vicina larvae".
- **Cross-paragraph keywords**: Make the question's keywords appear in 6+ paragraphs so grep returns too many hits to be useful.
- **Negation traps**: Text says "unlike site gamma, which had no recorded closure date" — grep for "closure date" hits the wrong entity.

### 2. `slice` — Range Extraction

**What it solves:** Extracting a specific region once located.

**How to defeat it:**
- **Distributed facts**: The answer requires combining information from line 47 AND line 312. No single slice contains the full answer.
- **Misleading locality**: The "obvious" slice region contains a decoy answer; the real fact is 200 lines away.

### 3. `count` — Occurrence Counting

**What it solves:** "How many times does X appear?" or counting matching lines.

**How to defeat it:**
- **Semantic counting**: "How many sites showed dangerous contamination?" requires understanding thresholds, not counting string matches.
- **Overlapping mentions**: An entity is mentioned by name once but referred to pronominally 4 more times.

### 4. `split` — Text Splitting

**What it solves:** Breaking text into paragraphs or sections for parallel processing.

**How to defeat it:**
- **No delimiters**: Write the haystack as a single continuous block with no paragraph breaks (or use inconsistent delimiters).
- **Boundary-straddling needles**: Place critical information across a split boundary — half the fact in one chunk, half in the next.

### 5. `chunk` — Sized Chunking

**What it solves:** Breaking text into fixed-size pieces for map operations.

**How to defeat it:**
- **Chunk-boundary attacks**: If chunks are ~500 tokens, place the needle sentence spanning tokens 498-502. Neither chunk has the complete fact.
- **Size-dependent answers**: The answer changes depending on how much context is visible (e.g., "the largest value" requires seeing ALL values).

### 6. `combine` — Merging Results

**What it solves:** Aggregating results from parallel sub-calls.

**How to defeat it:**
- **Contradictory sub-results**: Design the text so that different chunks, read in isolation, support different answers. Only the FULL context resolves the ambiguity.
- **Order-dependent logic**: "The site mentioned AFTER gamma" — combine can't preserve source ordering reliably.

### 7. `rlm_call` — Recursive Sub-calls

**What it solves:** Delegating reasoning to a child orchestrator.

**How to defeat it:**
- **Context loss**: The child only sees what's passed to it. If the parent strips context to save tokens, the child can't reason about the full document.
- **Multi-hop chains**: Answering requires A→B→C inference where each hop needs different context that no single sub-call will have.

### 8. `map` — Parallel Recursive

**What it solves:** Processing chunks in parallel.

**How to defeat it:**
- **Global comparisons**: "Which site had the highest value?" requires comparing across ALL chunks. Each parallel call only sees its own chunk.
- **Sequential dependency**: "The value that was recorded AFTER the team left gamma" — ordering matters, but map is unordered.

### 9. `eval` — Python in Sandbox

**What it solves:** Precise computation — parsing numbers, doing math, regex.

**How to defeat it:**
- **Semantic extraction prerequisites**: Eval can compute `max(47, 12, 130, 2.3)`, but only if the right numbers are extracted first. If the extraction step fails, eval gets garbage in.
- **Ambiguous units**: "34 milligrams per liter" vs "34 parts per million" — eval can't resolve unit semantics.
- **Natural language math**: "roughly three times the expected density" — eval needs the expectation AND the multiplier, both embedded in prose.

---

## The 6 Hardest Test Patterns

### Pattern 1: "Distributed Superlative" (defeats grep + chunk + map)

The answer requires finding the MAX/MIN/FIRST/LAST across entities scattered throughout the entire text. No single chunk contains enough information.

**Example:**
- Haystack: 40 paragraphs, each about a different engineering project. Each mentions a completion year.
- Needle: One paragraph mentions a budget of $X.
- Question: "What was the budget of the project completed most recently?"
- Why it's hard: `map` over chunks can find individual completion years, but `combine` must track (year, budget) pairs, and the budget is only stated for ONE project — the model must identify which project is "most recent" globally, then retrieve a value that may be in a completely different chunk than the date it depends on.

### Pattern 2: "Cross-Reference Dereference" (defeats grep + slice)

The answer requires following a chain of references across the document, like a pointer dereference.

**Example:**
- Paragraph 12: "The facility code for the Norwegian site is NOR-7."
- Paragraph 31: "Facility NOR-7 recorded output of 840 megawatts."
- Paragraph 38: "The facility with the highest output used cooling water from a nearby fjord."
- Question: "What country hosts the facility that uses fjord cooling water?"
- Why it's hard: grep("fjord") → paragraph 38, but it doesn't name the facility. grep("country") hits many paragraphs. The model must chain: fjord → highest output → 840MW → NOR-7 → Norwegian. That's 4 hops across 3 paragraphs.

### Pattern 3: "Negation Filter" (defeats grep + eval)

The answer requires EXCLUDING entities that match a condition, not finding ones that do.

**Example:**
- Haystack: 5 research stations described with their equipment lists.
- Question: "Which station did NOT have a spectrometer?"
- Why it's hard: grep("spectrometer") finds the 4 stations that DO have one. The answer is the station NOT in that set. This requires: (1) enumerate all stations, (2) find which mention spectrometers, (3) compute the set difference. `eval` could do step 3, but steps 1-2 require semantic extraction that grep alone can't reliably provide if stations are described using varying terminology.

### Pattern 4: "Chunk Boundary Straddle" (defeats chunk + map)

Critical information is split across any reasonable chunk boundary.

**Example:**
- End of chunk N: "...the third sample, collected at the eastern ridge, contained an unusually high"
- Start of chunk N+1: "concentration of 4.7 milligrams per liter of dissolved lithium carbonate, which Dr. Solvaard attributed to..."
- Question: "What was the lithium carbonate concentration at the eastern ridge?"
- Why it's hard: Chunk N has "eastern ridge" but no concentration. Chunk N+1 has the concentration but no location. Neither chunk alone can answer the question. The `combine` step would need to merge partial sentences, which is fragile.

### Pattern 5: "Temporal Reasoning Under Alias" (defeats grep, requires multi-hop)

Entities are referred to by different names at different points, and the answer requires temporal ordering.

**Example:**
- Paragraph 8: "Project Meridian launched in March with a team of 12."
- Paragraph 19: "By summer, the initiative — now internally called 'Compass' after a rebranding — had expanded to 31 members."
- Paragraph 34: "Compass was formally dissolved in November."
- Question: "How many months did Project Meridian operate?"
- Why it's hard: grep("Project Meridian") hits paragraph 8 only. grep("dissolved") hits paragraph 34 but references "Compass." The model must: (1) discover that Meridian = Compass, (2) find launch date (March), (3) find dissolution date (November), (4) compute duration (8 months). Steps 1-3 require reading three different paragraphs, and step 1 requires understanding a rename.

### Pattern 6: "Conditional Value Extraction" (defeats simple grep + slice)

The correct value depends on a condition stated elsewhere in the document.

**Example:**
- Paragraph 15: "Under standard conditions, the reactor operates at 350 degrees. Under emergency protocol, the threshold drops to 280 degrees."
- Paragraph 42: "On September 14, the facility declared a Level 2 emergency."
- Question: "What was the reactor's operating temperature threshold on September 14?"
- Why it's hard: grep("temperature") → paragraph 15, which has TWO values. grep("September 14") → paragraph 42, which has no temperature. The correct answer (280°) requires connecting the emergency declaration to the conditional temperature, across paragraphs.

---

## Composite "Nightmare" Test

A single test that combines patterns 1, 2, 5, and 6 to defeat all 9 operations simultaneously:

**Design principles:**
1. No unique keyword overlap between question and needle (defeats grep)
2. Answer requires comparing values across 4+ entities (defeats chunk+map without global context)
3. One entity is renamed mid-document (defeats simple grep traversal)
4. The correct value is conditional on a state declared in a different paragraph (defeats slice)
5. Multiple paragraphs contain similar numeric values as decoys (defeats eval without perfect extraction)
6. The needle straddles what would be a natural chunk boundary (defeats chunk)

This composite is what should go in the actual eval. The individual patterns are useful for diagnosing WHICH operation is failing.

---

## Recommended Eval Matrix

| Test ID | Pattern | Primary Op Defeated | Secondary Op Defeated | Reasoning Hops |
|---------|---------|--------------------|-----------------------|----------------|
| T1 | Distributed Superlative | map, chunk | combine | 2 |
| T2 | Cross-Reference Dereference | grep, slice | rlm_call (context loss) | 4 |
| T3 | Negation Filter | grep | eval (extraction prerequisite) | 3 |
| T4 | Chunk Boundary Straddle | chunk, map | combine | 1 |
| T5 | Temporal Alias | grep | eval (alias resolution) | 4 |
| T6 | Conditional Value | grep, slice | rlm_call (split context) | 2 |
| T7 | Nightmare Composite | ALL | ALL | 5+ |
