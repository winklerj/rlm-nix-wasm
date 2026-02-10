------------------------------ MODULE RLMOperations ----------------------------
EXTENDS Naturals, Sequences

\* ========== OPERATION INVARIANTS ==========
\* These describe properties that must hold for ALL possible inputs.

\* O1: slice returns a contiguous substring of the input
\*     Formally: slice(text, s, e) = SubSeq(text, s+1, e)
\*     Implies: Len(slice(text, s, e)) <= Len(text)
SliceIsSubstring == TRUE  \* Verified via Hypothesis -- see test

\* O2: grep returns only lines present in the input
\*     Formally: \A line \in Lines(grep(text, pat)): line \in Lines(text)
GrepIsSubset == TRUE  \* Verified via Hypothesis

\* O3: count returns a non-negative integer
\*     Formally: count(text, mode) \in Nat
CountNonNegative == TRUE  \* Verified via Hypothesis

\* O4: chunk preserves all content -- no lines lost
\*     Formally: Lines(Join(chunk(text, n))) = Lines(text)
ChunkPreservesContent == TRUE  \* Verified via Hypothesis

\* O5: chunk produces at most n pieces
\*     Formally: Len(chunk(text, n)) <= n
ChunkBounded == TRUE  \* Verified via Hypothesis

\* O6: split then rejoin recovers the original text
\*     Formally: Join(split(text, delim), delim) = text
SplitRoundTrip == TRUE  \* Verified via Hypothesis

\* O7: all DSL ops are pure -- same input always produces same output
\*     Formally: \A op, args, b: op(args, b) = op(args, b)
OpPurity == TRUE  \* Verified via Hypothesis

\* ========== CACHE INVARIANTS ==========

\* C1: Cache key is deterministic -- same inputs always produce the same key
\*     Formally: make_cache_key(op, args, h) = make_cache_key(op, args, h)
CacheKeyDeterministic == TRUE  \* Verified via Hypothesis

\* C2: Cache round-trip -- put then get returns the value
\*     Formally: (put(k, v) ; get(k)) = v
CacheRoundTrip == TRUE  \* Verified via Hypothesis

\* C3: Different operations produce different cache keys (with high probability)
\*     Formally: op1 /= op2 \/ args1 /= args2 => key1 /= key2
CacheKeyDistinct == TRUE  \* Verified via Hypothesis

\* ========== PARSER INVARIANTS ==========

\* P1: Parsing a well-formed action's JSON recovers the original action
\*     Formally: parse(to_json(action)) = action
ParseRoundTrip == TRUE  \* Verified via Hypothesis

\* P2: Parsing garbage raises ParseError
\*     Formally: \A s \notin ValidJSON: parse(s) raises ParseError
ParseRejectsGarbage == TRUE  \* Verified via Hypothesis

\* ========== EVAL INVARIANTS ==========

\* E1: With finite fuel, execution always terminates (possibly with TimeoutError)
EvalTerminatesWithFuel == TRUE  \* Verified via Hypothesis + Wasm

\* E2: Variable injection is faithful -- injected vars are accessible in code
EvalVariableInjection == TRUE  \* Verified via Hypothesis + Wasm

\* E3: Deterministic code produces identical results across runs
EvalDeterministic == TRUE  \* Verified via Hypothesis + Wasm

\* E4: Executing code does not mutate the caller's bindings dict
EvalIsolation == TRUE  \* Verified via Hypothesis

================================================================================
