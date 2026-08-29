# Trace analysis: OOLONG trec_coarse 262K (50/50) and 524K (10/50), max-depth 1, Muse-Glimmer-30B

Sources: `/tmp/oolong-trec-d1-262144.log`, `/tmp/oolong-trec-d1-524288.log`, result JSONL, leaf cache
`~/.cache/rlm-nix-wasm` (5,323 labeled leaf outputs from the fixed-prompt era), llama-server `/props`,
and a 5-call raw-HTTP probe of leaf-call token usage.

## Where the time goes

| type (262K) | n | wall s | root LLM s | leaf calls | leaf s / 3 slots |
|---|---|---|---|---|---|
| LABEL (counting) | 4 | ~900 | ~50 | 128 | ~850 |
| COMPARISON (counting) | 27 | ~1250 | ~55 | ~300 | ~1200 |
| NUMERIC | 12 | 1241 | 55 | 187 | 1178 |
| user-group (15-item) | 7 | 28–105 | 20–85 | 1–15 | <60 |

95% of wall time is map leaf calls. A leaf call on a 50-line chunk: 1.6K in, **~1,260 out**, 15–55 s.
Probe (same chunk, temp 0):

| chat_template_kwargs | completion tokens | visible labels | reasoning_content | s |
|---|---|---|---|---|
| reasoning_strength=low (what the harness sends) | 1,259 | ~300 tok | ~950 tok | 14.9 |
| none | 2,802 | ~300 | ~2,500 | 38.4 |
| enable_thinking=false | 3,062 | ~300 | ~2,750 | 43.9 |
| reasoning_strength=off / none | 1,322–1,359 | ~300 | ~1,000 | 20–22 |

The chat template renders `reasoning_strength` as free text ("Reasoning strength: <value>", default high);
there is no thinking-off switch. `low` is already the floor. **~75% of every leaf call is unavoidable
reasoning** on this model. Server has `total_slots: 3`, so `RLM_MAX_PARALLEL_JOBS` > 3 does nothing.

Leaf cost ≈ 25 output tokens per data line (≈0.2 s/line per slot, loaded). End-to-end the measured
rate is strikingly linear: 879 s / 6,380 lines = 0.138 s/line at 262K, 1,732 s / 12,760 = 0.136 s/line
at 524K. Duration is linear in lines with a hard floor of ~0.14 s/line. Projection at 50 tasks each:
1M ≈ 2 days, 2M ≈ 4 days, 4M ≈ 8 days.

### Duration tail = retry pathology, not slow leaves
- 1114: 6,133 s. Three eval errors, then the SAME map re-run in 5 commit cycles with
  n=128, 127, 200, 100, **6374** (one piece per line) → 4,087 leaf calls.
- 1111 (3,538 s), 1142 (3,034 s), 1145 (3,082 s), 1131 (2,510 s), 1109 (1,715 s): 5 commit cycles,
  384–520 leaf calls — the model re-runs the map to "verify" a tally it doesn't trust.
- Eval errors per run: 131K 15, 262K 14, 524K 7. At 131K/262K about half are `NameError: name 'context'
  is not defined` (model forgot `inputs`; none so far at 524K), the rest are binding-name typos and `'str' object has no attribute ...`
  (treating the JSON-serialized map result as a dict).

## Correctness

### One mechanism: leaf classifier class bias
NUMERIC pred vs gold, 262K (fixed code):

| class | pred / gold | bias |
|---|---|---|
| numeric value | 1000/997, 934/936 | ~0 |
| entity | 536/258, 1284/1304 | over (noisy) |
| description & abstract concept | 1606/1305, 1500/1304 | **+15–25% over** |
| human being | 1484/1699, 725/834 | −13% under |
| location | 965/1118, 52/33 | −14% under (noisy) |
| abbreviation | 634/997, 1470/1963 | **−25–36% under** |

131K (old combine code, directions corroborate): description 553/352, 761/577 over; abbreviation
264/319, location 458/571, 301/351, human 393/447 under; numeric 318/398.

The same bias explains the other failures:
- LABEL 0.38: over-counting "description" flips the argmax (1100, 1121 predicted description; gold human being).
- Missed ties (1105, 1135): opposite-sign biases on the two classes move a 997/997 pair far outside any
  threshold — no tie window fixes this.
- User-group tasks (15 items, one map call per item): a single mislabel flips the answer (1121, 1122, 1124, 1147).

Scoring is exponential, so a −25% count scores 0; NUMERIC 0.08 is entirely this.

### Smaller leaf-quality defects (measured over 5,323 new-era leaf outputs)
- 68 outputs (1.3%) contain a run of ≥20 identical labels — degeneration mid-chunk.
- Label leakage: `description` 6,084 (vs 54,742 full form), `skip` 555, `abstract concept` 291.
  Exact-match tallies drop these; a prefix/fuzzy normaliser in the tally would recover ~2% of labels.
- No truncation: only 1 output has skipped line numbers; maxN modes (25/50/64/100) are the chunk sizes used.
- Chunk sizing follows the prompt (50 lines at 262K, n=128; 50 or 100 at 524K).

## Ranked improvements

Correctness
1. **Leaf classifier prompt quality** — biggest lever by far. Have the root include a one-line
   definition + example per label in the map prompt (system-prompt guidance: "when labeling into a
   fixed label set, define each label with an example"). Needs a calibration experiment: ~200 TREC
   validation questions with gold coarse labels (join on question text), 4 prompt variants
   (plain names / names+definitions+examples / single-letter codes / codes+definitions), measure
   per-class confusion and latency. ~8–16 leaf calls; run when the server is idle between context sizes.
2. **Tally normalisation**: prefix-match labels in the eval example (`description` → full form, drop
   `skip`/blank). Small, free.
3. Nothing to gain from the tie threshold until (1) lands.

Duration
4. **Fan-out guardrail** in the harness: reject `map` over > N pieces (or pieces < ~5 lines) with an
   error that steers to larger chunks; refuse a commit that repeats an identical map prompt already
   bound. Kills the p95 (1114-style 4,000-call runs) without touching the median.
5. **Eval ergonomics**: auto-inject `context` and all current bindings into the sandbox (removes
   ~half the eval errors that trigger retry cycles).
6. **Single-letter label codes** in map prompts: visible output 300 → ~150 tokens per leaf, i.e. ~12%
   of leaf time. Might also reduce degeneration runs. Modest.
7. Reasoning cannot be turned off on this model; leaf cost floor stays ~25 tok/line. The 1M–4M runs
   are 2/4/8 days each at 50 tasks — consider `--limit` for those sizes.

## Addendum (same day): leaf-classifier calibration against TREC gold

Joined every line of both 262K windows (ids 9, 10) to `CogComp/trec` coarse labels — the join is
exact (reproduces gold counts 997/1699/1305/1118/997/258 for window 9). Scripts in
`thoughts/shared/research/2026-08-27-calib/`: `calib_join.py`, `calib_run.py`, `match_outputs.py`.

**Actual run outputs** (2,937 cached leaf outputs matched to their chunk by gold agreement,
140K labels): accuracy **0.899**; pred/gold ratio description 1.20, entity 1.16, abbreviation 0.88,
human 0.88, location 0.88, numeric 1.01. Dominant confusion: abbreviation → description (3,474),
entity → description (2,517), human → entity (2,373). Matches the NUMERIC bias table exactly.

**Controlled prompts** (same model, 200 gold lines in 4 chunks of 50, temp 0, reasoning low):

| variant | acc | abbreviation recall | notes |
|---|---|---|---|
| A plain label names | 0.935 | 0.94 | |
| B names + definition + example each | 0.940 | 0.91 | human recall 0.98 |
| C single-letter codes | 0.925 | 0.91 | tokens −7%, no speed gain (reasoning dominates) |
| D codes + definitions | **0.950** | 0.91 | human 1.00, entity 1.00 |
| T "which category does the question belong to" | 0.920 | 0.88 | topic framing hurts |
| binary `'abbreviation' or 'other'` | — | 0.91 | fine |
| binary `'description…' or 'other'` | — | recall 0.02 | vague-label binary prompts collapse |

Conclusions: the root-written map prompts lose ~5 points of accuracy vs a plain prompt and ~5 more
vs definitions; no generic prompt reproduces the run bias, so the exact wording matters and was not
logged (fixed: verbose now prints the full map prompt). Degeneration runs retracted — 266/286 long
identical runs are `other` in binary outputs, which is legitimate. Single-letter codes give no
duration benefit. Guidance now: copy the header's label set and criterion ("type of the answer")
verbatim, define each label with an example.
