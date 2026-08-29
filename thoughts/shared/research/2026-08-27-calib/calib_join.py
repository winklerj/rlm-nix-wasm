"""Offline: join a 262K oolong window's lines to TREC coarse gold labels."""
import re, json, collections, sys
from datasets import load_dataset
from rlm.eval.datasets import load_oolong_synth_tasks
trec = load_dataset("CogComp/trec", split="train+test", revision="refs/convert/parquet")
names = trec.features["coarse_label"].names  # ABBR, ENTY, DESC, HUM, LOC, NUM
m = {"ABBR":"abbreviation","ENTY":"entity","DESC":"description and abstract concept","HUM":"human being","LOC":"location","NUM":"numeric value"}
gold = {}
for r in trec:
    gold[r["text"].strip()] = m[names[r["coarse_label"]]]
tasks = load_oolong_synth_tasks("trec_coarse", 262144)
t = next(t for t in tasks if t.id == 1115)  # numeric value, gold 997
lines = [l for l in t.context_text.splitlines() if l.startswith("Date:")]
qs = [l.split("Instance:",1)[1].strip() for l in lines]
hit = [gold.get(q) for q in qs]
print("lines", len(lines), "joined", sum(1 for h in hit if h))
c = collections.Counter(h for h in hit if h); print(c)
print("expected numeric value gold 997 ->", c["numeric value"])
json.dump([{"q":q,"label":g} for q,g in zip(qs,hit) if g], open(sys.argv[1],"w"))
