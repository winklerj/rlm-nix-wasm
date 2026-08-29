import os,re,glob,json,collections,time
from datasets import load_dataset
from rlm.eval.datasets import load_oolong_synth_tasks
from rlm.ops.text import op_chunk
from rlm.ops.values import parse_list_value
LABELS=['entity','location','numeric value','abbreviation','human being','description and abstract concept']
trec=load_dataset("CogComp/trec",split="train+test",revision="refs/convert/parquet")
names=trec.features["coarse_label"].names
m={"ABBR":"abbreviation","ENTY":"entity","DESC":"description and abstract concept","HUM":"human being","LOC":"location","NUM":"numeric value"}
gold={r["text"].strip():m[names[r["coarse_label"]]] for r in trec}
tasks=load_oolong_synth_tasks("trec_coarse",262144)
windows={}
for t in tasks: windows.setdefault(t.context_window_id,t.context_text)
print("windows",sorted(windows))
pieces=[]  # (window, idx, [gold labels per Date line])
for w,ctx in windows.items():
    for i,piece in enumerate(parse_list_value(op_chunk({"input":"context","n":128},{"context":ctx}))):
        g=[gold.get(l.split("Instance:",1)[1].strip()) for l in piece.splitlines() if l.startswith("Date:")]
        pieces.append((w,i,g))
print("pieces",len(pieces), "sizes", collections.Counter(len(p[2]) for p in pieces).most_common(4))
def norm(s):
    s=s.strip().lower().strip("*`.'\"")
    return next((L for L in LABELS if s and (L.startswith(s) or s.startswith(L))), s)
lab=re.compile(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$")
C=os.path.expanduser("~/.cache/rlm-nix-wasm"); cut=time.mktime(time.strptime("2026-08-26 15:35","%Y-%m-%d %H:%M")); end=time.mktime(time.strptime("2026-08-27 06:31","%Y-%m-%d %H:%M"))
conf=collections.Counter(); bin_stats=collections.defaultdict(collections.Counter); n_out=0; n_match=0; unmatched_sizes=collections.Counter()
for p in glob.glob(C+"/*/*/*"):
    mt=os.path.getmtime(p)
    if not (cut<mt<end) or os.path.getsize(p)>20000: continue
    lines=open(p,errors="replace").read().strip().splitlines()
    got={}
    for l in lines:
        mm=lab.match(l)
        if mm: got[int(mm.group(1))]=norm(mm.group(2))
    if len(got)<40 or len(got)>60: continue
    n_out+=1
    labs=[got.get(i) for i in range(1,max(got)+1)]
    known=[l for l in labs if l in LABELS]
    is_binary = len(set(known))==1 and labs.count("other")>0
    best=(0,None)
    for w,i,g in pieces:
        if abs(len(g)-len(labs))>3: continue
        if is_binary:
            tgt=known[0]; agree=sum(1 for a,b in zip(labs,g) if (a==tgt)==(b==tgt))
        else:
            agree=sum(1 for a,b in zip(labs,g) if a==b)
        if agree>best[0]: best=(agree,(w,i,g))
    thr=0.9 if is_binary else 0.6
    if best[0] < thr*min(len(labs), len(best[1][2]) if best[1] else 1): unmatched_sizes[len(labs)]+=1; continue
    n_match+=1; g=best[1][2]
    if is_binary:
        tgt=known[0]
        for a,b in zip(labs,g):
            bin_stats[tgt][("gold" if b==tgt else "notgold", "pred" if a==tgt else "notpred")]+=1
    else:
        for a,b in zip(labs,g):
            if b: conf[(b,a)]+=1
print("outputs",n_out,"matched",n_match,"unmatched sizes",unmatched_sizes.most_common(5))
tot=sum(conf.values()); acc=sum(n for (g,p_),n in conf.items() if g==p_)/tot
print(f"full-label outputs: {tot} labels, acc={acc:.3f}")
for L in LABELS:
    g=sum(n for (gg,_),n in conf.items() if gg==L); pr=sum(n for (_,pp),n in conf.items() if pp==L); tp=conf[(L,L)]
    print(f"  {L:34s} gold={g:5d} pred={pr:5d} ratio={pr/g if g else 0:.2f} recall={tp/g if g else 0:.2f} precision={tp/pr if pr else 0:.2f}")
print("top confusions:",sorted(((g,p_,n) for (g,p_),n in conf.items() if g!=p_),key=lambda x:-x[2])[:10])
for tgt,c in bin_stats.items():
    tp=c[("gold","pred")]; fn=c[("gold","notpred")]; fp=c[("notgold","pred")]
    print(f"binary {tgt}: gold={tp+fn} pred={tp+fp} recall={tp/(tp+fn) if tp+fn else 0:.2f} precision={tp/(tp+fp) if tp+fp else 0:.2f}")
