import re,sys,json,statistics
log=sys.argv[1]; res=sys.argv[2]
R={str(r["id"]):r for r in (json.loads(l) for l in open(res) if l.strip())}
tasks=[];cur=None
for line in open(log):
    m=re.match(r"Task \d+/50 \(id=(\d+), type=(\w+)",line)
    if m:
        cur={"id":m.group(1),"type":m.group(2),"root_calls":0,"root_s":0,"root_in":0,"root_out":0,"leaf":[], "explore":0,"commit":0,"ops":[],"errors":[],"chunk_n":[]}
        tasks.append(cur); continue
    if cur is None: continue
    m=re.match(r"\s*LLM call #(\d+): ([\d.]+)s, ([\d,]+) in \+ ([\d,]+) out",line)
    if m:
        n=int(m.group(1));s=float(m.group(2));i=int(m.group(3).replace(",",""));o=int(m.group(4).replace(",",""))
        if n==1 and i<8000 and cur["root_calls"]>=1: cur["leaf"].append((s,i,o))
        else: cur["root_calls"]+=1;cur["root_s"]+=s;cur["root_in"]+=i;cur["root_out"]+=o
        continue
    m=re.match(r"EXPLORE step (\d+)/\d+: (\w+)\(",line)
    if m: cur["explore"]+=1;cur["ops"].append("E:"+m.group(2));continue
    m=re.match(r"COMMIT cycle (\d+)/\d+: (\d+) ops",line)
    if m: cur["commit"]+=1;continue
    m=re.match(r"\s+\d+\. (\w+)\(",line)
    if m:
        cur["ops"].append("C:"+m.group(1))
        mm=re.search(r"chunk\(.*n=(\d+)",line)
        if mm: cur["chunk_n"].append(int(mm.group(1)))
        continue
    m=re.match(r"(\w+Error): (.*)",line)
    if m: cur["errors"].append(m.group(1)+": "+m.group(2)[:80])
print(f"{'id':>5} {'type':<10} {'sc':>4} {'wall':>6} {'root#':>5} {'rootS':>6} {'rootIn':>7} {'leaf#':>5} {'leafS':>7} {'lfOut':>6} {'E':>2} {'C':>2} chunk errs")
agg={}
for t in tasks:
    r=R.get(t["id"]); 
    if not r: continue
    wall=float(r["elapsed_s"]); leafs=sum(x[0] for x in t["leaf"]); lo=statistics.mean(x[2] for x in t["leaf"]) if t["leaf"] else 0
    print(f"{t['id']:>5} {t['type']:<10} {float(r['score']):>4.2f} {wall:>6.0f} {t['root_calls']:>5} {t['root_s']:>6.0f} {t['root_in']:>7} {len(t['leaf']):>5} {leafs:>7.0f} {lo:>6.0f} {t['explore']:>2} {t['commit']:>2} {t['chunk_n']} {len(t['errors'])}")
    a=agg.setdefault(t["type"],[]);a.append((wall,t["root_s"],leafs/3,len(t["leaf"]),t["explore"],t["commit"]))
print()
for k,v in agg.items():
    n=len(v);print(k,n,"wall %.0f rootS %.0f leafS/3 %.0f leaf# %.0f explore %.1f commit %.1f"%tuple(sum(x[i] for x in v)/n for i in range(6)))
errs={}
for t in tasks:
    for e in t["errors"]: errs[e[:60]]=errs.get(e[:60],0)+1
for e,c in sorted(errs.items(),key=lambda x:-x[1]): print(c,e)
