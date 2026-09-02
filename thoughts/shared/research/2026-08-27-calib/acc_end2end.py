import os,json,urllib.request,re,collections,random,math,concurrent.futures as cf
gold=json.load(open("calib_gold.json"))
random.seed(11); pool=random.sample(gold,1000)
LABELS=['entity','location','numeric value','abbreviation','human being','description and abstract concept']
DEFS={"abbreviation":"the answer is an abbreviation or its expansion (e.g. 'What does NASA stand for?')",
 "entity":"the answer is a thing: animal, plant, food, product, color, event, language, creative work, substance, etc. (e.g. 'What is the largest fish?')",
 "description and abstract concept":"asks for a definition, description, reason or manner: 'What is X?', 'Why...', 'How does...' (e.g. 'What is a hemophiliac?')",
 "human being":"the answer is a person, group of people or organization (e.g. 'Who invented the telephone?')",
 "location":"the answer is a place: city, country, mountain, state, address (e.g. 'What is the capital of Peru?')",
 "numeric value":"the answer is a number, count, date, time, distance, money, percentage, etc. (e.g. 'How many feet are in a mile?')"}
DEFTXT="\n".join(f"- '{l}': {DEFS[l]}" for l in LABELS)
BULK=("For each line in the input, classify the question by the type of its answer into exactly one of these labels:\n"+DEFTXT+
 "\nOutput exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
def call(content,maxtok):
    body={"model":"/models/Qwen3.8-27B-UD-Q6_K_XL.gguf","temperature":0,"max_tokens":maxtok,
      "messages":[{"role":"user","content":content}]}
    req=urllib.request.Request("http://192.168.1.217:11434/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=90))["choices"][0]["message"]["content"] or ""
        except Exception: pass
    return ""
def norm(s):
    s=s.strip().lower().strip("*`.'\"")
    return next((L for L in LABELS if s and (L.startswith(s) or s.startswith(L))), s)
chunks=[pool[i:i+50] for i in range(0,1000,50)]
def bulk_one(c):
    txt="\n".join(f"Date: Jan 01, 2024 || User: 10000 || Instance: {r['q']}" for r in c)
    out=call(f"Query: {BULK}\n\nContext:\n{txt}",6000)
    return {int(m.group(1)):norm(m.group(2)) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$",out,re.M)}
with cf.ThreadPoolExecutor(max_workers=3) as ex: bulk=list(ex.map(bulk_one,chunks))
assigned=[]  # (item, gold, bulk_label)
for ci,c in enumerate(chunks):
    for i,r in enumerate(c,1): assigned.append((r["q"],r["label"],bulk[ci].get(i,"<missing>")))
def careful_one(q):
    c=call(f"Classify this question by the type of its answer into exactly one of these labels:\n{DEFTXT}\n\nQuestion: {q}\n\nAnswer with the label only.",200)
    return norm(c.splitlines()[-1] if c.strip() else "")
# ACC with sample_per_label=20 (as the op defaults) AND full-referee upper bound
rng=random.Random(0)
by={} 
for q,g,b in assigned: by.setdefault(b,[]).append((q,g))
raw=collections.Counter(b for _,_,b in assigned)
goldc=collections.Counter(g for _,g,_ in assigned)
for K in (20,60):
    samp=[]
    for lab,grp in sorted(by.items()): samp.extend((lab,q) for q,_ in rng.sample(grp,min(K,len(grp))))
    with cf.ThreadPoolExecutor(max_workers=3) as ex: ref=list(ex.map(careful_one,[q for _,q in samp]))
    strat={}
    for (lab,_),t in zip(samp,ref): strat.setdefault(lab,collections.Counter())[t]+=1
    corr=collections.Counter()
    for lab,n in raw.items():
        rows=strat.get(lab,{lab:1}); k=sum(rows.values())
        for t,h in rows.items(): corr[t]+=n*h/k
    err_raw=sum(abs(raw[L]-goldc[L]) for L in LABELS)
    err_cor=sum(abs(corr[L]-goldc[L]) for L in LABELS)
    print(f"K={K}: total-abs-error raw={err_raw} corrected={err_cor:.0f}")
    for L in LABELS: print(f"   {L:34s} gold={goldc[L]:3d} raw={raw[L]:3d} corrected={corr[L]:6.1f}")
