import os,json,urllib.request,re,collections,random,concurrent.futures as cf
gold=json.load(open("calib_gold.json"))
random.seed(11); pool=random.sample(gold,1000)
LABELS=['entity','location','numeric value','abbreviation','human being','description and abstract concept']
DEFS={"abbreviation":"the answer is an abbreviation or its expansion (e.g. 'What does NASA stand for?')",
 "entity":"the answer is a thing: animal, plant, food, product, color, event, language, creative work, substance, etc. (e.g. 'What is the largest fish?')",
 "description and abstract concept":"asks for a definition, description, reason or manner: 'What is X?', 'Why...', 'How does...' (e.g. 'What is a hemophiliac?')",
 "human being":"the answer is a person, group of people or organization (e.g. 'Who invented the telephone?')",
 "location":"the answer is a place: city, country, mountain, state, address (e.g. 'What is the capital of Peru?')",
 "numeric value":"the answer is a number, count, date, time, distance, money, percentage, etc. (e.g. 'How many feet are in a mile?')"}
def prompt(order):
    return ("For each line in the input, classify the question by the type of its answer into exactly one of these labels:\n"+
      "\n".join(f"- '{l}': {DEFS[l]}" for l in order)+
      "\nOutput exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
def call(content,maxtok=6000):
    body={"model":"/models/Qwen3.8-27B-UD-Q6_K_XL.gguf","temperature":0,"max_tokens":maxtok,"messages":[{"role":"user","content":content}]}
    req=urllib.request.Request("http://192.168.1.217:11434/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=180))["choices"][0]["message"]["content"] or ""
        except Exception: pass
    return ""
def norm(s):
    s=s.strip().lower().strip("*`.'\"")
    return next((L for L in LABELS if s and (L.startswith(s) or s.startswith(L))), s)
def run(order,batch):
    chunks=[pool[i:i+batch] for i in range(0,1000,batch)]
    P=prompt(order)
    def one(c):
        txt="\n".join(f"Date: Jan 01, 2024 || User: 10000 || Instance: {r['q']}" for r in c)
        out=call(f"Query: {P}\n\nContext:\n{txt}")
        return {int(m.group(1)):norm(m.group(2)) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$",out,re.M)}
    with cf.ThreadPoolExecutor(max_workers=3) as ex: outs=list(ex.map(one,chunks))
    preds=[]
    for ci,c in enumerate(chunks):
        for i,r in enumerate(c,1): preds.append(outs[ci].get(i,"<missing>"))
    return preds
goldc=collections.Counter(r["label"] for r in pool)
def report(name,counts,preds=None):
    err=sum(abs(counts[L]-goldc[L]) for L in LABELS)
    acc=f" acc={sum(p==r['label'] for p,r in zip(preds,pool))/1000:.3f}" if preds else ""
    print(f"{name:28s} total-abs-error={err:6.1f}{acc}  "+" ".join(f"{L[:5]}={counts[L]:.0f}" for L in LABELS),flush=True)
O1=LABELS; O2=list(reversed(LABELS)); rng=random.Random(3); O3=LABELS[:]; rng.shuffle(O3)
res={}
for name,order,batch in (("order1/50",O1,50),("order2/50",O2,50),("order3/50",O3,50),("order1/20",O1,20),("order1/10",O1,10)):
    preds=run(order,batch); res[name]=preds
    report(name,collections.Counter(preds),preds)
print("gold                        "+" ".join(f"{L[:5]}={goldc[L]}" for L in LABELS))
c12=collections.Counter(); c123=collections.Counter()
for L in LABELS:
    c12[L]=(collections.Counter(res["order1/50"])[L]+collections.Counter(res["order2/50"])[L])/2
    c123[L]=sum(collections.Counter(res[k])[L] for k in ("order1/50","order2/50","order3/50"))/3
report("avg(order1,order2)",c12); report("avg(order1,2,3)",c123)
# per-item majority of 3 orders
maj=[collections.Counter(v).most_common(1)[0][0] for v in zip(res["order1/50"],res["order2/50"],res["order3/50"])]
report("majority(3 orders)",collections.Counter(maj),maj)
json.dump(res,open("order_batch_preds.json","w"))
