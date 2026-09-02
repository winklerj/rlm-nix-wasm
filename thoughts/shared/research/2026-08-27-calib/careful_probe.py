import os,json,urllib.request,collections,concurrent.futures as cf
gold=json.load(open("calib_gold.json"))
LABELS=['entity','location','numeric value','abbreviation','human being','description and abstract concept']
DEFS={"abbreviation":"the answer is an abbreviation or its expansion (e.g. 'What does NASA stand for?')",
 "entity":"the answer is a thing: animal, plant, food, product, color, event, language, creative work, substance, etc. (e.g. 'What is the largest fish?')",
 "description and abstract concept":"asks for a definition, description, reason or manner: 'What is X?', 'Why...', 'How does...' (e.g. 'What is a hemophiliac?')",
 "human being":"the answer is a person, group of people or organization (e.g. 'Who invented the telephone?')",
 "location":"the answer is a place: city, country, mountain, state, address (e.g. 'What is the capital of Peru?')",
 "numeric value":"the answer is a number, count, date, time, distance, money, percentage, etc. (e.g. 'How many feet are in a mile?')"}
DEFTXT="\n".join(f"- '{l}': {DEFS[l]}" for l in LABELS)
def call(msgs,maxtok=200):
    body={"model":"/models/Qwen3.8-27B-UD-Q6_K_XL.gguf","temperature":0,"max_tokens":maxtok,"messages":msgs}
    req=urllib.request.Request("http://192.168.1.217:11434/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    last=None
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req,timeout=60))["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last=e
    print("call failed:",last,flush=True); return ""
def norm(s):
    s=s.strip().lower().strip("*`.'\"")
    return next((L for L in LABELS if s and (L.startswith(s) or s.startswith(L))), s)
done=[0]
def careful_one(r):
    done[0]+=1
    if done[0]%25==0: print(f"{done[0]} done",flush=True)
    c=call([{"role":"user","content":f"Classify this question by the type of its answer into exactly one of these labels:\n{DEFTXT}\n\nQuestion: {r['q']}\n\nAnswer with the label only."}])
    return norm(c.splitlines()[-1] if c.strip() else "")
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    preds=list(ex.map(careful_one,gold))
conf=collections.Counter(); ok=0
for r,p in zip(gold,preds):
    conf[(r["label"],p)]+=1; ok+=(p==r["label"])
out={"acc":ok/len(gold),"conf":{f"{g}->{p}":n for (g,p),n in conf.items()}}
json.dump(out,open("careful_probe_results.json","w"),indent=1)
print(f"careful single-item: acc={ok/len(gold):.3f}  (bulk B_defs was 0.905)")
for L in LABELS:
    tp=conf[(L,L)]; g=sum(n for (gg,_),n in conf.items() if gg==L); pr=sum(n for (_,pp),n in conf.items() if pp==L)
    print(f"   {L:34s} gold={g:3d} pred={pr:3d} recall={tp/g:.2f}")
print("top confusions:",sorted(((g,p,n) for (g,p),n in conf.items() if g!=p and n>=2),key=lambda x:-x[2])[:6])
