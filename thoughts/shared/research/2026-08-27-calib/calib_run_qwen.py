import os
TEMP=float(os.environ.get("CALIB_TEMP","0"))
import json,time,urllib.request,re,sys,collections,random
S=sys.argv[1]; gold=json.load(open(S+"/calib_gold.json"))
random.seed(7); idx=list(range(len(gold))); random.shuffle(idx)
CHUNKS=[[gold[i] for i in idx[k*50:(k+1)*50]] for k in range(4)]
LABELS=['entity','location','numeric value','abbreviation','human being','description and abstract concept']
DEFS={"abbreviation":"the answer is an abbreviation or its expansion (e.g. 'What does NASA stand for?')",
 "entity":"the answer is a thing: animal, plant, food, product, color, event, language, creative work, substance, etc. (e.g. 'What is the largest fish?')",
 "description and abstract concept":"asks for a definition, description, reason or manner: 'What is X?', 'Why...', 'How does...' (e.g. 'What is a hemophiliac?')",
 "human being":"the answer is a person, group of people or organization (e.g. 'Who invented the telephone?')",
 "location":"the answer is a place: city, country, mountain, state, address (e.g. 'What is the capital of Peru?')",
 "numeric value":"the answer is a number, count, date, time, distance, money, percentage, etc. (e.g. 'How many feet are in a mile?')"}
CODES=dict(zip(LABELS,"ABCDEF"))
def fmt(chunk): return "\n".join(f"Date: Jan 01, 2024 || User: 10000 || Instance: {r['q']}" for r in chunk)
def prompt(variant):
    if variant=="A_plain":
        return ("For each line in the input, classify the question into exactly one of: "+", ".join(f"'{l}'" for l in LABELS)+
         ". Output exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
    if variant=="B_defs":
        return ("For each line in the input, classify the question by the type of its answer into exactly one of these labels:\n"+
         "\n".join(f"- '{l}': {DEFS[l]}" for l in LABELS)+
         "\nOutput exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
    if variant=="C_codes":
        return ("For each line in the input, classify the question into exactly one code: "+", ".join(f"{CODES[l]}={l}" for l in LABELS)+
         ". Output exactly one line per input line, in order, formatted '<line number>: <code>'. No other text.")
    if variant=="D_codes_defs":
        return ("For each line in the input, classify the question by the type of its answer into exactly one code:\n"+
         "\n".join(f"- {CODES[l]} = '{l}': {DEFS[l]}" for l in LABELS)+
         "\nOutput exactly one line per input line, in order, formatted '<line number>: <code>'. No other text.")
def prompt_extra(variant):
    if variant=="E_sharp":
        D=dict(DEFS)
        D["abbreviation"]=("the question is about an abbreviation or acronym: what it stands for, means, or how "
            "something is abbreviated (e.g. 'What does NASA stand for?', 'What is the abbreviation for Texas?'). "
            "Such questions are 'abbreviation' even though they ask 'what is/does/means'")
        D["description and abstract concept"]=("the answer is an explanation, definition, reason or manner: 'What is X?' "
            "wanting a definition, 'Why...', 'How does...' — but NOT questions about what an acronym stands for "
            "(those are 'abbreviation') and NOT 'What is <thing>?' where a specific named thing is the answer "
            "(those are 'entity')")
        D["entity"]=("the answer is a specific named thing: animal, plant, food, product, color, event, language, "
            "creative work, substance (e.g. 'What is the largest fish?', 'What comic strip features Beasley?')")
        return ("For each line in the input, classify the question by the type of its answer into exactly one of these labels:\n"+
         "\n".join(f"- '{l}': {D[l]}" for l in LABELS)+
         "\nOutput exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
    if variant=="F_abbr_only":
        D=dict(DEFS)
        D["abbreviation"]=("the question asks about an abbreviation or acronym: what it stands for, what it means, "
            "or how something is abbreviated (e.g. 'What does NASA stand for?', 'What is the abbreviation for Texas?'). "
            "Asking what an acronym stands for is 'abbreviation', not 'description and abstract concept'")
        return ("For each line in the input, classify the question by the type of its answer into exactly one of these labels:\n"+
         "\n".join(f"- '{l}': {D[l]}" for l in LABELS)+
         "\nOutput exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
    return None

def call(p,chunk):
    body={"model":os.environ.get("CALIB_MODEL","/models/Qwen3.8-27B-UD-Q6_K_XL.gguf"),"temperature":TEMP,"max_tokens":6000,
      "messages":[{"role":"system","content":"Answer the following query based on the provided context. Be precise and concise."},
                  {"role":"user","content":f"Query: {p}\n\nContext:\n{fmt(chunk)}"}]}
    req=urllib.request.Request(os.environ.get("CALIB_URL","http://192.168.1.217:11434/v1")+"/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    t=time.time(); r=json.load(urllib.request.urlopen(req,timeout=900)); el=time.time()-t
    return r["choices"][0]["message"]["content"] or "", r["usage"]["completion_tokens"], el
def norm(s,codes):
    s=s.strip().lower().strip("*`.'\"")
    if codes:
        inv={v.lower():k for k,v in CODES.items()}; return inv.get(s[:1], s)
    return next((L for L in LABELS if s and (L.startswith(s) or s.startswith(L))), s)
out={}
variants=sys.argv[2].split(",") if len(sys.argv)>2 else ["A_plain","B_defs","E_sharp"]
for v in variants:
    p=prompt_extra(v) or prompt(v); conf=collections.Counter(); toks=0; secs=0; nlab=0
    for chunk in CHUNKS:
        c,tk,el=call(p,chunk); toks+=tk; secs+=el
        got={int(m.group(1)):norm(m.group(2),v.startswith(("C","D"))) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$",c,re.M)}
        nlab+=len(got)
        for i,r in enumerate(chunk,1): conf[(r["label"],got.get(i,"<missing>"))]+=1
    acc=sum(n for (g,p_),n in conf.items() if g==p_)/200
    per={}
    for L in LABELS:
        tp=conf[(L,L)]; g=sum(n for (gg,_),n in conf.items() if gg==L); pr=sum(n for (_,pp),n in conf.items() if pp==L)
        per[L]={"gold":g,"pred":pr,"recall":round(tp/g,2) if g else None}
    out[v]={"acc":acc,"labels_returned":nlab,"tokens":toks,"secs":round(secs,1),"per_class":per,
            "confusions":sorted(((g,p_,n) for (g,p_),n in conf.items() if g!=p_ and n>=2),key=lambda x:-x[2])[:8]}
    print(f"{v}: acc={acc:.3f} labels={nlab}/200 tokens={toks} secs={secs:.0f}")
    for L in LABELS: print(f"   {L:34s} gold={per[L]['gold']:3d} pred={per[L]['pred']:3d} recall={per[L]['recall']}")
    print("   top confusions (gold->pred):",out[v]["confusions"])
    sys.stdout.flush()
json.dump(out,open(S+"/calib_results.json","w"),indent=1)
# --- binary variants
if len(sys.argv)>2 and sys.argv[2]=="binary":
    for target in ["abbreviation","description and abstract concept"]:
        p=(f"Label EVERY line below as '{target}' or 'other'. Output exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.")
        conf=collections.Counter(); toks=0; secs=0
        for chunk in CHUNKS:
            c,tk,el=call(p,chunk); toks+=tk; secs+=el
            got={int(m.group(1)):norm(m.group(2),False) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$",c,re.M)}
            for i,r in enumerate(chunk,1): conf[(r["label"]==target, got.get(i)==target)]+=1
        tp=conf[(True,True)]; fn=conf[(True,False)]; fp=conf[(False,True)]
        print(f"BINARY {target}: gold={tp+fn} pred={tp+fp} recall={tp/(tp+fn):.2f} precision={tp/(tp+fp) if tp+fp else 0:.2f} tokens={toks} secs={secs:.0f}")
if len(sys.argv)>2 and sys.argv[2]=="topic":
    variants={"T_topic":"For each line in the input, determine which category the question belongs to. Categories: "+", ".join(f"'{l}'" for l in LABELS)+". Output exactly one line per input line, in order, formatted '<line number>: <label>'. No other text.",
     "U_about":"For each question line below, classify what the question is about into one of: "+", ".join(f"'{l}'" for l in LABELS)+". Output exactly one line per input line, in order, formatted '<line number>: <label>'. No other text."}
    for v,p in variants.items():
        conf=collections.Counter()
        for chunk in CHUNKS:
            c,tk,el=call(p,chunk)
            got={int(m.group(1)):norm(m.group(2),False) for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$",c,re.M)}
            for i,r in enumerate(chunk,1): conf[(r["label"],got.get(i,"<missing>"))]+=1
        acc=sum(n for (g,p_),n in conf.items() if g==p_)/200
        print(f"{v}: acc={acc:.3f}")
        for L in LABELS:
            g=sum(n for (gg,_),n in conf.items() if gg==L); pr=sum(n for (_,pp),n in conf.items() if pp==L); tp=conf[(L,L)]
            print(f"   {L:34s} gold={g:3d} pred={pr:3d} recall={tp/g if g else 0:.2f}")
        print("   top confusions:",sorted(((g,p_,n) for (g,p_),n in conf.items() if g!=p_ and n>=2),key=lambda x:-x[2])[:6])
