import os
import json,time,urllib.request,re,glob,os
C=os.path.expanduser("~/.cache/rlm-nix-wasm")
chunk=None
for p in glob.glob(C+"/*/*/*"):
    if 3000<os.path.getsize(p)<9000:
        t=open(p,errors="replace").read()
        ls=[l for l in t.splitlines() if l.startswith("Date:")]
        if len(ls)>=50: chunk="\n".join(ls[:50]); break
prompt=("For each line in the input, classify the question into exactly one of: 'entity', 'location', 'numeric value', "
 "'abbreviation', 'human being', 'description and abstract concept'. Output exactly one line per input line, in order, "
 "formatted '<line number>: <label>'. No other text.")
def call(tag,extra):
    body={"model":"/models/Muse-Glimmer-30B-UD-Q5_K_M.gguf","temperature":0,"max_tokens":4000,
      "messages":[{"role":"system","content":"Answer the following query based on the provided context. Be precise and concise."},
                  {"role":"user","content":f"Query: {prompt}\n\nContext:\n{chunk}"}]}
    body.update(extra)
    req=urllib.request.Request("http://192.168.1.247:11434/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    t=time.time(); r=json.load(urllib.request.urlopen(req,timeout=600)); el=time.time()-t
    m=r["choices"][0]["message"]; c=m.get("content") or ""
    rc=m.get("reasoning_content") or m.get("reasoning") or ""
    n=len(re.findall(r"^\s*\d+\s*:",c,re.M))
    print(f"{tag}: {el:.1f}s usage={r['usage']} labels={n} content_chars={len(c)} reasoning_chars={len(rc)} think_tag={'<think>' in c} extra_keys={[k for k in m if k not in ('role','content')]}")
    print("   head:",c[:120].replace("\n"," | "))
    if rc: print("   reasoning head:",rc[:200].replace("\n"," | "))
#call("A reasoning=low",{"chat_template_kwargs":{"reasoning_strength":"low"}})
#call("B no kwargs",{})
#call("C reasoning=off",{"chat_template_kwargs":{"reasoning_strength":"off"}})
print("=== round 2")
call("D enable_thinking=false",{"chat_template_kwargs":{"enable_thinking":False}})
call("E thinking=false + reasoning_strength=none",{"chat_template_kwargs":{"thinking":False,"reasoning_strength":"none"}})
