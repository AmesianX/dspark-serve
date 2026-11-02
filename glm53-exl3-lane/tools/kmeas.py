import json,urllib.request,time,statistics,sys
import _lane; K=_lane.api_key()
def metrics():
    r=urllib.request.Request("http://localhost:8000/metrics",headers={"Authorization":"Bearer "+K})
    t=urllib.request.urlopen(r,timeout=30).read().decode(); d={"dr":0,"ac":0}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_drafts_total"): d["dr"]=float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["ac"]=float(l.split()[-1])
    return d
def one(msg,n=300):
    b=metrics()
    p={"model":"glm-5.3-flash","messages":[{"role":"user","content":msg}],"max_tokens":n,"temperature":0,
       "ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p).encode(),
        {"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(r,timeout=380).read()); el=time.time()-t0
    a=metrics(); dr=max(a["dr"]-b["dr"],1); ac=a["ac"]-b["ac"]; c=d["usage"]["completion_tokens"]
    return c/el, ac/dr, dr/el
P={"hash-map":"Explain how a hash map works, why collisions happen, and how open addressing differs from chaining.",
   "한국어산문":"해시맵이 어떻게 동작하는지, 충돌이 왜 생기는지 자세히 설명해줘.",
   "한국어코드":"파이썬으로 LRU 캐시를 구현하고 각 메서드를 설명해줘."}
one("warmup",40)
tag=sys.argv[1] if len(sys.argv)>1 else "?"
for name,msg in P.items():
    rs=[one(msg) for _ in range(3)]
    ts=statistics.median([x[0] for x in rs]); ac=statistics.median([x[1] for x in rs]); sp=statistics.median([x[2] for x in rs])
    print(f"[{tag}] {name:8s} {ts:5.1f} t/s | accept/step {ac:5.3f} | 스텝 {sp:5.1f}/s ({1000/sp:5.1f}ms)",flush=True)
