# teacher-forcing: BF16 생성문을 assistant 턴으로 넣고 prompt_logprobs 로 동일 문맥 하의 토큰 logprob 를 비교
import json,urllib.request,sys
import _lane; K=_lane.api_key(); tag=sys.argv[1]
ref=json.load(open("lp_bf16.json")); out={}
for m,toks in ref.items():
    text="".join(t["t"] for t in toks)
    p={"model":"glm-5.3-flash","messages":[{"role":"user","content":m},{"role":"assistant","content":text}],"max_tokens":1,"temperature":0,
       "prompt_logprobs":1,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=180).read())
    pl=d.get("prompt_logprobs") or d["choices"][0].get("prompt_logprobs")
    seq=[]
    for e in pl:
        if not e: continue
        # e: {token_id: {"logprob":..,"rank":..,"decoded_token":..}} — 실제 토큰은 rank 필드로 식별
        items=list(e.values()); act=max(items,key=lambda x:1 if x.get("rank")==1 else 0)  # 첫 항목이 실제 토큰
        first=items[0]; seq.append({"lp":first["logprob"],"rank":first.get("rank"),"tok":first.get("decoded_token")})
    out[m]=seq[-len(toks):]   # 뒤쪽 = assistant 턴
json.dump(out,open(f"tf_{tag}.json","w"),ensure_ascii=False); print(f"teacher-forced 저장 tf_{tag}.json: {sum(len(v) for v in out.values())} 토큰")
