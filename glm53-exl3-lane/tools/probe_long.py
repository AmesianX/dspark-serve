import json,urllib.request,sys,re
import _lane; K=_lane.api_key(); tag=sys.argv[1]
head=open(sys.argv[2]).read()
Q2=f"다음은 src/tools/skill_tool.rs 의 처음 150줄이다.\n```rust\n{head}\n```\n이 파일의 import(use) 경로에 버그가 있는가? 있으면 어떤 줄인지, 없으면 '없음'이라고만 답해라."
claims=0
for i in range(8):
    p={"model":"glm-5.3-flash","messages":[{"role":"user","content":f"(세션 {i}) "+Q2}],"max_tokens":300,"temperature":0.6,"top_p":0.95,"top_k":40,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p,ensure_ascii=False).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=300).read()); dd=d["choices"][0]; c=(dd["message"].get("content") or "").strip()
    claim=("없음" not in c[:60]); claims+=claim
    if i==0: print(f"[{tag}] prompt_tokens={d['usage']['prompt_tokens']}")
    print(f"[{tag}] #{i+1}: {'버그 주장' if claim else '없음'} | {c[:90].replace(chr(10),' ')!r}",flush=True)
print(f"[{tag}] 가짜 버그 주장 {claims}/8")
