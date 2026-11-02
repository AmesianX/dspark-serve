import json,urllib.request,sys
import _lane; K=_lane.api_key(); tag=sys.argv[1]
P=["'괭' 이 글자를 자모로 분해해줘.","해시맵이 어떻게 동작하는지 설명해줘.","조선시대 과학기술의 발전을 설명해줘.","파이썬으로 LRU 캐시를 구현해줘.",
   "Explain why the sky is blue.","Write a Python function that reverses a linked list.","양자역학에 기여한 과학자 5명을 설명해줘.","서울에서 부산까지 가는 방법을 알려줘.",
   "SQL 인젝션이 무엇이고 어떻게 막는지 설명해줘.","List three concrete recommendations for a robotics startup facing winter navigation failures."]
out={}
for i,m in enumerate(P):
    p={"model":"glm-5.3-flash","messages":[{"role":"user","content":m}],"max_tokens":120,"temperature":0,"logprobs":True,"top_logprobs":5,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=180).read())
    toks=[{"t":x["token"],"lp":x["logprob"],"top":{y["token"]:y["logprob"] for y in x.get("top_logprobs",[])}} for x in d["choices"][0]["logprobs"]["content"]]
    out[m]=toks
json.dump(out,open(f"lp_{tag}.json","w"),ensure_ascii=False); print(f"logprob 저장: lp_{tag}.json ({sum(len(v) for v in out.values())} 토큰)")
