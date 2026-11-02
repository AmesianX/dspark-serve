import json,urllib.request,time,sys
import _lane; K=_lane.api_key()
base=open(_lane.LANE+"/launch.sh").read()+open(_lane.LANE+"/overlay/exl3.py").read()
def run(tag, nchars, salt):
    txt=(str(salt)+" ")+base[:nchars]
    p={"model":"glm-5.3-flash","messages":[{"role":"user","content":"다음 코드를 한 줄로 요약해줘:\n"+txt}],"max_tokens":8,"temperature":0,"stream":True,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    t0=time.time(); f=urllib.request.urlopen(r,timeout=600); ttft=None
    for line in f:
        if line.startswith(b"data:") and b"content" in line and ttft is None: ttft=time.time()-t0
    # prompt tokens via non-stream usage
    p2=dict(p); p2["stream"]=False; p2["max_tokens"]=1
    r2=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p2).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r2,timeout=600).read()); pt=d["usage"]["prompt_tokens"]
    print(f"[{tag}] prompt_tokens={pt} TTFT={ttft:.2f}s -> prefill {pt/ttft:.0f} t/s")
for i,(n) in enumerate((40000,40000,90000)):
    run(f"run{i} {n//1000}k-chars", n, salt=1000+i)
