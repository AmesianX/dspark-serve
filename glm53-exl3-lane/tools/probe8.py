import json,urllib.request,sys,re
import _lane; K=_lane.api_key(); tag=sys.argv[1]
head=open("/tmp/claude-1000/-home-user/8fab0ddc-66bb-493f-910e-4f4c6ba538ad/scratchpad/probe_head.rs").read()
truth=set(re.findall(r"^use ([^;]+);",head,re.M))
Q1=f"다음은 src/tools/skill_tool.rs 의 처음 40줄이다.\n```rust\n{head}\n```\n이 파일의 `use` 문 경로를 있는 그대로 한 줄에 하나씩 나열해라. 파일에 없는 것은 절대 쓰지 마라. 설명 없이 목록만."
Q2=f"다음은 src/tools/skill_tool.rs 의 처음 40줄이다.\n```rust\n{head}\n```\n이 파일의 import 경로에 버그가 있는가? 있으면 어떤 줄인지, 없으면 '없음'이라고만 답해라."
fab_tot=miss_tot=0; bugclaims=0
for i in range(8):
    for qi,q in ((2,Q2),):
        p={"model":"glm-5.3-flash","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0.6,"top_p":0.95,"top_k":40}
        r=urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(p,ensure_ascii=False).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json"})
        dd=json.loads(urllib.request.urlopen(r,timeout=300).read())["choices"][0]; c=dd["message"].get("content") or ""; fr=dd["finish_reason"]; rl=len(dd["message"].get("reasoning") or "")
        if qi==1:
            got=set(x.strip().rstrip(";").strip("`").replace("use ","").strip() for x in re.findall(r"(?:use\s+)?(?:crate|std|serde_json|tokio|regex|once_cell|anyhow)::[^\n`;]+",c))
            fab=[g for g in got if g not in truth]; miss=[t for t in truth if t not in got]; fab_tot+=len(fab); miss_tot+=len(miss)
            print(f"[{tag}] 열거#{i+1}: 출력{len(got)} 지어냄{len(fab)} 누락{len(miss)} {fab[:2]}",flush=True)
        else:
            claim=("없음" not in c[:40]) ; bugclaims+=claim
            print(f"[{tag}] 버그질문#{i+1}: {'버그 있다고 주장' if claim else '없음'} | finish={fr} 생각{rl}자 | {c[:80].replace(chr(10),' ')!r}",flush=True)
print(f"[{tag}] 합계: 지어낸 import {fab_tot}, 누락 {miss_tot}, 가짜 버그 주장 {bugclaims}/8")
