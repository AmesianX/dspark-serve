import json, urllib.request, time, statistics, sys
sys.path.insert(0, "/home/user/work/dspark-serve/glm53-exl3-lane/tools")
import _lane; K = _lane.api_key()
def metrics():
    r = urllib.request.Request("http://localhost:8000/metrics", headers={"Authorization": "Bearer " + K})
    t = urllib.request.urlopen(r, timeout=30).read().decode(); d = {"dr": 0, "ac": 0}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_drafts_total"): d["dr"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["ac"] = float(l.split()[-1])
    return d
def one(msg, n=300):
    b = metrics()
    p = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "temperature": 0,
         "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(p).encode(),
                               {"Authorization": "Bearer " + K, "Content-Type": "application/json"})
    t0 = time.time(); d = json.loads(urllib.request.urlopen(r, timeout=380).read()); el = time.time() - t0
    a = metrics(); dr = max(a["dr"] - b["dr"], 1)
    return d["usage"]["completion_tokens"] / el, (a["ac"] - b["ac"]) / dr, dr / el
P = ["Explain how a hash map works, why collisions happen, and how open addressing differs from chaining.",
     "해시맵이 어떻게 동작하는지, 충돌이 왜 생기는지 자세히 설명해줘.",
     "파이썬으로 LRU 캐시를 구현하고 각 메서드를 설명해줘.",
     "Write a Rust function that parses an ISO-8601 date and explain the error handling.",
     "TCP 3-way handshake와 4-way 종료 과정을 단계별로 설명해줘.",
     "Describe the differences between processes and threads with examples in C.",
     "SQL에서 인덱스가 언제 도움이 되고 언제 해가 되는지 예시와 함께 설명해줘.",
     "Implement binary search in Go and discuss off-by-one pitfalls.",
     "쿠버네티스에서 Deployment와 StatefulSet의 차이를 설명해줘.",
     "Explain the CAP theorem and give a concrete database example for each tradeoff."]
one("warmup", 40)
tag = sys.argv[1]; steps = []; tps = []
for msg in P:
    rs = [one(msg) for _ in range(2)]
    sp = statistics.median([x[2] for x in rs]); steps.append(1000 / sp); tps.append(statistics.median([x[0] for x in rs]))
print(f"[{tag}] 10 prompts: mean step {statistics.mean(steps):.2f}ms (min {min(steps):.1f} max {max(steps):.1f}) | mean t/s {statistics.mean(tps):.2f}", flush=True)
print(f"[{tag}] per-prompt step ms: " + " ".join(f"{x:.1f}" for x in steps), flush=True)
