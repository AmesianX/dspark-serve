import json, os, threading, time, urllib.request
K = os.environ["K"]; T0 = time.time(); out = {}
def req(tag, prio, ntok, delay):
    time.sleep(delay)
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": f"파이썬으로 {tag} 관련 유틸리티 함수를 여러 개 구현하고 설명해줘."}],
            "max_tokens": ntok, "temperature": 0, "ignore_eos": True, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if prio is not None: body["priority"] = prio
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                               {"Authorization": "Bearer " + K, "Content-Type": "application/json"})
    sent = time.time(); first = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for line in resp:
            if first is None and b'"content"' in line and b'"content":""' not in line: first = time.time()
    out[tag] = {"prio": prio, "sent": round(sent - T0, 1), "first_token": round(first - T0, 1), "wait": round(first - sent, 1), "done": round(time.time() - T0, 1)}
th = [threading.Thread(target=req, args=(f"fill{i}", 1, n, 0)) for i, n in enumerate((300, 600, 900, 1200))]
th += [threading.Thread(target=req, args=("A_sub_early", 1, 50, 2.0)), threading.Thread(target=req, args=("B_main_late", None, 50, 3.0))]
[t.start() for t in th]; [t.join() for t in th]
for k in sorted(out, key=lambda k: out[k]["first_token"]): print(k, out[k])
