"""Batch-invariance probe run. phase=alone: X decodes alone. phase=batch: three
short requests are decoding, X joins. Dumps land in /cache/bi (host
~/glm53-vllm-cache/bi) and are moved to <phase>/ afterwards by the caller."""
import json, sys, threading, time, uuid, urllib.request
import client
from harness import BODY, QUESTION
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
import os
TGT = os.environ.get("BI_TARGET", "A")
X = big(f"E5-{TGT}")
def req(prompt, n, out, key):
    b = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": n,
         "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(client.URL, json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + client.KEY})
    out[key] = json.loads(urllib.request.urlopen(r, timeout=3600).read())["choices"][0]["message"]["content"]
phase = sys.argv[1]; out = {}
if phase == "warm":                      # put X in the GPU prefix cache (not probed: MINPOS)
    req(X, 1, out, "X")
elif phase == "alone":
    req(X, int(sys.argv[2]) if len(sys.argv) > 2 else 30, out, "X")
elif phase == "warm4":                 # A,B,C,D into the GPU prefix cache, one by one
    for t in "ABCD":
        req(big(f"E5-{t}"), 1, out, t)
elif phase == "co":                     # C alone first, then A,B,D together -> A's tail is co-prefilled
    FIRST = os.environ.get("BI_FIRST", "C")
    tc = threading.Thread(target=req, args=(big(f"E5-{FIRST}"), 60, out, FIRST)); tc.start(); time.sleep(0.3)
    CO = os.environ.get("BI_CO", ",".join(t for t in "ABCD" if t != FIRST)).split(",")
    ts = [threading.Thread(target=req, args=(big(f"E5-{t}"), 60, out, t)) for t in CO]
    [t.start() for t in ts]; [t.join() for t in ts]; tc.join(); out["X"] = out.get(TGT)
elif phase == "mix":                   # X decodes while two cold prompts prefill
    def cold(f): req(f"[콜드 {uuid.uuid4()}]\n" + BODY[: int(len(BODY) * f)] + QUESTION, 4, out, f"c{f}")
    tx = threading.Thread(target=req, args=(X, 60, out, "X")); tx.start(); time.sleep(3)
    cs = [threading.Thread(target=cold, args=(f,)) for f in (0.08, 0.12)]
    for t in cs: t.start(); time.sleep(1.5)
    tx.join(); [t.join() for t in cs]
else:
    others = [threading.Thread(target=req, args=(f"파이썬으로 {t} 를 구현하고 설명해줘.", 400, out, t))
              for t in ("LRU 캐시", "트라이", "레이트 리미터")]
    [t.start() for t in others]; time.sleep(4)
    req(X, 30, out, "X"); [t.join() for t in others]
print(json.dumps({"phase": phase, "X": (out.get("X") or "")[:120]}, ensure_ascii=False))
