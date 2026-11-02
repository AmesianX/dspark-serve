#!/usr/bin/env python3
import os
"""유저 상설 평가 세트: ① 과학자 질문 ② 괭테스트(자모 분해, 정답 ㄱ+ㅙ+ㅇ)."""
import json, time, sys, urllib.request

BASE, MODEL = "http://127.0.0.1:8000/v1", "solar-open2-250b"
KEY = os.environ["VLLM_API_KEY"]

def chat(messages, max_tokens=2000, timeout=900):
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps({"model": MODEL, "messages": messages,
                         "temperature": 1.0, "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = json.load(r)
    dt = time.time() - t0
    m = b["choices"][0]["message"]
    u = b.get("usage", {})
    out = u.get("completion_tokens", 0)
    reasoning = m.get("reasoning_content") or ""
    return (m.get("content") or ""), reasoning, out, dt, out / dt if dt else 0

print("=" * 70)
print("① 과학자 질문")
print("=" * 70)
c, th, out, dt, tps = chat([{"role": "user",
    "content": "양자역학에 기여한 과학자 10인을 나열해줘"}])
print(f"[{dt:.1f}s, out={out}, 생각={len(th)}자] **{tps:.2f} t/s**\n")
print(c)

print("\n" + "=" * 70)
print("② 괭테스트 (정답: ㄱ + ㅙ + ㅇ)")
print("=" * 70)
msgs = [{"role": "user", "content": "'괭'이라는 글자를 자모로 분해하면?"}]
c2, th2, out2, dt2, tps2 = chat(msgs)
print(f"[턴1 {dt2:.1f}s, out={out2}, 생각={len(th2)}자] **{tps2:.2f} t/s**")
print(c2)
msgs += [{"role": "assistant", "content": c2},
         {"role": "user", "content": "초성, 중성, 종성을 각각 하나씩만 정확히 답해줘."}]
c3, th3, out3, dt3, tps3 = chat(msgs)
print(f"\n[턴2 {dt3:.1f}s, out={out3}, 생각={len(th3)}자] **{tps3:.2f} t/s**")
print(c3)
ok = ("ㄱ" in c3 and "ㅙ" in c3 and "ㅇ" in c3)
print(f"\n>>> 괭테스트 판정: {'통과' if ok else '실패'} (ㄱ/ㅙ/ㅇ 동시 포함 여부)")
