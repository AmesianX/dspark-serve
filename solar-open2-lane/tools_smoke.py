#!/usr/bin/env python3
"""Solar Open 2 기동 직후 스모크: 한국어 품질 / 툴콜 non-ASCII / t/s."""
import json, os, sys, time, urllib.request

BASE = "http://127.0.0.1:8000/v1"
# [2026-11-13] 평문 키 제거 — 이 리포는 공개(github.com/AmesianX/dspark-serve).
KEY = os.environ.get("VLLM_API_KEY") or ""
if not KEY:
    raise SystemExit("VLLM_API_KEY 를 환경변수로 줄 것 (solar-open2-lane/solar-serve.env 참조)")
MODEL = "solar-open2-250b"

def call(payload, timeout=600):
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.time() - t0

def rate(body, dt):
    u = body.get("usage", {})
    out = u.get("completion_tokens", 0)
    return out, u.get("prompt_tokens", 0), (out / dt if dt else 0)

def show(tag, body, dt):
    out, pin, tps = rate(body, dt)
    msg = body["choices"][0]["message"]
    txt = (msg.get("content") or "")
    print(f"\n[{tag}] {dt:.1f}s  in={pin} out={out}  **{tps:.2f} t/s**")
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print("  tool_call:", tc["function"]["name"], tc["function"]["arguments"][:300])
    print("  content:", txt[:400].replace("\n", " "))
    bad = [c for c in txt if c == "�"]
    if bad: print(f"  ⚠️ U+FFFD {len(bad)}개")
    return txt

# 1) 한국어 품질 + 글자 중간 라틴 전환 탐침
b, dt = call({"model": MODEL, "messages": [
    {"role": "user", "content": "한국의 사계절을 각 계절마다 세 문장씩, 순우리말을 많이 써서 설명해줘."}],
    "temperature": 1.0, "max_tokens": 700})
show("한국어", b, dt)

# 2) 툴콜 non-ASCII 인자 (DSV4에서 잘림 버그가 났던 지점)
tools = [{"type": "function", "function": {
    "name": "save_note", "description": "메모를 저장한다",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title", "body"]}}}]
b, dt = call({"model": MODEL, "messages": [
    {"role": "user", "content": "save_note 툴로 제목은 '가을 서리', 본문은 한국어 500자 분량 수필을 저장해줘."}],
    "tools": tools, "tool_choice": "auto", "temperature": 1.0, "max_tokens": 1500})
show("툴콜-한글", b, dt)

# 3) 디코드 속도 (긴 출력)
b, dt = call({"model": MODEL, "messages": [
    {"role": "user", "content": "이진 탐색 트리를 파이썬으로 구현하고 각 메서드를 한국어 주석으로 설명해줘."}],
    "temperature": 1.0, "max_tokens": 1200})
show("decode", b, dt)
