#!/usr/bin/env python3
"""DSV4 라이브 서버 처리량 측정. 단발 숫자의 들쭉날쭉을 없애려고 만들었다.

측정 원칙 (전에 데인 것들):
  * SSE 청크 수 ≠ 토큰 수. 속도는 반드시 usage 의 prompt_tokens/completion_tokens 로 낸다.
  * prefix 캐시가 붙으면 prefill 이 공짜가 된다. 매 요청 앞에 난수 nonce 를 박아 무력화한다.
  * TTFT 는 prefill + 큐 + 스케줄링이다. decode 는 첫 토큰 이후 구간으로만 낸다.
  * ignore_eos 로 출력 길이를 고정해야 회차 간 비교가 성립한다.
서버 경유로만 측정한다 — GB10 은 통합메모리라 옆에 GPU 프로세스를 띄우면 박스가 죽는다.
"""
import argparse, json, os, random, statistics, string, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
KEY  = os.environ["KEY"]
MODEL = os.environ.get("MODEL", "DSV4-Flash")

WORDS = ("time person year way day thing man world life hand part child eye woman place work week case "
         "point government company number group problem fact water room mother area money story fish month "
         "book job word business issue side kind head house service friend father power hour game line end").split()

def nonce(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def filler(approx_tokens):
    # 영어 단어 1개 ≈ 1.3 토큰. 넉넉히 잡고 usage 로 실제값을 읽는다.
    n = max(1, int(approx_tokens / 1.3))
    return ' '.join(random.choices(WORDS, k=n))

def one(prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None; t_last = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            # 생각모드가 켜져 있으면 델타가 content 가 아니라 reasoning_content 로 온다.
            # content 만 보면 첫 토큰을 영영 못 잡아 전 회차가 실패로 떨어진다.
            d = (ch[0].get("delta") or {}) if ch else {}
            if d.get("content") or d.get("reasoning_content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                t_last = now
    total = time.perf_counter() - t0
    if usage is None or ttft is None:
        return None
    pt = usage["prompt_tokens"]; ct = usage["completion_tokens"]
    dec_span = (t_last - t0) - ttft
    return {
        "prompt_tokens": pt, "completion_tokens": ct,
        "ttft": ttft, "total": total,
        "prefill_tps": pt / ttft if ttft > 0 else 0,
        "decode_tps": (ct - 1) / dec_span if ct > 1 and dec_span > 0 else 0,
    }

def stats(vals):
    vals = [v for v in vals if v]
    if not vals: return "n/a"
    m = statistics.mean(vals); md = statistics.median(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"평균 {m:7.2f}  중앙 {md:7.2f}  σ {sd:6.2f}  최소 {min(vals):7.2f}  최대 {max(vals):7.2f}"

def run_case(label, in_tok, out_tok, conc, reps):
    print(f"\n### {label}  (입력~{in_tok} / 출력{out_tok} / 동시{conc} / {reps}회)", flush=True)
    pf, dc, tt = [], [], []
    for i in range(reps):
        prompts = [f"[{nonce()}] {filler(in_tok)}\n\n위 단어 목록과 무관하게, 아무 이야기나 계속 쓰라."
                   for _ in range(conc)]
        with ThreadPoolExecutor(conc) as ex:
            rs = [r for r in ex.map(lambda p: one(p, out_tok), prompts) if r]
        if not rs:
            print(f"  회차 {i+1}: 실패", flush=True); continue
        pf += [r["prefill_tps"] for r in rs]
        dc += [r["decode_tps"] for r in rs]
        tt += [r["ttft"] for r in rs]
        print(f"  회차 {i+1}: 실입력 {rs[0]['prompt_tokens']:6d} tok  "
              f"decode {statistics.mean(r['decode_tps'] for r in rs):6.2f} t/s  "
              f"TTFT {statistics.mean(r['ttft'] for r in rs):6.2f}s", flush=True)
    print(f"  prefill t/s : {stats(pf)}", flush=True)
    print(f"  decode  t/s : {stats(dc)}", flush=True)
    print(f"  TTFT    (s) : {stats(tt)}", flush=True)
    return {"label": label, "in": in_tok, "out": out_tok, "conc": conc,
            "prefill": pf, "decode": dc, "ttft": tt}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    random.seed()
    print(f"모델 {MODEL} @ {BASE}   반복 {a.reps}회", flush=True)
    cases = [
        # (라벨, 입력토큰, 출력토큰, 동시)
        ("A 단일 디코딩 (짧은 입력)",      200,   256, 1),
        ("B 단일 디코딩 (긴 출력)",        200,   512, 1),
        ("C 프리필 4k",                   4000,    8, 1),
        ("D 프리필 16k",                 16000,    8, 1),
        ("E 프리필 64k",                 64000,    8, 1),
        ("F 긴 컨텍스트 디코딩 16k",     16000,   256, 1),
        ("G 동시 2",                       200,   256, 2),
        ("H 동시 4",                       200,   256, 4),
    ]
    out = [run_case(*c, reps=a.reps) for c in cases]
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n원자료: {a.out}", flush=True)
