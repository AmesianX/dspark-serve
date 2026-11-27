"""Final light verification of the prefix cache (F1/F2/F5). Usage:
    python3 tools/kvtest/final_light.py before <state.json>   # F1 + F2
    ./vllm-glm53-exl3.sh RESTARTALL
    python3 tools/kvtest/final_light.py after  <state.json>   # F5 (NVMe restore)
F3 = tools/bi/ab_cache_namespace_probe.py, F4 = cm2 through the gateway, F6 = tools/kmeas.py.
"""
import concurrent.futures as cf, json, os, sys, time, uuid, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lane import URL, api_key  # noqa: E402

H = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'} {name}  {extra}")
    if not cond:
        FAILS.append(name)


def gen(msgs, n=48, salt=None, stream_ttft=True):
    body = {"model": "glm-5.3-flash", "temperature": 0, "max_tokens": n, "logprobs": True, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if salt:
        body["cache_salt"] = salt
    t0 = time.time(); ttft = None; text = ""; lps = []; usage = {}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H)
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
                lps += [x["logprob"] for x in ((c.get("logprobs") or {}).get("content") or [])]
    total = time.time() - t0
    n_out = usage.get("completion_tokens", 0)
    tps = n_out / (total - ttft) if ttft and total > ttft and n_out > 1 else 0
    return {"text": text, "lp": lps, "ttft": round(ttft or total, 2), "tps": round(tps, 1), "ptok": usage.get("prompt_tokens")}


def convo(nonce):
    # ~18K tokens: below the first 18,432-token checkpoint segment, where temp-0
    # output is bit-stable run to run. Above it a second determinism issue was
    # found on 2026-11-27 (40K cold vs cold diverge, no cache involved) -- kept
    # out of this cache check on purpose, tracked separately.
    rows = "\n".join(f"[{nonce}] 기록 {i}: 서버 점검 결과와 후속 조치 메모." for i in range(650))
    return [{"role": "system", "content": "너는 운영 보조다."},
            {"role": "user", "content": rows + "\n\n위 기록을 세 문장으로 요약해줘."}]


phase, path = sys.argv[1], sys.argv[2]
if phase == "before":
    n = uuid.uuid4().hex[:10]
    m = convo(n)
    print("[F1] 콜드 → 같은 요청")
    a = gen(m); b = gen(m)
    print(f"  콜드 TTFT {a['ttft']}s → 재요청 {b['ttft']}s ({a['ptok']} tok), 디코드 {b['tps']} t/s")
    check("재요청이 GPU 적중(TTFT 1/5 미만)", b["ttft"] < a["ttft"] / 5)
    check("출력 비트 동일", a["text"] == b["text"] and a["lp"] == b["lp"])
    print("[F2] 같은 요청 + 디코드 1 + 콜드 프리필 1 동시")
    other = convo(uuid.uuid4().hex[:10])
    with cf.ThreadPoolExecutor(3) as ex:
        fd = ex.submit(gen, [{"role": "user", "content": "1부터 200까지 숫자를 한 줄에 하나씩 써."}], 300)
        time.sleep(1.0)
        fc = ex.submit(gen, other, 16)
        fm = ex.submit(gen, m)
        c = fm.result(); fd.result(); fc.result()
    check("동시 처리 중에도 출력 비트 동일", c["text"] == a["text"] and c["lp"] == a["lp"])
    json.dump({"nonce": n, "text": a["text"], "lp": a["lp"], "cold": a["ttft"], "warm": b["ttft"]}, open(path, "w"))
elif phase == "after":
    st = json.load(open(path))
    print("[F5] 재기동 후 같은 요청 (GPU 비었음 → NVMe)")
    r = gen(convo(st["nonce"]))
    print(f"  TTFT {r['ttft']}s (콜드 {st['cold']}s, GPU 적중 {st['warm']}s), 디코드 {r['tps']} t/s")
    check("NVMe 복원(콜드의 1/2 미만)", r["ttft"] < st["cold"] / 2)
    check("재기동 전과 출력 비트 동일", r["text"] == st["text"] and r["lp"] == st["lp"])
print()
print("FAIL: " + ", ".join(FAILS) if FAILS else "ALL PASS")
sys.exit(1 if FAILS else 0)
