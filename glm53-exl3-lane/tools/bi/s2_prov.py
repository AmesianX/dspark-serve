# 2단계(캐시 출처) 시험: 같은 프롬프트 X를 부하 조건만 바꿔 콜드 프리필(요청마다 새 cache_salt → 캐시 미적중,
# 모델 입력은 동일)하고 (A) X의 출력, (B) 그 캐시를 이어 쓰는 후속 질문(X+고정 꼬리, 같은 salt)의 출력이
# 혼자 프리필한 기준과 같은지 본다. LONG_PREFILL=4608·MNBT=9236 → 프리필 3개 이상 동시면 3번째가 자투리 청크.
import json, os, sys, threading, time, urllib.request, uuid
S = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8000/v1/chat/completions"; KEY = os.environ["K"]
BODY = open(os.path.join(S, "p60k.txt")).read(); L = len(BODY)
Q = "\n\n위 코드들의 전체 구조를 한국어로 5줄 요약해라."
X = BODY[: int(L * 0.22)] + Q
Y = BODY[int(L * 0.30): int(L * 0.52)] + Q
Z = BODY[int(L * 0.60): int(L * 0.82)] + Q
SUF = "\n\n[후속] 위 코드에서 가장 위험해 보이는 함수 하나를 골라 이유를 3줄로 설명해라."
DEC = "1부터 600까지 숫자를 쉼표로 구분해 빠짐없이 나열해라. 다른 말은 하지 마라."
N = 160; OUT = os.path.join(S, "logs", "s2_prov.jsonl")

def ask(prompt, n, salt):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": n, "stream": True, "stream_options": {"include_usage": True}, "cache_salt": salt,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); ttft = tl = None; out = []; u = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:]); u = d.get("usage") or u
            for c in d.get("choices", []):
                t = (c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning_content") or "")
                if t:
                    if ttft is None: ttft = time.time() - t0
                    out.append(t); tl = time.time()
    n_out = u["completion_tokens"]; cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
    tps = (n_out - 1) / (tl - t0 - ttft) if n_out > 1 else None
    return {"pt": u["prompt_tokens"], "cached": cached, "ttft": round(ttft, 2), "n": n_out,
            "tps": tps and round(tps, 1), "text": "".join(out)}

def diff_at(a, b):
    if a == b: return None
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return i

def cond(name, peers):
    """peers: list of (delay_s_before_X(negative=before), prompt, n). X starts at t=0."""
    salt = f"s2-{name}-{uuid.uuid4()}"; res = {}; th = []
    order = sorted([(d, p, n, i) for i, (d, p, n) in enumerate(peers)] + [(0.0, X, N, "X")], key=lambda r: r[0])
    dmin = order[0][0]; t0 = time.time()
    def run(key, p, n, s): res[key] = ask(p, n, s)
    for d, p, n, k in order:
        wait = t0 + (d - dmin) - time.time()
        if wait > 0: time.sleep(wait)
        s = salt if k == "X" else f"s2-peer-{uuid.uuid4()}"
        t = threading.Thread(target=run, args=(k, p, n, s)); t.start(); th.append(t)
    [t.join() for t in th]
    return salt, res["X"]

def log(rec):
    with open(OUT, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    phase = sys.argv[1]
    CONDS = {
        "R1": [], "R2": [],
        "D1": [(-3.0, DEC, 900)],
        "D3": [(-4.0, DEC, 900), (-3.0, DEC, 900), (-2.0, DEC, 900)],
        "P2": [(-1.0, Y, 8)],
        "P3L": [(-2.0, Y, 8), (-1.0, Z, 8)],
        "P3F": [(1.0, Y, 8), (2.0, Z, 8)],
        "MX": [(-3.0, DEC, 900), (-1.0, Y, 8)],
    }
    if phase == "A":
        ask(DEC, 4, f"warm-{uuid.uuid4()}")  # 기동 후 첫 요청 버리기
        ref = None; salts = {}
        for name, peers in CONDS.items():
            salt, r = cond(name, peers); salts[name] = salt
            if ref is None: ref = r["text"]
            rec = {"phase": "A", "cond": name, "salt": salt, **r, "eq_R1": r["text"] == ref, "diff_at": diff_at(r["text"], ref)}
            log(rec); print(json.dumps({k: v for k, v in rec.items() if k not in ("text", "salt")}, ensure_ascii=False), flush=True)
        json.dump(salts, open(os.path.join(S, "logs", "s2_salts.json"), "w"))
    elif phase == "C":
        # C: 재기동 후(GPU 캐시 비고 NVMe만 남음) 같은 salt로 X, X+SUF → 재기동 전 A·B 출력과 같은가 (NVMe 출처)
        prev = {}
        for l in open(OUT):
            d = json.loads(l)
            if d["phase"] in ("A", "B"): prev[(d["phase"], d["cond"])] = d["text"]
        salts = json.load(open(os.path.join(S, "logs", "s2_salts.json")))
        ask(DEC, 4, f"warm-{uuid.uuid4()}")
        for name, salt in salts.items():
            for ph, p, key in (("A", X, name), ("B", X + SUF, f"hit-{name}")):
                r = ask(p, N, salt)
                rec = {"phase": "C" + ph, "cond": name, **r, "eq_before_restart": r["text"] == prev[(ph, key)]}
                log(rec); print(json.dumps({k: v for k, v in rec.items() if k != "text"}, ensure_ascii=False), flush=True)
    else:
        # B: 후속 질문 X+SUF. hit-* = X 캐시(조건별 출처)를 이어 씀(512 단위 꼬리 적중 → 격자 밖에서 프리필 시작),
        #    cold* = 같은 입력을 새 salt로 통째 콜드, junc = X 앞부분을 공유하고 격자 밖에서 갈라진 W를 먼저 넣은 뒤 X+SUF.
        salts = json.load(open(os.path.join(S, "logs", "s2_salts.json")))
        runs = [(f"hit-{k}", v, None) for k, v in salts.items()]
        runs += [("cold1", f"s2-cold-{uuid.uuid4()}", None), ("cold2", f"s2-cold-{uuid.uuid4()}", None)]
        W = X[: int(len(X) * 0.61)] + "\n\n위 코드의 import 목록만 나열해라."
        runs += [("junc", f"s2-junc-{uuid.uuid4()}", W)]
        ref = {}
        for name, salt, pre in runs:
            if pre: ask(pre, 4, salt)
            r = ask(X + SUF, N, salt)
            grp = "hit" if name.startswith("hit") else "cold"
            ref.setdefault(grp, r["text"])
            rec = {"phase": "B", "cond": name, **r, "eq_hitR1": r["text"] == ref["hit"], "diff_at": diff_at(r["text"], ref["hit"]),
                   "eq_cold1": r["text"] == ref.get("cold", r["text"])}
            log(rec); print(json.dumps({k: v for k, v in rec.items() if k != "text"}, ensure_ascii=False), flush=True)
