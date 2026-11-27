"""Live probe: /ab prefix-cache namespaces, every mode, GPU and NVMe.

Usage:
    python3 tools/bi/ab_cache_namespace_probe.py live    <state.json>
    ./vllm-glm53-exl3.sh RESTARTALL       # GPU cache gone, NVMe tier survives
    python3 tools/bi/ab_cache_namespace_probe.py restart <state.json>

Fresh random prefixes, temperature 0, one request at a time. "fresh" means the
same request with a random cache_salt (no cache at all) -- the reference.

live:
  leak  : ab=1 prime -> ab=0 reuse must equal ab=0 fresh       (no leak)
  on    : ab=1 x3 on one prefix -> 2nd/3rd fast                (/ab on cached)
  auto  : ab=0 turn 1 -> ab=1 turn 2 (continuation) fast        (reads default prefix)
          -> ab=0 turn 2 equals fresh                           (no leak)
  mix   : (setup only) ab=0 turn 1 -> ab=1 turn 2 on a separate prefix
restart (after RESTARTALL, GPU empty -> NVMe; restore is per 4608-token chunk):
  mix   : ab=0 turn 2 = the default-path control: restores turn 1's default chunks,
          recomputes the rest, equals fresh                     (NVMe does not leak)
          ab=1 turn 2 restores default chunks AND its own tagged chunks -> clearly
          faster than the control                               (mixed restore)
  on    : ab=1 on the /ab on prefix (no default key exists at all) restores
          from tagged chunks, equals a fresh ab=1 computation    (tagged chain)
"""
import json, os, sys, time, uuid, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lane import URL, api_key  # noqa: E402

H = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
W = ("amber basalt cobalt dune ember fjord granite harbor iris jade kelp lumen "
     "marble nectar onyx pine quartz reef slate tundra").split()
FAILS = []


def call(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H)
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d, round(time.time() - t, 1)


def sysmsg(nonce, lines=600):
    return "\n".join(f"[{nonce}] 목록 {i}: " + " ".join(W[(i * 7 + j) % len(W)] for j in range(9))
                     for i in range(lines))


def gen(messages, ab, salt=None, n=32):
    body = {"model": "glm-5.3-flash", "temperature": 0, "max_tokens": n, "logprobs": True,
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {"ab_scale": ab}}
    if salt:
        body["cache_salt"] = salt
    d, dt = call(body)
    c = d["choices"][0]
    return {"text": c["message"]["content"], "lp": [x["logprob"] for x in c["logprobs"]["content"]],
            "dt": dt, "ptok": d["usage"]["prompt_tokens"]}


def same(a, b):
    return a["text"] == b["text"] and a["lp"] == b["lp"]


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'} {name}  {extra}")
    if not cond:
        FAILS.append(name)


def turn1(nonce):
    return [{"role": "system", "content": sysmsg(nonce)},
            {"role": "user", "content": "위 목록이 무엇인지 한 문장으로 설명해줘."}]


def turn2(nonce, a1):
    # The new user message is long (~6K tokens) on purpose: the ab request that
    # answers it must leave at least one COMPLETE NVMe chunk (4608 tokens) of
    # its own tagged KV, or the restart check cannot tell "default chunks +
    # own chunks" from "default chunks only".
    extra = "\n".join(f"[{nonce}] 추가 {i}: " + " ".join(W[(i * 3 + j) % len(W)] for j in range(9))
                      for i in range(200))
    return turn1(nonce) + [{"role": "assistant", "content": a1},
                           {"role": "user", "content": extra + "\n이 추가 목록까지 보고, 이런 목록을 어디에 쓰는지 하나만 말해줘."}]


phase, state_path = sys.argv[1], sys.argv[2]

if phase == "live":
    st = {}
    print("[leak] ab=1 이 먼저 채운 접두부")
    n = uuid.uuid4().hex[:12]
    p = gen(turn1(n), 1)
    reuse = gen(turn1(n), 0)
    fresh = gen(turn1(n), 0, salt=uuid.uuid4().hex)
    print(f"  prime ab=1 {p['dt']}s, ab=0 reuse {reuse['dt']}s, ab=0 fresh {fresh['dt']}s ({p['ptok']} tok)")
    check("ab=0 재사용 == 새 계산 (누출 없음)", same(reuse, fresh))

    print("[on] /ab on: ab=1 만 연속")
    n = uuid.uuid4().hex[:12]
    st["on"] = n
    ts = [gen(turn1(n), 1)["dt"] for _ in range(3)]
    print(f"  ab=1 x3: {ts}")
    check("/ab on 두 번째부터 캐시 적중", ts[1] < ts[0] / 3 and ts[2] < ts[0] / 3)

    print("[auto] ab=0 턴1 → ab=1 턴2(이어쓰기) → ab=0 턴2")
    n = uuid.uuid4().hex[:12]
    t1 = gen(turn1(n), 0, n=40)
    m2 = turn2(n, t1["text"])
    st["auto"] = {"nonce": n, "a1": t1["text"]}
    a = gen(m2, 1)
    z = gen(m2, 0)
    zf = gen(m2, 0, salt=uuid.uuid4().hex)
    print(f"  턴1 ab=0 {t1['dt']}s | 턴2 ab=1 {a['dt']}s | 턴2 ab=0 {z['dt']}s | 턴2 ab=0 fresh {zf['dt']}s ({zf['ptok']} tok)")
    # It still recomputes its own new turn (~6K tokens here), so the bar is
    # "well under a fresh computation", not a fixed ratio of it.
    check("auto 첫 요청(ab=1)이 기본 접두부를 읽음", a["dt"] < zf["dt"] * 0.5, f"{a['dt']}s vs fresh {zf['dt']}s")
    check("auto 다음 ab=0 == 새 계산 (누출 없음)", same(z, zf))
    check("auto 다음 ab=0 도 접두부 재사용", z["dt"] < zf["dt"] / 2, f"{z['dt']}s")

    print("[mix 준비] 재기동 시험용 — ab=0 턴1 → ab=1 턴2 만 (ab=0 턴2 는 보내지 않는다)")
    # A separate prefix: an ab=0 turn 2 here would legitimately store the tail
    # under default keys, and the restart control could then restore it too.
    n = uuid.uuid4().hex[:12]
    t1 = gen(turn1(n), 0, n=40)
    a = gen(turn2(n, t1["text"]), 1)
    print(f"  턴1 ab=0 {t1['dt']}s | 턴2 ab=1 {a['dt']}s ({a['ptok']} tok)")
    st["mix"] = {"nonce": n, "a1": t1["text"]}
    json.dump(st, open(state_path, "w"))

elif phase == "restart":
    st = json.load(open(state_path))
    n, a1 = st["mix"]["nonce"], st["mix"]["a1"]
    m2 = turn2(n, a1)
    print("[restart/mix] GPU 비어 있음 → NVMe (ab=1 을 ab=0 보다 먼저)")
    zf = gen(m2, 0, salt=uuid.uuid4().hex)
    a = gen(m2, 1)
    z = gen(m2, 0)
    print(f"  턴2 fresh {zf['dt']}s | 턴2 ab=1 {a['dt']}s | 턴2 ab=0 {z['dt']}s ({zf['ptok']} tok)")
    check("ab=0 == 새 계산 (NVMe 누출 없음)", same(z, zf))
    check("ab=0 대조군이 기본 청크를 복원", z["dt"] < zf["dt"] * 0.75, f"{z['dt']}s vs fresh {zf['dt']}s")
    check("ab=1 이 기본+자기 청크로 대조군보다 확실히 빠름", a["dt"] < z["dt"] * 0.7,
          f"{a['dt']}s vs 대조 {z['dt']}s")
    print("[restart/on]")
    t = gen(turn1(st["on"]), 1)
    f = gen(turn1(st["on"]), 1, salt=uuid.uuid4().hex)
    print(f"  ab=1 {t['dt']}s vs fresh {f['dt']}s")
    check("/ab on 체인이 NVMe tag 청크로 복원", t["dt"] < f["dt"] * 0.75, f"{t['dt']}s vs fresh {f['dt']}s")
    check("NVMe 에서 받은 ab=1 출력 == ab=1 새 계산", same(t, f))

print()
print("FAIL: " + ", ".join(FAILS) if FAILS else "ALL PASS")
sys.exit(1 if FAILS else 0)
