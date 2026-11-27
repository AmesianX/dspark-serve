"""Live probe: does an ab_scale=1 request's prefix-cache KV leak into ab_scale=0 requests?

Usage: python3 tools/bi/ab_cache_leak_probe.py      (URL / API_KEY as in tools/_lane.py)

Two fresh prefixes (random nonce, > 4608 tokens so the hybrid KDA checkpoint grid
gives a cache hit), sent one at a time at temperature 0:

  control  C: ab=0 prime -> ab=0 reuse -> ab=0 with a random cache_salt (fresh)
  test     T: ab=1 prime -> ab=0 reuse -> ab=0 with a random cache_salt (fresh)

Control must be bit-identical (cache reuse itself changes nothing). If the test's
reuse differs from its fresh computation, the KV computed under ab=1 was served
to an ab=0 request: the per-request switch leaks through the prefix cache.
"""
import json, os, sys, time, uuid, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lane import URL, api_key  # noqa: E402

H = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def call(path, body=None, method="POST"):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers=H, method=method)
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), time.time() - t


MODEL = call("/v1/models", method="GET")[0]["data"][0]["id"]
WORDS = ("amber basalt cobalt dune ember fjord granite harbor iris jade kelp lumen "
         "marble nectar onyx pine quartz reef slate tundra umber velvet willow yarrow zinc").split()


def prefix(nonce: str, lines: int = 600) -> str:
    rows = [f"[{nonce}] 목록 {i}: " + " ".join(WORDS[(i * 7 + j) % len(WORDS)] for j in range(9))
            for i in range(lines)]
    return "\n".join(rows)


def gen(system: str, ab: int, salt: str | None = None, n: int = 32):
    body = {"model": MODEL, "temperature": 0, "max_tokens": n, "logprobs": True,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "위 목록이 무엇인지 한 문장으로 설명하고, 이런 걸 어디에 쓰는지 말해줘."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {"ab_scale": ab}}
    if salt:
        body["cache_salt"] = salt
    d, dt = call("/v1/chat/completions", body)
    c = d["choices"][0]
    u = d.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"text": c["message"]["content"], "lp": [t["logprob"] for t in c["logprobs"]["content"]],
            "ptok": u.get("prompt_tokens"), "cached": cached, "dt": round(dt, 1)}


def show(tag, r):
    print(f"  {tag:<22} prompt={r['ptok']} cached={r['cached']} {r['dt']}s | {r['text'][:70]!r}")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, abs(x - y)
    return None, 0.0


def run(label, prime_ab):
    sysmsg = prefix(uuid.uuid4().hex[:12])
    print(f"[{label}] prime ab={prime_ab}")
    p = gen(sysmsg, prime_ab); show(f"prime ab={prime_ab}", p)
    reuse = gen(sysmsg, 0); show("ab=0 reuse", reuse)
    fresh = gen(sysmsg, 0, salt=uuid.uuid4().hex); show("ab=0 fresh(salt)", fresh)
    same = reuse["lp"] == fresh["lp"] and reuse["text"] == fresh["text"]
    i, d = first_diff(reuse["lp"], fresh["lp"])
    print(f"  reuse == fresh ? {same}" + ("" if same else f"  (first diff at token {i}, |dlp|={d:.4g})"))
    return same


ok_ctrl = run("control", 0)
ok_test = run("test", 1)
print()
print("control (ab=0 cache reuse is exact):", "PASS" if ok_ctrl else "FAIL — reuse itself not exact; test below is inconclusive")
print("test    (ab=1 KV not served to ab=0):", "PASS — no leak" if ok_test else "LEAK — ab=0 request reused KV computed under ab=1")
