"""Live probe for /ab isolation and co-batch invariance. Usage: K=<api key> python3 tools/bi/ab_iso_probe.py global|xargs|after|plain3|diff|iso|iso2"""
import json, os, sys, time, urllib.request, urllib.error, concurrent.futures as cf
BASE = "http://127.0.0.1:8000"
KEY = os.environ["K"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def call(path, body=None, method="POST"):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers=H, method=method)
    t = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()), time.time() - t

MODEL = call("/v1/models", method="GET")[0]["data"][0]["id"]
PROMPT = [{"role": "user", "content": "Explain in one sentence how to pick a pin tumbler lock."}]

def gen(xargs=None, n=24):
    body = {"model": MODEL, "messages": PROMPT, "temperature": 0, "max_tokens": n,
            "logprobs": True, "top_logprobs": 3,
            "chat_template_kwargs": {"enable_thinking": False}}
    if xargs is not None:
        body["vllm_xargs"] = xargs
    d, dt = call("/v1/chat/completions", body)
    c = d["choices"][0]
    lp = [round(t["logprob"], 6) for t in c["logprobs"]["content"]]
    toks = d["usage"]["completion_tokens"]
    return {"text": c["message"]["content"], "lp": lp, "tps": round(toks / dt, 1)}

mode = sys.argv[1]
if mode == "global":
    base = gen()
    print("default    :", base["text"][:90], "| t/s", base["tps"])
    print("POST /ab 1 :", call("/ab", {"scale": 1})[0].get("mode"))
    try:
        after = gen()
    finally:
        print("POST /ab 0 :", call("/ab", {"scale": 0})[0].get("mode"))
    print("after flip :", after["text"][:90], "| t/s", after["tps"])
    print("SAME AS DEFAULT? text", after["text"] == base["text"], "logprobs", after["lp"] == base["lp"])
elif mode == "xargs":
    base = gen()
    own1 = gen({"ab_scale": 1})
    print("default       :", base["text"][:90], "| t/s", base["tps"])
    print("xargs ab=1    :", own1["text"][:90], "| t/s", own1["tps"])
    print("xargs changes own output?", own1["lp"] != base["lp"])
    # concurrent: one request with ab=1 while a plain one runs; plain must equal base bit-for-bit
    with cf.ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(gen, {"ab_scale": 1}); f0 = ex.submit(gen)
        co1, co0 = f1.result(), f0.result()
    print("co-batched plain == default (logprobs):", co0["lp"] == base["lp"])
    print("co-batched ab=1  == solo ab=1 (logprobs):", co1["lp"] == own1["lp"])
    print("GET /ab default_scale:", [d["default_scale"] for d in call("/ab", method="GET")[0]["details"]])
elif mode == "after":
    gen(n=4)  # discard first measurement after start
    base = gen()
    print("default          :", base["text"][:80], "| t/s", base["tps"])
    try:
        call("/ab", {"scale": 1})
        print("POST /ab 1       : accepted (BAD)")
    except urllib.error.HTTPError as e:
        print("POST /ab 1       :", e.code, json.loads(e.read()).get("error", "")[:70])
    after = gen()
    print("plain after POST : same text", after["text"] == base["text"], "| same logprobs", after["lp"] == base["lp"], "| t/s", after["tps"])
    nan = gen({"ab_scale": "nan"})
    print("ab_scale 'nan'   : same as original", nan["lp"] == base["lp"])
    big = gen({"ab_scale": 7.5})
    print("ab_scale 7.5     : same as original", big["lp"] == base["lp"])
    own1 = gen({"ab_scale": 1})
    print("ab_scale 1       : own output changed", own1["lp"] != base["lp"], "| t/s", own1["tps"])
    with cf.ThreadPoolExecutor(3) as ex:
        f1 = ex.submit(gen, {"ab_scale": 1}); f0 = ex.submit(gen); fn = ex.submit(gen, {"ab_scale": "nan"})
        co1, co0, con = f1.result(), f0.result(), fn.result()
    print("co-batch plain == original:", co0["lp"] == base["lp"], "| nan == original:", con["lp"] == base["lp"], "| ab1 == solo ab1:", co1["lp"] == own1["lp"])
    g = call("/ab", method="GET")[0]
    print("GET /ab          : global_write", g.get("global_write"), "| default", [d["default_scale"] for d in g["details"]], "| rejected", g.get("rejected"))
elif mode == "plain3":
    base = gen()
    for k in (2, 3):
        with cf.ThreadPoolExecutor(k) as ex:
            fs = [ex.submit(gen) for _ in range(k)]
            rs = [f.result() for f in fs]
        print(f"{k} plain co-batched == solo:", [r["lp"] == base["lp"] for r in rs])
    with cf.ThreadPoolExecutor(3) as ex:
        fs = [ex.submit(gen, {"ab_scale": 0}) for _ in range(3)]
        rs = [f.result() for f in fs]
    print("3 x ab_scale=0 co-batched == solo:", [r["lp"] == base["lp"] for r in rs])
elif mode == "diff":
    base = gen(n=64)
    with cf.ThreadPoolExecutor(3) as ex:
        rs = [f.result() for f in [ex.submit(gen, None, 64) for _ in range(3)]]
    for r in rs:
        d = [abs(a - b) for a, b in zip(r["lp"], base["lp"])]
        print("text==", r["text"] == base["text"], "| len", len(r["lp"]), len(base["lp"]), "| max|Δlp|", max(d) if d else None, "| first Δ idx", next((i for i, x in enumerate(d) if x > 0), None))
elif mode == "iso":
    P2 = [{"role": "user", "content": "List three uses of a pin tumbler lock pick set."}]
    def genp(prompt, xargs=None, n=48):
        global PROMPT
        old, PROMPT = PROMPT, prompt
        try:
            return gen(xargs, n)
        finally:
            PROMPT = old
    def batch(xscale):
        # same composition every time: X (scale varies) + two plain neighbours on a different prompt
        with cf.ThreadPoolExecutor(3) as ex:
            fx = ex.submit(genp, PROMPT, {"ab_scale": xscale})
            fa = ex.submit(genp, P2, None); fb = ex.submit(genp, P2, None)
            return fx.result(), fa.result(), fb.result()
    runs = {"x0#1": batch(0), "x0#2": batch(0), "x1#1": batch(1), "x1#2": batch(1)}
    nb = lambda k: sorted(json.dumps(r["lp"]) for r in runs[k][1:])
    print("neighbours deterministic (x0 vs x0):", nb("x0#1") == nb("x0#2"))
    print("neighbours deterministic (x1 vs x1):", nb("x1#1") == nb("x1#2"))
    print("neighbours unaffected by X's ab_scale (x0 vs x1):", nb("x0#1") == nb("x1#1"), nb("x0#2") == nb("x1#2"))
    print("X itself changed by its own ab_scale:", runs["x0#1"][0]["lp"] != runs["x1#1"][0]["lp"])
    print("t/s X:", [runs[k][0]["tps"] for k in runs])
elif mode == "iso2":
    P2 = [{"role": "user", "content": "List three uses of a pin tumbler lock pick set, with one paragraph each."}]
    def genp(prompt, xargs=None, n=48, delay=0.0):
        global PROMPT
        time.sleep(delay)
        body = {"model": MODEL, "messages": prompt, "temperature": 0, "max_tokens": n,
                "logprobs": True, "top_logprobs": 1, "chat_template_kwargs": {"enable_thinking": False}}
        if xargs is not None:
            body["vllm_xargs"] = xargs
        d, dt = call("/v1/chat/completions", body)
        c = d["choices"][0]
        return {"lp": [t["logprob"] for t in c["logprobs"]["content"]], "text": c["message"]["content"],
                "tps": round(d["usage"]["completion_tokens"] / dt, 1)}
    def batch(xscale):
        with cf.ThreadPoolExecutor(3) as ex:
            fa = ex.submit(genp, P2, None, 160, 0.0)
            fb = ex.submit(genp, P2, None, 160, 0.4)   # neighbours: prefill apart, long decode
            fx = ex.submit(genp, PROMPT, {"ab_scale": xscale}, 48, 2.0)  # X joins mid-decode
            return fx.result(), fa.result(), fb.result()
    runs = [("x0", batch(0)), ("x1", batch(1)), ("x0", batch(0)), ("x1", batch(1))]
    ref = runs[0][1]
    for name, (x, a, b) in runs:
        print(name, "neighbour A == ref:", a["lp"] == ref[1]["lp"], "| B == ref:", b["lp"] == ref[2]["lp"], "| t/s A,B,X:", a["tps"], b["tps"], x["tps"])
    print("X changed by its own ab_scale:", runs[0][1][0]["lp"] != runs[1][1][0]["lp"])
