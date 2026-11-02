#!/usr/bin/env python3
"""Prefix-cache corruption probe for the kpool tail-seed stride bug (upstream #57477).

Run right after a fresh RESTARTALL so the target prompts own the lowest block ids
(the region the old dense seed kernel scribbles into). For each target:
  1. base   : greedy continuation, cold (fills the prefix cache)
  2. seed   : unrelated long prefills (each one runs the tail-seed kernel)
  3. cached : same target again, served from the prefix cache
  4. fresh  : same target with a cache_salt -> full re-prefill, same prompt
base == fresh is the determinism control. cached != base (with fresh == base)
means the cached blocks changed underneath us.
"""
import argparse, glob, json, os, random, time, urllib.request, uuid

URL = os.environ.get("URL", "http://localhost:8000")
import _lane
KEY = _lane.api_key()


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def corpus(src, n_chars, rng):
    files = sorted(glob.glob(os.path.join(src, "**/*.py"), recursive=True))
    rng.shuffle(files)
    out, size = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        out.append(f"# ==== {os.path.relpath(f, src)} ====\n{t}\n")
        size += len(t)
        if size >= n_chars:
            break
    return "".join(out)[:n_chars]


def cached_total():
    req = urllib.request.Request(URL + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
    for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines():
        if line.startswith("vllm:prompt_tokens_cached_total"):
            return float(line.rsplit(" ", 1)[1])
    return None


def complete(prompt, max_tokens, salt=None):
    c0 = cached_total()
    body = {"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "logprobs": 1}
    if salt:
        body["cache_salt"] = salt
    t = time.time()
    r = post("/v1/completions", body)
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    u = r.get("usage", {})
    c1 = cached_total()
    cached = None if c0 is None or c1 is None else int(c1 - c0)  # sequential client only
    return {"text": c["text"], "tokens": lp.get("tokens", []),
            "lps": lp.get("token_logprobs", []), "prompt_tokens": u.get("prompt_tokens"),
            "cached": cached, "dur": round(time.time() - t, 1), "finish": c.get("finish_reason")}


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/work/vllm/vllm"))
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--target-chars", type=int, default=120_000)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-chars", type=int, default=40_000)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="kpool_probe.jsonl")
    a = ap.parse_args()

    rng = random.Random(1234)  # same prompts on every build -> A/B comparable
    targets = [corpus(a.src, a.target_chars, rng) + "\n# Continue the file:\n" for _ in range(a.targets)]
    seed_rng = random.Random(99)
    log = open(a.out, "a")
    base = [complete(t, a.max_tokens) for t in targets]  # cold, first after restart -> low block ids
    for r in range(a.rounds):
        for s in range(a.seeds):  # unique prompts, never share a prefix with targets
            p = f"[seed {a.tag} r{r} s{s} {uuid.uuid4()}]\n" + corpus(a.src, a.seed_chars, seed_rng)
            complete(p, 4)
        for i, t in enumerate(targets):
            cached = complete(t, a.max_tokens)
            fresh = complete(t, a.max_tokens, salt=str(uuid.uuid4()))
            rec = {"tag": a.tag, "round": r, "target": i,
                   "prompt_tokens": base[i]["prompt_tokens"],
                   "cached_tokens": cached["cached"], "fresh_cached_tokens": fresh["cached"],
                   "base_eq_fresh": base[i]["tokens"] == fresh["tokens"],
                   "cached_eq_base": cached["tokens"] == base[i]["tokens"],
                   "div_cached": first_div(base[i]["tokens"], cached["tokens"]),
                   "div_fresh": first_div(base[i]["tokens"], fresh["tokens"]),
                   "bang_cached": cached["text"].count("!"), "bang_base": base[i]["text"].count("!"),
                   "lp0": [base[i]["lps"][:1], cached["lps"][:1], fresh["lps"][:1]],
                   "base": base[i]["text"][:160], "cached": cached["text"][:160], "fresh": fresh["text"][:160]}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
