#!/usr/bin/env python3
"""Regression check for the '!' collapse (prompt tail chunk replaying the decode graph).

On the unfixed runner every prompt whose length is 5 (mod 512) and longer than
one chunk returned NaN last-position logits (the server answers HTTP 400
"Out of range float values are not JSON compliant: nan" when logprobs are
requested), sampled as token 0 = '!'. Other residues were clean.
Each length is sent cold (fresh cache_salt), max_tokens=1.

    API_KEY=... python3 tools/prefill_tail_nan_check.py
Exit status 1 if any length returns NaN.
"""
import json, os, random, sys, urllib.error, urllib.request, uuid

URL = os.environ.get("URL", "http://localhost:8000")
import _lane
KEY = _lane.api_key()
LENGTHS = [1029, 4613, 9221, 13317, 17925, 30725, 61445] + [17920 + r for r in (1, 2, 3, 4, 6, 7, 8)]


def nan_at(ids):
    body = {"model": "glm-5.3-flash", "prompt": ids, "max_tokens": 1, "temperature": 0,
            "logprobs": 1, "cache_salt": str(uuid.uuid4())}
    req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    try:
        urllib.request.urlopen(req, timeout=1800).read()
        return "ok"
    except urllib.error.HTTPError as e:
        return "NaN" if "nan" in e.read().decode() else f"HTTP {e.code}"


def main():
    rng = random.Random(0)
    # Plain token ids from the ordinary vocabulary range; content does not matter.
    ids = [rng.randrange(1000, 150000) for _ in range(max(LENGTHS))]
    bad = []
    for n in LENGTHS:
        r = nan_at(ids[:n])
        print(f"{n:7d}  mod512={n % 512:3d}  {r}", flush=True)
        if r != "ok":
            bad.append(n)
    print("FAIL" if bad else "PASS", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
