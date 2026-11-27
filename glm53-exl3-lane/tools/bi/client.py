import json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
KEY = os.environ["K"]

def ask(prompt, max_tokens=200, tag="", thinking=False):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); ttft = None; out = []; usage = None; t_last = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                txt = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if txt:
                    if ttft is None: ttft = time.time() - t0
                    out.append(txt); t_last = time.time()
    n = usage["completion_tokens"] if usage else None
    cached = (usage or {}).get("prompt_tokens_details") or {}
    dec = (n - 1) / (t_last - t0 - ttft) if n and n > 1 and t_last else None
    res = {"tag": tag, "prompt_tokens": usage and usage["prompt_tokens"], "cached": cached.get("cached_tokens"),
           "ttft_s": round(ttft, 2), "completion_tokens": n, "decode_tps": dec and round(dec, 1),
           "text": "".join(out)}
    return res

if __name__ == "__main__":
    prompt = open(sys.argv[1]).read()
    r = ask(prompt, int(sys.argv[3]) if len(sys.argv) > 3 else 200, sys.argv[2])
    json.dump(r, open(sys.argv[2] + ".json", "w"), ensure_ascii=False, indent=1)
    print({k: v for k, v in r.items() if k != "text"})
