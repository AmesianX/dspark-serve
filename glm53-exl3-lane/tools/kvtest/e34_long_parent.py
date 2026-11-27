"""E3: ~450K-token parent survives a restart (NVMe restore time at scale).
E4: parent + 3 concurrent sub-agents (~133K each, priority=1), parent returns;
then 700K more traffic pushes the parent out of GPU -> NVMe restore.
phase=cold   : P cold                          (before restart)
phase=after  : P after restart, E4 sequence    (after restart)"""
import json, sys, threading, time, uuid
from harness import BODY, QUESTION, ask, MemFloor, slots, log
import client
OUT, phase = sys.argv[1], sys.argv[2]
def big(tag, reps):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * (int(reps) + 1))[: int(len(BODY) * reps)] + QUESTION
P = big("E3-parent", 5.45)                 # ~450K tokens
mem = MemFloor()
if phase == "cold":
    r = ask(P, 100, "P450_cold"); r["slots"] = slots(); log(r, OUT)
else:
    cold = next(json.loads(l)["text"] for l in open(OUT) if '"P450_cold"' in l)
    r = ask(P, 100, "P450_after_restart"); r["same_as_cold"] = r["text"] == cold; log(r, OUT)
    # E4: three sub-agents at once (priority 1), each ~133K, generating 300 tokens
    subs = [big(f"E4-sub{i}", 1.62) for i in range(3)]
    res = [None] * 3
    def run(i):
        body_ask = client.ask.__wrapped__ if hasattr(client.ask, "__wrapped__") else None
        res[i] = ask_prio(subs[i], 300, f"sub{i}")
    def ask_prio(p, n, tag):
        import urllib.request, os
        b = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": n,
             "stream": True, "stream_options": {"include_usage": True}, "priority": 1,
             "chat_template_kwargs": {"enable_thinking": False}}
        rq = urllib.request.Request(client.URL, json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + client.KEY})
        t0 = time.time(); first = None; last = None; usage = None
        with urllib.request.urlopen(rq, timeout=3600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                d = json.loads(line[6:])
                if d.get("usage"): usage = d["usage"]
                if any((c.get("delta") or {}).get("content") for c in d.get("choices", [])):
                    last = time.time(); first = first or last
        n = usage["completion_tokens"]
        return {"tag": tag, "prompt_tokens": usage["prompt_tokens"], "ttft_s": round(first - t0, 1), "decode_tps": round((n - 1) / (last - first), 1)}
    th = [threading.Thread(target=run, args=(i,)) for i in range(3)]; t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    for x in res: log(x, OUT)
    r = ask(P, 100, "P450_after_subagents"); r["same_as_cold"] = r["text"] == cold; log(r, OUT)
    # heavier: ~700K of other traffic (sequential), then the parent again
    for i in range(8):
        log(ask(big(f"E4-flood{i}", 1.07), 16, f"flood{i}"), OUT)
    r = ask(P, 100, "P450_after_flood"); r["same_as_cold"] = r["text"] == cold; r["slots"] = slots(); log(r, OUT)
log({"tag": f"mem_floor_{phase}", "mem_floor_gb": mem.report()}, OUT)
