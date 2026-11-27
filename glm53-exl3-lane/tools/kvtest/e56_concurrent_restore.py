"""E5: four ~150K conversations restored concurrently after a restart.
E6: before that restart, rank-0 damage — B: one MLA chunk's payload flipped;
C: one MLA chunk's header zeroed. Expected: A,D restore; B recomputes (load
failure); C restores up to the damaged chunk. All outputs byte-identical."""
import json, sys, threading, time, uuid
from harness import BODY, QUESTION, ask, MemFloor, slots, log
OUT, phase = sys.argv[1], sys.argv[2]
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
NAMES = ["A", "B", "C", "D"]
mem = MemFloor()
if phase == "cold":
    for n in NAMES:
        t0 = time.time(); r = ask(big(f"E5-{n}"), 100, f"{n}_cold"); r["window"] = [t0, time.time()]; log(r, OUT)
else:
    cold = {}
    for l in open(OUT):
        d = json.loads(l)
        if d["tag"].endswith("_cold") and "text" in d: cold[d["tag"][0]] = d["text"]
    res = {}
    def run(n):
        r = ask(big(f"E5-{n}"), 100, f"{n}_concurrent_after_restart"); r["same_as_cold"] = r["text"] == cold[n]; res[n] = r
    th = [threading.Thread(target=run, args=(n,)) for n in NAMES]; [t.start() for t in th]; [t.join() for t in th]
    for n in NAMES: log(res[n], OUT)
log({"tag": f"mem_floor_{phase}", "mem_floor_gb": mem.report()}, OUT)
