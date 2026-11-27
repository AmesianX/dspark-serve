"""E2: NVMe store LRU under pressure (store temporarily ~100 slots; one ~82K
conversation needs ~29 slots: 17 MLA chunks + 4 boundaries x 3 KDA groups).
phase=fill   : C1,C2,C3 cold; C1 again (touch); C4 cold  -> C2's TAIL should go.
phase=verify : (after restart) C2 (partial restore), C1, C4 (full) — outputs
               must equal their cold outputs."""
import json, sys
from harness import prompt, ask, slots, log, MemFloor
OUT, phase = sys.argv[1], sys.argv[2]
P = sys.argv[3] if len(sys.argv) > 3 else "E2"
mem = MemFloor()
if phase == "fill":
    for c in ("C1", "C2", "C3"):
        r = ask(prompt(f"{P}-{c}"), 80, f"{c}_cold"); log(r, OUT)
    r = ask(prompt(f"{P}-C1"), 80, "C1_touch"); log(r, OUT)
    r = ask(prompt(f"{P}-C4"), 80, "C4_cold"); r["slots"] = slots(); log(r, OUT)
else:
    cold = {json.loads(l)["tag"]: json.loads(l)["text"] for l in open(OUT) if '"_cold"' in l or "_cold" in json.loads(l)["tag"]}
    for c in ("C4", "C1", "C3", "C2"):   # survivors first: a full restore stores nothing new
        r = ask(prompt(f"{P}-{c}"), 80, f"{c}_after_restart"); r["same_as_cold"] = r["text"] == cold.get(f"{c}_cold"); log(r, OUT)
log({"tag": f"mem_floor_{phase}", "mem_floor_gb": mem.report()}, OUT)
