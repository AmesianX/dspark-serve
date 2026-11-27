"""E1: runtime eviction -> NVMe -> restore, same server process (no restart).
Parent P (~82K) is answered, then N distinct ~82K conversations flood the GPU
pool (1.27M tokens) so P is evicted from GPU; P is asked again and must come
back from NVMe in seconds with byte-identical output."""
import sys, time
from harness import prompt, ask, MemFloor, slots, log
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 18
mem = MemFloor()
p = ask(prompt("E1-parent"), 120, "P_cold"); p["slots"] = slots(); log(p, OUT)
p2 = ask(prompt("E1-parent"), 120, "P_gpu_hit"); p2["same_as_cold"] = p2["text"] == p["text"]; log(p2, OUT)
t0 = time.time()
for i in range(N):
    r = ask(prompt(f"E1-flood-{i}"), 16, f"flood{i}"); r["elapsed_s"] = round(time.time() - t0); log(r, OUT)
r = ask(prompt("E1-parent"), 120, "P_after_flood"); r["same_as_cold"] = r["text"] == p["text"]
r["slots"] = slots(); r["mem_floor_gb"] = mem.report(); log(r, OUT)
r2 = ask(prompt("E1-flood-0"), 16, "flood0_again"); log(r2, OUT)
mem.stop = True
