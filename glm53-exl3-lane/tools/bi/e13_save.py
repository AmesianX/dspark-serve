import json, sys, threading, time, uuid
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
N = 120; mode = sys.argv[1]; out = sys.argv[2]
for t in ("E5-A", "E5-B", "E5-D", "E5-Czzz12"):
    ask(big(t), 2, t)
r = {"alone": {t: ask(big(t), N, t)["text"] for t in ("E5-A", "E5-B", "E5-D")}}
res = {}
def run(t): res[t] = ask(big(t), N, t)["text"]
t0 = threading.Thread(target=run, args=("E5-Czzz12",)); t0.start(); time.sleep(0.3)
ts = [threading.Thread(target=run, args=(t,)) for t in ("E5-A", "E5-B", "E5-D")]; [t.start() for t in ts]; [t.join() for t in ts]; t0.join()
r["co"] = {t: res[t] for t in ("E5-A", "E5-B", "E5-D")}
json.dump(r, open(out, "w"), ensure_ascii=False)
if len(sys.argv) > 3:
    ref = json.load(open(sys.argv[3]))
    for t in ("E5-A", "E5-B", "E5-D"):
        print(f"{t}: alone==ref_alone {r['alone'][t] == ref['alone'][t]}  co==ref_alone {r['co'][t] == ref['alone'][t]}  co==alone {r['co'][t] == r['alone'][t]}", flush=True)
