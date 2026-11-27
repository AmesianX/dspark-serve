import threading, uuid, time
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
N = 120
for t in ("E5-A", "E5-B", "E5-D", "E5-Czzz12"):
    ask(big(t), 2, t)
before = {t: ask(big(t), N, t)["text"] for t in ("E5-A", "E5-B", "E5-D")}
res = {}
def run(t): res[t] = ask(big(t), N, t)["text"]
t0 = threading.Thread(target=run, args=("E5-Czzz12",)); t0.start(); time.sleep(0.3)
ts = [threading.Thread(target=run, args=(t,)) for t in ("E5-A", "E5-B", "E5-D")]; [t.start() for t in ts]; [t.join() for t in ts]; t0.join()
after = {t: ask(big(t), N, t)["text"] for t in ("E5-A", "E5-B", "E5-D")}
for t in ("E5-A", "E5-B", "E5-D"):
    print(f"{t}: co==before {res[t] == before[t]}   after==before {after[t] == before[t]}   after==co {after[t] == res[t]}", flush=True)
