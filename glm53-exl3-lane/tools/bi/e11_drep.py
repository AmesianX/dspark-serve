import threading, uuid, time
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
N = 200
alone = [ask(big("E5-D"), N, "D")["text"] for _ in range(5)]
print("D alone x5 distinct outputs:", len(set(alone)), flush=True)
co = []
for _ in range(3):
    res = {}
    def run(t): res[t] = ask(big(t), N, t)["text"]
    t0 = threading.Thread(target=run, args=("E5-Czzz12",)); t0.start(); time.sleep(0.3)
    ts = [threading.Thread(target=run, args=(t,)) for t in ("E5-A", "E5-B", "E5-D")]; [t.start() for t in ts]; [t.join() for t in ts]; t0.join()
    co.append(res)
for i, r in enumerate(co):
    print(f"co {i}: D==alone {r['E5-D'] == alone[0]}  A==? B==? (vs co0) A {r['E5-A'] == co[0]['E5-A']} B {r['E5-B'] == co[0]['E5-B']} D {r['E5-D'] == co[0]['E5-D']}", flush=True)
