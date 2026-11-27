# 온도0 결정성 종단 시험: A의 출력이 (1) 혼자 반복 (2) 다른 요청이 콜드 프리필하는 동안 (3) ABCD 동시 — 모두 같은가
import threading, time, uuid, sys
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
def cold(frac):
    return f"[콜드 {uuid.uuid4()}]\n" + BODY[: int(len(BODY) * frac)] + QUESTION
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 200
seq = {n: ask(big(f"E5-{n}"), NTOK, n)["text"] for n in "ABCD"}
seq2 = ask(big("E5-A"), NTOK, "A")["text"]
print("A alone repeat identical:", seq2 == seq["A"], flush=True)
for trial in range(3):
    res = {}
    def runA(): res["A"] = ask(big("E5-A"), NTOK, "A")
    ta = threading.Thread(target=runA); ta.start()
    time.sleep(2.0 + trial)
    tb = [threading.Thread(target=lambda f=f: ask(cold(f), 8, "cold")) for f in (0.08, 0.12)]
    for t in tb: t.start(); time.sleep(1.5)
    ta.join(); [t.join() for t in tb]
    r = res["A"]
    print(f"[mix {trial}] A during cold prefills == alone: {r['text'] == seq['A']}  (A {r.get('decode_tps')} t/s)", flush=True)
for trial in range(2):
    res = {}
    def run(n): res[n] = ask(big(f"E5-{n}"), NTOK, n)["text"]
    th = [threading.Thread(target=run, args=(n,)) for n in "ABCD"]; [t.start() for t in th]; [t.join() for t in th]
    print(f"[conc {trial}] ABCD concurrent == sequential:", {n: res[n] == seq[n] for n in "ABCD"}, flush=True)
