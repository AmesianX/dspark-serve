import threading, time, uuid
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
N = 120
alone = {n: ask(big(f"E5-{n}"), N, n)["text"] for n in "AB"}
def pair(delay):
    res = {}
    def run(n): res[n] = ask(big(f"E5-{n}"), N, n)
    ta = threading.Thread(target=run, args=("A",)); tb = threading.Thread(target=run, args=("B",))
    ta.start(); time.sleep(delay); tb.start(); ta.join(); tb.join()
    return {n: (res[n]["text"] == alone[n], res[n].get("ttft_s")) for n in "AB"}
print("A+B simultaneous:", pair(0.0), flush=True)
print("B 3s later     :", pair(3.0), flush=True)
print("A+B simultaneous:", pair(0.0), flush=True)
