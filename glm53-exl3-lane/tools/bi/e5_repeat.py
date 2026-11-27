import threading, uuid
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
def conc():
    res = {}
    def run(n): res[n] = ask(big(f"E5-{n}"), 100, n)["text"]
    th = [threading.Thread(target=run, args=(n,)) for n in "ABCD"]; [t.start() for t in th]; [t.join() for t in th]
    return res
seq1 = {n: ask(big(f"E5-{n}"), 100, n)["text"] for n in "ABCD"}
seq2 = {n: ask(big(f"E5-{n}"), 100, n)["text"] for n in "ABCD"}
runs = [conc() for _ in range(3)]
print("sequential run1 == run2:", {n: seq1[n] == seq2[n] for n in "ABCD"})
for i in (1, 2):
    print(f"concurrent run{i+1} == run1:", {n: runs[i][n] == runs[0][n] for n in "ABCD"})
