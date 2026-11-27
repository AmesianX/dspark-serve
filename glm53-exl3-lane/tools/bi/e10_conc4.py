import threading, uuid
from harness import BODY, QUESTION, ask
def big(tag, reps=1.8):
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + (BODY * 2)[: int(len(BODY) * reps)] + QUESTION
N = 200
TAGS = ["A", "B", "Cz", "D"]
TAGMAP = {"A": "E5-A", "B": "E5-B", "Cz": "E5-Czzz12", "D": "E5-D"}
seq = {n: ask(big(TAGMAP[n]), N, n) for n in TAGS}
for trial in range(2):
    res = {}
    def run(n): res[n] = ask(big(TAGMAP[n]), N, n)
    th = [threading.Thread(target=run, args=(n,)) for n in TAGS]; [t.start() for t in th]; [t.join() for t in th]
    for n in TAGS:
        print(f"[conc {trial}] {n}: same={res[n]['text'] == seq[n]['text']}  cached seq={seq[n]['cached']} conc={res[n]['cached']}  ttft seq={seq[n]['ttft_s']} conc={res[n]['ttft_s']}", flush=True)
