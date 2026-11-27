# 가벼운 서버 재검증: S1 캐시 GPU 적중 / S2 동시 결정성 / S5(재기동 후) NVMe 복원
import json, sys, threading, time, uuid
sys.path.insert(0, "/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad/kvtest")
from harness import BODY, QUESTION, ask
REF = "/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad/cmtest/light_ref.json"
P = "[세션 LIGHT-1]\n" + BODY[: len(BODY) // 4] + QUESTION          # ~20K tokens
N = 150
phase = sys.argv[1]
if phase == "pre":
    a = ask(P, N, "cold"); b = ask(P, N, "hit")
    print(f"S1 cold ttft {a['ttft_s']}s ({a['prompt_tokens']} tok) -> repeat ttft {b['ttft_s']}s | same text {a['text'] == b['text']} | decode {b['decode_tps']} t/s", flush=True)
    res = {}
    def r(tag, prompt, n): res[tag] = ask(prompt, n, tag)
    th = [threading.Thread(target=r, args=("dec", "파이썬으로 큐를 구현하고 설명해줘.", 400)),
          threading.Thread(target=r, args=("cold", f"[콜드 {uuid.uuid4()}]\n" + BODY[: len(BODY) // 12] + QUESTION, 8))]
    th[0].start(); time.sleep(2); th[1].start(); time.sleep(0.5)
    c = ask(P, N, "conc"); [t.join() for t in th]
    print(f"S2 same prompt with 1 decoding + 1 cold prefill concurrently: same text {c['text'] == a['text']}", flush=True)
    json.dump({"text": a["text"], "ttft_cold": a["ttft_s"]}, open(REF, "w"), ensure_ascii=False)
else:
    ref = json.load(open(REF)); d = ask(P, N, "after-restart")
    print(f"S5 after RESTARTALL: ttft {d['ttft_s']}s (cold was {ref['ttft_cold']}s) | same text as before restart {d['text'] == ref['text']}", flush=True)
