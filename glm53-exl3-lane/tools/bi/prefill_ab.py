import sys, time
from harness import prompt, ask
tag = sys.argv[1]
for i in range(int(sys.argv[2])):
    r = ask(prompt(f"PF-{tag}-{i}-{time.time()}"), 4, "x")
    print(f"[{tag}] prompt {r['prompt_tokens']} ttft {r['ttft_s']}s -> prefill {r['prompt_tokens']/r['ttft_s']:.0f} t/s", flush=True)
