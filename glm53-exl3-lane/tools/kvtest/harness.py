import json, os, subprocess, threading, time, uuid
from client import ask
S = os.path.dirname(os.path.abspath(__file__))
BODY = open(os.path.join(S, "p60k.txt")).read()          # ~82K tokens of code
QUESTION = "\n\n위 코드들의 전체 구조를 한국어로 5줄 요약해라."
WORKER = subprocess.run(["bash", "-c", "source ~/work/dspark-serve-kvoff/glm53-exl3-lane/glm53-serve.env >/dev/null 2>&1; echo $WORKER_SSH"],
                        capture_output=True, text=True).stdout.strip()

def prompt(tag, frac=1.0):
    """Distinct conversation: a unique first line forks the prefix hash chain."""
    body = BODY[: int(len(BODY) * frac)]
    return f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n" + body + QUESTION

class MemFloor:
    """Lowest MemAvailable (GB) seen on head and worker while running."""
    def __init__(self, every=5):
        self.every, self.min = every, {"head": 1e9, "worker": 1e9}; self.stop = False
        threading.Thread(target=self.loop, daemon=True).start()
    def sample(self):
        h = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1048576
        w = subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, "awk '/MemAvailable/{print $2}' /proc/meminfo"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        self.min["head"] = min(self.min["head"], h)
        if w: self.min["worker"] = min(self.min["worker"], int(w) / 1048576)
    def loop(self):
        while not self.stop:
            try: self.sample()
            except Exception: pass
            time.sleep(self.every)
    def report(self):
        return {k: round(v, 2) for k, v in self.min.items()}

def slots():
    """Valid slot headers per rank (from inside the containers)."""
    code = open(os.path.join(S, "count.py")).read()
    h = subprocess.run(["docker", "exec", "-i", "vllm_glm53", "python3", "-"], input=code, capture_output=True, text=True).stdout.strip()
    w = subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, "docker exec -i vllm_glm53 python3 -"], input=code,
                       capture_output=True, text=True).stdout.strip()
    return h.split("slots:")[-1].strip(), w.split("slots:")[-1].strip()

def log(rec, path):
    with open(path, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "text"}, ensure_ascii=False), flush=True)
