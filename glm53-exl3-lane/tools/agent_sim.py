#!/usr/bin/env python3
"""cm2-shaped agent traffic to reproduce the '!' (token 0) reasoning collapse.

Each session replays what codemason sends: its real system prompt and 18 tool
schemas (captured from a CODEMASON_DEBUG request body), vendor sampling
(temperature 1.0 / top_p 0.95), reasoning_effort=high, clear_thinking=true,
repetition_detection {8,512,4}, streaming, reasoning round-tripped inside the
tool loop. Tools run read-only against a real source tree, so prompts grow the
way a coding session grows and later requests re-hit blocks written during
earlier decodes.

On a collapse (a run of '!' in reasoning/content, or finish_reason=repetition)
the exact request body is saved and replayed: twice as-is (prefix cache hit)
and twice with a fresh cache_salt (same prompt, cache bypassed).
"""
import argparse, glob as globmod, json, os, random, re, shlex, subprocess, threading, time, urllib.request, uuid

URL = os.environ.get("URL", "http://localhost:8000")
import _lane
KEY = _lane.api_key()
LOCK = threading.Lock()

TASKS = [
    "/repo 는 vLLM 소스 트리다. 하이브리드(Mamba/선형어텐션) 모델에서 prefix caching 이 상태 체크포인트를 어떻게 저장하고 재사용하는지 관련 파일을 전부 직접 읽고 추적해서, 함수 단위로 흐름을 정리한 분석 보고서를 써줘. 추측하지 말고 코드를 근거로.",
    "/repo 의 V2 model runner(vllm/v1/worker/gpu/)에서 MTP speculative decoding 이 draft 생성, verify, rejection, 상태 롤백을 어떻게 하는지 코드를 끝까지 읽고 단계별로 설명해줘. 관련 파일을 빠짐없이 열어봐.",
    "/repo/vllm/v1/core/kv_cache_utils.py 와 kv_cache_coordinator.py, single_type_kv_cache_manager.py 를 처음부터 끝까지 정독하고 버그 가능성이 있는 곳을 전부 찾아 근거와 함께 목록으로 정리해줘.",
    "/repo 에서 GLM-5.3-Flash(glm5next) 모델 구현과 sparse attention indexer(kpool) 경로를 전부 읽고, 프리필과 디코드에서 인덱서 캐시가 어떻게 쓰이고 읽히는지 설명해줘. 파일을 실제로 열어 확인할 것.",
    "/repo 의 스케줄러(vllm/v1/core/sched/)가 chunked prefill 과 prefix cache hit 를 어떻게 다루는지 코드를 정독하고, mamba align 모드의 청크 분할 규칙을 예시 숫자와 함께 설명해줘.",
]
FOLLOWUPS = [
    "좋아. 방금 설명한 부분과 연결된 다른 파일들도 더 열어서 빠진 경로가 없는지 확인해줘.",
    "테스트 코드(tests/)에서 이 동작을 검증하는 케이스를 찾아 읽고, 보고서와 어긋나는 점이 있으면 고쳐줘.",
    "지금까지 내용 중 확신이 낮은 주장 세 개를 골라 코드를 다시 읽어 검증해줘.",
    "엣지 케이스(빈 입력, 블록 경계, 동시 요청) 관점에서 다시 한 번 코드를 따라가 봐.",
]

READONLY = {"ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "sed", "awk", "sort", "uniq",
            "git", "tree", "file", "stat", "du", "echo", "pwd", "python3", "cut", "tr", "nl", "diff"}


def http(path, body, stream=False, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return urllib.request.urlopen(req, timeout=timeout)


def metric(name):
    try:
        req = urllib.request.Request(URL + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
        for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines():
            if line.startswith(name):
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return None


def bang_run(s):
    m = max((len(x) for x in re.findall(r"!+", s or "")), default=0)
    return m


def stream_chat(body):
    """Return dict(reasoning, content, tool_calls, finish, usage)."""
    out = {"reasoning": "", "content": "", "tc": {}, "finish": None, "usage": {}}
    with http("/v1/chat/completions", body, stream=True) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            try:
                ev = json.loads(s[5:])
            except Exception:
                continue
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                out["reasoning"] += d.get("reasoning") or d.get("reasoning_content") or ""
                out["content"] += d.get("content") or ""
                for t in d.get("tool_calls") or []:
                    e = out["tc"].setdefault(t.get("index", 0), {"id": None, "name": "", "args": ""})
                    if t.get("id"):
                        e["id"] = t["id"]
                    f = t.get("function") or {}
                    e["name"] += f.get("name") or ""
                    e["args"] += f.get("arguments") or ""
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
    return out


def mapped(p, repo):
    p = p or "/repo"
    if p.startswith("/repo"):
        p = repo + p[len("/repo"):]
    elif not p.startswith("/"):
        p = os.path.join(repo, p)
    rp = os.path.realpath(p)
    return rp if rp.startswith(os.path.realpath(repo)) else None


def run_tool(name, args, repo):
    cap = 30000
    try:
        if name == "Read":
            p = mapped(args.get("file_path"), repo)
            if not p or not os.path.isfile(p):
                return "File does not exist."
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            off = max(int(args.get("offset") or 1), 1)
            lim = int(args.get("limit") or 2000)
            seg = lines[off - 1: off - 1 + lim]
            return "\n".join(f"{i:6d}\t{l[:2000]}" for i, l in enumerate(seg, off))[:cap]
        if name == "Grep":
            cmd = ["rg", "--no-heading", "-n" if args.get("-n", True) else ""]
            mode = args.get("output_mode", "files_with_matches")
            if mode == "files_with_matches":
                cmd = ["rg", "-l"]
            elif mode == "count":
                cmd = ["rg", "-c"]
            if args.get("-i"):
                cmd.append("-i")
            if args.get("glob"):
                cmd += ["-g", args["glob"]]
            for k in ("-A", "-B", "-C"):
                if args.get(k):
                    cmd += [k, str(args[k])]
            p = mapped(args.get("path"), repo) or repo
            cmd += ["-e", args.get("pattern", ""), p]
            r = subprocess.run([c for c in cmd if c], capture_output=True, text=True, timeout=30)
            return (r.stdout.replace(repo, "/repo") or "No matches found")[:cap]
        if name == "Glob":
            base = mapped(args.get("path"), repo) or repo
            hits = sorted(globmod.glob(os.path.join(base, args.get("pattern", "*")), recursive=True))[:200]
            return ("\n".join(h.replace(repo, "/repo") for h in hits) or "No files found")[:cap]
        if name == "Bash":
            cmd = args.get("command", "")
            if re.search(r"[;&`>]|\$\(|\brm\b|\bmv\b|\bcp\b|\btee\b|\bdd\b", cmd):
                return "Error: only read-only commands are allowed in this sandbox."
            for seg in cmd.split("|"):
                toks = shlex.split(seg.replace("/repo", repo))
                if not toks or toks[0] not in READONLY:
                    return f"Error: command not allowed in this sandbox: {toks[:1]}"
                if toks[0] == "sed" and "-i" in toks:
                    return "Error: in-place edits are not allowed."
            r = subprocess.run(cmd.replace("/repo", repo), shell=True, cwd=repo, capture_output=True,
                               text=True, timeout=30)
            return ((r.stdout + r.stderr).replace(repo, "/repo") or "(no output)")[:cap]
        if name in ("Edit", "Write", "NotebookEdit"):
            return "The file has been updated successfully."
        if name == "TodoWrite":
            return "Todos have been modified successfully."
        return f"{name} is not available in this environment."
    except Exception as e:
        return f"Error: {e}"


class Sim:
    def __init__(self, a, tmpl):
        self.a, self.tmpl = a, tmpl
        self.stop = False
        os.makedirs(a.outdir, exist_ok=True)
        self.log = open(os.path.join(a.outdir, "requests.jsonl"), "a")
        self.n_req = self.n_collapse = 0

    def write(self, rec):
        with LOCK:
            self.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log.flush()
            print(json.dumps({k: rec[k] for k in rec if k not in ("tail_r", "tail_c")}, ensure_ascii=False), flush=True)

    def body(self, messages, **kw):
        b = {k: v for k, v in self.tmpl.items() if k not in ("messages", "cache_prompt")}
        b["messages"] = messages
        b["max_tokens"] = self.a.max_tokens
        b["repetition_detection"] = {"min_pattern_size": 8, "max_pattern_size": 512, "min_count": 4}
        b.update(kw)
        return b

    def replay(self, body, tag):
        res = []
        for kind in ("same", "same", "salt", "salt"):
            b = dict(body, stream=True, max_tokens=self.a.replay_tokens)
            if kind == "salt":
                b["cache_salt"] = str(uuid.uuid4())
            t = time.time()
            try:
                o = stream_chat(b)
            except Exception as e:
                res.append({"kind": kind, "err": repr(e)[:200]})
                continue
            res.append({"kind": kind, "bang_r": bang_run(o["reasoning"]), "bang_c": bang_run(o["content"]),
                        "finish": o["finish"], "r_len": len(o["reasoning"]), "c_len": len(o["content"]),
                        "head_r": o["reasoning"][:120], "dur": round(time.time() - t, 1)})
        return res

    def session(self, sid):
        rng = random.Random(sid * 7919 + int(time.time()))
        while not self.stop:
            task = TASKS[sid % len(TASKS)] if rng.random() < 0.7 else rng.choice(TASKS)
            sys_msg = self.tmpl["messages"][0]
            messages = [sys_msg, {"role": "user", "content": task}]
            turn = 0
            while not self.stop and turn < self.a.max_turns:
                turn += 1
                body = self.body(messages)
                nan0 = metric("vllm:corrupted_requests")
                t = time.time()
                try:
                    o = stream_chat(body)
                except Exception as e:
                    self.write({"sid": sid, "turn": turn, "err": repr(e)[:300]})
                    time.sleep(5)
                    break
                nan1 = metric("vllm:corrupted_requests")
                u = o["usage"]
                br, bc = bang_run(o["reasoning"]), bang_run(o["content"])
                collapsed = br >= 30 or bc >= 30 or o["finish"] == "repetition"
                with LOCK:
                    self.n_req += 1
                rec = {"ts": time.strftime("%H:%M:%S"), "sid": sid, "turn": turn,
                       "prompt": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                       "r_len": len(o["reasoning"]), "c_len": len(o["content"]), "ntc": len(o["tc"]),
                       "finish": o["finish"], "bang_r": br, "bang_c": bc,
                       "nan_delta": None if nan0 is None or nan1 is None else nan1 - nan0,
                       "dur": round(time.time() - t, 1), "collapsed": collapsed,
                       "tail_r": o["reasoning"][-200:], "tail_c": o["content"][-200:]}
                self.write(rec)
                if collapsed:
                    with LOCK:
                        self.n_collapse += 1
                        k = self.n_collapse
                    path = os.path.join(self.a.outdir, f"collapse_{k:03d}_s{sid}_t{turn}.json")
                    json.dump({"body": body, "out": {kk: (vv if kk != "tc" else list(vv.values())) for kk, vv in o.items()},
                               "rec": rec}, open(path, "w"), ensure_ascii=False)
                    rr = self.replay(body, path)
                    self.write({"ts": time.strftime("%H:%M:%S"), "sid": sid, "turn": turn, "replay_of": path, "replay": rr})
                    break  # start a fresh session; the saved body is the reproducer
                tcs = [v for _, v in sorted(o["tc"].items()) if v["name"]]
                am = {"role": "assistant", "content": o["content"] or ""}
                if o["reasoning"]:
                    am["reasoning_content"] = o["reasoning"]
                if tcs:
                    am["tool_calls"] = []
                    for i, v in enumerate(tcs):
                        v["id"] = v["id"] or f"call_{uuid.uuid4().hex[:24]}"
                        am["tool_calls"].append({"id": v["id"], "type": "function",
                                                 "function": {"name": v["name"], "arguments": v["args"] or "{}"}})
                messages.append(am)
                if tcs:
                    for v in tcs:
                        try:
                            args = json.loads(v["args"] or "{}")
                        except Exception:
                            args = {}
                        messages.append({"role": "tool", "tool_call_id": v["id"],
                                         "content": run_tool(v["name"], args, self.a.repo)})
                else:
                    messages.append({"role": "user", "content": rng.choice(FOLLOWUPS)})
                if (u.get("prompt_tokens") or 0) > self.a.max_prompt:
                    break

    def run(self):
        th = [threading.Thread(target=self.session, args=(i,), daemon=True) for i in range(self.a.sessions)]
        for i, t in enumerate(th):
            t.start()
            time.sleep(self.a.stagger)
        t_end = time.time() + self.a.minutes * 60
        try:
            while time.time() < t_end and any(t.is_alive() for t in th):
                if self.a.stop_after and self.n_collapse >= self.a.stop_after:
                    break
                time.sleep(5)
        finally:
            self.stop = True
            print(json.dumps({"summary": True, "requests": self.n_req, "collapses": self.n_collapse}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="captured cm2 request body (json)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--replay-tokens", type=int, default=600)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--max-prompt", type=int, default=300_000)
    ap.add_argument("--stagger", type=float, default=20)
    ap.add_argument("--stop-after", type=int, default=0)
    a = ap.parse_args()
    Sim(a, json.load(open(a.template))).run()


if __name__ == "__main__":
    main()
