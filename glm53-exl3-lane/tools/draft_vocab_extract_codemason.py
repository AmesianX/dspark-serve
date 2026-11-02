"""Extract model-GENERATED text from codemason transcripts for the draft-vocab corpus.

Only what the model produced goes in: assistant text, thinking, and tool_use arguments
(tool results are inputs, not generations). Sessions are split 80/20 (seeded) into a
frequency ('train') part and a held-out real-usage evaluation part.

Usage (host, no GPU):
  python3 tools/draft_vocab_extract_codemason.py <workdir>/cm.json
Pipeline: draft_vocab_corpus_gen.py -> this -> draft_vocab_build.py (see its header).
"""
import glob, json, os, random, sys

out_path = sys.argv[1] if len(sys.argv) > 1 else "cm.json"
fs = sorted(glob.glob(os.path.expanduser("~/.codemason/projects/**/*.jsonl"), recursive=True))
random.Random(0).shuffle(fs)
cut = int(len(fs) * 0.8)
out = []
for i, f in enumerate(fs):
    split = "train" if i < cut else "eval"
    parts = []
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        c = (d.get("message") or {}).get("content")
        if isinstance(c, str):
            parts.append(c)
            continue
        for b in c or []:
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "thinking":
                parts.append(b.get("thinking", ""))
            elif t == "tool_use":
                parts.append(json.dumps(b.get("input", {}), ensure_ascii=False))
    txt = "\n".join(p for p in parts if p)
    if txt.strip():
        out.append({"domain": "codemason_real", "split": split, "prompt": os.path.basename(f), "text": txt})
json.dump(out, open(out_path, "w"), ensure_ascii=False)
tr = sum(len(o["text"]) for o in out if o["split"] == "train")
ev = sum(len(o["text"]) for o in out if o["split"] == "eval")
print(f"sessions with output: {len(out)}  train chars {tr:,}  eval chars {ev:,} -> {out_path}")
