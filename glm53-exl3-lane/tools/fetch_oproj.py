#!/usr/bin/env python3
"""Pull only the 46 ``self_attn.o_proj`` tensors (3.89 GB) of a GLM-5.3-Flash tr3-4bpw checkpoint.

The abliterated and the original checkpoint differ ONLY in these BF16 tensors
(every other tensor and every runtime file is byte identical), so either side
can be rebuilt from the other plus this one file. HTTP Range requests read the
tensors straight out of the HuggingFace shards (the safetensors header gives the
byte offsets) — no 175 GB download.

    # original o_proj — to turn an abliterated install into the original model
    python3 tools/fetch_oproj.py --source original   --out ~/Models/glm53-oproj/o_proj_original.safetensors
    # abliterated o_proj — the second input of tools/build_ab_lora.py on a fresh install
    python3 tools/fetch_oproj.py --source abliterated --out ~/Models/glm53-oproj/o_proj_abliterated.safetensors
    # same file from a local snapshot of that source instead of the network
    python3 tools/fetch_oproj.py --source abliterated --local <snapshot dir> --out <file>

The output is checked against the known md5 of each source.
"""
import argparse, hashlib, json, os, struct, sys, time, urllib.request

SOURCES = {
    # repo, pinned revision, md5 of the resulting file
    "original": ("brandonmusic/GLM-5.3-Flash-tr3-4bpw",
                 "a5fee929cf4888b1824323e33e8a19b60129e025",
                 "5d1f22fde91e8644e5f9555398620298"),
    "abliterated": ("lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated",
                    "c8f58e6aa9117c73607d692978b22f091d80450c",
                    "a7b574f9345822b4998da63de278ec2e"),
}


def rng(url, a, b, tries=5):
    for t in range(tries):
        try:
            q = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(q, timeout=300).read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


class Remote:
    def __init__(self, repo, rev):
        self.base = f"https://huggingface.co/{repo}/resolve/{rev}"

    def index(self):
        return json.loads(urllib.request.urlopen(f"{self.base}/model.safetensors.index.json", timeout=120).read())

    def header(self, shard):
        u = f"{self.base}/{shard}"
        n = struct.unpack("<Q", rng(u, 0, 7))[0]
        return json.loads(rng(u, 8, 8 + n - 1)), 8 + n

    def read(self, shard, a, b):
        return rng(f"{self.base}/{shard}", a, b - 1)


class Local:
    def __init__(self, root):
        self.root = root

    def index(self):
        return json.load(open(os.path.join(self.root, "model.safetensors.index.json")))

    def header(self, shard):
        with open(os.path.join(self.root, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n)), 8 + n

    def read(self, shard, a, b):
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(a)
            return f.read(b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--local", help="read the tensors from this local snapshot of --source instead of HuggingFace")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    repo, rev, want = SOURCES[a.source]
    src = Local(a.local) if a.local else Remote(repo, rev)
    wmap = src.index()["weight_map"]
    targets = sorted(k for k in wmap if "o_proj" in k)
    shards = sorted({wmap[k] for k in targets})
    print(f"o_proj tensors {len(targets)} in {len(shards)} shards from {repo}", flush=True)
    hdrs = {s: src.header(s) for s in shards}

    # Header first (every tensor's size is known from the shard headers), then
    # stream each tensor straight to disk — never holds the 3.89 GB in memory.
    meta, plan, off = {}, [], 0
    for k in targets:
        hdr, base = hdrs[wmap[k]]
        t = hdr[k]
        s0, s1 = t["data_offsets"]
        meta[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + s1 - s0]}
        plan.append((k, wmap[k], base + s0, base + s1))
        off += s1 - s0
    meta["__metadata__"] = {"source_repo": repo, "note": "non-abliterated o_proj tensors only"
                            if a.source == "original" else "abliterated o_proj tensors only"}
    hj = json.dumps(meta).encode()
    hj += b" " * ((-len(hj)) % 8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for i, (k, shard, b0, b1) in enumerate(plan, 1):
            data = src.read(shard, b0, b1)
            assert len(data) == b1 - b0, f"{k} length mismatch"
            f.write(data)
            print(f"  {i}/{len(plan)} {k} {len(data) / 1e6:.1f}MB", flush=True)
    os.replace(tmp, a.out)
    h = hashlib.md5()
    with open(a.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    md5 = h.hexdigest()
    verdict = "" if want is None else (" OK" if md5 == want else f" MISMATCH (expected {want})")
    print(f"\ndone: {a.out} {os.path.getsize(a.out) / 1e9:.2f} GB md5 {md5}{verdict}")
    sys.exit(1 if want is not None and md5 != want else 0)


if __name__ == "__main__":
    main()
