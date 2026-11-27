import glob, os, sys, torch
def load(d):
    rows = {}                                     # (pos, tok) -> {(layer, tag): tensor}
    for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
        b = torch.load(f)
        for i, p in enumerate(b["pos"].tolist()):
            key = (p, b["emb"][i].view(torch.int16).numpy().tobytes())
            if key in rows: continue              # first time this (position, token) was computed
            rows[key] = {k: v[i] for k, v in b["act"].items()}
    return rows
A, B = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(A) & set(B))
print("rows alone", len(A), "batch", len(B), "common (pos,tok)", len(common))
order = lambda k: (999 if k[0] == -1 else k[0], k[1])
first_div = {}
for key in common:
    a, b = A[key], B[key]
    for lk in sorted(a, key=order):
        if lk not in b: continue
        if not torch.equal(a[lk], b[lk]):
            d = (a[lk].float() - b[lk].float()).abs().max().item()
            first_div[key] = (lk, d)
            break
    else:
        first_div[key] = None
same = sum(1 for v in first_div.values() if v is None)
print("rows identical through final norm:", same, "/", len(common))
from collections import Counter
c = Counter(v[0] for v in first_div.values() if v)
for lk, n in sorted(c.items(), key=lambda x: order(x[0]))[:12]:
    print("first divergence at", lk, "rows", n)
for key in common[:8]:
    print(key[0], first_div[key])
