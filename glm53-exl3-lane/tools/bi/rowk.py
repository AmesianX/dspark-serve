import glob, os, sys, torch
def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
        b = torch.load(f)
        for i, p in enumerate(b["pos"].tolist()):
            if 148100 <= p <= 148115:
                key = (p, b["emb"][i].view(torch.int16).numpy().tobytes())
                rows.setdefault(key, {k: v[i] for k, v in b["act"].items() if k[0] == 3})
    return rows
A = load(sys.argv[1]); B = load(sys.argv[2])
for key in sorted(A):
    if key not in B: continue
    diffs = [lk[1] for lk in sorted(A[key]) if lk in B[key] and not torch.equal(A[key][lk], B[key][lk])]
    print(key[0], diffs[:6])
