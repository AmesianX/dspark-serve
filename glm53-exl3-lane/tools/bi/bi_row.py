import glob, os, sys, torch
def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
        b = torch.load(f)
        for i, p in enumerate(b["pos"].tolist()):
            key = (p, b["emb"][i].view(torch.int16).numpy().tobytes())
            rows.setdefault(key, {k: v[i] for k, v in b["act"].items()})
    return rows
A = load(sys.argv[1]); B = load(sys.argv[2]); P = int(sys.argv[3]); L = [int(x) for x in sys.argv[4].split(",")]
for key in A:
    if key[0] != P or key not in B: continue
    for lk in sorted(A[key]):
        if lk[0] in L and lk in B[key]:
            a, b = A[key][lk].float(), B[key][lk].float()
            print(lk, "equal" if torch.equal(A[key][lk], B[key][lk]) else f"maxdiff {(a-b).abs().max().item():.2e}")
