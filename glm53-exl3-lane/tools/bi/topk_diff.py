import glob, os, sys, torch
def load(d, P):
    for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
        b = torch.load(f)
        for i, p in enumerate(b["pos"].tolist()):
            if p == P and (3, "2a_idx_topk") in b["act"]:
                return b["act"][(3, "2a_idx_topk")][i].long(), b["emb"][i]
for P in [148136, 148137, 148150, 148200]:
    a, ea = load(sys.argv[1], P); b, eb = load(sys.argv[2], P)
    sa = set(a[a >= 0].tolist()); sb = set(b[b >= 0].tolist())
    only_a = sorted(sa - sb); only_b = sorted(sb - sa)
    print(P, "n", len(sa), len(sb), "| only alone:", only_a[:20], "| only co:", only_b[:20], "| order-equal-on-common:", [x for x in a.tolist() if x in sb][:0] == [], flush=True)
