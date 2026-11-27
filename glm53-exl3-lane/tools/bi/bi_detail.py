import glob, os, sys, torch
def load(d):
    rows = {}; occ = {}
    for fi, f in enumerate(sorted(glob.glob(os.path.join(d, "*.pt")))):
        b = torch.load(f); n = b["pos"].numel()
        for i, p in enumerate(b["pos"].tolist()):
            key = (p, b["emb"][i].view(torch.int16).numpy().tobytes())
            occ.setdefault(key, []).append((fi, n))
            if key in rows: continue
            rows[key] = ({k: v[i] for k, v in b["act"].items()}, fi, b.get("m_total", n))
    return rows, occ
A, oa = load(sys.argv[1]); B, ob = load(sys.argv[2])
order = lambda k: (999 if k[0] == -1 else k[0], k[1])
for key in sorted(set(A) & set(B)):
    a, fa, na = A[key]; b, fb, nb = B[key]
    for lk in sorted(a, key=order):
        if lk in b and not torch.equal(a[lk], b[lk]):
            if lk[1] in ("1b_a_in_proj", "1b_b_f_b_proj", "1b_f_o_proj") or key[0] < 148962:
                print(f"pos {key[0]} first@{lk} maxdiff {(a[lk].float()-b[lk].float()).abs().max().item():.2e} "
                      f"alone: file{fa} rows_in_fwd={na} occ={len(oa[key])} | batch: file{fb} rows_in_fwd={nb} occ={len(ob[key])}")
            break
