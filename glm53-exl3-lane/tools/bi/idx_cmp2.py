import glob, os, sys, torch
fa, fb = sorted(glob.glob(sys.argv[1] + "/*.pt")), sorted(glob.glob(sys.argv[2] + "/*.pt"))
B = [(f, torch.load(f)) for f in fb]
for x in fa[:12]:
    A = torch.load(x)
    m = [(f, b) for f, b in B if torch.equal(b["q"], A["q"])]
    if not m:
        # nearest by q bytes mismatch count
        best = min(B, key=lambda fb_: (fb_[1]["q"] != A["q"]).sum().item())
        print(os.path.basename(x), "NO exact q match; nearest", os.path.basename(best[0]), "q bytes differ", (best[1]["q"] != A["q"]).sum().item()); continue
    f, b = m[0]; out = []
    for key in ("w", "k", "ks", "logits"):
        ea, eb = A[key], b[key]
        if ea.shape != eb.shape: out.append(f"{key}: shape {tuple(ea.shape)} vs {tuple(eb.shape)}"); continue
        if torch.equal(ea, eb): out.append(f"{key}: equal"); continue
        if key in ("k", "ks"):
            rows = (ea != eb).reshape(ea.shape[0], -1).any(-1).nonzero().flatten()
            out.append(f"{key}: {rows.numel()} pools differ, first {rows[:8].tolist()} last {rows[-3:].tolist()}")
        elif key == "logits":
            d = (ea - eb).abs(); out.append(f"logits maxdiff {d.max().item():.3e} n_diff {(d > 0).sum().item()} first {(d>0).nonzero().flatten()[:6].tolist()}")
        else: out.append(f"{key}: differ")
    print(os.path.basename(x), "<->", os.path.basename(f), " | ".join(out), flush=True)
