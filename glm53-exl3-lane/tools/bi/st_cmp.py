import glob, os, sys, torch
a, b = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(a, "*.pt"))):
    n = os.path.basename(f); g = os.path.join(b, n)
    if not os.path.exists(g): continue
    x, y = torch.load(f), torch.load(g)
    eq = torch.equal(x["state"], y["state"])
    print(n, "equal" if eq else f"DIFF maxdiff {(x['state']-y['state']).abs().max().item():.3e}", "has_init", x["has_init"], y["has_init"])
