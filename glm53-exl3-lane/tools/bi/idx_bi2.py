import torch
import torch.nn.functional as F
torch.manual_seed(0); dev = "cuda"
X = torch.randn(4096, 4096, device=dev).to(torch.bfloat16)
Wnk = torch.randn(16, 4096, device=dev) * .02
Wb = (torch.randn(144, 4096, device=dev) * .02).to(torch.bfloat16)
fns = {"fp32 weights": lambda c: (Wnk @ c.float().t()).t().clone(memory_format=torch.contiguous_format),
       "bf16 wk_proj": lambda c: F.linear(c, Wb)}
for name, f in fns.items():
    # target rows 0..63, placed at every offset of a 1024-row (and 16-row) call; filler = other rows
    for n in (16, 1024):
        ref = f(X[:n])
        bad = 0
        for off in range(0, n, max(1, n // 64)):
            c = X[1000:1000 + n].clone(); c[off] = X[0]
            if not torch.equal(f(c)[off], ref[0]): bad += 1
        print(f"{name:14s} fixed {n:5d}-row call: row value depends on position/neighbours in {bad} of {len(range(0, n, max(1, n // 64)))} placements")
    big = f(X[:2975]); print(f"{name:14s} 2975-row call rows equal to fixed-1024 result: {sum(torch.equal(big[r], f(torch.cat([X[r:r+1], X[:1023]]))[0]) for r in range(0, 2975, 97))}/{len(range(0,2975,97))}")
