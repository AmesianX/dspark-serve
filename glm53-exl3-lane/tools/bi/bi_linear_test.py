import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/w"); from bi_linear import bi_linear
torch.manual_seed(0); dev = "cuda"
X = torch.randn(9300, 4096, device=dev).to(torch.bfloat16)
for name, N, dt in (("wk 144", 144, torch.bfloat16), ("gate 128", 128, torch.bfloat16), ("router 288", 288, torch.bfloat16), ("weights fp32 16", 16, torch.float32)):
    W = (torch.randn(N, 4096, device=dev) * .02).to(dt); Xd = X.to(dt)
    ref = bi_linear(Xd[:1], W)[0]
    bad = 0
    for M in (1, 2, 5, 12, 16, 17, 100, 994, 1024, 2975, 4613, 9236):
        y = bi_linear(Xd[:M], W)
        if not torch.equal(y[0], ref): bad += 1
    pos = sum(not torch.equal(bi_linear(torch.roll(Xd[:2975], o, 0), W)[o], ref) for o in range(0, 2975, 37))
    err = (bi_linear(Xd[:64], W).float() - F.linear(Xd[:64], W).float()).abs().max().item()
    def tm(f, it=200):
        for _ in range(10): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(it): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
    tb = {M: tm(lambda: bi_linear(Xd[:M], W)) for M in (5, 16, 9236)}
    tc = {M: tm(lambda: F.linear(Xd[:M], W)) for M in (5, 16, 9236)}
    print(f"{name:16s} M-variance rows {bad}/12  position-variance {pos}/{len(range(0,2975,37))}  max|bi-cublas| {err:.1e} | "
          f"bi {tb[5]:.1f}/{tb[16]:.1f}/{tb[9236]:.0f}us  cublas {tc[5]:.1f}/{tc[16]:.1f}/{tc[9236]:.0f}us (M=5/16/9236)")
