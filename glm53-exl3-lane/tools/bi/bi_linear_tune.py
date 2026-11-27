import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/w"); from bi_linear import bi_linear
torch.manual_seed(0); dev = "cuda"
X = torch.randn(9300, 4096, device=dev).to(torch.bfloat16)
def tm(f, it=100):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
cfgs = [(bm, bn, bk, nw, ns) for bm in (16, 32, 64) for bn in (16, 32, 64) for bk in (64, 128, 256) for nw in (4, 8) for ns in (3, 4)]
for name, N, dt, prec in (("wk 144", 144, torch.bfloat16, "tf32"), ("router 288", 288, torch.bfloat16, "tf32"), ("weights fp32 16", 16, torch.float32, "tf32x3")):
    W = (torch.randn(N, 4096, device=dev) * .02).to(dt); Xd = X.to(dt)
    c5, c9 = tm(lambda: F.linear(Xd[:5], W)), tm(lambda: F.linear(Xd[:9236], W), 20)
    res = []
    for bm, bn, bk, nw, ns in cfgs:
        if dt == torch.float32 and bk > 128: continue
        cfg = (bm, bn, bk, nw, ns, prec)
        try:
            t5 = tm(lambda: bi_linear(Xd[:5], W, cfg=cfg), 50); t9 = tm(lambda: bi_linear(Xd[:9236], W, cfg=cfg), 10)
        except Exception as e:
            continue
        res.append((max(t5 / c5, t9 / c9), t5, t9, cfg))
    res.sort(key=lambda r: r[0])
    print(f"{name}: cublas M5 {c5:.1f}us M9236 {c9:.0f}us")
    for r in res[:3]:
        print(f"   cfg {r[3]}  M5 {r[1]:.1f}us  M9236 {r[2]:.0f}us  worst-ratio {r[0]:.2f}")
