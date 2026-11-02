import torch, time, sys
sys.path.insert(0, "/ours")
from skinny_gemm import skinny_gemm
torch.manual_seed(0); dev = "cuda"; COPIES = 48   # 48 x 2.4MB > L2 -> DRAM-cold
def bench(f, n):
    for i in range(8): f(i)
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(n): f(i)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
ok = True
for M in (1, 5, 20, 64):
    for (N, K, dt) in ((64, 4096, torch.float32), (192, 4096, torch.bfloat16), (128, 4096, torch.bfloat16), (288, 4096, torch.bfloat16)):
        x = torch.randn(M, K, device=dev, dtype=dt)
        ws = [torch.randn(N, K, device=dev, dtype=dt) * 0.05 for _ in range(COPIES)]
        pad = [torch.empty(4 << 20, device=dev, dtype=torch.uint8) for _ in range(COPIES)]  # L2 flush filler
        ref32 = x.float() @ ws[0].float().t(); y = skinny_gemm(x, ws[0], out_dtype=torch.float32)
        rel = ((y - ref32).abs().max() / ref32.abs().max()).item()
        if dt == torch.float32:
            wts = [w.t().contiguous() for w in ws]
            base = lambda i: torch.mm(x, wts[i % COPIES])
        else:
            base = lambda i: torch.nn.functional.linear(x, ws[i % COPIES])
        ours = lambda i: skinny_gemm(x, ws[i % COPIES], out_dtype=torch.float32)
        t_new = bench(ours, 192); t_old = bench(base, 192)
        tol = 1e-4 if dt == torch.float32 else 2e-2; flag = "OK " if rel < tol else "BAD"; ok &= rel < tol
        print(f"{flag} M={M:2d} N={N:3d} {str(dt)[6:]:8s} rel-err={rel:.1e}  ours={t_new:6.1f}us  torch(cold)={t_old:6.1f}us")
print("ALL OK" if ok else "FAILURES")
