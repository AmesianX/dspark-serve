import time, torch, triton, triton.language as tl
@triton.jit
def _rowdot(x_ptr, w_ptr, y_ptr, K, sxm, swn, sym, BLOCK_K: tl.constexpr):
    m = tl.program_id(0); n = tl.program_id(1)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk; msk = kk < K
        a = tl.load(x_ptr + m * sxm + kk, mask=msk, other=0.0).to(tl.float32)
        b = tl.load(w_ptr + n * swn + kk, mask=msk, other=0.0)
        acc += a * b
    tl.store(y_ptr + m * sym + n, tl.sum(acc, axis=0))
def rowdot(x, w, BK=1024, nw=4):
    M, K = x.shape; N = w.shape[0]
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _rowdot[(M, N)](x, w, y, K, x.stride(0), w.stride(0), y.stride(0), BLOCK_K=BK, num_warps=nw)
    return y
torch.manual_seed(0); dev = "cuda"
X = torch.randn(9300, 4096, device=dev).to(torch.bfloat16)
W = torch.randn(16, 4096, device=dev) * .02
ref1 = rowdot(X[:1], W)[0]
badM = [M for M in (2, 5, 12, 16, 17, 994, 2975, 9236) if not torch.equal(rowdot(X[:M], W)[0], ref1)]
r = 1000; rr = rowdot(X[r:r + 1], W)[0]
badP = sum(not torch.equal(rowdot(X[s:s + 2975], W)[r - s], rr) for s in range(0, r + 1, 97))
exact = (W @ X[:64].float().t()).t()
err = (rowdot(X[:64], W) - exact).abs().max().item()
def tm(f, it=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for BK, nw in ((512, 2), (1024, 4), (2048, 4), (4096, 8)):
    print(f"BK={BK} warps={nw}: M5 {tm(lambda: rowdot(X[:5], W, BK, nw)):.1f}us  M16 {tm(lambda: rowdot(X[:16], W, BK, nw)):.1f}us  M9236 {tm(lambda: rowdot(X[:9236], W, BK, nw), 20):.0f}us")
cast_mm = lambda M: (W @ X[:M].float().t()).t().contiguous()
print(f"current cuBLAS (cast+mm): M5 {tm(lambda: cast_mm(5)):.1f}us  M9236 {tm(lambda: cast_mm(9236), 20):.0f}us")
print(f"M-variance {badM}  position-variance {badP}  max|rowdot-fp32mm| {err:.2e}")
