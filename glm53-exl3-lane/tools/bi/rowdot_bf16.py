import time, torch, triton, triton.language as tl, torch.nn.functional as F
@triton.jit
def _rowdot(x_ptr, w_ptr, y_ptr, K, N, sxm, swn, sym, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    m = tl.program_id(0); pn = tl.program_id(1)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N); rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(x_ptr + m * sxm + kk, mask=kk < K, other=0.0).to(tl.float32)
        b = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K), other=0.0).to(tl.float32)
        acc += a[None, :] * b
    tl.store(y_ptr + m * sym + rn, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=rn < N)
def rowdot(x, w, BK, BN, nw):
    M, K = x.shape; N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _rowdot[(M, triton.cdiv(N, BN))](x, w, y, K, N, x.stride(0), w.stride(0), y.stride(0), BLOCK_K=BK, BLOCK_N=BN, num_warps=nw)
    return y
torch.manual_seed(0); dev = "cuda"
X = torch.randn(64, 4096, device=dev).to(torch.bfloat16)
def tm(f, it=300):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for N in (144, 128, 288):
    W = (torch.randn(N, 4096, device=dev) * .02).to(torch.bfloat16)
    best = []
    for BK in (256, 512, 1024):
        for BN in (1, 2, 4, 8):
            for nw in (1, 2, 4):
                try: best.append((tm(lambda: rowdot(X[:5], W, BK, BN, nw)), BK, BN, nw))
                except Exception: pass
    best.sort()
    t, BK, BN, nw = best[0]
    ref = rowdot(X[:1], W, BK, BN, nw)[0]
    inv = all(torch.equal(rowdot(X[:M], W, BK, BN, nw)[0], ref) for M in (2, 5, 12, 16, 33, 64))
    t16 = tm(lambda: rowdot(X[:16], W, BK, BN, nw))
    print(f"N={N}: best rowdot BK={BK} BN={BN} warps={nw}  M5 {t:.1f}us M16 {t16:.1f}us  invariant {inv} | cuBLAS M5 {tm(lambda: F.linear(X[:5], W)):.1f}us M16 {tm(lambda: F.linear(X[:16], W)):.1f}us")
