# vLLM 배치불변 모드(Hopper/Blackwell 분기)와 같은 cuBLAS 설정에서 작은 GEMM이 행 불변인지 + 속도
import os, sys, time, torch, torch.nn.functional as F
mode = sys.argv[1]
if mode == "nosplitk":
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.manual_seed(0); dev = "cuda"
X = torch.randn(9300, 4096, device=dev).to(torch.bfloat16)
def tm(f, it=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for name, N, dt in (("wk 144", 144, torch.bfloat16), ("gate 128", 128, torch.bfloat16), ("router 288", 288, torch.bfloat16), ("weights fp32 16", 16, torch.float32)):
    W = (torch.randn(N, 4096, device=dev) * .02).to(dt); Xd = X.to(dt)
    lin = (lambda x: F.linear(x, W)) if dt != torch.float32 else (lambda x: (W @ x.t()).t())
    ref = lin(Xd[:1])[0]
    badM = [M for M in (2, 3, 5, 12, 16, 17, 64, 100, 994, 1024, 2975, 4613, 9236) if not torch.equal(lin(Xd[:M])[0], ref)]
    # neighbour/position: row r inside various calls (sliced views at odd offsets, as in the server)
    r = 1000; refr = lin(Xd[r:r + 1])[0]
    badP = sum(not torch.equal(lin(Xd[s:s + M])[r - s], refr) for M in (1024, 2975) for s in range(r - M + 1 if r - M + 1 > 0 else 0, r + 1, 97))
    t5, t16, t9 = tm(lambda: lin(Xd[:5])), tm(lambda: lin(Xd[:16])), tm(lambda: lin(Xd[:9236]), 20)
    print(f"[{mode}] {name:16s} M-variance at {badM}  slice-position variance {badP}  time M5 {t5:.1f}us M16 {t16:.1f}us M9236 {t9:.0f}us", flush=True)
