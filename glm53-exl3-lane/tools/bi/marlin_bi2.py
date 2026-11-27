import torch, time
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.batch_invariant import matmul_persistent
from vllm.scalar_type import scalar_types
torch.manual_seed(0); dev="cuda"
N, K = 12800, 4096
w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
wref, qw, sc = marlin_quant_fp8_torch(w, 128)   # wref: [K,N] bf16 dequant
ws = marlin_make_workspace_new(torch.device(dev))
def g(x):
    return ops.marlin_gemm(a=x, c=None, b_q_weight=qw, b_bias=None, b_scales=sc, a_scales=None,
        global_scale=None, b_zeros=None, g_idx=None, perm=None, workspace=ws,
        b_q_type=scalar_types.float8_e4m3fn, size_m=x.shape[0], size_n=N, size_k=K,
        use_atomic_add=False, use_fp32_reduce=True)
X = torch.randn(2048, K, dtype=torch.bfloat16, device=dev)
single = torch.cat([g(X[r:r+1]) for r in range(16)])
ok = all(torch.equal(g(X[:M])[r], single[r]) for M in range(1,17) for r in range(M))
print("Marlin M<=16: every row == its M=1 result:", ok)
# row placed at different offsets inside a 16 block
perm_ok = all(torch.equal(g(torch.roll(X[:16], s, 0))[s], single[0]) for s in range(16))
print("Marlin M=16 position-invariant:", perm_ok)
# persistent matmul invariance
wt = wref.contiguous()  # [K,N]
base = matmul_persistent(X[:1], wt)[0]
bad = [M for M in [1,2,5,16,17,33,64,100,257,994,1003,1024,2048] if not torch.equal(matmul_persistent(X[:M], wt)[0], base)]
print("persistent matmul row0 differs at M:", bad)
# timing: decode-ish M
def tm(f, it=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it*1e3
for M in [5, 12, 16, 20, 32, 64, 128, 512, 994]:
    x = X[:M]
    t_m = tm(lambda: g(x))
    t_c = tm(lambda: [g(x[i:i+16]) for i in range(0, M, 16)])
    t_p = tm(lambda: matmul_persistent(x, wt))
    t_b = tm(lambda: torch.nn.functional.linear(x, wt.t()))
    print(f"M={M:4d}  marlin {t_m:.3f}ms  marlin16chunks {t_c:.3f}ms  persistent(bf16) {t_p:.3f}ms  cublas(bf16) {t_b:.3f}ms")
