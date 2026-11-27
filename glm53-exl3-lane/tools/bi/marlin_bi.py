import torch, sys, inspect
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.scalar_type import scalar_types
print(inspect.signature(ops.marlin_gemm))
torch.manual_seed(0)
dev = "cuda"
def run(N, K, Ms, fp32_reduce=True):
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
    _, qw, sc = marlin_quant_fp8_torch(w, 128)
    ws = marlin_make_workspace_new(torch.device(dev))
    X = torch.randn(max(Ms), K, dtype=torch.bfloat16, device=dev)
    def g(x):
        return ops.marlin_gemm(a=x, c=None, b_q_weight=qw, b_bias=None, b_scales=sc, a_scales=None,
            global_scale=None, b_zeros=None, g_idx=None, perm=None, workspace=ws,
            b_q_type=scalar_types.float8_e4m3fn, size_m=x.shape[0], size_n=N, size_k=K,
            use_atomic_add=False, use_fp32_reduce=fp32_reduce)
    ref = g(X[:1])[0]
    # rerun-determinism at fixed M
    a = g(X[:37]); b = g(X[:37]); print(f"  same-M rerun identical: {torch.equal(a,b)}")
    bad = []
    for M in Ms:
        o = g(X[:M])[0]
        if not torch.equal(o, ref): bad.append(M)
    print(f"N={N} K={K} fp32red={fp32_reduce}: row0 differs from M=1 at M in {bad[:40]}{'...' if len(bad)>40 else ''} ({len(bad)}/{len(Ms)})")
Ms = list(range(1, 130)) + [192, 256, 320, 512, 994, 1003, 1024]
for N, K in [(4096, 4096), (12800, 4096), (4096, 6144), (24576, 4096)]:
    run(N, K, Ms)
