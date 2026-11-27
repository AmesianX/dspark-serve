"""[ours] Batch-invariant small-N linear: y = x @ w.T with a launch config that
depends only on (N, K, dtype), never on M. Each program owns BLOCK_M rows x
BLOCK_N cols and walks the full K in fixed BLOCK_K steps (no split-K), so a
row's result is the same whatever M is and wherever the row sits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bi_linear_kernel(x_ptr, w_ptr, y_ptr, M, N, K,
                      sxm, sxk, swn, swk, sym, syn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :] * sxk,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(w_ptr + rn[None, :] * swn + kk[:, None] * swk,
                    mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn,
             acc.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def bi_linear(x: torch.Tensor, w: torch.Tensor, out_dtype=None, cfg=None) -> torch.Tensor:
    """x [M, K], w [N, K] -> [M, N]. cfg=(BM, BN, BK, warps, stages, prec) must not depend on M."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    if M == 0:
        return y
    if cfg is None:
        cfg = (16, 16, 128, 4, 3, "tf32x3" if x.dtype == torch.float32 else "tf32")
    BM, BN, BK, nw, ns, prec = cfg
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _bi_linear_kernel[grid](x, w, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                            y.stride(0), y.stride(1), BLOCK_M=BM, BLOCK_N=BN,
                            BLOCK_K=BK, PREC=prec, num_warps=nw, num_stages=ns)
    return y
