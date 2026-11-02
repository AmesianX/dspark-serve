# [ours] 2026-09-03 GB10 디코드용 스키니 GEMM: out[M,N] = x[M,K] @ w[N,K]^T, M<=64.
# 동기: cuBLAS 가 M=5·N<=320 에서 고르는 커널(split-K 32 wmma / 1블록 fp32 gemmSN)이 DRAM-cold 1MB 읽기에 50~130us.
# 설계: (N/8 x split-K) 그리드로 48 SM 을 채우고 fp32 FMA 로 누적(bf16 입력도 fp32 누적 = cuBLAS bf16 경로와 동일 산술,
#       fp32 입력은 tf32 없이 정확). split-K 부분합 [S,M,N] 을 두 번째 커널이 고정 순서로 합산+캐스트(결정적). launch 2회.
import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_partial_kernel(
    x_ptr, w_ptr, p_ptr, M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_ps, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_per, BLOCK_K):
        kk = k_start + k0 + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk, mask=m_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk, mask=n_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    p_ptrs = p_ptr + pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    tl.store(p_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _skinny_reduce_kernel(p_ptr, out_ptr, M, N, stride_ps, stride_pm, stride_pn, stride_om, stride_on,
                          SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + s * stride_ps + m * stride_pm + n * stride_pn, mask=mask, other=0.0)
    tl.store(out_ptr + m * stride_om + n * stride_on, acc.to(out_ptr.dtype.element_ty), mask=mask)


_SKINNY_MAX_M = 64


def skinny_gemm_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.is_cuda and x.dim() == 2 and w.dim() == 2 and x.shape[1] == w.shape[1]
        and x.shape[0] <= _SKINNY_MAX_M and x.dtype == w.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.shape[1] % 512 == 0
    )


def skinny_gemm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [M,K], w [N,K] (nn.Linear 레이아웃) -> [M,N]. 미지원 형상은 torch 로 폴백."""
    if not skinny_gemm_supported(x, w):
        y = torch.nn.functional.linear(x, w)
        return y.to(out_dtype) if out_dtype is not None else y
    M, K = x.shape
    N = w.shape[0]
    out_dtype = out_dtype or x.dtype
    BLOCK_M = max(8, triton.next_power_of_2(M))
    BLOCK_K = 64 if BLOCK_M <= 16 else (32 if BLOCK_M <= 32 else 16)
    BLOCK_N = 8
    n_blocks = triton.cdiv(N, BLOCK_N)
    split_k = 1
    for cand in (8, 4, 2):
        if n_blocks * cand <= 256 and K % (cand * BLOCK_K) == 0 and n_blocks * cand >= 48:
            split_k = cand
            break
    partial = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    _skinny_partial_kernel[(n_blocks, split_k)](
        x, w, partial, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=split_k, num_warps=4,
    )
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    BLOCK = 1024
    _skinny_reduce_kernel[(triton.cdiv(M * N, BLOCK),)](
        partial, out, M, N, partial.stride(0), partial.stride(1), partial.stride(2), out.stride(0), out.stride(1),
        SPLIT_K=split_k, BLOCK=BLOCK, num_warps=4,
    )
    return out
