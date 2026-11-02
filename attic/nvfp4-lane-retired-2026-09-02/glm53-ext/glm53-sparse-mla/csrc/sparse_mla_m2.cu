// M2: sparse MLA (NoPE, tail_dim == 0) for sm_120/121 -- tensor cores.
//
// Same structure and same numerics as M1; the two GEMMs now run on bf16 tensor
// cores via mma.sync.m16n8k16. sm_120/121 are consumer/embedded Blackwell: no
// tcgen05, no TMEM, no TMA, no wgmma, so the warp-level mma.sync family is the
// tensor-core path available. (TileLang concedes the same point by disabling
// TMA lowering and warp specialization for this target.)
//
// Layout, per block: one token, 32 heads, 256 threads = 8 warps.
//   GEMM1  S[32h, 32i] = Q[32h, 512d] . KV[32i, 512d]^T
//          M=32 -> 2 m-tiles, N=32 -> 4 n-tiles  =>  8 warp-tiles for 8 warps,
//          each reducing the full K=512 itself, so no cross-warp reduction.
//   GEMM2  acc[32h, 512d] += P[32h, 32i] . KV[32i, 512d]
//          warp w owns m-tile w/4 and 16 of the 64 d-tiles => 64 acc regs/lane.
//
// Shared memory (padded, single KV stage):
//   Q  32 x 520 bf16 = 33280      KV 32 x 520 bf16 = 33280
//   S  32 x  36 f32  =  4608      P  32 x  40 bf16 =  2560
//   + 3 x 32 f32 scratch          => 69632 B, comfortably under the 101376 ceiling.
//
// The kPad = 8 halves (4 words) is load-bearing, not decoration: fragment loads
// address rows (lane>>2) at a stride of 520 halves = 260 words, and 260 % 32 = 4,
// so the 8 rows of a fragment land on 8 distinct bank groups and the 32 lanes
// cover 32 distinct banks. Remove the pad and every fragment load conflicts.

#include <cuda_bf16.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int kHeads = 32;
constexpr int kDim = 512;
constexpr int kBI = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kQS = kDim + 8;     // Q row stride, halves
constexpr int kKS = kDim + 8;     // KV row stride, halves
constexpr int kSS = kBI + 4;      // S row stride, floats
constexpr int kPS = kBI + 8;      // P row stride, halves
constexpr float kLog2e = 1.44269504f;
constexpr float kNegBig = -(1 << 30);

__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  uint32_t r;
  uint16_t l = *reinterpret_cast<const uint16_t*>(&lo);
  uint16_t h = *reinterpret_cast<const uint16_t*>(&hi);
  r = (uint32_t)l | ((uint32_t)h << 16);
  return r;
}

__global__ __launch_bounds__(kThreads) void sparse_mla_m2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ KV,
    const int32_t* __restrict__ Indices,
    __nv_bfloat16* __restrict__ O,
    int topk,
    float sm_scale) {
  extern __shared__ char smem_raw[];
  __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* kv_s = q_s + kHeads * kQS;
  float* s_s = reinterpret_cast<float*>(kv_s + kBI * kKS);
  __nv_bfloat16* p_s = reinterpret_cast<__nv_bfloat16*>(s_s + kHeads * kSS);
  float* alpha = reinterpret_cast<float*>(p_s + kHeads * kPS);
  float* m_state = alpha + kHeads;
  float* sum_state = m_state + kHeads;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token = blockIdx.x;

  const int mt = warp >> 2;          // GEMM1/GEMM2 m-tile: heads [mt*16, +16)
  const int nt = warp & 3;           // GEMM1 n-tile: indices [nt*8, +8)
  const int lr = lane >> 2;          // fragment row within an 8-row group
  const int lc = (lane & 3) * 2;     // fragment column pair

  // ---- stage Q once; it is reused by every index tile ---------------------
  for (int e = tid; e < kHeads * kDim; e += kThreads) {
    const int h = e / kDim, d = e % kDim;
    q_s[h * kQS + d] = Q[(int64_t)token * kHeads * kDim + (int64_t)h * kDim + d];
  }
  if (tid < kHeads) {
    m_state[tid] = kNegBig;
    sum_state[tid] = 0.f;
  }

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  const int32_t* idx_row = Indices + (int64_t)token * topk;
  const int num_tiles = topk / kBI;
  __syncthreads();

  for (int tile = 0; tile < num_tiles; ++tile) {
    // ---- gather the KV tile ----------------------------------------------
    for (int e = tid; e < kBI * kDim; e += kThreads) {
      const int r = e / kDim, d = e % kDim;
      const int32_t raw = idx_row[tile * kBI + r];
      const int64_t row = raw < 0 ? 0 : (int64_t)raw;   // clamp: KV[-1] is OOB
      kv_s[r * kKS + d] = KV[row * kDim + d];
    }
    __syncthreads();

    // ---- GEMM1 ------------------------------------------------------------
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    {
      const __nv_bfloat16* qa = q_s + (mt * 16 + lr) * kQS;
      const __nv_bfloat16* qb = q_s + (mt * 16 + lr + 8) * kQS;
      const __nv_bfloat16* kb = kv_s + (nt * 8 + lr) * kKS;
#pragma unroll 4
      for (int k = 0; k < kDim; k += 16) {
        const uint32_t a0 = ld32(qa + k + lc);
        const uint32_t a1 = ld32(qb + k + lc);
        const uint32_t a2 = ld32(qa + k + lc + 8);
        const uint32_t a3 = ld32(qb + k + lc + 8);
        const uint32_t b0 = ld32(kb + k + lc);
        const uint32_t b1 = ld32(kb + k + lc + 8);
#ifndef SKIP_GEMM1
        mma_m16n8k16(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1);
#endif
      }
    }
    s_s[(mt * 16 + lr) * kSS + nt * 8 + lc] = s0;
    s_s[(mt * 16 + lr) * kSS + nt * 8 + lc + 1] = s1;
    s_s[(mt * 16 + lr + 8) * kSS + nt * 8 + lc] = s2;
    s_s[(mt * 16 + lr + 8) * kSS + nt * 8 + lc + 1] = s3;
    __syncthreads();

    // ---- online softmax, one owner thread per head ------------------------
    // Identical arithmetic to M1: exp2(x*c - m*c), two multiplies, matching the
    // reference (tilelang_kernel.py:405-407) rather than exp2((x-m)*c).
    if (tid < kHeads) {
      const int h = tid;
      const float m_prev = m_state[h];
      float m_cur = m_prev;
      float row[kBI];
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const bool valid = idx_row[tile * kBI + i] >= 0;
        row[i] = valid ? s_s[h * kSS + i] : -INFINITY;
        m_cur = fmaxf(m_cur, row[i]);   // reduce_max(clear=False)
      }
      const float c = sm_scale * kLog2e;
      const float a = exp2f((m_prev - m_cur) * c);
      float ssum = 0.f;
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const float p = isinf(row[i]) ? 0.f : exp2f(row[i] * c - m_cur * c);
        p_s[h * kPS + i] = __float2bfloat16(p);
        ssum += p;
      }
      alpha[h] = a;
      m_state[h] = m_cur;
      sum_state[h] = sum_state[h] * a + ssum;
    }
    __syncthreads();

    // ---- rescale, then GEMM2 ---------------------------------------------
    const float a_lo = alpha[mt * 16 + lr];
    const float a_hi = alpha[mt * 16 + lr + 8];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= a_lo; acc[j][1] *= a_lo;
      acc[j][2] *= a_hi; acc[j][3] *= a_hi;
    }

#ifndef SKIP_GEMM2
    {
      const __nv_bfloat16* pa = p_s + (mt * 16 + lr) * kPS;
      const __nv_bfloat16* pb = p_s + (mt * 16 + lr + 8) * kPS;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int i0 = ks * 16;
        const uint32_t a0 = ld32(pa + i0 + lc);
        const uint32_t a1 = ld32(pb + i0 + lc);
        const uint32_t a2 = ld32(pa + i0 + lc + 8);
        const uint32_t a3 = ld32(pb + i0 + lc + 8);
        // B wants KV[k=index][n=dim] with n contiguous, i.e. the transpose of
        // the smem layout: two halves from adjacent index rows, same column.
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int d = ((warp & 3) * 16 + j) * 8 + lr;
          const uint32_t b0 = pack2(kv_s[(i0 + lc) * kKS + d],
                                    kv_s[(i0 + lc + 1) * kKS + d]);
          const uint32_t b1 = pack2(kv_s[(i0 + lc + 8) * kKS + d],
                                    kv_s[(i0 + lc + 9) * kKS + d]);
          mma_m16n8k16(acc[j][0], acc[j][1], acc[j][2], acc[j][3],
                       a0, a1, a2, a3, b0, b1);
        }
      }
    }
#endif
    __syncthreads();
  }

  // ---- epilogue ----------------------------------------------------------
  // Fully masked token -> sumexp == 0 -> return exactly zero, not 0/0 = NaN.
  const float dlo = sum_state[mt * 16 + lr];
  const float dhi = sum_state[mt * 16 + lr + 8];
  const float ilo = dlo > 0.f ? 1.f / dlo : 0.f;
  const float ihi = dhi > 0.f ? 1.f / dhi : 0.f;
  __nv_bfloat16* o_base = O + (int64_t)token * kHeads * kDim;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = ((warp & 3) * 16 + j) * 8 + lc;
    __nv_bfloat16* olo = o_base + (int64_t)(mt * 16 + lr) * kDim;
    __nv_bfloat16* ohi = o_base + (int64_t)(mt * 16 + lr + 8) * kDim;
    olo[d] = __float2bfloat16(acc[j][0] * ilo);
    olo[d + 1] = __float2bfloat16(acc[j][1] * ilo);
    ohi[d] = __float2bfloat16(acc[j][2] * ihi);
    ohi[d + 1] = __float2bfloat16(acc[j][3] * ihi);
  }
}

}  // namespace

void sparse_mla_m2(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                   double sm_scale, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda() && out.is_cuda());
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 2 && indices.dim() == 2);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && kv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous());
  TORCH_CHECK(q.size(1) == kHeads && q.size(2) == kDim && kv.size(1) == kDim);

  const int T = q.size(0);
  const int topk = indices.size(1);
  TORCH_CHECK(topk % kBI == 0);
  TORCH_CHECK(indices.size(0) == T && out.sizes() == q.sizes());

  const size_t smem = sizeof(__nv_bfloat16) * (kHeads * kQS + kBI * kKS + kHeads * kPS)
                    + sizeof(float) * (kHeads * kSS + 3 * kHeads);
  TORCH_CHECK(smem <= 101376, "smem ", smem, " exceeds the sm_12x opt-in ceiling");
  cudaFuncSetAttribute(sparse_mla_m2_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  if (T == 0) return;
  sparse_mla_m2_kernel<<<T, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      topk, (float)sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(glm53_sparse_mla_m2, m) {
  m.def("sparse_mla_m2(Tensor q, Tensor kv, Tensor indices, float sm_scale, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(glm53_sparse_mla_m2, CUDA, m) {
  m.impl("sparse_mla_m2", &sparse_mla_m2);
}
