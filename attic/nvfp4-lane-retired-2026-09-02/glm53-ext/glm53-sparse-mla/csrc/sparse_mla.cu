// glm53_sparse_mla -- sparse MLA (NoPE, tail_dim == 0) for sm_120 / sm_121.
//
// Serves GLM-5.3-Flash's MLA geometry, which no vLLM backend supports on
// capability 12.x: kv_lora_rank=512, qk_rope_head_dim=0 (so pe_dim==0),
// v_head_dim=256, index_topk=2048, 32 q-heads per rank at TP=2.
//
// This is the single-pass kernel used for BOTH prefill and decode. That is not a
// simplification: causality is the indexer's job (the reference computes
// max_kv_i = q_i and never reads it), so prefill and decode tokens are
// indistinguishable to the kernel and a mixed batch runs as one grid.
//
// Derived from the M5 milestone -- see README.md for the measured evolution and
// for the two defects this fixes in the TileLang kernel it replaces.
//
//
// With M4 the gather is fully hidden -- ablation at T=2048: gather alone 1.01 ms
// of a 3.27 ms kernel, and the parts now sum to MORE than the whole, so the
// pipeline genuinely overlaps. What is left is compute-bound, and the cost is
// not the mma instructions but feeding them: GEMM1 issued six 32-bit shared
// loads per mma, and GEMM2's B operand -- which needs KV transposed -- was
// assembled by hand from four 16-bit loads per register, eight per mma.
//
// ldmatrix replaces those: one .x4 for the four A registers, one .x2 for the two
// B registers, one .x2.trans for GEMM2's transposed B.
//
// kPad = 8 halves keeps all three conflict-free: rows sit 520 halves = 260 words
// apart and 260 % 32 = 4, so the eight rows an ldmatrix addresses land on eight
// distinct bank groups and the 32 lanes cover 32 banks.
//
// M4 header follows.
//
// After M3 the gather was still 58% of the kernel (1.94 ms of 3.35 ms at
// T=2048) -- and it is a LATENCY problem, not a bandwidth one: the KV address
// depends on an index that is itself a global load, and with only 8 chunks per
// thread there is no ILP to hide it behind.
//
// Shared-memory double buffering is the textbook fix and does not fit: a second
// KV stage costs 33280 B and takes the block to 107392 B against the 101376 B
// sm_12x ceiling. So the pipeline lives in REGISTERS instead -- tile i+1 is
// loaded global->register while tile i computes, then stored register->smem at
// the top of the next iteration. Costs 8 uint4 = 32 registers/lane (M3 used 128
// of 255, so there is room) and zero extra shared memory.
//
// Chunk mapping is deliberately left as M3's strided one (thread t owns chunks
// t, t+256, t+512, ...). Giving each thread 8 consecutive chunks of one row was
// tried first -- it halves the index loads and looks better coalesced -- but it
// puts all 8 threads of a row 128 B apart, i.e. exactly 32 words, so every
// shared-memory store lands on the same four banks and the tile write serialises
// 8 ways. Measured 1.35x SLOWER than M3. The strided mapping also keeps 8
// independent index loads in flight per thread, which is memory-level
// parallelism the single-index version gives up.
//
// M3 header follows.
//
// Ablation at T=2048 said the tensor cores were never the problem: GEMM1 cost
// 0.89 ms and GEMM2 0.60 ms of a 5.72 ms kernel, while gather + softmax alone
// cost 3.94 ms (69%). The gather moved KV two bytes at a time and re-read the
// index from global for every one of the 64 elements a thread copied. Here it
// moves 16 bytes at a time (uint4) with one index load per 8-element chunk, and
// the epilogue packs adjacent d-pairs into 32-bit stores.
//
// M2 header follows.
// sparse MLA (NoPE, tail_dim == 0) for sm_120/121 -- tensor cores.
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

// c10 rather than ATen/cuda/CUDAContext.h: the latter pulls in
// CUDAContextLight.h -> cusparse.h, and the deployed vLLM image ships a CUDA
// toolkit without the cuSPARSE headers, so including it fails the build there.
// All we need is the current stream and the launch check.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

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

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3, const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1,
                                            const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
                                                  const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
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

constexpr int kChunksPerRow = kDim / 8;                        // 64 uint4 per row
constexpr int kPFChunks = kBI * kChunksPerRow / kThreads;      // 8 per thread

__device__ __forceinline__ void load_tile(uint4 (&pf)[kPFChunks],
                                          const __nv_bfloat16* __restrict__ KV,
                                          const int32_t* __restrict__ idx_row,
                                          int tile, int tid) {
#pragma unroll
  for (int i = 0; i < kPFChunks; ++i) {
    const int c = tid + i * kThreads;
    const int32_t raw = idx_row[tile * kBI + c / kChunksPerRow];
    const int64_t row = raw < 0 ? 0 : (int64_t)raw;   // clamp: KV[-1] is OOB
    pf[i] = *reinterpret_cast<const uint4*>(KV + row * kDim + (c % kChunksPerRow) * 8);
  }
}

__global__ __launch_bounds__(kThreads) void sparse_fwd_kernel(
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
  {
    constexpr int kQChunks = kDim / 8;
    const __nv_bfloat16* q_base = Q + (int64_t)token * kHeads * kDim;
#pragma unroll
    for (int c = tid; c < kHeads * kQChunks; c += kThreads) {
      const int h = c / kQChunks, dc = c % kQChunks;
      *reinterpret_cast<uint4*>(q_s + h * kQS + dc * 8) =
          *reinterpret_cast<const uint4*>(q_base + (int64_t)h * kDim + dc * 8);
    }
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

  uint4 pf[kPFChunks];
  load_tile(pf, KV, idx_row, 0, tid);             // prime the pipeline

  for (int tile = 0; tile < num_tiles; ++tile) {
    // ---- gather the KV tile ----------------------------------------------
    // publish the tile prefetched during the previous iteration
    __syncthreads();                     // everyone finished reading kv_s
#pragma unroll
    for (int i = 0; i < kPFChunks; ++i) {
      const int c = tid + i * kThreads;
      *reinterpret_cast<uint4*>(kv_s + (c / kChunksPerRow) * kKS
                                + (c % kChunksPerRow) * 8) = pf[i];
    }
    __syncthreads();

    // issue the next tile's global loads now; GEMM1 + softmax + GEMM2 below hide
    // the index->KV dependent latency.
    if (tile + 1 < num_tiles) load_tile(pf, KV, idx_row, tile + 1, tid);

    // ---- GEMM1 ------------------------------------------------------------
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    {
      // ldmatrix.x4: lane L supplies row (L & 15) of the 16x16 A tile, column
      // block (L >> 4) * 8, which returns r0..r3 in exactly the a0..a3 order
      // mma.m16n8k16 expects (rows 0-7/8-15 x cols 0-7/8-15).
      const __nv_bfloat16* qa = q_s + (mt * 16 + (lane & 15)) * kQS + (lane >> 4) * 8;
      // B is [n = index][k = dim], already the fragment's (row, col) order, so
      // plain non-trans .x2 works: lanes 0-7 give k 0-7 and lanes 8-15 k 8-15
      // for the same eight index rows.
      const __nv_bfloat16* kb = kv_s + (nt * 8 + (lane & 7)) * kKS + ((lane >> 3) & 1) * 8;
#pragma unroll 4
      for (int k = 0; k < kDim; k += 16) {
        uint32_t a0, a1, a2, a3, b0, b1;
        ldmatrix_x4(a0, a1, a2, a3, qa + k);
        ldmatrix_x2(b0, b1, kb + k);
        mma_m16n8k16(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1);
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

    {
      const __nv_bfloat16* pa = p_s + (mt * 16 + (lane & 15)) * kPS + (lane >> 4) * 8;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int i0 = ks * 16;
        uint32_t a0, a1, a2, a3;
        ldmatrix_x4(a0, a1, a2, a3, pa + i0);
        // B wants KV[k = index][n = dim] with n as the fragment row -- the
        // transpose of the [index][dim] smem layout, which is exactly what
        // ldmatrix.trans does. One instruction replaces eight 16-bit loads.
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int d0 = ((warp & 3) * 16 + j) * 8;
          uint32_t b0, b1;
          ldmatrix_x2_trans(b0, b1, kv_s + (i0 + (lane & 15)) * kKS + d0);
          mma_m16n8k16(acc[j][0], acc[j][1], acc[j][2], acc[j][3],
                       a0, a1, a2, a3, b0, b1);
        }
      }
    }
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
    // acc[j][0]/[1] and [2]/[3] are adjacent in d, so pack each pair into one
    // 32-bit store instead of four 16-bit ones.
    *reinterpret_cast<uint32_t*>(olo + d) =
        pack2(__float2bfloat16(acc[j][0] * ilo), __float2bfloat16(acc[j][1] * ilo));
    *reinterpret_cast<uint32_t*>(ohi + d) =
        pack2(__float2bfloat16(acc[j][2] * ihi), __float2bfloat16(acc[j][3] * ihi));
  }
}

}  // namespace

void sparse_fwd(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                   double sm_scale, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda() && out.is_cuda());
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 2 && indices.dim() == 2);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && kv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous());
  // The warp decomposition is fixed to 32 heads/rank: 2 m-tiles x 4 n-tiles =
  // 8 warp-tiles for 8 warps, each reducing the full K itself. 64 heads (TP=1)
  // would need 114944 B of shared memory, over the 101376 B sm_12x ceiling, so
  // this model needs TP >= 2 regardless.
  TORCH_CHECK(q.size(1) == kHeads, "expected ", kHeads,
              " heads per rank (TP=2 for GLM-5.3-Flash), got ", q.size(1));
  TORCH_CHECK(q.size(2) == kDim && kv.size(1) == kDim,
              "expected kv_lora_rank ", kDim, " with qk_rope_head_dim 0 (NoPE)");

  const int T = q.size(0);
  const int topk = indices.size(1);
  TORCH_CHECK(topk % kBI == 0);
  TORCH_CHECK(indices.size(0) == T && out.sizes() == q.sizes());

  const size_t smem = sizeof(__nv_bfloat16) * (kHeads * kQS + kBI * kKS + kHeads * kPS)
                    + sizeof(float) * (kHeads * kSS + 3 * kHeads);
  TORCH_CHECK(smem <= 101376, "smem ", smem, " exceeds the sm_12x opt-in ceiling");
  cudaFuncSetAttribute(sparse_fwd_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  if (T == 0) return;
  sparse_fwd_kernel<<<T, kThreads, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      topk, (float)sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(glm53_sparse_mla, m) {
  m.def("sparse_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(glm53_sparse_mla, CUDA, m) {
  m.impl("sparse_fwd", &sparse_fwd);
}
