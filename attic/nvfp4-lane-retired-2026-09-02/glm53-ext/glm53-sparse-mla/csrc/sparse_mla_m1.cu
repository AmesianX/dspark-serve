// M1: sparse MLA (NoPE, tail_dim == 0) for sm_120/121 -- CORRECTNESS ONLY.
//
// Scalar math, naive shared-memory staging, no cp.async, no tensor cores. The
// point of this stage is to pin down indexing, masking, the online-softmax
// recurrence and the epilogue against the fp32 reference. Tensor cores (M1b)
// and the cp.async pipeline (M2) are layered on top of this exact structure.
//
// Contract (identical to SGLang's sparse_attention_fwd_kernel_v1):
//   Q       [T, H, D]     bf16
//   KV      [R, D]        bf16   -- one MQA head, serves as BOTH K and V
//   Indices [T, TOPK]     int32  -- physical row ids; -1 is the mask sentinel
//   O       [T, H, D]     bf16
//
// Causality is NOT applied: the reference computes max_kv_i = q_i and never
// reads it. The only mask is Indices >= 0. Causality belongs to the indexer,
// which is what keeps the index axis layout-agnostic so vLLM paged slot ids can
// be passed in raw.
//
// Two deliberate divergences from the reference, both bug fixes:
//   * the gather clamps the -1 sentinel to row 0 instead of reading KV[-1],
//     which is an out-of-bounds read 1024 B before the tensor base;
//   * a fully masked token returns exactly zero rather than 0/0 = NaN.

#include <cuda_bf16.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int kHeads = 32;      // 64 q-heads / TP2
constexpr int kDim = 512;       // kv_lora_rank, tail_dim == 0
constexpr int kBI = 32;         // index tile
constexpr int kThreads = 256;   // 8 warps
constexpr int kPad = 8;         // smem row pad, keeps banks apart
constexpr int kDPerThread = kDim * kHeads / kThreads;   // 64
constexpr int kThreadsPerHead = kThreads / kHeads;      // 8
constexpr float kLog2e = 1.44269504f;

// Sentinel matching the reference's m_i init (tilelang_kernel.py:355). Using
// -INFINITY here would make (m_prev - m) = -inf - -inf = NaN on the first tile.
constexpr float kNegBig = -(1 << 30);

__global__ __launch_bounds__(kThreads) void sparse_mla_m1_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ KV,
    const int32_t* __restrict__ Indices,
    __nv_bfloat16* __restrict__ O,
    int topk,
    int64_t kv_rows,
    float sm_scale) {
  extern __shared__ char smem_raw[];
  __nv_bfloat16* kv_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [kBI][kDim + kPad]
  float* s_tile = reinterpret_cast<float*>(kv_tile + kBI * (kDim + kPad));  // [kHeads][kBI]
  float* p_tile = s_tile + kHeads * kBI;                                 // [kHeads][kBI]
  float* alpha = p_tile + kHeads * kBI;                                  // [kHeads]
  float* m_state = alpha + kHeads;                                       // [kHeads]
  float* sum_state = m_state + kHeads;                                   // [kHeads]

  const int tid = threadIdx.x;
  const int token = blockIdx.x;

  // Thread -> (head, dim-slice). 8 threads cover one head's 512 dims, 64 each.
  const int head = tid / kThreadsPerHead;
  const int lane_in_head = tid % kThreadsPerHead;
  const int d0 = lane_in_head * kDPerThread;

  const __nv_bfloat16* q_row = Q + (int64_t)token * kHeads * kDim + (int64_t)head * kDim;
  float q_reg[kDPerThread];
#pragma unroll
  for (int i = 0; i < kDPerThread; ++i) q_reg[i] = __bfloat162float(q_row[d0 + i]);

  float acc_o[kDPerThread];
#pragma unroll
  for (int i = 0; i < kDPerThread; ++i) acc_o[i] = 0.f;

  if (tid < kHeads) {
    m_state[tid] = kNegBig;
    sum_state[tid] = 0.f;
  }
  __syncthreads();

  const int32_t* idx_row = Indices + (int64_t)token * topk;
  const int num_tiles = topk / kBI;

  for (int tile = 0; tile < num_tiles; ++tile) {
    // ---- stage the KV tile (naive: plain global loads) --------------------
    // 32 rows x 512 halves = 16384 elements over 256 threads = 64 each.
    for (int e = tid; e < kBI * kDim; e += kThreads) {
      const int r = e / kDim;
      const int d = e % kDim;
      const int32_t raw = idx_row[tile * kBI + r];
      // Clamp the sentinel: KV[-1] is an OOB read. The column is masked to
      // -inf below, so the value fetched here never reaches the output.
      const int64_t row = raw < 0 ? 0 : (int64_t)raw;
      kv_tile[r * (kDim + kPad) + d] = KV[row * kDim + d];
    }
    __syncthreads();

    // ---- GEMM1: S[h][i] = dot(Q[h], KV[i]) --------------------------------
    // Each thread reduces its own 64 dims for all kBI indices, then the 8
    // threads of a head combine via shuffle (they are contiguous, hence in one
    // warp: 32 lanes = 4 heads x 8).
    for (int i = 0; i < kBI; ++i) {
      float part = 0.f;
      const __nv_bfloat16* kv_row = kv_tile + i * (kDim + kPad) + d0;
#pragma unroll
      for (int e = 0; e < kDPerThread; ++e) part += q_reg[e] * __bfloat162float(kv_row[e]);
#pragma unroll
      for (int off = kThreadsPerHead / 2; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off, kThreadsPerHead);
      if (lane_in_head == 0) s_tile[head * kBI + i] = part;
    }
    __syncthreads();

    // ---- online softmax, one owner thread per head ------------------------
    if (tid < kHeads) {
      const int h = tid;
      float m_prev = m_state[h];
      float m_cur = m_prev;
      float row[kBI];
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const bool valid = idx_row[tile * kBI + i] >= 0;
        row[i] = valid ? s_tile[h * kBI + i] : -INFINITY;
        // Matches reduce_max(clear=False): a fully masked tile leaves m
        // UNCHANGED rather than dragging it to -inf.
        m_cur = fmaxf(m_cur, row[i]);
      }
      // exp2(x*c - m*c), NOT exp2((x-m)*c): the reference emits two multiplies
      // (tilelang_kernel.py:405-407) and matching it avoids a systematic drift.
      const float c = sm_scale * kLog2e;
      const float a = exp2f((m_prev - m_cur) * c);
      float ssum = 0.f;
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const float p = isinf(row[i]) ? 0.f : exp2f(row[i] * c - m_cur * c);
        p_tile[h * kBI + i] = p;
        ssum += p;
      }
      alpha[h] = a;
      m_state[h] = m_cur;
      sum_state[h] = sum_state[h] * a + ssum;
    }
    __syncthreads();

    // ---- GEMM2: acc_o[h][d] = acc_o[h][d] * alpha + sum_i P[h][i] KV[i][d] --
    const float a = alpha[head];
#pragma unroll
    for (int e = 0; e < kDPerThread; ++e) acc_o[e] *= a;
    for (int i = 0; i < kBI; ++i) {
      const float p = p_tile[head * kBI + i];
      if (p == 0.f) continue;
      const __nv_bfloat16* kv_row = kv_tile + i * (kDim + kPad) + d0;
#pragma unroll
      for (int e = 0; e < kDPerThread; ++e) acc_o[e] += p * __bfloat162float(kv_row[e]);
    }
    __syncthreads();
  }

  // ---- epilogue ----------------------------------------------------------
  // A fully masked token has sumexp == 0. Return exactly zero: the reference
  // divides unconditionally and produces NaN, which poisons the residual
  // stream for the rest of the sequence.
  const float denom = sum_state[head];
  const float inv = denom > 0.f ? 1.f / denom : 0.f;
  __nv_bfloat16* o_row = O + (int64_t)token * kHeads * kDim + (int64_t)head * kDim;
#pragma unroll
  for (int e = 0; e < kDPerThread; ++e) o_row[d0 + e] = __float2bfloat16(acc_o[e] * inv);
}

}  // namespace

void sparse_mla_m1(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                   double sm_scale, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda() && out.is_cuda());
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 2 && indices.dim() == 2 && out.dim() == 3);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && kv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous());
  TORCH_CHECK(q.size(1) == kHeads, "M1 is fixed at ", kHeads, " heads");
  TORCH_CHECK(q.size(2) == kDim && kv.size(1) == kDim, "M1 is fixed at dim ", kDim);

  const int T = q.size(0);
  const int topk = indices.size(1);
  TORCH_CHECK(topk % kBI == 0, "topk must be a multiple of ", kBI);
  TORCH_CHECK(indices.size(0) == T && out.sizes() == q.sizes());

  const size_t smem = sizeof(__nv_bfloat16) * kBI * (kDim + kPad)
                    + sizeof(float) * (2 * kHeads * kBI + 3 * kHeads);
  TORCH_CHECK(smem <= 101376, "smem ", smem, " exceeds the sm_12x opt-in ceiling");
  cudaFuncSetAttribute(sparse_mla_m1_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  if (T == 0) return;
  sparse_mla_m1_kernel<<<T, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      topk, kv.size(0), (float)sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(glm53_sparse_mla_m1, m) {
  m.def("sparse_mla_m1(Tensor q, Tensor kv, Tensor indices, float sm_scale, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(glm53_sparse_mla_m1, CUDA, m) {
  m.impl("sparse_mla_m1", &sparse_mla_m1);
}
