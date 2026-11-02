#!/usr/bin/env bash
# ============================================================================
#  Qwen3.8-Flash-Next (FP8) serve command — 컨테이너 안에서 /serve.sh 로 실행됨
#  (serve-dspark.sh 와 같은 패턴: 이미지의 baked CMD 를 안 쓰고 우리가 전부 통제)
#
#  근거 (2026-09-01 조사):
#   - vLLM 공식 레시피      https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
#   - 2x DGX Spark 실전배포  https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
#                           https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe
#
#  이 모델은 DSV4 와 공통점이 거의 없다. serve-dspark.sh 를 복사하면 안 되는 이유:
#   48층 중 36층 Gated DeltaNet(선형 어텐션) + 12층 full attention(QSA 인덱서),
#   512 전문가 MoE, 별도 51B n-gram PLE 테이블, MLA 아님.
# ============================================================================
set -euo pipefail

# persistent caches on the host-mounted /cache (JIT/triton/inductor survive restarts)
export XDG_CACHE_HOME=/cache/xdg
export TRITON_CACHE_DIR=/cache/triton-cache-rank${NODE_RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache-rank${NODE_RANK:-0}
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[qwen38next] NODE_RANK=${NODE_RANK} mtp=${SPEC_TOKENS} GID=${NCCL_IB_GID_INDEX:-?}" \
     "model_len=${MAX_MODEL_LEN} gpu_mem=${GPU_MEM} seqs=${MAX_NUM_SEQS} eager=${EAGER:-0}"

# --- CUDA graph / compile ----------------------------------------------------
# torch.compile 은 양쪽 레시피 모두 끈다(mode 0). CUDA graph 는 보고가 갈린다:
#   MiaAI-Lab  : FULL_DECODE_ONLY 로 dual spark 검증, decode 55.8 t/s
#   x00byte    : --enforce-eager 아니면 노드 락업 경험 → eager 고정 권고
# 기본은 측정치가 있는 FULL_DECODE_ONLY. 락업이 보이면 EAGER=1 로 즉시 회피.
if [ "${EAGER:-0}" = "1" ]; then
  COMPILE_ARGS="--enforce-eager"
else
  COMPILE_ARGS="--compilation-config {\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}"
fi

# --- YaRN (262144 초과일 때만) ------------------------------------------------
# 네이티브 262144 이하에서 YaRN 을 켜면 정확도만 깎인다. 길이로 자동 판단.
if [ "${MAX_MODEL_LEN}" -gt 262144 ]; then
  YARN_ARGS="--hf-overrides {\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":262144}}"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
else
  YARN_ARGS=""
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port "${PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --distributed-executor-backend mp \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --kv-cache-dtype auto \
  --load-format safetensors --safetensors-load-strategy lazy \
  --enable-chunked-prefill --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER}" --reasoning-parser qwen3 \
  ${COMPILE_ARGS} \
  ${YARN_ARGS} \
  --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port 25000 \
  ${HEADLESS:+--headless} \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}"

# ---------------------------------------------------------------------------
# 플래그별 근거 (지우지 말 것)
#
# --enable-expert-parallel   FP8 체크포인트에 필수. weight_block_size=[128,128] 인데
#                            moe_intermediate_size=640 이라 TP2 로 쪼개면 320,
#                            320 % 128 != 0 → 평범한 TP 는 깨진다. 공식 문서의
#                            "Plain TP8 is incompatible with the FP8 checkpoint;
#                            use TEP8" 과 같은 이유. EP 는 전문가를 통째로 나눠
#                            갖기 때문에 640 이 그대로 유지된다.
#                            (shared_expert 는 modules_to_not_convert = BF16, 무관)
# --all2all-backend          노드 간 EP. NVLink 없는 2박스 RoCE 구성의 선택지.
#   allgather_reducescatter
# --kv-cache-dtype auto      이 체크포인트는 bf16 KV 만 지원. fp8 KV 주면 깨진다.
#                            (DSV4 는 fp8 KV 였음 — 그대로 가져오면 안 됨)
# --no-enable-flashinfer-    공식 레시피 명시. autotune 켜면 기동에서 문제.
#   autotune
# --safetensors-load-strategy 172.76 GiB 로드 중 통합메모리 고갈 방지.
#   lazy                      GB10 은 host RAM == GPU RAM 이라 로드 피크가 곧 위험.
# --speculative-config mtp   config.json 에 mtp 1층 내장. num_speculative_tokens 는
#                            3 권장 — 같은 MTP 층을 여러 번 forward 하는 구조라
#                            토큰 수를 올릴수록 수용률이 떨어지고, QSA 백엔드가
#                            multi-step draft decode 를 융합하지 못한다.
# PLE 패치 불필요             공개 레시피들이 필수라던 ple_layer.py 패치는 NVFP4
#                            체크포인트 전용 문제다(PLE 만 FP8 이라 vLLM 이
#                            BF16 102GB 로 만들다 크래시). 우리 FP8 체크포인트는
#                            전체가 FP8 직렬화(F8_E4M3 162.5 GiB + BF16 10.2 GiB,
#                            PLE 테이블 47.7 GiB 도 F8_E4M3)라 해당 없음.
# VLLM_PLE_CPU_OFFLOAD       쓰지 않는다. GB10 은 통합메모리라 CPU 로 내려도 같은
#                            128GB 를 먹는다. 공개 레시피도 이 하드웨어에선 끄라고 함.
# ---------------------------------------------------------------------------
