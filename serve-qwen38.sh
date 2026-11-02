#!/usr/bin/env bash
# Qwen3.8-27B serve command — mounted into the container and run via `--entrypoint bash /serve.sh`
# (serve-dspark.sh 패턴 그대로: 이미지의 baked CMD 를 안 쓰고 우리가 인자를 전부 통제한다)
#
# DSV4 와 다른 점 (뺀 것):
#   - MLA/b12x 계열 인자 전부 제외 (--attention-backend, --moe-backend). Qwen3.8 은 dense + 하이브리드 선형어텐션
#   - deepseek 전용 파서 3종 제외 (--tokenizer-mode/--tool-call-parser/--reasoning-parser)
#   - 2노드 인자 제외 (--nnodes/--node-rank/--master-addr). 단일 박스
#   - --kv-cache-dtype fp8 제외: 공식 FP8 체크포인트는 KV 양자화를 안 쓴다(품질 트레이드 금지 방침)
# 유지한 것: 영속 캐시(JIT/triton/inductor), async 스케줄링, chunked prefill, prefix caching
set -euo pipefail

export XDG_CACHE_HOME=/cache/xdg
export TRITON_CACHE_DIR=/cache/triton-cache-qwen38
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache-qwen38
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [ "${ASYNC_SCHED:-1}" = "1" ]; then ASYNC_ARG="--async-scheduling"; else ASYNC_ARG=""; fi
if [ "${EAGER:-0}" = "1" ]; then EAGER_ARG="--enforce-eager"; else EAGER_ARG=""; fi

# 스펙 디코딩: SPEC=none | mtp | dspark
#   mtp    — 체크포인트 안의 MTP 헤드 사용 (공식 FP8 / unsloth NVFP4 둘 다 포함)
#   dspark — 별도 draft 모델 필요 (DRAFT_PATH 로 지정)
SPEC_ARG=""
case "${SPEC:-none}" in
  mtp)    SPEC_ARG="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-3}}" ;;
  dspark) SPEC_ARG="--speculative-config {\"method\":\"dspark\",\"model\":\"${DRAFT_PATH}\",\"num_speculative_tokens\":${SPEC_TOKENS:-3},\"draft_sample_method\":\"probabilistic\"}" ;;
esac

echo "[qwen38] model=${MODEL_PATH} spec=${SPEC:-none}/${SPEC_TOKENS:-3} len=${MAX_MODEL_LEN} gpu_mem=${GPU_MEM} eager=${EAGER:-0}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --api-key "${VLLM_API_KEY}" \
  --load-format auto \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${GRAPH_CAP}" \
  `# DSV4 의 FULL_AND_PIECEWISE + custom_ops=all 은 Qwen3.8(하이브리드 mamba)에서
   # dynamo 추적 실패로 죽는다. COMPILE_CFG 로 덮어쓸 수 있게 두되 기본은 vLLM 기본값.` \
  ${COMPILE_CFG:+--compilation-config "$COMPILE_CFG"} \
  ${ASYNC_ARG} ${EAGER_ARG} \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  ${SPEC_ARG}
