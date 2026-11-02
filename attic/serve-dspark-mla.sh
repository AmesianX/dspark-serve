#!/usr/bin/env bash
# DSpark serve command — mounted into the container and run via `--entrypoint bash /serve.sh`
# (mirrors dsv4-serve/serve.sh pattern: we do NOT rely on the image's baked CMD, so WE
# fully control --max-model-len etc.). Body is the production-3.7 image CMD, with:
#   - max-model-len 393216 -> 524288 (512K; 1M은 프리필 워크스페이스 고수위로 박스 OOM, 2026-08-25)
#   - max-num-batched-tokens -> 16384 (env MAX_NUM_BATCHED_TOKENS)
#   - RoCE GID: honored from env (NCCL_IB_HCA=mlx5_0 / NCCL_IB_GID_INDEX=3 passed by
#     vllm-dspark.sh); image's rocep* probe is skipped since we override the CMD.
#   - JIT/compile caches -> persistent /cache (else /tmp = ephemeral, slow restarts)
set -euo pipefail

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_PCIE_ALLREDUCE_BACKEND=cpp

# persistent caches on the host-mounted /cache (JIT/triton/inductor survive restarts)
export XDG_CACHE_HOME=/cache/xdg
export TRITON_CACHE_DIR=/cache/triton-cache-rank${NODE_RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache-rank${NODE_RANK:-0}
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[dspark] NODE_RANK=${NODE_RANK} SPEC=dspark/${SPEC_TOKENS} KV=${KV_DTYPE:-fp8} seqs=${MAX_NUM_SEQS} cap=${GRAPH_CAP} GID=${NCCL_IB_GID_INDEX:-?} model_len=${MAX_MODEL_LEN:-524288} thinking=${THINKING:-false}"
if [ "${ASYNC_SCHED:-1}" = "1" ]; then ASYNC_ARG="--async-scheduling"; else ASYNC_ARG=""; fi
# thinking ON. effort는 'max'일 때만 프리픽스가 주입되고(encoding_dsv4.py:262) low/medium/high는
# 렌더 결과가 완전히 동일하다 — 즉 max만 피하면 된다(이미지 기본 max = 답 앞 리즈닝 과다).
# REASONING_EFFORT=low|medium|high|max, THINKING=false로 끌 수도
if [ "${THINKING:-true}" = "true" ]; then
  THINK_ARGS="--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=${REASONING_EFFORT:-low}"
else
  THINK_ARGS="--default-chat-template-kwargs.thinking=false"
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name DSV4-Flash \
  --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --api-key "${VLLM_API_KEY}" \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" --block-size 256 --load-format auto \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN:-524288}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${GRAPH_CAP}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  ${ASYNC_ARG} --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --enable-auto-tool-choice \
  ${THINK_ARGS} \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --moe-backend b12x \
  --disable-custom-all-reduce \
  --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port 25000 \
  ${HEADLESS:+--headless} \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}"
