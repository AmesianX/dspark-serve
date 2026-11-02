#!/usr/bin/env bash
# ============================================================================
#  Qwen3.8-27B 단일 박스 vLLM 서빙 제어
#  — vllm-dspark-lv0731.sh 구조를 따름: 오케스트레이터가 serve-qwen38.sh 를
#    마운트해 `--entrypoint bash /serve.sh` 로 실행. 이미지의 baked CMD 안 씀.
#
#  기존 DSV4(vllm_dspark / vllm_ds4)와 완전 분리: 별도 컨테이너(vllm_qwen38),
#  별도 포트(8400), 별도 캐시. DSV4 파일·컨테이너 무영향.
#
#  Commands: START | STOP | RESTART | STATUS | LOG [n]
#
#  실험용 변종 (env):
#    MODEL=fp8|nvfp4|gguf        기본 fp8 (공식). nvfp4 = unsloth 혼합정밀도
#    SPEC=none|mtp|dspark        기본 none. dspark 는 DRAFT 지정 필요
#    DRAFT=radixark             dspark draft 선택
#    EAGER=1                     cudagraph 없이 빠른 기동 (첫 검증용)
#  예:  SPEC=mtp ./vllm-qwen38.sh RESTART
#       MODEL=nvfp4 SPEC=none ./vllm-qwen38.sh RESTART
# ============================================================================
set -uo pipefail

IMG="${IMG:-aidendle94/sparkrun-vllm-ds4-gb10:production-3.7}"   # Qwen3_5ForConditionalGeneration / Qwen3_5MTP 지원 확인됨
SERVE="${SERVE:-$HOME/work/dspark-serve/serve-qwen38.sh}"
CONTAINER=vllm_qwen38
CACHE_DIR=$HOME/work/dspark-serve/cache-qwen38
API_KEY='test1234@X'
PORT="${PORT:-8400}"

case "${MODEL:-fp8}" in
  fp8)   MODEL_DIR="$HOME/Models/Qwen--Qwen3.8-27B-FP8" ;;
  nvfp4) MODEL_DIR="$HOME/Models/unsloth--Qwen3.8-27B-NVFP4" ;;
  awq)   MODEL_DIR="$HOME/Models/abihsoro--Qwen3.8-27B-AWQ-INT4" ;;
  *)     echo "MODEL 은 fp8|nvfp4|awq"; exit 1 ;;
esac
case "${DRAFT:-radixark}" in
  radixark) DRAFT_DIR="$HOME/Models/RadixArk--Qwen3.8-27B-DSpark" ;;
  *)        echo "DRAFT 는 radixark"; exit 1 ;;
esac

# --- serve knobs -----------------------------------------------------------
SPEC="${SPEC:-none}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 통합메모리 121G. llama-server(8385)가 18G 를 쓰는 중이면 0.45 이하로.
GPU_MEM="${GPU_MEM:-0.45}"
GRAPH_CAP="${GRAPH_CAP:-24}"
ASYNC_SCHED="${ASYNC_SCHED:-1}"
EAGER="${EAGER:-0}"

start() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  mkdir -p "$CACHE_DIR"
  local dmount=()
  [ "$SPEC" = "dspark" ] && dmount=(-v "$DRAFT_DIR":/draft:ro)
  docker run -d --name "$CONTAINER" \
    --network host --ipc host --shm-size 32g \
    --ulimit memlock=-1 --gpus all --restart no \
    -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/cache -v "$SERVE":/serve.sh:ro \
    "${dmount[@]}" \
    `# 이미지는 DSV4 전용으로 튜닝돼 있다. Qwen3.8(dense + 하이브리드 선형어텐션)에
     # 걸리는 것들을 무력화한다:
     #   V2_MODEL_RUNNER  : mamba_cache_mode='align' 미지원 (AssertionError)
     #   B12X_MOE         : DSV4 MoE 전용, Qwen 은 dense
     #   PREFIX_CACHE_RETENTION_INTERVAL : scheduler_block_size(784)의 배수여야 함 -> 0 으로 해제
     #   DSPARK_*         : DSV4 draft 전용 튜닝값
     #   PCIE_ALLREDUCE / NCCL_*  : 2노드 RoCE 용. 단일 박스라 불필요(장치명도 stale)` \
    -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_AOT_COMPILE=0 \
    -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
    -e VLLM_DSPARK_FP8_LM_HEAD= -e VLLM_DSPARK_FP8_MAIN_PROJ= -e VLLM_DSPARK_KV_QAT= \
    -e VLLM_DSPARK_IMPL= -e VLLM_DSPARK_CONFIDENCE_SCHEDULER= \
    -e VLLM_DSPARK_MARKOV_W2_BF16= -e VLLM_DSPARK_MARKOV_W2_FP8= \
    -e VLLM_DSPARK_REPLICATE_MARKOV_W1= -e VLLM_DSPARK_REPLICATE_MARKOV_W2= \
    -e NCCL_IB_HCA= -e NCCL_SOCKET_IFNAME= -e NCCL_NET= \
    -e MODEL_PATH=/model -e DRAFT_PATH=/draft -e PORT="$PORT" -e VLLM_API_KEY="$API_KEY" \
    -e SPEC="$SPEC" -e SPEC_TOKENS="$SPEC_TOKENS" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    -e GPU_MEM="$GPU_MEM" -e GRAPH_CAP="$GRAPH_CAP" \
    -e ASYNC_SCHED="$ASYNC_SCHED" -e EAGER="$EAGER" -e COMPILE_CFG="${COMPILE_CFG:-}" \
    --entrypoint bash "$IMG" /serve.sh >/dev/null \
  && echo "START: $CONTAINER  model=${MODEL:-fp8} spec=$SPEC port=$PORT gpu_mem=$GPU_MEM"
}

case "${1:-STATUS}" in
  START)   start ;;
  STOP)    docker stop -t 10 "$CONTAINER" >/dev/null 2>&1 && echo "STOP: 완료" || echo "STOP: 안 떠 있음"
           docker rm -f "$CONTAINER" >/dev/null 2>&1 || true ;;
  RESTART) docker stop -t 10 "$CONTAINER" >/dev/null 2>&1; docker rm -f "$CONTAINER" >/dev/null 2>&1; start ;;
  LOG)     docker logs --tail "${2:-60}" "$CONTAINER" 2>&1 ;;
  STATUS)  docker ps -a --filter "name=$CONTAINER" --format '{{.Names}}\t{{.Status}}'
           curl -s -m 3 -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo ;;
  *)       echo "usage: $0 {START|STOP|RESTART|STATUS|LOG [n]}"; exit 1 ;;
esac
