#!/usr/bin/env bash
set -euo pipefail

# Solar Open 2 (250B-A15B) Nota NVFP4, TP2 across two DGX Spark (GB10) boxes.
#
# 이미지 = 공식 vllm/vllm-openai:v0.22.0-aarch64 (sm_120+PTX → sm_121 동작)
# 오버레이 = UpstageAI/vllm v0.22.0-solar-open2 커밋의 파이썬 파일(overlay/SOURCE).
#   포크는 v0.22.0 + 1커밋, CUDA/C++ 변경 0 → 이미지 무수정, 파일만 bind-mount.
#
# ★ W4A16 강제: 체크포인트는 W4A4(NVFP4 activations)지만 GB10 디코드는 대역폭
#   바운드라 A4 이득이 없고 활성값 양자화 오차만 남는다. 두 백엔드를 marlin 으로
#   고정하면 input_global_scale 을 버리고 FP4 가중치 × BF16 활성값으로 돈다.
#   - --linear-backend marlin : MarlinNvFp4LinearKernel (apply_fp4_marlin_linear)
#   - --moe-backend marlin    : nvfp4_w4a16_moe_quant_config (a13/a2 scale=None)
#   빼면 sm_121 에서 FlashInfer/CUTLASS W4A4 가 자동 선택된다.
#
# 2노드는 무조건 vllm-solar-open2.sh RESTARTALL (워커 먼저, 헤드는 600초 안에).

NODE_RANK="${1:?usage: launch.sh <0|1>   (0=head/API, 1=worker)}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

LANE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
[[ -f "$LANE/solar-serve.env" ]] && . "$LANE/solar-serve.env"
# LANE_VARIANT=b12x → solar-serve-b12x.env 를 덧씌운다(A/B 용).
[[ -n "${LANE_VARIANT:-}" && -f "$LANE/solar-serve-$LANE_VARIANT.env" ]] && . "$LANE/solar-serve-$LANE_VARIANT.env"

HEAD_RAIL_IP="${HEAD_RAIL_IP:-10.0.1.1}"
WORKER_RAIL_IP="${WORKER_RAIL_IP:-10.0.1.2}"
NCCL_IF="${NCCL_IF:-enp1s0f0np0}"
NCCL_HCA="${NCCL_HCA:-mlx5_0}"
NCCL_SUBNET="${NCCL_SUBNET:-10.0.1.0/24}"
PORT="${PORT:-8000}"
MPORT="${MPORT:-29522}"

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/home/user/Models/nota-ai--Solar-Open2-250B-Nota-NVFP4}"
MODEL_PATH="/models/solar-open2"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-aarch64}"
NAME="${NAME:-vllm_solar}"
SERVED_NAME="${SERVED_NAME:-solar-open2-250b}"
MAX_LEN="${MAX_LEN:-524288}"
MAX_SEQS="${MAX_SEQS:-4}"
GMU="${GMU:-0.85}"
MNBT="${MNBT:-8192}"
KV_DTYPE="${KV_DTYPE:-}"              # 빈값 = auto(BF16). softmax 12층만 KV → 토큰당 48KB
LINEAR_BACKEND="${LINEAR_BACKEND:-marlin}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
EAGER="${EAGER:-0}"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/solar-vllm-cache}"

case "$NODE_RANK" in
  0) HOST_IP="$HEAD_RAIL_IP"; HEADLESS="" ;;
  1) HOST_IP="$WORKER_RAIL_IP"; HEADLESS="--headless" ;;
esac
ip -o addr show | grep -q "$HOST_IP/" || { echo "this host does not own $HOST_IP (rank $NODE_RANK)" >&2; exit 2; }
test -f "$MODEL_HOST_PATH/config.json" || { echo "weights not found at $MODEL_HOST_PATH" >&2; exit 2; }

# RoCE v2 GID 는 하드코딩 금지(08-04 사고) — 이 박스 레일 IP 에 맞는 인덱스를 찾는다
if [[ -z "${NCCL_GID_INDEX:-}" ]]; then
  _want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' ${HOST_IP//./ })
  for _g in "/sys/class/infiniband/$NCCL_HCA/ports/1/gids/"*; do
    [[ "$(cat "$_g" 2>/dev/null)" == "$_want" ]] || continue
    _idx="${_g##*/}"
    [[ "$(cat "/sys/class/infiniband/$NCCL_HCA/ports/1/gid_attrs/types/$_idx" 2>/dev/null)" == *"v2"* ]] || continue
    NCCL_GID_INDEX="$_idx"; break
  done
  [[ -n "${NCCL_GID_INDEX:-}" ]] || { echo "no RoCE v2 GID for $HOST_IP on $NCCL_HCA" >&2; exit 2; }
fi

OVERLAY_VOLS=()
while IFS= read -r -d '' f; do
  rel="${f#"$LANE/overlay/"}"
  [[ "$rel" == "SOURCE" ]] && continue
  OVERLAY_VOLS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
done < <(find "$LANE/overlay" -type f -name '*.py' -print0)

OPT_ARGS=()
# 템플릿 로짓 프로세서. 2026-09-24: 이 프로세서의 apply() 가 디코딩 중
# solar_open2.py:916 (torch.tensor H2D) 에서 영구 정지하는 것을 py-spy 로 확인.
# LOGITS_PROC=0 으로 빼고 원인 격리 가능.
[[ "${LOGITS_PROC:-1}" != "0" ]] && OPT_ARGS+=(--logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor)
[[ -n "$KV_DTYPE" ]] && OPT_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[[ -n "$MNBT" ]] && OPT_ARGS+=(--max-num-batched-tokens "$MNBT")
[[ -n "${KV_BYTES:-}" ]] && OPT_ARGS+=(--kv-cache-memory-bytes "$KV_BYTES")
# 비동기 스케줄링. 2026-09-24: 양 랭크가 gpu_input_batch.py:1042
# (async_copy_ready_event.synchronize) 에서 영구 정지하는 것을 py-spy 로 확인.
# ASYNC_SCHED=0 이면 끈다.
[[ "${ASYNC_SCHED:-1}" == "0" ]] && OPT_ARGS+=(--no-async-scheduling)
[[ "$EAGER" != "0" ]] && OPT_ARGS+=(--enforce-eager)

mkdir -p "$CACHE_HOST_PATH"/jit/{triton,flashinfer,vllm}

# 메모리 궤적 기록(2초 간격). 기동 실패 원인 추적용 — 컨테이너가 사라져도 남는다.
MEMLOG="${MEMLOG:-$LANE/memwatch-rank$NODE_RANK.log}"
[[ -f "$LANE/.memwatch-rank$NODE_RANK.pid" ]] && kill -9 "$(cat "$LANE/.memwatch-rank$NODE_RANK.pid")" 2>/dev/null || true
setsid bash -c '
  while :; do
    printf "%s %s\n" "$(date +%H:%M:%S)" \
      "$(awk "/MemTotal|MemAvailable|MemFree|Cached:/{printf \"%s=%.1fGB \", \$1, \$2/1048576}" /proc/meminfo)"
    sleep 2
  done' > "$MEMLOG" 2>/dev/null < /dev/null &
echo $! > "$LANE/.memwatch-rank$NODE_RANK.pid"
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --oom-score-adj 1000 \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/jit/triton:/root/.triton" \
  -v "$CACHE_HOST_PATH/jit/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_HOST_PATH/jit/vllm:/root/.cache/vllm" \
  "${OVERLAY_VOLS[@]}" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTHONUNBUFFERED=1 \
  ${FI_SAMPLER:+-e VLLM_USE_FLASHINFER_SAMPLER="$FI_SAMPLER"} \
  ${NVFP4_GEMM:+-e VLLM_NVFP4_GEMM_BACKEND="$NVFP4_GEMM"} \
  -e SOLAR_MEM_PROBE="${SOLAR_MEM_PROBE:-0}" -e SOLAR_MEM_RECLAIM="${SOLAR_MEM_RECLAIM:-1}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT_S:-1800}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e VLLM_API_KEY="${VLLM_API_KEY:?VLLM_API_KEY 필수 (solar-serve.env)}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_IB_GID_INDEX="$NCCL_GID_INDEX" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_SUBNET" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF" \
  -e TP_SOCKET_IFNAME="$NCCL_IF" -e MN_IF_NAME="$NCCL_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --entrypoint vllm \
  "$IMAGE" \
    serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 2 \
    --linear-backend "$LINEAR_BACKEND" --moe-backend "$MOE_BACKEND" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
    "${OPT_ARGS[@]}" \
    --default-chat-template-kwargs '{"think_render_option":"preserved"}' \
    --reasoning-parser solar_open2 \
    --tool-call-parser solar_open2 --enable-auto-tool-choice \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_RAIL_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP gid=$NCCL_GID_INDEX linear=$LINEAR_BACKEND moe=$MOE_BACKEND len=$MAX_LEN seqs=$MAX_SEQS"
