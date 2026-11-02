#!/usr/bin/env bash
# ============================================================================
#  Qwen3.8-Flash-Next (FP8) 2-box vLLM serving control
#  — 양쪽 박스에서 같은 스크립트를 돌린다. vllm-dspark-lv0731.sh 의 구조를
#    그대로 물려받되(RoCE GID 자동탐지 / memwatch / STOPALL·RESTARTALL),
#    모델·이미지·컨테이너 이름이 전부 달라서 DSV4 운영에는 무영향.
#
#  Role auto-detected from RoCE IP:
#    10.0.1.1 = MASTER (rank0, serves :8000)   10.0.1.2 = SLAVE (rank1, headless)
#
#  Commands: START | STOP | RESTART | RESTARTALL | STOPALL | STATUS
#
#  ⚠ 2노드는 항상 RESTARTALL/STOPALL 로. 마스터 없이 Up 인 슬레이브는 좀비다.
# ============================================================================
set -uo pipefail

IMG="${IMG:-vllm/vllm-openai:qwen38-flash-next}"   # 공식 이미지. PyPI 설치는 미지원(공식 명시)
SERVE="${SERVE:-$HOME/work/dspark-serve/serve-qwen38next.sh}"
CONTAINER=vllm_qwen38next             # DSV4(vllm_dspark)와 다른 이름
OTHER_CONTAINERS="vllm_dspark vllm_ds4"   # GPU 회수 대상 (컨테이너만 stop, 파일 무영향)
MODEL_DIR="${MODEL_DIR:-/home/user/Models/Qwen--Qwen3.8-Flash-Next-FP8}"
CACHE_DIR=/home/user/work/dspark-serve/cache-qwen38next
IFACE=enp1s0f0np0
MASTER_IP=10.0.1.1; SLAVE_IP=10.0.1.2; SLAVE_SSH=10.0.1.2
SELF="$HOME/work/dspark-serve/vllm-qwen38next.sh"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8"
PIDFILE=/tmp/vllm_qwen38next_memwatch.pid
API_KEY="${API_KEY:?API 키를 환경변수로 지정하세요}"  # 기존 클라이언트 호환

# --- serve knobs (env-overridable) -----------------------------------------
SPEC_TOKENS="${SPEC_TOKENS:-3}"                  # MTP. 3 초과는 수용률 하락(같은 층 반복 forward)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"         # 실측 확정값(2026-09-01). 네이티브는 262144지만 아래 GPU_MEM 참조.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"   # 8192 → 2048. 프리필 워크스페이스가 GPU 예산 밖에서 잡힌다.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"                # KV 4.1 GiB / 131072 = 동시성 1.88x. 4는 과하다.
GPU_MEM="${GPU_MEM:-0.80}"
#   ── 2026-09-01 실측으로 확정. 이 값이 유일한 진짜 레버다. ──
#   FP8 가중치는 88.18 GiB/노드(체크포인트 172.76 GiB / 2). GB10 통합메모리라
#   vLLM 예산(GPU_MEM x 121.69 GiB)을 뺀 나머지가 곧 호스트 여유이고, 프리필
#   워크스페이스와 CUDA 그래프 캡처는 그 예산 "밖에서" 잡힌다.
#
#   실측 3회:
#     0.85 / len 262144 → KV  6.93 GiB, 기동은 되나 12k 프리필에서 MemAvailable
#                          2GB → memwatch 가 선제 stop.
#     0.85 / len 131072 → KV 10.53 GiB. 컨텍스트를 줄였더니 엔진이 남는 공간을
#                          KV 로 더 먹어서 호스트 여유가 오히려 줄었고,
#                          CUDA 그래프 캡처 중에 사망. → MAX_MODEL_LEN 은 레버가 아니다.
#     0.80 / len 131072 → KV  4.10 GiB(246,873 토큰, 동시성 1.88x), 호스트 15GB.
#                          기동 OK, 12k/30k 프리필 통과, decode 42.75 t/s(코드).  ← 채택
#
#   NVFP4(가중치 64.45 GiB/노드, KV 35 GiB)와 달리 FP8 173GB 는 2x110GB 박스에
#   "겨우" 들어가는 크기다. 공개 dual-Spark 레시피가 전부 NVFP4 인 이유가 이것.
#   올릴 생각이 들면 엔진이 찍는 "Available KV cache memory" 실측만 믿을 것 — 외삽 금지.
EAGER="${EAGER:-0}"                              # 1 = --enforce-eager (락업 회피용 탈출구)
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"          # 공식 레시피값. spark 레시피들은 qwen3_coder 사용
WATCH_MIN_GB="${WATCH_MIN_GB:-3}"
IB_HCA="${IB_HCA:-mlx5_0}"                       # 장치명은 mlx5_*. 공개 레시피의 rocep* 는 우리 박스에서 STALE

# --- role -------------------------------------------------------------------
ips="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
if   grep -qx "$MASTER_IP" <<<"$ips"; then ROLE=MASTER; RANK=0; HEADLESS=""; HOST_IP=$MASTER_IP
elif grep -qx "$SLAVE_IP"  <<<"$ips"; then ROLE=SLAVE;  RANK=1; HEADLESS=1; HOST_IP=$SLAVE_IP
else ROLE=UNKNOWN; fi

# --- RoCE GID 자동 탐지 (2026-08-04 사고 대응, vllm-dspark-lv0731.sh 에서 이식) ---
# GID INDEX 를 3 으로 박으면 조용히 깨진다. 커널이 슬롯을 동적 배정하기 때문에
# .66 은 [3], .67 은 [4] 인 상황이 실제로 났고, 빈 슬롯을 가리키면 ibv_modify_qp 가
# 실패 → 워커 기동 실패 → 도커 재시작 루프 → 매번 172GB 재독으로 박스 먹통.
gid_index() {
    local hca="${1%%,*}" ip="$2" base i g t
    base="/sys/class/infiniband/$hca/ports/1"
    [ -d "$base/gids" ] || { echo ""; return; }
    for i in $(seq 0 15); do
        g="$(cat "$base/gids/$i" 2>/dev/null)" || continue
        t="$(cat "$base/gid_attrs/types/$i" 2>/dev/null)"
        [ "$t" = "RoCE v2" ] || continue
        case "$g" in
            *ffff:$(printf '%02x%02x:%02x%02x' ${ip//./ })) echo "$i"; return ;;
        esac
    done
    echo ""
}
require_gid() {
    GID_INDEX="$(gid_index "$IB_HCA" "$HOST_IP")"
    if [ -n "$GID_INDEX" ]; then
        echo "[$ROLE] RoCE GID index=$GID_INDEX ($IB_HCA, $HOST_IP)"; return 0
    fi
    echo "[$ROLE] RoCE GID 없음 → $HOST_IP 재등록 시도"
    sudo ip addr del "$HOST_IP/24" dev "$IFACE" 2>/dev/null
    sudo ip addr add "$HOST_IP/24" dev "$IFACE" 2>/dev/null
    for _ in 1 2 3 4 5; do
        sleep 1
        GID_INDEX="$(gid_index "$IB_HCA" "$HOST_IP")"
        [ -n "$GID_INDEX" ] && { echo "[$ROLE] 복구됨: GID index=$GID_INDEX"; return 0; }
    done
    echo "ERR: $IB_HCA 에서 $HOST_IP 의 RoCE v2 GID 를 복구하지 못했습니다." >&2
    exit 1
}

# --- 페이지 캐시 비우기 (GB10 통합메모리 필수) --------------------------------
# 공개 레시피 양쪽 모두 "mandatory". 안 하면 172GB 로드 중 허위 CUDA OOM 이 난다.
drop_caches() { sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
                && echo "[$ROLE] page cache dropped" || echo "[$ROLE] WARN: drop_caches 실패(sudo?)"; }

# --- memory watchdog (박스 OOM 리부팅 방지) ----------------------------------
start_watch() {
    stop_watch
    nohup bash -c '
        min='"$WATCH_MIN_GB"'; cont='"$CONTAINER"'
        while true; do
            avail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
            if [ "${avail:-99}" -lt "$min" ]; then
                echo "$(date +%T) MemAvailable ${avail}GB < ${min}GB -> docker stop -t 0 $cont (OOM pre-empt)" >> /tmp/vllm_qwen38next_memwatch.log
                docker stop -t 0 "$cont" >> /tmp/vllm_qwen38next_memwatch.log 2>&1
                sleep 5
            fi
            sleep 0.2
        done' >> /tmp/vllm_qwen38next_memwatch.log 2>&1 &
    echo $! > "$PIDFILE"; disown
    echo "[$ROLE] memwatch on (MemAvailable >= ${WATCH_MIN_GB}GB)"
}
stop_watch() { [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; }; }

# --- local container lifecycle ---------------------------------------------
start_local() {
    [ "$ROLE" = UNKNOWN ] && { echo "ERR: role unknown (iface $IFACE not on $MASTER_IP/$SLAVE_IP)"; exit 1; }
    if docker ps --filter name="$CONTAINER" --format '{{.Status}}' | grep -q Up; then
        echo "[$ROLE] START: already running (use RESTART)"; return 0
    fi
    mkdir -p "$CACHE_DIR"
    require_gid
    # GPU 하나 → 다른 서빙 컨테이너 내려서 회수 (컨테이너만, 파일 무영향)
    for c in $OTHER_CONTAINERS; do docker rm -f "$c" >/dev/null 2>&1 || true; done
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    drop_caches
    start_watch
    # on-failure:3 — unless-stopped 는 기동 실패까지 영원히 되살려서 박스를 먹통으로
    # 만든다(2026-08-04). 정상 크래시 3회까지만 복구하고 그 다음은 사람이 로그를 본다.
    docker run -d --name "$CONTAINER" \
        --restart on-failure:3 \
        --network host --ipc host --shm-size 64g --privileged \
        --ulimit memlock=-1 --gpus all --cap-add SYS_NICE \
        --device /dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
        -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/cache -v "$SERVE":/serve.sh:ro \
        -e MODEL_PATH=/model -e PORT=8000 -e TP_SIZE=2 -e SPEC_TOKENS="$SPEC_TOKENS" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e GPU_MEM="$GPU_MEM" -e EAGER="$EAGER" \
        -e TOOL_PARSER="$TOOL_PARSER" \
        -e VLLM_API_KEY="$API_KEY" \
        -e NODE_RANK="$RANK" -e MASTER_ADDR="$MASTER_IP" -e VLLM_HOST_IP="$HOST_IP" \
        -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$GID_INDEX" \
        -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \
        ${HEADLESS:+-e HEADLESS=1} \
        --entrypoint bash "$IMG" /serve.sh >/dev/null \
      && echo "[$ROLE] START: $CONTAINER up (rank $RANK, len $MAX_MODEL_LEN, seqs $MAX_NUM_SEQS, mtp $SPEC_TOKENS)"
}
stop_local() {
    stop_watch
    docker stop -t 5 "$CONTAINER" >/dev/null 2>&1 && echo "[$ROLE] STOP: $CONTAINER stopped" || echo "[$ROLE] STOP: not running"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
status_local() {
    local st w avail model
    st="$(docker ps --filter name="$CONTAINER" --format '{{.Status}} ({{.Image}})')"
    w=down; [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && w="up"
    avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    echo "[$ROLE] qwen38next: ${st:-(not running)} | memwatch: $w | MemAvailable: ${avail}GB"
    if [ "$ROLE" = MASTER ]; then
        echo -n "[SLAVE] "; $SSH "$SLAVE_SSH" "docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "(unreachable)"
        model="$(curl -s -m4 -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)"
        [ -n "$model" ] && echo "[API] :8000 READY (model=$model)" || echo "[API] :8000 not ready (loading?)"
    fi
}
logs_local() { docker logs --tail "${2:-80}" -f "$CONTAINER"; }

require_master() { [ "$ROLE" = MASTER ] || { echo "ERR: run this on MASTER ($MASTER_IP)"; exit 1; }; }
fwd() { echo "IMG='$IMG' SERVE='$SERVE' MODEL_DIR='$MODEL_DIR' SPEC_TOKENS='$SPEC_TOKENS' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' MAX_NUM_SEQS='$MAX_NUM_SEQS' GPU_MEM='$GPU_MEM' EAGER='$EAGER' TOOL_PARSER='$TOOL_PARSER' WATCH_MIN_GB='$WATCH_MIN_GB' IB_HCA='$IB_HCA' API_KEY='$API_KEY'"; }

case "${1:-STATUS}" in
    START)   start_local ;;
    STOP)    stop_local ;;
    RESTART) stop_local; sleep 2; start_local ;;
    STATUS)  status_local ;;
    LOGS)    logs_local "$@" ;;
    STOPALL)
        require_master
        echo "== Qwen3.8-Next STOPALL =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local ;;
    RESTARTALL)
        require_master
        echo "== Qwen3.8-Next RESTARTALL: stop both =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local
        echo "== start slave(.67) then master(.66) =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF START" || echo "[WARN] slave START failed — starting master anyway"
        sleep 3
        start_local
        echo "== done. 'STATUS'로 확인. 콜드스타트 ~11분(가중치 172GB 로드) ==" ;;
    *)
        echo "Qwen3.8-Flash-Next vLLM control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | RESTARTALL | STOPALL | STATUS | LOGS"
        exit 1 ;;
esac
