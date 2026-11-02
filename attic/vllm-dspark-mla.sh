#!/usr/bin/env bash
# ============================================================================
#  DSpark 2-box vLLM serving control  (run the SAME script on both boxes)
#  — mirrors dsv4-serve/vllm.sh structure: orchestrator mounts serve-dspark.sh
#    and runs it via `--entrypoint bash /serve.sh`, so WE control --max-model-len
#    (1M) instead of the image's baked 384K CMD.
#
#  기존 MTP(vllm_ds4 / gb10-local / dsv4-serve)는 완전 무영향: 별도 컨테이너
#  (vllm_dspark), 별도 이미지(production-3.7), 별도 디렉토리.
#
#  Role auto-detected from RoCE IP:
#    10.0.1.1 = MASTER (rank0, serves :8000)   10.0.1.2 = SLAVE (rank1, headless)
#
#  Commands: START | STOP | RESTART | RESTARTALL | STOPALL | STATUS
# ============================================================================
set -uo pipefail

IMG="${IMG:-aidendle94/sparkrun-vllm-ds4-gb10:production-3.7}"   # dspark + b12x 내장
SERVE="${SERVE:-$HOME/work/dspark-serve/serve-dspark-mla.sh}"
CONTAINER=vllm_dspark                 # MTP(vllm_ds4)와 다른 이름
MTP_CONTAINER=vllm_ds4                # GPU 회수용 (컨테이너만 stop, dsv4-serve 파일 무영향)
MODEL_DIR="${MODEL_DIR:-/home/user/Models/DeepSeek-V4-Flash-DSpark-Abliterated}"  # 정착: 거부제거(abliterated). env로 override 가능. fwd()로 슬레이브 자동전파
CACHE_DIR=/home/user/work/dspark-serve/cache
IFACE=enp1s0f0np0
MASTER_IP=10.0.1.1; SLAVE_IP=10.0.1.2; SLAVE_SSH=10.0.1.2
SELF="$HOME/work/dspark-serve/vllm-dspark-mla.sh"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8"
PIDFILE=/tmp/vllm_dspark_memwatch.pid
API_KEY="${API_KEY:?API 키를 환경변수로 지정하세요}"
# --- serve knobs (env-overridable) -----------------------------------------
SPEC_TOKENS="${SPEC_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"             # 512K (1M은 프리필 고수위로 OOM, 2026-08-25)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEM="${GPU_MEM:-0.85}"                           # 0.82: OOM 안 남(실측). KV 1.05x/1M(0.85=1.38x), 대신 가용메모리 +3.7GB(0.85=8.4GB→0.82=12.1GB). 업무서버 안정 우선
B12X_MICRO_MAC="${B12X_MICRO_MAX_ACTIVE_CLUSTERS:-12}" # GB10 48-SM 스윕 실측 최적(2026-07-06): 48클램프 43.32 / 8→46.98 / 12→47.22(+9%) / 16→46.26
B12X_DYNAMIC_MAC="${B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS:-}"
GRAPH_CAP="${GRAPH_CAP:-24}"
KV_DTYPE="${KV_DTYPE:-fp8}"                          # fp8(범용) | fp8_ds_mla(MLA 전용 패킹). 빌드 지원: config/cache.py
ASYNC_SCHED="${ASYNC_SCHED:-1}"
THINKING="${THINKING:-true}"                         # 폭주는 0731 모델 고유 증상이고 현 Abliterated에선 발생 안 함(실측). false=생각 끔
REASONING_EFFORT="${REASONING_EFFORT:-high}"         # THINKING=true일 때만 유효.
                                                     # encoding_dsv4.py:262 / GGUF 템플릿 모두 'max'일 때만 프리픽스를 주입한다
                                                     # → low/medium/high 는 렌더가 전부 동일(무주입)이고 max 만 실제 지시가 붙는다.
                                                     # ⚠️ 그런데 max 는 실사용에 못 쓴다(2026-08-13 실측). 프리픽스가 없는 high
                                                     #    에서도 이미 추론 26,271 토큰 / 7만 자까지 간다. max 는 "shortcuts 금지,
                                                     #    모든 중간단계·기각가설을 빠짐없이 써라"를 지시하므로 여기서 더 늘어나
                                                     #    응답이 실용 범위를 벗어난다. 필요할 때만 요청 단위
                                                     #    chat_template_kwargs 로 주고, 기본값은 high 로 둔다.
WATCH_MIN_GB="${WATCH_MIN_GB:-2}"                    # MemAvailable 바닥; 밑돌면 vLLM만 선제 stop (박스 리부팅 방지)
IB_HCA="${IB_HCA:-mlx5_0}"                           # RoCE 레일 1개. 유저 결정 2026-07-27: 일관성 우선.
                                                     # 4레일 A/B 실측(각 n=8/n=6) — 평균은 4레일이 근소 우세하나 전부 노이즈 범위:
                                                     #   prefill 12.5k  1rail 1680.4 σ15.0  vs  4rail 1690.5 σ83.3(이봉 1610/1770)
                                                     #   decode  6x400  1rail 46.56        vs  4rail 46.94
                                                     #   → 평균 +0.6~0.8%(=노이즈) 사려고 분산 5배·최저값 하락을 감수할 이유 없음.
                                                     # 근본 이유 = 배선 예산: prefill 31k cross-node 트래픽 22.8GB는 ttft의 4.56%뿐
                                                     #   (4레일 이론 상한 -3.4%), decode는 0.65%. prefill은 95% compute-bound.
                                                     # 4레일 전환: IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 ./vllm-dspark.sh RESTARTALL

# --- role -------------------------------------------------------------------
ips="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
if   grep -qx "$MASTER_IP" <<<"$ips"; then ROLE=MASTER; RANK=0; HEADLESS=""; HOST_IP=$MASTER_IP
elif grep -qx "$SLAVE_IP"  <<<"$ips"; then ROLE=SLAVE;  RANK=1; HEADLESS=1; HOST_IP=$SLAVE_IP
else ROLE=UNKNOWN; fi

# --- RoCE GID 자동 탐지 --------------------------------------------------------
# NCCL_IB_GID_INDEX 를 3 으로 박아두면 조용히 깨진다. 커널은 GID 슬롯을 동적으로
# 배정해서, IPv6 주소가 붙었다 빠지면 RoCE v2 항목이 다른 인덱스로 밀린다.
# 2026-08-04 실제 사고: .66 은 [3], .67 은 [4] 였고 [3] 이 빈 슬롯이라
# ibv_modify_qp 가 Invalid argument 로 실패 → 워커 기동 실패 → 도커가 되살림 →
# 69회 재시작 루프. 메모리는 108GB 남아 있었고 워치독은 조용했다(호스트 RAM 문제가
# 아니므로). 그래서 숫자를 박지 않고 이 노드의 IP 로 실측해서 찾는다.
gid_index() {
    local hca="${1%%,*}" ip="$2" base i g t
    base="/sys/class/infiniband/$hca/ports/1"
    [ -d "$base/gids" ] || { echo ""; return; }
    for i in $(seq 0 15); do
        g="$(cat "$base/gids/$i" 2>/dev/null)" || continue
        t="$(cat "$base/gid_attrs/types/$i" 2>/dev/null)"
        # RoCE v2 + IPv4-mapped 주소가 이 노드의 RDMA IP 와 일치하는 슬롯
        [ "$t" = "RoCE v2" ] || continue
        case "$g" in
            *ffff:$(printf '%02x%02x:%02x%02x' ${ip//./ })) echo "$i"; return ;;
        esac
    done
    echo ""
}

# 시작 전 게이트: GID 를 못 찾으면 아예 띄우지 않는다. 띄워봐야 NCCL 에서 죽고
# 재시작 루프로 번지는데, 그 루프는 매번 156GB 를 다시 읽어 박스를 느리게 만든다.
require_gid() {
    GID_INDEX="$(gid_index "$IB_HCA" "$HOST_IP")"
    if [ -n "$GID_INDEX" ]; then
        echo "[$ROLE] RoCE GID index=$GID_INDEX ($IB_HCA, $HOST_IP)"
        return 0
    fi
    # 자가복구: 슬롯이 비어 있으면 IP 를 다시 등록해 GID 를 재생성한다. 커널이
    # 비어 있는 가장 낮은 슬롯에 다시 넣으므로 보통 이걸로 붙는다. 사람을 부르지
    # 않는 이유 = 이 상태로 기동하면 NCCL 이 죽고 도커가 되살리는 루프가 되는데,
    # 그 루프가 매번 156GB 를 다시 읽어 박스를 못 쓰게 만든다(2026-08-04 사고).
    echo "[$ROLE] RoCE GID 없음 → $HOST_IP 재등록 시도"
    sudo ip addr del "$HOST_IP/24" dev "$IFACE" 2>/dev/null
    sudo ip addr add "$HOST_IP/24" dev "$IFACE" 2>/dev/null
    for _ in 1 2 3 4 5; do
        sleep 1
        GID_INDEX="$(gid_index "$IB_HCA" "$HOST_IP")"
        [ -n "$GID_INDEX" ] && { echo "[$ROLE] 복구됨: GID index=$GID_INDEX"; return 0; }
    done
    echo "ERR: $IB_HCA 에서 $HOST_IP 의 RoCE v2 GID 를 복구하지 못했습니다." >&2
    echo "     확인: for i in 0 1 2 3 4; do echo \$i \$(cat /sys/class/infiniband/${IB_HCA%%,*}/ports/1/gids/\$i); done" >&2
    exit 1
}

# --- memory watchdog (박스 OOM 리부팅 방지, dsv4-serve/vllm.sh 포팅) ----------
start_watch() {
    stop_watch
    nohup bash -c '
        min='"$WATCH_MIN_GB"'; cont='"$CONTAINER"'
        while true; do
            avail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
            if [ "${avail:-99}" -lt "$min" ]; then
                echo "$(date +%T) MemAvailable ${avail}GB < ${min}GB -> docker stop -t 0 $cont (OOM pre-empt)" >> /tmp/vllm_dspark_memwatch.log
                docker stop -t 0 "$cont" >> /tmp/vllm_dspark_memwatch.log 2>&1
                sleep 5
            fi
            sleep 0.2
        done' >> /tmp/vllm_dspark_memwatch.log 2>&1 &
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
    start_watch
    # GPU 하나 → 이 노드의 MTP 컨테이너 내려서 GPU 회수 (컨테이너만, 파일 무영향)
    docker rm -f "$MTP_CONTAINER" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER"     >/dev/null 2>&1 || true
    # on-failure:3 — unless-stopped 는 기동 실패까지 영원히 되살려서, 2026-08-04 에
    # GID 문제로 69회 재시작하며 매번 156GB 를 읽어 박스를 먹통으로 만들었다.
    # 정상 크래시는 3회까지 복구하고, 그 이상이면 멈춰서 사람이 로그를 보게 한다.
    docker run -d --name "$CONTAINER" \
        --restart on-failure:3 \
        --network host --ipc host --shm-size 64g --privileged \
        --ulimit memlock=-1 --gpus all \
        --device /dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
        -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/cache -v "$SERVE":/serve.sh:ro \
        -v "$HOME/work/dspark-serve/reasoning-patch/engine_protocol.py":/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/openai/engine/protocol.py:ro \
        -v "$HOME/work/dspark-serve/reasoning-patch/chat_protocol.py":/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py:ro \
        -v "$HOME/work/dspark-serve/parser-patch/deepseekv32_tool_parser.py":/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/deepseekv32_tool_parser.py:ro \
        -e MODEL_PATH=/model -e PORT=8000 -e TP_SIZE=2 -e SPEC_TOKENS="$SPEC_TOKENS" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e GPU_MEM="$GPU_MEM" -e GRAPH_CAP="$GRAPH_CAP" -e ASYNC_SCHED="$ASYNC_SCHED" \
        -e KV_DTYPE="$KV_DTYPE" \
        -e THINKING="$THINKING" -e REASONING_EFFORT="$REASONING_EFFORT" \
        -e VLLM_API_KEY="$API_KEY" \
        -e NODE_RANK="$RANK" -e MASTER_ADDR="$MASTER_IP" -e VLLM_HOST_IP="$HOST_IP" \
        -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$GID_INDEX" \
        -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \
        -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
        -e VLLM_DSPARK_LOCAL_ARGMAX=1 -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
        -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 -e VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1 \
        ${B12X_MICRO_MAC:+-e B12X_MICRO_MAX_ACTIVE_CLUSTERS=$B12X_MICRO_MAC} \
        ${B12X_DYNAMIC_MAC:+-e B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=$B12X_DYNAMIC_MAC} \
        ${HEADLESS:+-e HEADLESS=1} \
        --entrypoint bash "$IMG" /serve.sh >/dev/null \
      && echo "[$ROLE] START: $CONTAINER up (rank $RANK, model_len $MAX_MODEL_LEN, batched $MAX_NUM_BATCHED_TOKENS)"
}
stop_local() {
    stop_watch
    docker stop -t 5 "$CONTAINER" >/dev/null 2>&1 && echo "[$ROLE] STOP: $CONTAINER stopped" || echo "[$ROLE] STOP: not running"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
status_local() {
    local st w avail
    st="$(docker ps --filter name="$CONTAINER" --format '{{.Status}} ({{.Image}})')"
    w=down; [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && w="up"
    avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    echo "[$ROLE] dspark: ${st:-(not running)} | memwatch: $w | MemAvailable: ${avail}GB"
    if [ "$ROLE" = MASTER ]; then
        echo -n "[SLAVE] "; $SSH "$SLAVE_SSH" "docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "(unreachable)"
        model="$(curl -s -m4 -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)"
        [ -n "$model" ] && echo "[API] :8000 READY (model=$model)" || echo "[API] :8000 not ready (loading?)"
    fi
}

require_master() { [ "$ROLE" = MASTER ] || { echo "ERR: run this on MASTER ($MASTER_IP)"; exit 1; }; }
fwd() { echo "IMG='$IMG' SERVE='$SERVE' MODEL_DIR='$MODEL_DIR' SPEC_TOKENS='$SPEC_TOKENS' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' GPU_MEM='$GPU_MEM' WATCH_MIN_GB='$WATCH_MIN_GB' B12X_MICRO_MAX_ACTIVE_CLUSTERS='$B12X_MICRO_MAC' B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS='$B12X_DYNAMIC_MAC' THINKING='$THINKING' REASONING_EFFORT='$REASONING_EFFORT' IB_HCA='$IB_HCA' KV_DTYPE='$KV_DTYPE' MAX_NUM_SEQS='$MAX_NUM_SEQS' GRAPH_CAP='$GRAPH_CAP' ASYNC_SCHED='$ASYNC_SCHED'"; }

case "${1:-STATUS}" in
    START)   start_local ;;
    STOP)    stop_local ;;
    RESTART) stop_local; sleep 2; start_local ;;
    STATUS)  status_local ;;
    STOPALL)
        require_master
        echo "== DSpark STOPALL =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local ;;
    RESTARTALL)
        require_master
        echo "== DSpark RESTARTALL: stop both =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local
        echo "== start slave(.67) then master(.66) =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF START" || echo "[WARN] slave START failed — starting master anyway"
        sleep 3
        start_local
        echo "== done. 'STATUS'로 확인, API는 로드+JIT로 ~5분 뒤 :8000 ready ==" ;;
    *)
        echo "DSpark vLLM control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | RESTARTALL | STOPALL | STATUS"
        exit 1 ;;
esac
