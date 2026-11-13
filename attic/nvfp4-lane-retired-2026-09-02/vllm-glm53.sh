#!/usr/bin/env bash
# ============================================================================
#  GLM-5.3-Flash NVFP4 2-box vLLM serving control (양쪽 박스에서 같은 스크립트)
#  vllm-dspark-lv0731.sh 구조 상속: RoCE GID 자동탐지 / memwatch / STOPALL·RESTARTALL
#  DSV4·Qwen 레인과 컨테이너·이미지·디렉터리가 전부 달라 상호 무영향.
#
#    10.0.1.1 = MASTER (rank0, :8000)   10.0.1.2 = SLAVE (rank1, headless)
#    Commands: START | STOP | RESTART | RESTARTALL | STOPALL | STATUS | LOGS
#
#  ⚠ 2노드는 항상 RESTARTALL/STOPALL. 마스터 없이 Up 인 슬레이브는 좀비다.
# ============================================================================
set -uo pipefail

IMG="${IMG:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"  # sm_121 day-0 버그 7건 패치됨
SERVE="${SERVE:-$HOME/work/dspark-serve/serve-glm53.sh}"
CONTAINER=vllm_glm53
OTHER_CONTAINERS="vllm_dspark vllm_ds4 vllm_qwen38next"   # GPU 회수 (컨테이너만, 파일 무영향)
MODEL_DIR="${MODEL_DIR:-/home/user/Models/RedHatAI--GLM-5.3-Flash-NVFP4}"
#   vcruz305 원본이 아니라 conv1d-split 판을 쓴다: vcruz305 는 q/k/v conv1d 를 하나로
#   융합해놔서 vLLM 로더가 KeyError 로 죽는다(공식 zai-org 는 3개 분리). 융합본이
#   정확히 [q|k|v] 연결임을 공식 체크포인트와 비트단위 대조로 확인(maxdiff 0.000000)한 뒤
#   34개를 102개로 되돌린 하드링크 사본이다. 나머지는 원본과 동일.
#   vcruz305 를 고른 이유: BF16 원본에서 직접 뜬 NVFP4(대부분은 FP8 에서 뜬 이중
#   양자화), MTP(layer45) 를 BF16 으로 유지(텐서 889개), 그리고 per-projection
#   input_scale 36,288 개가 실재한다. 레시피들이 쓰는 LibertAIDAI/RedHatAI 와 다르다.
DRAFT_DIR="${DRAFT_DIR:-/home/user/Models/incoai--GLM-5.3-Flash-DFlash2}"  # SPEC_MODE=dflash 일 때만
CACHE_DIR=/home/user/work/dspark-serve/cache-glm53
IFACE=enp1s0f0np0
MASTER_IP=10.0.1.1; SLAVE_IP=10.0.1.2; SLAVE_SSH=10.0.1.2
SELF="$HOME/work/dspark-serve/vllm-glm53.sh"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8"
PIDFILE=/tmp/vllm_glm53_memwatch.pid
# [2026-11-13] 평문 키 제거 — 이 리포는 공개(github.com/AmesianX/dspark-serve).
# 은퇴한 스크립트라 자동 조회도 하지 않는다. 쓸 일이 생기면 환경변수로 줄 것.
API_KEY="${API_KEY:?API_KEY 를 환경변수로 줄 것 (glm53-exl3-lane/glm53-serve.env 의 VLLM_API_KEY)}"

# --- serve knobs ------------------------------------------------------------
SPEC_MODE="${SPEC_MODE:-mtp}"                    # mtp | dflash | off
SPEC_TOKENS="${SPEC_TOKENS:-4}"                  # SPEC_MODE=mtp 일 때. dflash 는 7 고정.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # LibertAI 실배포값. 262144 는 우리 체크포인트에선 못 버틴다(아래).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"  # 포럼: 8192 는 드라이버 OOM. 4096 이 상한.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
GPU_MEM="${GPU_MEM:-0.80}"
#   ── 2026-09-01 실측 + LibertAI 기록으로 확정 ──
#   우리 가중치는 90.86 GiB/노드(엔진 실측). LibertAI 가 같은 모델에서 잰 91.03 과
#   사실상 동일하므로 그들의 메모리 지도가 그대로 적용된다:
#     0.88 -> 박스 전체 OOM-kill (device 107.09 + vllm anon-rss 11.86 + OS 3 = 122 > 121.69)
#     0.82 -> host 프로세스가 15.7 GiB 로 커지며 한 번 더 OOM
#     0.79 -> 바닥 (weights 91.03 + peak activation 4.9 = 95.93 이 들어가야 함)
#     0.80 -> device 97.35, KV ~1.4 GiB, host 여유 ~24 GiB   ← 실배포값
#   우리 실측(2026-09-01): 0.85/262144 는 가중치 로드는 되지만 프로파일링 구간에서
#   MemAvailable 2GB 로 떨어져 memwatch 가 선제 stop. tonyd2wild 의 0.85/262144 는
#   RedHatAI 체크포인트 기준이라 그대로 이식하면 안 된다.
#   ⚠ GB10 은 통합메모리라 vLLM 호스트 프로세스 3개(APIServer/EngineCore/Worker)의
#   RSS 가 이 분수 "밖에서" 산다. 엔진이 찍는 "Available KV cache memory" 실측만 믿을 것.
BLOCK_SIZE="${BLOCK_SIZE:-2304}"                 # kpool(64)·MLA정렬(128) 공배수. 틀리면 조용히 망가짐.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"          # 비우면 auto→fp8_ds_mla→pe_dim==64 assert 로 사망
KV_BYTES="${KV_BYTES-3221225472}"                # ⚠ ':-' 가 아니라 '-' 다. 빈 값(KV_BYTES=)을 넘기면
                                                 # 진짜로 고정 해제된다. ':-' 였을 때 빈 값이 기본값으로
                                                 # 되돌아가 "고정 안 함"이 조용히 무시됐다(2026-09-02).
                                                 # 고정하면 프로파일링을 건너뛰어 기동이 안전하고,
                                                 # 해제하면 vLLM 이 남은 예산만큼 KV 를 잡는다
                                                 # (tonyd2wild 권고: "Never pin --kv-cache-memory").
THINKING="${THINKING:-false}"
EFFORT="${EFFORT:-high}"                          # 생각 노력: high 권장(미지정=Max=폭주)                    # tonyd2wild 기본값: 생각 off
EAGER="${EAGER:-1}"                              # 0 = CUDA graph 사용(속도 레버, 기동 위험 증가)
ASYNC_SCHED="${ASYNC_SCHED:-0}"                  # 1 = --async-scheduling
MM="${MM:-off}"                                  # off = 비전 프로파일링·입력 차단(기동 OOM 회피)
MOE_BACKEND="${MOE_BACKEND:-marlin}"
#   ── 2026-09-02 실측으로 marlin 확정 ──
#   flashinfer_cutlass 는 이 이미지에서 아예 빌드가 안 된다:
#     "fatal error: nvrtc.h: No such file or directory"
#     (flashinfer 0.6.18 의 fused_moe_120 CUTLASS JIT, 97개 타깃 중 1번째에서 실패)
#   포럼 보고로도 이 모델에서 FLASHINFER_CUTLASS 는 출력이 깨진다("corruption").
#   marlin 은 이 JIT 경로를 타지 않는다. 대신 FP4->FP16 repack 으로 전문가 메모리를
#   더 쓰는데, tonyd2wild 가 swap 을 필수라고 못박은 이유가 바로 그 repack 이다
#   ("UVM driver livelock during MoE repacking"). swap 이 켜져 있어야 안전하다.
WATCH_MIN_GB="${WATCH_MIN_GB:-3}"
IB_HCA="${IB_HCA:-mlx5_0}"

# --- role -------------------------------------------------------------------
ips="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
if   grep -qx "$MASTER_IP" <<<"$ips"; then ROLE=MASTER; RANK=0; HEADLESS=""; HOST_IP=$MASTER_IP
elif grep -qx "$SLAVE_IP"  <<<"$ips"; then ROLE=SLAVE;  RANK=1; HEADLESS=1; HOST_IP=$SLAVE_IP
else ROLE=UNKNOWN; fi

# --- RoCE GID 자동 탐지 (2026-08-04 사고 대응) --------------------------------
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

# --- 페이지 캐시 비우기 (GB10 통합메모리, 레시피가 "every time" 이라고 명시) ----
drop_caches() { sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
                && echo "[$ROLE] page cache dropped" || echo "[$ROLE] WARN: drop_caches 실패(sudo?)"; }

# --- swap 확인 --------------------------------------------------------------
# 레시피 기록: swap 을 완전히 끄면 MoE repack 중 UVM 드라이버가 livelock 된다.
# swappiness=0 으로 두되 swap 자체는 살려둘 것.
check_swap() {
    local s; s=$(awk '/SwapTotal/{print int($2/1024/1024)}' /proc/meminfo)
    if [ "${s:-0}" -eq 0 ]; then
        echo "[$ROLE] ⚠ swap 이 0 입니다. 레시피는 swap 을 켜두라고 기록합니다"
        echo "         (완전 비활성 시 MoE repack 중 UVM 드라이버 livelock)."
    else
        echo "[$ROLE] swap ${s}GB, swappiness=$(cat /proc/sys/vm/swappiness)"
    fi
}

# --- memory watchdog --------------------------------------------------------
start_watch() {
    stop_watch
    # ⚠ swap 인지 워치독. MemAvailable 만 보면 안 된다 — 2026-09-01 실측:
    # swap 32GB 를 켰는데도 SwapFree 30GB 를 남긴 채 MemAvailable 3.7GiB 에서
    # 이 워치독이 컨테이너를 내려버렸다. 커널은 swap 으로 넘길 여유가 있었는데
    # 우리 안전장치가 병목이 된 것. 진짜 OOM 은 (MemAvailable + SwapFree) 가
    # 바닥날 때 일어나므로 그 합으로 판정한다.
    nohup bash -c '
        min='"$WATCH_MIN_GB"'; cont='"$CONTAINER"'
        while true; do
            read -r avail swapfree < <(awk "/MemAvailable/{a=\$2}/SwapFree/{s=\$2}END{print int(a/1048576), int(s/1048576)}" /proc/meminfo)
            head=$(( ${avail:-99} + ${swapfree:-0} ))
            if [ "$head" -lt "$min" ]; then
                echo "$(date +%T) headroom ${head}GB (avail ${avail} + swapfree ${swapfree}) < ${min}GB -> docker stop -t 0 $cont (OOM pre-empt)" >> /tmp/vllm_glm53_memwatch.log
                docker stop -t 0 "$cont" >> /tmp/vllm_glm53_memwatch.log 2>&1
                sleep 5
            fi
            sleep 0.2
        done' >> /tmp/vllm_glm53_memwatch.log 2>&1 &
    echo $! > "$PIDFILE"; disown
    echo "[$ROLE] memwatch on (MemAvailable+SwapFree >= ${WATCH_MIN_GB}GB)"
}
stop_watch() { [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; }; }

# --- lifecycle --------------------------------------------------------------
start_local() {
    [ "$ROLE" = UNKNOWN ] && { echo "ERR: role unknown (iface $IFACE not on $MASTER_IP/$SLAVE_IP)"; exit 1; }
    if docker ps --filter name="$CONTAINER" --format '{{.Status}}' | grep -q Up; then
        echo "[$ROLE] START: already running (use RESTART)"; return 0
    fi
    if [ "$SPEC_MODE" = dflash ] && [ ! -d "$DRAFT_DIR" ]; then
        echo "ERR: SPEC_MODE=dflash 인데 드래프터가 없습니다: $DRAFT_DIR"
        echo "     incoai/GLM-5.3-Flash-DFlash2 는 게이트 모델(수동 승인)입니다."; exit 1
    fi
    mkdir -p "$CACHE_DIR"
    require_gid
    check_swap
    for c in $OTHER_CONTAINERS; do docker rm -f "$c" >/dev/null 2>&1 || true; done
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    drop_caches
    start_watch
    docker run -d --name "$CONTAINER" \
        --restart on-failure:3 \
        --network host --ipc host --shm-size 64g --privileged \
        --ulimit memlock=-1 --gpus all --cap-add SYS_NICE \
        --device /dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
        -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/cache -v "$SERVE":/serve.sh:ro \
        -v "$HOME/work/dspark-serve/glm53-patches":/patches:ro \
        -v "$HOME/work/dspark-serve/glm53-patches/sparse_attn_indexer_kpool_sm121.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
        -e THINKING="$THINKING" \
        -e EFFORT="$EFFORT" \
        ${SPEC_MODE:+$([ "$SPEC_MODE" = dflash ] && echo "-v $DRAFT_DIR:/draft:ro")} \
        -e MODEL_PATH=/model -e PORT=8000 -e TP_SIZE=2 \
        -e SPEC_MODE="$SPEC_MODE" -e SPEC_TOKENS="$SPEC_TOKENS" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e GPU_MEM="$GPU_MEM" \
        -e BLOCK_SIZE="$BLOCK_SIZE" -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e MOE_BACKEND="$MOE_BACKEND" -e MM="$MM" -e KV_BYTES="$KV_BYTES" -e EAGER="$EAGER" -e ASYNC_SCHED="$ASYNC_SCHED" \
        -e VLLM_API_KEY="$API_KEY" \
        -e NODE_RANK="$RANK" -e MASTER_ADDR="$MASTER_IP" -e VLLM_HOST_IP="$HOST_IP" \
        -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$GID_INDEX" \
        -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \
        ${HEADLESS:+-e HEADLESS=1} \
        --entrypoint bash "$IMG" /serve.sh >/dev/null \
      && echo "[$ROLE] START: $CONTAINER up (rank $RANK, len $MAX_MODEL_LEN, spec $SPEC_MODE)"
}
stop_local() {
    stop_watch
    docker stop -t 5 "$CONTAINER" >/dev/null 2>&1 && echo "[$ROLE] STOP: $CONTAINER stopped" || echo "[$ROLE] STOP: not running"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
status_local() {
    local st w avail
    st="$(docker ps --filter name="$CONTAINER" --format '{{.Status}}')"
    w=down; [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && w="up"
    avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    echo "[$ROLE] glm53: ${st:-(not running)} | memwatch: $w | MemAvailable: ${avail}GB"
    if [ "$ROLE" = MASTER ]; then
        echo -n "[SLAVE] "; $SSH "$SLAVE_SSH" "docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "(unreachable)"
        # /health 로 볼 것 — /v1/models 는 엔진이 죽어도 200 을 준다(레시피 기록)
        if curl -sf -m4 http://localhost:8000/health >/dev/null 2>&1; then
            echo "[API] :8000 HEALTHY"
        else
            echo "[API] :8000 not healthy (loading?)"
        fi
    fi
}
logs_local() { docker logs --tail "${2:-80}" -f "$CONTAINER"; }

require_master() { [ "$ROLE" = MASTER ] || { echo "ERR: run this on MASTER ($MASTER_IP)"; exit 1; }; }
fwd() { echo "IMG='$IMG' SERVE='$SERVE' MODEL_DIR='$MODEL_DIR' DRAFT_DIR='$DRAFT_DIR' SPEC_MODE='$SPEC_MODE' SPEC_TOKENS='$SPEC_TOKENS' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' MAX_NUM_SEQS='$MAX_NUM_SEQS' GPU_MEM='$GPU_MEM' BLOCK_SIZE='$BLOCK_SIZE' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' MOE_BACKEND='$MOE_BACKEND' MM='$MM' KV_BYTES='$KV_BYTES' EAGER='$EAGER' THINKING='$THINKING' EFFORT='$EFFORT' ASYNC_SCHED='$ASYNC_SCHED' WATCH_MIN_GB='$WATCH_MIN_GB' IB_HCA='$IB_HCA' API_KEY='$API_KEY'"; }

case "${1:-STATUS}" in
    START)   start_local ;;
    STOP)    stop_local ;;
    RESTART) stop_local; sleep 2; start_local ;;
    STATUS)  status_local ;;
    LOGS)    logs_local "$@" ;;
    STOPALL)
        require_master
        echo "== GLM-5.3 STOPALL =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local ;;
    RESTARTALL)
        require_master
        echo "== GLM-5.3 RESTARTALL: stop both =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF STOP" || echo "[WARN] slave STOP failed"
        stop_local
        echo "== start slave(.67), 25s 후 master(.66) =="
        $SSH "$SLAVE_SSH" "$(fwd) bash $SELF START" || echo "[WARN] slave START failed"
        sleep 25          # 레시피 명시: 워커 먼저, 약 25초 뒤 헤드
        start_local
        echo "== done. STATUS 로 확인. 콜드스타트 ~15분(샤드 로드 지배) ==" ;;
    *)
        echo "GLM-5.3-Flash vLLM control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | RESTARTALL | STOPALL | STATUS | LOGS"
        exit 1 ;;
esac
