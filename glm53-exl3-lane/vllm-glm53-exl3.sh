#!/usr/bin/env bash
# ============================================================================
#  GLM-5.3-Flash EXL3 (원본 tr3-4bpw + 런타임 /ab 검열해제 스위치) — 2×DGX Spark 조작 스크립트
#  헤드에서 실행해 양 노드를 제어한다. 사이트 값은 glm53-serve.env ① 블록.
#
#  레인: Entrpi 킷 v2.3-tier1 + 우리 오버레이(dense Marlin W8A16, lm_head 포함)
#    - 런처   : <레인>/launch.sh (양 노드 동일 경로, Entrpi 런처 복사본 + 우리 수정 — launch.sh.diff)
#    - 설정   : <레인>/glm53-serve.env (양 노드) — MTP-4·dense/lm_head/eh_proj Marlin W8A16·프리필 BF16 이중경로·KV 12GB(⛔고정, 병렬 서브에이전트 측정값)
#    - 오버레이: <레인>/overlay/ (양 노드). 프로파일: NSYS=1 ./vllm-glm53-exl3.sh RESTARTALL
#    - 실측   : 디코드 영어 ~35 / 한국어 산문 ~39 / 코드 ~42 t/s, 스텝 ~80ms, 프리필 12k ~1,350 t/s (10-05). 문서 = README.md
#
#  ⚠️ 2노드는 무조건 RESTARTALL. 워커만 살아있으면 랑데부 대기 좀비(10분 날림).
#  ⚠️ 서빙 중 헤드에서 cargo build 등 무거운 작업 금지(여유 6GB, OOM 으로 워커 사망).
#  ⚠️ 같은 박스에 다른 vLLM 서버(우리 박스의 vllm_dspark 등)가 떠 있으면 같이 못 뜬다 — OTHER 목록이 막는다.
# ============================================================================
set -uo pipefail
LANE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# 사이트 값(워커 ssh 주소, API 키)은 env 파일 한 곳에서 읽는다.
GLM53_ENV="${GLM53_ENV:-$LANE_DIR/glm53-serve.env}"
[[ -f "$GLM53_ENV" ]] && . "$GLM53_ENV"
WORKER="${WORKER:-${WORKER_SSH:?WORKER_SSH 를 glm53-serve.env 에 설정할 것}}"
NAME=vllm_glm53
LAUNCH="${LAUNCH:-$LANE_DIR/launch.sh}"
KEY="${API_KEY:-${VLLM_API_KEY:?VLLM_API_KEY 를 glm53-serve.env 에 설정할 것}}"
HEALTH="http://localhost:8000/health"
OTHER="vllm_dspark vllm_qwen38next"     # 같이 뜨면 안 되는 컨테이너

wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }
WATCH_MIN_GB="${WATCH_MIN_GB:-3}"   # MemAvailable 바닥. 밑돌면 vLLM 컨테이너만 선제 stop (박스 보호)
PIDFILE=/tmp/vllm_glm53exl3_memwatch.pid
WLOG=/tmp/vllm_glm53exl3_memwatch.log

# --- memory watchdog (딥식이 vllm-dspark-lv0731.sh 의 start_watch 포팅, 양 노드) ----
# 각 노드에서 실행되는 스니펫. 인자: min cont pidfile log
_WATCH_SNIPPET='min=$1; cont=$2; pf=$3; log=$4
[ -f "$pf" ] && { kill "$(cat "$pf")" 2>/dev/null || true; rm -f "$pf"; }
[ -f /tmp/vllm_glm53_memwatch.pid ] && { kill "$(cat /tmp/vllm_glm53_memwatch.pid)" 2>/dev/null || true; rm -f /tmp/vllm_glm53_memwatch.pid; }  # 옛 NVFP4 레인 감시자 정리
nohup bash -c "
  while true; do
    avail=\$(awk \"/MemAvailable/{print int(\\\$2/1024/1024)}\" /proc/meminfo)
    if [ \"\${avail:-99}\" -lt $min ]; then
      echo \"\$(date +%T) MemAvailable \${avail}GB < ${min}GB -> docker stop -t 0 $cont (OOM pre-empt)\" >> $log
      docker stop -t 0 $cont >> $log 2>&1; sleep 5
    fi
    sleep 0.2
  done" >> "$log" 2>&1 &
echo $! > "$pf"; disown'
watch_start() {
  bash -c "$_WATCH_SNIPPET" _ "$WATCH_MIN_GB" "$NAME" "$PIDFILE" "$WLOG" && echo "[HEAD]   memwatch on (MemAvailable >= ${WATCH_MIN_GB}GB)"
  wssh "bash -c '$_WATCH_SNIPPET' _ $WATCH_MIN_GB $NAME $PIDFILE $WLOG" && echo "[WORKER] memwatch on (MemAvailable >= ${WATCH_MIN_GB}GB)"
}
watch_stop() {
  [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; echo "[HEAD]   memwatch off"; }
  wssh "[ -f $PIDFILE ] && { kill \$(cat $PIDFILE) 2>/dev/null || true; rm -f $PIDFILE; echo '[WORKER] memwatch off'; }" || true
}
watch_status() {
  local h=down w=down
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && h=up
  wssh "[ -f $PIDFILE ] && kill -0 \$(cat $PIDFILE) 2>/dev/null" && w=up
  echo "memwatch: HEAD=$h WORKER=$w (bar ${WATCH_MIN_GB}GB, log $WLOG)"
}
# 워커는 파일 사본이라 헤드에서 브랜치/커밋이 바뀌면 어긋난다 → 기동 전에 반드시 대조(끄기: SYNC_CHECK=0)
sync_check() { [[ "${SYNC_CHECK:-1}" == "0" ]] && return 0; "$LANE_DIR/sync-lane-67.sh" --check || { echo "⛔ 기동 중단: 레인 불일치"; exit 3; }; }
dropcaches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; }

stop_head()   { docker rm -f $NAME >/dev/null 2>&1 && echo "[HEAD]   STOP: $NAME" || echo "[HEAD]   STOP: not running"; }
stop_worker() { wssh "docker rm -f $NAME >/dev/null 2>&1 && echo '[WORKER] STOP: $NAME' || echo '[WORKER] STOP: not running'"; }

check_others() {
  for c in $OTHER; do
    if docker ps --format '{{.Names}}' | grep -qx "$c"; then echo "⛔ $c 가 떠 있음. 먼저 내릴 것."; exit 2; fi
  done
}
start_worker() { wssh "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; $LAUNCH 1 2>&1 | tail -1 | cut -c1-90" | sed 's/^/[WORKER] /'; }
start_head()   { check_others; dropcaches; $LAUNCH 0 2>&1 | tail -1 | cut -c1-90 | sed 's/^/[HEAD]   /'; }

# 서빙 중인 모델과 abliteration 스위치 상태. 모델은 원본 하나이므로 실제로 알고
# 싶은 것은 스위치이고, 그건 서버에게 묻는다(우리 추측이 아니라 서버의 값).
model_label() {
  local src; src=$(docker inspect $NAME --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null | grep -m1 'Models/')
  [[ -z "$src" ]] && { echo "없음"; return; }
  local name="원본"; [[ "$src" == *Abliterated* ]] && name="abliterated(대조용)"
  local ab; ab=$(curl -s -m3 -H "Authorization: Bearer $KEY" "${HEALTH%/health}/ab" 2>/dev/null)
  case "$ab" in
    *'"scale":1'*) echo "$name + /ab ON(검열 해제)" ;;
    *'"scale":0'*) echo "$name + /ab off" ;;
    *)             echo "$name (스위치 없음)" ;;
  esac
}

wait_ready() {
  local t=0
  until curl -sf -m5 -H "Authorization: Bearer $KEY" $HEALTH >/dev/null 2>&1; do
    docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "⛔ 헤드 컨테이너 종료됨. LOGS 확인."; return 1; }
    sleep 15; t=$((t+15)); [ $t -ge 900 ] && { echo "⛔ 15분 내 미기동"; return 1; }
  done
  echo "✅ API READY ($(date +%T), ${t}s) — 모델 glm-5.3-flash :8000 — $(model_label)"
  docker logs $NAME 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens|Maximum concurrency[^|]*' | tail -2 | sed 's/^/   /'
}

status() {
  echo "== HEAD ==";   docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}' | grep . || echo "없음"
  echo "== WORKER ($WORKER) =="; wssh "docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'" | grep . || echo "없음"
  curl -s -o /dev/null -w "health=%{http_code}\n" -m5 -H "Authorization: Bearer $KEY" $HEALTH
  echo "모델: $(model_label)"
  docker inspect $NAME --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -E 'DENSE_FP8_KERNEL|LMHEAD_FP8|KV_CACHE_MEMORY' | tr '\n' ' '; echo
  free -g | awk 'NR==2{print "[HEAD]   가용 "$7"GB"}'; wssh "free -g | awk 'NR==2{print \"[WORKER] 가용 \"\$7\"GB\"}'"
  watch_status
}

case "${1:-}" in
  RESTARTALL) echo "== GLM-5.3 EXL3 RESTARTALL =="; sync_check; stop_head; stop_worker; sleep 3; watch_start; start_worker; sleep 20; start_head; wait_ready ;;
  STOPALL)    echo "== GLM-5.3 EXL3 STOPALL =="; stop_head; stop_worker; watch_stop ;;
  STATUS)     status ;;
  LOGS)       docker logs -f --tail 100 $NAME ;;
  LOGSW)      wssh "docker logs -f --tail 100 $NAME" ;;
  START)      echo "⚠️ 단독 START 는 워커가 이미 랑데부 대기 중일 때만. 보통은 RESTARTALL."; sync_check; watch_start; start_head; wait_ready ;;
  WATCH)      watch_start; watch_status ;;      # 재시작 없이 감시자만 (재)가동
  UNWATCH)    watch_stop ;;
  STOP)       stop_head; watch_stop ;;
  *) echo "사용법: $0 RESTARTALL | STOPALL | STATUS | LOGS | LOGSW | WATCH | UNWATCH | START | STOP"; exit 1 ;;
esac
