#!/usr/bin/env bash
# Solar Open 2 (Nota NVFP4, W4A16 marlin) 2노드 제어.
#  ⚠️ 2노드는 무조건 RESTARTALL. 워커만 살아 있으면 랑데부 대기 좀비(600초).
#  ⚠️ GLM5.3(vllm_glm53)·DSV4(vllm_dspark)와 동시에 못 뜬다(통합메모리). 먼저 그쪽 STOPALL.
set -uo pipefail
LANE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LAUNCH="$LANE/launch.sh"
NAME="${NAME:-vllm_solar}"
WORKER="${WORKER:-10.0.1.2}"
PORT="${PORT:-8000}"
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }

others_running() {
  local o; o=$( { docker ps --format '{{.Names}}'; wssh "docker ps --format '{{.Names}}'"; } | grep -E '^vllm_' | grep -v "^$NAME$" | sort -u)
  [[ -n "$o" ]] && { echo "다른 vLLM 서빙이 떠 있음: $o — 먼저 내릴 것"; return 0; }; return 1
}
stop_head()   { docker rm -f "$NAME" >/dev/null 2>&1 && echo "[HEAD]   STOP: $NAME" || echo "[HEAD]   STOP: not running"; }
stop_worker() { wssh "docker rm -f $NAME >/dev/null 2>&1 && echo '[WORKER] STOP: $NAME' || echo '[WORKER] STOP: not running'"; }
# 양노드가 반드시 같은 값을 받아야 하는 노브(다르면 KV 스펙 불일치로 죽는다).
FORWARD_VARS="LANE_VARIANT SOLAR_MEM_PROBE SOLAR_MEM_RECLAIM FI_SAMPLER EAGER LOGITS_PROC ASYNC_SCHED MAX_LEN KV_BYTES MAX_SEQS GMU MNBT MOE_BACKEND LINEAR_BACKEND NVFP4_GEMM EXEC_TIMEOUT_S"
_fwd_env() { local v out=""; for v in $FORWARD_VARS; do [[ -n "${!v:-}" ]] && out+="$v=${!v} "; done; printf '%s' "$out"; }
start_worker(){ wssh "$(_fwd_env)$LAUNCH 1 2>&1 | tail -1" | sed 's/^/[WORKER] /'; }
start_head()  { "$LAUNCH" 0 2>&1 | tail -1 | sed 's/^/[HEAD]   /'; }
wait_ready() {
  echo "[HEAD]   준비 대기 (로그: $0 LOGS)"
  for i in $(seq 1 360); do
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && { echo "[HEAD]   READY (${i}0초 이내)"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "[HEAD]   컨테이너 죽음:"; docker logs --tail 30 "$NAME" 2>&1; return 1; }
    sleep 10
  done; echo "[HEAD]   타임아웃"; return 1
}
status() {
  echo "== HEAD(.66) ==";   docker ps -a --filter "name=^$NAME$" --format '{{.Names}} {{.Status}}' | grep . || echo "없음"
  echo "== WORKER(.67) =="; wssh "docker ps -a --filter name=^$NAME\$ --format '{{.Names}} {{.Status}}'" | grep . || echo "없음"
  free -g | awk 'NR==2{print "[HEAD]   가용 "$7"GB"}'; wssh "free -g | awk 'NR==2{print \"[WORKER] 가용 \"\$7\"GB\"}'"
}
case "${1:-}" in
  RESTARTALL) echo "== Solar Open 2 RESTARTALL =="; others_running && exit 1
              stop_head; stop_worker; sleep 3; start_worker; sleep 20; start_head; wait_ready ;;
  STOPALL)    echo "== Solar Open 2 STOPALL =="; stop_head; stop_worker ;;
  STATUS)     status ;;
  LOGS)       docker logs -f --tail 100 "$NAME" ;;
  LOGSW)      wssh "docker logs -f --tail 100 $NAME" ;;
  *) echo "사용법: $0 RESTARTALL | STOPALL | STATUS | LOGS | LOGSW"; exit 1 ;;
esac
