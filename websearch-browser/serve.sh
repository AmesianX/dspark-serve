#!/usr/bin/env bash
# ============================================================================
#  serve.sh — browser-lane 기동/정지
#
#  ⛔ 메모리 하드캡이 필수다. 이 박스는 vLLM 이 117GB 를 쥐고 있고 가용 ~4GB,
#     GB10 은 통합메모리라 OOM 이 나면 박스 전체가 죽는다. 캡이 있으면 레인만
#     죽고 박스는 산다. systemd-run --scope 로 MemoryMax 를 걸어 띄운다.
#
#  브라우저는 상주하지 않는다 — 유휴 LANE_IDLE_EXIT(기본 180초)에 프로세스가
#  죽고, 다음 질의에 다시 뜬다. 프로필(=신원)은 디스크에 남으므로 신원은 유지된다.
#
#  바인딩: 127.0.0.1 + 172.18.0.1(nginx-network 게이트웨이). SearXNG 컨테이너가
#  172.18.0.1:8099 로 닿는다. ⛔ 0.0.0.0 금지.
#
#  Commands: start | stop | restart | status | logs | test | selftest
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
STATE="$ROOT/state"
PIDF="$STATE/lane.pid"
LOG="$STATE/lane.log"
TOKF="$STATE/token"
PORT="${LANE_PORT:-8099}"
MEMMAX="${LANE_MEMMAX:-1500M}"
IDENTITY="${LANE_IDENTITY:-default}"
TAG="[lane]"

mkdir -p "$STATE"

ensure_token() {
  if [ ! -f "$TOKF" ]; then
    openssl rand -hex 32 > "$TOKF"; chmod 600 "$TOKF"
    echo "$TAG 토큰 생성: $TOKF"
  fi
}

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

cmd_start() {
  running && { echo "$TAG 이미 돌고 있다 (pid $(cat "$PIDF"))"; exit 0; }
  ensure_token
  [ -x "$VENV/bin/python" ] || { echo "$TAG .venv 없음. uv venv .venv && uv pip install playwright"; exit 1; }
  command -v xvfb-run >/dev/null || { echo "$TAG xvfb-run 없음"; exit 1; }

  echo "$TAG 기동 (메모리캡 $MEMMAX, 신원 $IDENTITY, 포트 $PORT)"
  # headless-shell 은 UA 에 HeadlessChrome 이 박히므로 headful + xvfb 로 돈다.
  local inner="LANE_TOKEN=$(cat "$TOKF") LANE_IDENTITY=$IDENTITY LANE_PORT=$PORT \
    xvfb-run -a '$VENV/bin/python' '$ROOT/lane.py'"
  if command -v systemd-run >/dev/null 2>&1 && \
     systemd-run --user --scope -p MemoryMax=1M true >/dev/null 2>&1; then
    setsid systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p "MemorySwapMax=0" \
      --unit "browser-lane-$$" bash -c "$inner" >>"$LOG" 2>&1 &
  else
    echo "$TAG ⚠️ systemd-run --user 불가 → 캡 없이 띄운다. 가용 메모리를 직접 지켜볼 것."
    setsid bash -c "$inner" >>"$LOG" 2>&1 &
  fi
  echo $! > "$PIDF"
  sleep 4
  if running; then echo "$TAG pid $(cat "$PIDF")"; cmd_status; else echo "$TAG 기동 실패 — logs 확인"; tail -20 "$LOG"; exit 1; fi
}

cmd_stop() {
  # ⛔ pkill 금지. PID 로만 죽인다 (자기 자신·ssh 까지 죽인 사례 있음).
  if running; then
    local p; p="$(cat "$PIDF")"
    kill -TERM "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true
    sleep 2
    kill -0 "$p" 2>/dev/null && { kill -9 "-$p" 2>/dev/null || kill -9 "$p" 2>/dev/null || true; }
    echo "$TAG 정지 (pid $p)"
  else
    echo "$TAG 안 돌고 있다"
  fi
  rm -f "$PIDF"
  # 남은 chromium 은 PID 로 확인해서 정리 (대괄호 패턴으로 자기매칭 회피)
  local leftover
  leftover="$(ps -o pid=,args= -e 2>/dev/null | grep "[i]dentities/$IDENTITY" | awk '{print $1}' || true)"
  [ -n "$leftover" ] && { echo "$TAG 잔여 chromium 정리: $leftover"; kill -9 $leftover 2>/dev/null || true; }
  true
}

cmd_status() {
  running && echo "$TAG RUNNING pid $(cat "$PIDF")" || echo "$TAG STOPPED"
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" \
    | "$VENV/bin/python" -c 'import json,sys
d=json.load(sys.stdin)
print(f"  신원={d["identity"]}  브라우저기동={d["browser_up"]}  유휴종료={d["idle_exit_s"]}s")
print(f"  신원 쿠키 {d["identity_cookies"]["count"]}개")
for k,v in d["engines"].items():
    print(f"  {k:8s} {v["status"]:15s} 질의{v["queries"]:3d} 챌린지{v["challenges"]:2d} {v["note"]}")' 2>/dev/null \
    || { echo "  ⚠️ health 파싱 실패 — 원문:"; curl -s --max-time 5 "http://127.0.0.1:$PORT/health"; echo; }
  echo "  가용 메모리: $(free -g | awk 'NR==2{print $7"GB"}')"
}

cmd_test() {
  local q="${1:-io_uring documentation}" e="${2:-google}"
  curl -s --max-time 90 -H "X-Lane-Token: $(cat "$TOKF")" \
    "http://127.0.0.1:$PORT/search?q=$(printf %s "$q" | sed 's/ /+/g')&engine=$e&n=10" \
    | "$VENV/bin/python" -c 'import json,sys
d=json.load(sys.stdin)
print("엔진:",d.get("engine"),"| 상태:",d.get("status"),"| 결과:",len(d.get("results",[])))
if d.get("error"): print("오류:",d["error"],d.get("note",""),d.get("screenshot",""))
if d.get("skipped"): print("건너뜀:",d["skipped"],d.get("note"))
for r in d.get("results",[])[:6]:
    w=" [wrapped]" if r.get("wrapped") else ""
    print(f"  - {r["title"][:62]}{w}\n      {r["url"][:76]}")'
}

cmd_selftest() {
  local e="${1:-google}"
  echo "$TAG 정직성 가드: 난수 질의 → 0건이어야 정상 (bing 교훈)"
  curl -s --max-time 90 -H "X-Lane-Token: $(cat "$TOKF")" \
    "http://127.0.0.1:$PORT/selftest?engine=$e" | "$VENV/bin/python" -m json.tool
}

case "${1:-}" in
  start)    cmd_start ;;
  stop)     cmd_stop ;;
  restart)  cmd_stop; sleep 1; cmd_start ;;
  status)   cmd_status ;;
  logs)     tail -n "${2:-60}" "$LOG" ;;
  test)     shift; cmd_test "$@" ;;
  selftest) shift; cmd_selftest "$@" ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
