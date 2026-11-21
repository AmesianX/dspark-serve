#!/usr/bin/env bash
# ============================================================================
#  ws.sh — websearch-index 제어
#
#  ⛔ 메모리 하드캡 필수. 이 박스는 vLLM 이 쥐고 있어 가용 ~4GB 이고 GB10 은
#     통합메모리라 OOM 이 박스 전체를 죽인다. 캡이 있으면 이놈만 죽는다.
#  ⛔ pkill 금지. PID 로만 죽인다.
#
#  start|stop|restart|status|collect|search|stats|subs|logs
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"; STATE="$ROOT/state"; LOGS="$ROOT/logs"
PORT="${INDEX_PORT:-8097}"
MEMMAX="${INDEX_MEMMAX:-600M}"
TOKF="$STATE/token"; TAG="[index]"
mkdir -p "$STATE" "$LOGS"
[ -f "$TOKF" ] || { openssl rand -hex 32 >"$TOKF"; chmod 600 "$TOKF"; echo "$TAG 토큰 생성"; }
TOK="$(cat "$TOKF")"

pidf() { echo "$STATE/$1.pid"; }
alive() { [ -f "$(pidf "$1")" ] && kill -0 "$(cat "$(pidf "$1")")" 2>/dev/null; }

launch() { # name  command...
  local n="$1"; shift
  alive "$n" && { echo "$TAG $n 이미 가동 (pid $(cat "$(pidf "$n")"))"; return 0; }
  local inner="INDEX_TOKEN=$TOK INDEX_PORT=$PORT $*"
  if command -v systemd-run >/dev/null 2>&1 && \
     systemd-run --user --scope -p MemoryMax=1M true >/dev/null 2>&1; then
    setsid systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p "MemorySwapMax=0" \
      --unit "wsindex-$n-$$" bash -c "$inner" >>"$LOGS/$n.log" 2>&1 &
  else
    echo "$TAG ⚠️ systemd-run 불가 → 캡 없이 띄운다"
    setsid bash -c "$inner" >>"$LOGS/$n.log" 2>&1 &
  fi
  echo $! > "$(pidf "$n")"
  echo "$TAG $n pid $(cat "$(pidf "$n")")"
}

halt() {
  local n="$1"
  if alive "$n"; then
    local p; p="$(cat "$(pidf "$n")")"
    kill -TERM "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true
    sleep 1
    kill -0 "$p" 2>/dev/null && { kill -9 "-$p" 2>/dev/null || kill -9 "$p" 2>/dev/null || true; }
    echo "$TAG $n 정지 (pid $p)"
  else echo "$TAG $n 안 돌고 있다"; fi
  rm -f "$(pidf "$n")"
}

api() { curl -s --max-time 120 -H "X-Index-Token: $TOK" "http://127.0.0.1:$PORT$1"; }
pp()  { "$VENV/bin/python" -m json.tool 2>/dev/null || cat; }

case "${1:-}" in
  start)
    # systemd 가 이 서비스를 들고 있으면 손대지 않는다. systemd 가 든 프로세스는
    # state/*.pid 를 쓰지 않으므로 alive() 가 못 보고, 그대로 두면 중복 기동 →
    # 8097 바인드 실패다. 소유자는 하나여야 한다. (2026-11-21)
    if command -v systemctl >/dev/null 2>&1 && \
       systemctl --user is-active --quiet websearch-index.service 2>/dev/null; then
      echo "$TAG ⛔ systemd 가 가동 중이다 — ws.sh start 는 중복 기동이다."
      echo "$TAG    제어: systemctl --user restart websearch-index websearch-collect"
      exit 1
    fi
    launch serve   "'$VENV/bin/python' '$ROOT/serve.py'"
    launch collect "'$VENV/bin/python' '$ROOT/collect.py' --interval 600"
    sleep 3; "$0" status ;;
  stop)    halt collect; halt serve ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status)
    for n in serve collect; do
      alive "$n" && echo "$TAG $n RUNNING pid $(cat "$(pidf "$n")")" || echo "$TAG $n STOPPED"
    done
    api /stats | pp
    echo "가용 메모리: $(free -g | awk 'NR==2{print $7"GB"}')" ;;
  collect) shift; "$VENV/bin/python" "$ROOT/collect.py" --once "$@" ;;
  search)  shift; q="$(printf %s "$*" | "$VENV/bin/python" -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.stdin.read().strip()))')"
           api "/search?q=$q&n=10" | pp ;;
  stats)   api /stats | pp ;;
  subs)    api /subs  | pp ;;
  logs)    tail -n "${2:-40}" "$LOGS"/*.log ;;
  *) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
