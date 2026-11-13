#!/usr/bin/env bash
# 비밀값은 비추적 파일에서 읽는다. 이 리포는 공개돼 있다
# (github.com/AmesianX/dspark-serve) — 2026-11-13 까지 AES 키가 이 줄에 평문으로
# 커밋돼 있었다. 키 소유가 곧 8443 의 자격증명이므로 평문 커밋은 게이트 무력화와
# 같다. gateway.env.example 을 gateway.env 로 복사해 값을 채운다.
GATEWAY_ENV="${GATEWAY_ENV:-$(dirname "$0")/gateway.env}"
if [ -z "${LLAMACPP_AES_KEY:-}" ] && [ -f "$GATEWAY_ENV" ]; then
  # shellcheck disable=SC1090
  . "$GATEWAY_ENV"
fi
export LLAMACPP_AES_KEY
[ -n "${LLAMACPP_AES_KEY:-}" ] || {
  echo "LLAMACPP_AES_KEY 를 찾을 수 없다 ($GATEWAY_ENV 확인, gateway.env.example 참조)" >&2
  exit 1
}
export VLLM_UPSTREAM=http://127.0.0.1:8000

# The gateway attaches the vLLM bearer token itself to decrypted requests, so the
# key must be in its environment. Source it from the lane's env file instead of
# pasting the secret a second time - one definition, one place to rotate it.
# That file uses `: "${VAR:=value}"` form, which systemd EnvironmentFile cannot
# read, so the unit runs this script (`fg`) rather than uvicorn directly.
LANE_ENV="${LANE_ENV:-$(dirname "$0")/glm53-exl3-lane/glm53-serve.env}"
if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$LANE_ENV" ]; then
  VLLM_API_KEY="$(set -a; . "$LANE_ENV" >/dev/null 2>&1; printf '%s' "${VLLM_API_KEY:-}")"
fi
export VLLM_API_KEY
[ -n "$VLLM_API_KEY" ] || { echo "VLLM_API_KEY 를 찾을 수 없다 ($LANE_ENV 확인)" >&2; exit 1; }

HOST=0.0.0.0
PORT=8443
APP=encrypt_gateway:app
# systemd gives this unit a minimal PATH that does not include ~/.local/bin, so
# resolve uvicorn here instead of relying on the caller's PATH.
UVICORN="${UVICORN:-$(command -v uvicorn 2>/dev/null || echo /home/user/.local/bin/uvicorn)}"
[ -x "$UVICORN" ] || { echo "uvicorn 실행파일을 찾을 수 없다: $UVICORN" >&2; exit 1; }

cd "$(dirname "$0")" || exit 1
PIDFILE="$(pwd)/gateway.pid"
LOGFILE="$(pwd)/gateway.log"

_running_pid() {
  # PID파일 우선, 없거나 죽었으면 포트 리스너에서 역추적
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    cat "$PIDFILE"; return 0
  fi
  local p
  p=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$p" ] && { echo "$p"; return 0; }
  return 1
}

run_fg() { exec "$UVICORN" "$APP" --host "$HOST" --port "$PORT"; }

start() {
  if pid=$(_running_pid); then
    echo "already running (pid=$pid, port=$PORT)"; return 0
  fi
  nohup "$UVICORN" "$APP" --host "$HOST" --port "$PORT" >>"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  sleep 1
  if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "started (pid=$(cat "$PIDFILE"), port=$PORT, log=$LOGFILE)"
  else
    echo "start FAILED — check $LOGFILE"; tail -n 20 "$LOGFILE"; return 1
  fi
}

stop() {
  if ! pid=$(_running_pid); then echo "not running"; rm -f "$PIDFILE"; return 0; fi
  echo "stopping pid=$pid ..."
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  if kill -0 "$pid" 2>/dev/null; then echo "SIGTERM 무응답 → SIGKILL"; kill -9 "$pid" 2>/dev/null; fi
  rm -f "$PIDFILE"; echo "stopped"
}

status() {
  if pid=$(_running_pid); then
    echo "RUNNING (pid=$pid, port=$PORT)"
    ss -ltnp 2>/dev/null | grep ":$PORT " || true
  else
    echo "STOPPED (port=$PORT)"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  ""|fg|foreground) run_fg ;;   # 인자 없이 실행 = 기존처럼 포그라운드
  *) echo "usage: $0 {start|stop|restart|status|fg}"; exit 2 ;;
esac
