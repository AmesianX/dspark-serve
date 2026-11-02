#!/usr/bin/env bash
# python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"
export LLAMACPP_AES_KEY="jCuJ9cvYtgXbl/Tp5zWBOXIJI02c3Bw3a0NN03XPp9E="
export VLLM_UPSTREAM=http://127.0.0.1:8000

HOST=0.0.0.0
PORT=8443
APP=encrypt_gateway:app

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

run_fg() { exec uvicorn "$APP" --host "$HOST" --port "$PORT"; }

start() {
  if pid=$(_running_pid); then
    echo "already running (pid=$pid, port=$PORT)"; return 0
  fi
  nohup uvicorn "$APP" --host "$HOST" --port "$PORT" >>"$LOGFILE" 2>&1 &
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
