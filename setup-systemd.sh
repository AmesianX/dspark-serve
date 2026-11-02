#!/usr/bin/env bash
# ============================================================================
#  각 머신에서 실행 → dspark-gateway systemd 서비스를 "이 머신에 맞게 생성+등록".
#  systemd 유닛은 머신마다 사용자/경로가 달라서 정적 파일 복사가 아니라 여기서
#  현재 환경(사용자·uvicorn 경로·작업디렉·키)을 감지해 생성한다. 멱등(재실행 안전).
#
#  사용:  ./setup-systemd.sh            # 게이트웨이 서비스 생성/갱신 + 시작
#         ./setup-systemd.sh status     # 상태만
#         ./setup-systemd.sh uninstall  # 제거(서비스 중지·disable·유닛 삭제)
#
#  sudo 로 실행하지 말 것 — 스크립트가 필요한 부분만 내부에서 sudo 를 호출한다
#  (그래야 whoami/$HOME/PATH 가 실제 사용자로 잡혀 경로가 올바르게 해석됨).
# ============================================================================
set -euo pipefail

SVC="${SVC:-dspark-gateway}"
PORT="${PORT:-8443}"
UNIT="/etc/systemd/system/${SVC}.service"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "sudo 없이 실행하세요: ./setup-systemd.sh  (내부에서 sudo 를 씁니다)"; exit 1
fi

# --- 현재 머신 환경 감지 ----------------------------------------------------
RUN_USER="$(whoami)"
UVICORN="$(command -v uvicorn || true)"
[ -z "$UVICORN" ] && UVICORN="$HOME/.local/bin/uvicorn"
if [ ! -x "$UVICORN" ]; then
  echo "uvicorn 을 못 찾음 ($UVICORN). 'pip install uvicorn' 후 다시 실행하세요."; exit 1
fi
# 키/업스트림은 run-gateway.sh 를 단일 출처로 추출 (env 로 override 가능)
AES_KEY="${LLAMACPP_AES_KEY:-$(grep -oP 'LLAMACPP_AES_KEY="\K[^"]+' "$DIR/run-gateway.sh" 2>/dev/null || true)}"
UPSTREAM="${VLLM_UPSTREAM:-$(grep -oP 'VLLM_UPSTREAM=\K\S+' "$DIR/run-gateway.sh" 2>/dev/null || echo 'http://127.0.0.1:8000')}"

status() { sudo systemctl status "$SVC" --no-pager -l 2>/dev/null | head -10 || echo "(미등록)"; }

uninstall() {
  sudo systemctl disable --now "$SVC" 2>/dev/null || true
  sudo rm -f "$UNIT"; sudo systemctl daemon-reload
  echo "$SVC 제거됨"; exit 0
}

case "${1:-install}" in
  status) status; exit 0 ;;
  uninstall) uninstall ;;
  install) : ;;
  *) echo "usage: $0 {install|status|uninstall}"; exit 2 ;;
esac

# --- :PORT 정리 (충돌 방지) --------------------------------------------------
# 이미 서비스가 있으면 systemctl 로 먼저 내린다(자동재시작 안 함). 그래야 아래
# run-gateway.sh stop 이 systemd 관리 프로세스를 죽여 flap 시키는 일이 없다.
sudo systemctl stop "$SVC" 2>/dev/null || true
# 남아있을 수 있는 (systemd 이전의) nohup 게이트웨이도 정리.
if [ -x "$DIR/run-gateway.sh" ]; then "$DIR/run-gateway.sh" stop 2>/dev/null || true; fi

echo "[setup] user=$RUN_USER  dir=$DIR  uvicorn=$UVICORN  port=$PORT"

# --- 유닛 생성 (이 머신 값으로) ---------------------------------------------
sudo tee "$UNIT" >/dev/null <<UNITEOF
[Unit]
Description=DSpark encrypt gateway (uvicorn :${PORT}, AES body-decrypt in front of vLLM)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${DIR}
Environment=LLAMACPP_AES_KEY=${AES_KEY}
Environment=VLLM_UPSTREAM=${UPSTREAM}
ExecStart=${UVICORN} encrypt_gateway:app --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=2
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNITEOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SVC"
echo "[setup] 게이트웨이 서비스 등록/시작 완료"
status

# --- vLLM 컨테이너가 이미 떠 있으면 재시작 정책도 적용 (있을 때만) -----------
if command -v docker >/dev/null 2>&1 && docker inspect vllm_dspark >/dev/null 2>&1; then
  docker update --restart unless-stopped vllm_dspark >/dev/null 2>&1 \
    && echo "[setup] vllm_dspark: --restart unless-stopped 적용 (재생성 없이)"
fi

echo "[setup] 완료. 로그: journalctl -u ${SVC} -f"
