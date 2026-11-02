#!/usr/bin/env bash
# ============================================================================
#  SearXNG serving control  (codemason WebSearch 의 0순위 백엔드)
#  — 자체 호스팅 메타서치. codemason 은 이 인스턴스의 JSON API 를 먼저 치고,
#    도달 불가/비활성이면 ddgs 스크래핑으로 fallback 한다
#    (codemason/tools/web_search_tool.py::_searxng_search).
#
#  llm.systemboan.net(vLLM DSV4)·open-webui(:8080)·qdrant 등과 완전 무영향:
#    별도 컨테이너(searxng), 별도 호스트 포트(:8888), 별도 config 디렉.
#
#  도메인: https://searxng.systemboan.net  →  Nginx Proxy Manager(:81 UI)
#          →  http://192.168.1.66:8888  (이 컨테이너)
#
#  Commands: start | stop | restart | status | logs | test
# ============================================================================
set -euo pipefail

# --- 설정 (전부 env override 가능) ------------------------------------------
NAME="${NAME:-searxng}"
HOST_IP="${HOST_IP:-192.168.1.66}"          # 8080 은 open-webui 점유 → 8888 사용
PORT="${PORT:-8888}"                        # 호스트 포트 (컨테이너 내부는 8080 고정)
IMG="${IMG:-searxng/searxng:latest}"
CONFIG_DIR="${CONFIG_DIR:-/home/user/searxng}"
DOMAIN="${DOMAIN:-searxng.systemboan.net}"
BASE_URL="${BASE_URL:-https://${DOMAIN}/}"

SETTINGS="${CONFIG_DIR}/settings.yml"
TAG="[searxng]"

# --- settings.yml 보장: json 포맷 + limiter off (codemason 프로그램 접근용) ---
ensure_settings() {
  mkdir -p "$CONFIG_DIR"
  if [ ! -f "$SETTINGS" ]; then
    local secret; secret="$(openssl rand -hex 32)"
    echo "$TAG settings.yml 생성 → $SETTINGS (json API 활성)"
    cat > "$SETTINGS" <<YAML
# codemason 자동 생성 — SearXNG 최소 설정.
# 기본값 위에 필요한 것만 덮어쓴다(use_default_settings). 자유롭게 수정 가능.
use_default_settings: true
general:
  instance_name: "codemason-searxng"
server:
  secret_key: "${secret}"
  limiter: false          # 프로그램적 JSON 요청 차단(429) 방지 — 내부 전용 인스턴스
  image_proxy: false
search:
  formats:
    - html
    - json                # ← codemason _searxng_search 가 요구. 없으면 계속 ddgs fallback
YAML
  else
    # 이미 있으면 사용자 편집 존중. 단, json 포맷 누락이면 경고(그럼 fallback 만 탐).
    if ! grep -qE '^\s*-\s*json\s*$' "$SETTINGS"; then
      echo "$TAG ⚠️  경고: $SETTINGS 에 'json' 포맷이 없음 → codemason 이 SearXNG 를 못 씀(ddgs fallback)."
      echo "$TAG     search.formats 아래에 '- json' 을 추가하고 restart 하세요."
    fi
  fi
}

exists() { docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; }
running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

start() {
  if running; then echo "$TAG 이미 실행 중: $NAME"; status; return 0; fi
  if exists;  then echo "$TAG 중지 상태 컨테이너 제거 후 재생성"; docker rm -f "$NAME" >/dev/null; fi
  ensure_settings
  echo "$TAG start → ${HOST_IP}:${PORT} (컨테이너 내부 8080), img=${IMG}"
  docker run -d --name "$NAME" --restart unless-stopped \
    -p "${HOST_IP}:${PORT}:8080" \
    -e "SEARXNG_BASE_URL=${BASE_URL}" \
    -v "${CONFIG_DIR}:/etc/searxng" \
    --label app=searxng \
    "$IMG"
  echo "$TAG 기동 대기..."; sleep 4; test || true
}

stop()    { echo "$TAG stop"; docker rm -f "$NAME" >/dev/null 2>&1 && echo "$TAG 제거됨" || echo "$TAG 없음"; }
restart() { stop; start; }
logs()    { docker logs -f "$NAME"; }

status() {
  echo "$TAG ---- status ----"
  docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo "$TAG domain : ${BASE_URL}"
  echo "$TAG local  : http://${HOST_IP}:${PORT}/search?q=...&format=json"
}

# JSON API 실제 응답 확인 (로컬 직결). 결과 개수 뜨면 정상.
test() {
  local url="http://${HOST_IP}:${PORT}/search?q=test&format=json"
  echo "$TAG test GET $url"
  local n
  n="$(curl -s -m 15 "$url" | { command -v jq >/dev/null && jq -r '.results | length' 2>/dev/null || grep -c '"url"'; })"
  if [ -n "${n:-}" ] && [ "${n:-0}" != "0" ]; then
    echo "$TAG ✅ OK — results=${n} (JSON API 정상)"
    echo "$TAG codemason 연결:  export CODEMASON_SEARXNG_URL=http://127.0.0.1:${PORT}   # 같은 서버면 프록시 우회가 더 빠름"
  else
    echo "$TAG ❌ JSON 응답 없음 — settings.yml 의 json 포맷/limiter, 컨테이너 로그(logs) 확인"
    return 1
  fi
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) restart ;;
  status)  status ;;
  logs)    logs ;;
  test)    test ;;
  *) echo "usage: $0 {start|stop|restart|status|logs|test}"; exit 1 ;;
esac
