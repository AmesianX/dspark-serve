#!/usr/bin/env bash
# ============================================================================
#  SearXNG serving control  (codemason WebSearch 의 0순위 백엔드)
#  — 자체 호스팅 메타서치. codemason 은 이 인스턴스의 JSON API 를 먼저 치고,
#    도달 불가/비활성이면 ddgs 스크래핑으로 fallback 한다
#    (codemason/tools/web_search_tool.py::_searxng_search).
#
#  ⛔ 호스트 포트를 퍼블리시하지 않는다 (2026-11-13 변경).
#     이전에는 `-p 192.168.1.66:8888:8080` 이었다. .66 은 공유기 DMZ 안이라
#     LAN IP 바인딩이 노출을 전혀 줄이지 않았고, 8888 은 **인증 없이**
#     인터넷에 열려 있었다 (http://220.85.97.203:8888 응답 확인).
#     지금은 NPM 이 nginx-network 로 `searxng:8080` 에 직접 닿으므로
#     호스트 포트가 필요 없다.
#
#  유일한 외부 경로:
#     https://searxng.systemboan.net   (NPM 443, vhost Advanced 의 x-api-key 게이트)
#     키 없으면 403. 상세는 edge/README.md + edge/npm/vhost-searxng-advanced.conf
#
#  정본 compose: edge/searxng/docker-compose.yml  (이 스크립트가 그걸 호출한다)
#
#  Commands: start | stop | restart | status | logs | test
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_DIR/edge/searxng/docker-compose.yml}"
NAME="${NAME:-searxng}"
NET="${NET:-nginx-network}"
CONFIG_DIR="${CONFIG_DIR:-/home/user/searxng}"
SETTINGS="${SETTINGS:-$CONFIG_DIR/settings.yml}"
TEMPLATE="${TEMPLATE:-$REPO_DIR/edge/searxng/settings.yml.example}"
DOMAIN="${DOMAIN:-searxng.systemboan.net}"
PROXY="${PROXY:-nginx-proxy-manager}"
TAG="[searxng]"

dc() { docker compose -f "$COMPOSE_FILE" "$@"; }

# --- 전제조건: 공용 네트워크 + settings.yml ---------------------------------
ensure_net() {
  docker network inspect "$NET" >/dev/null 2>&1 || {
    echo "$TAG 네트워크 $NET 생성"
    docker network create "$NET" >/dev/null
  }
}

ensure_settings() {
  mkdir -p "$CONFIG_DIR"
  if [ ! -f "$SETTINGS" ]; then
    echo "$TAG settings.yml 생성 → $SETTINGS (secret_key 새로 생성)"
    sed "s|__GENERATE_WITH_openssl_rand_hex_32__|$(openssl rand -hex 32)|" \
      "$TEMPLATE" > "$SETTINGS"
  elif ! grep -qE '^\s*-\s*json\s*$' "$SETTINGS"; then
    echo "$TAG ⚠️  경고: $SETTINGS 에 'json' 포맷이 없음 → codemason 이 SearXNG 를 못 씀(ddgs fallback)."
    echo "$TAG     search.formats 아래에 '- json' 을 추가하고 restart 할 것."
  fi
}

running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

start() {
  ensure_net
  ensure_settings
  echo "$TAG start (호스트 포트 퍼블리시 없음 — NPM 이 $NET 로 ${NAME}:8080 직결)"
  dc up -d
  echo "$TAG 기동 대기..."; sleep 5
  test || true
}

stop()    { echo "$TAG stop"; dc down; }
restart() { stop; start; }
logs()    { dc logs -f; }

status() {
  echo "$TAG ---- status ----"
  docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo "$TAG network : $(docker inspect "$NAME" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}({{$v.IPAddress}}) {{end}}' 2>/dev/null || echo '-')"
  echo "$TAG domain  : https://${DOMAIN}  (X-API-Key 필수)"
  echo "$TAG 호스트 포트: 없음 (의도된 것. edge/README.md 참조)"
}

# JSON API 확인. 호스트 포트가 없으므로 NPM 컨테이너에서 내부 이름으로 때린다.
# codemason 이 쓰는 공개 경로(TLS + 키)도 키를 찾을 수 있으면 함께 확인한다.
test() {
  local ok=0

  echo "$TAG [1/2] 내부 직결: ${PROXY} → http://${NAME}:8080"
  if docker ps --format '{{.Names}}' | grep -qx "$PROXY"; then
    local n
    n="$(docker exec "$PROXY" sh -c \
      "curl -s -m 15 'http://${NAME}:8080/search?q=test&format=json'" 2>/dev/null \
      | { command -v jq >/dev/null && jq -r '.results | length' 2>/dev/null || grep -c '"url"'; })" || n=0
    if [ "${n:-0}" != "0" ]; then echo "$TAG   ✅ results=${n}"; ok=1
    else echo "$TAG   ❌ JSON 응답 없음 — settings.yml 의 json 포맷/limiter, logs 확인"; fi
  else
    echo "$TAG   ⚠️  $PROXY 컨테이너가 없어 건너뜀"
  fi

  echo "$TAG [2/2] 공개 경로: https://${DOMAIN} (키 포함)"
  local key=""
  key="${CODEMASON_SEARXNG_API_KEY:-}"
  if [ -z "$key" ] && [ -f "$HOME/.codemason/config.json" ]; then
    key="$(python3 -c "import json;print(json.load(open('$HOME/.codemason/config.json')).get('searxng_api_key',''))" 2>/dev/null || true)"
  fi
  if [ -n "$key" ]; then
    local m
    m="$(curl -s -m 20 -H "X-API-Key: $key" \
      "https://${DOMAIN}/search?q=test&format=json" \
      | { command -v jq >/dev/null && jq -r '.results | length' 2>/dev/null || grep -c '"url"'; })" || m=0
    if [ "${m:-0}" != "0" ]; then echo "$TAG   ✅ results=${m} (TLS + 키 게이트 정상)"
    else echo "$TAG   ❌ 응답 없음 — 키 불일치(403) 또는 인증서 만료 확인"; fi
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 15 "https://${DOMAIN}/search?q=test&format=json")"
    [ "$code" = "403" ] && echo "$TAG   ✅ 키 없이 → 403 (거부 기본값 정상)" \
                        || echo "$TAG   🚨 키 없이 → ${code} — 게이트가 빠졌다! edge/npm/vhost-searxng-advanced.conf 확인"
  else
    echo "$TAG   ⚠️  키를 못 찾아 건너뜀 (CODEMASON_SEARXNG_API_KEY 또는 ~/.codemason/config.json)"
  fi

  [ "$ok" = "1" ]
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
