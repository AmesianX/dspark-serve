#!/usr/bin/env bash
# ============================================================================
#  Open WebUI serving control  (:8080 — DSV4 vLLM 앞단 채팅 UI)
#  — 컨테이너는 껍데기고, 대화기록/설정/업로드는 전부 named volume 에 있다.
#    그래서 이미지 교체 = 무손실 업그레이드. 볼륨만 지키면 된다.
#
#  ⚠️ 설정 출처: 이 스크립트의 기본값은 "돌고 있던 컨테이너를 docker inspect
#     해서 뜬 것"이다. ~/work/solivian/run_openwebui.sh 같은 다른 데 있는
#     스크립트는 폐기된 실험 잔재이고 WEBUI_AUTH=false(인증 해제)가 박혀 있으니
#     절대 참조하지 말 것.
#
#  ⚠️ 업그레이드 방향성: 0.10 → 0.11 처럼 마이너 올라가면 기동 시 DB 마이그레이션이
#     돌고 다운그레이드는 지원되지 않는다. 그래서 update 는 백업을 강제한다.
#     되돌리려면 restore 로 볼륨을 복원해야 한다(이미지만 되돌려선 안 됨).
#
#  다른 서비스와 무영향: vLLM(:8000)·게이트웨이(:8443)·searxng(:8888) 별개 컨테이너.
#  UPSTREAM 이 :8000(평문 vLLM)을 직접 보는 건 현재 배선 그대로 유지한 것.
#
#  Commands: start | stop | restart | status | logs | version | check
#            | update | backup | restore
# ============================================================================
set -euo pipefail

# --- 설정 (전부 env override 가능) ------------------------------------------
NAME="${NAME:-open-webui}"
PORT="${PORT:-8080}"                         # 호스트:컨테이너 동일
REPO="${REPO:-ghcr.io/open-webui/open-webui}"
# 라이브는 :main(브랜치 롤링 빌드)으로 떠 있었다. 기본은 :latest(안정 릴리스)로
# 둔다 — 롤링 데브 빌드를 상시로 물고 있을 이유가 없다.
# 예전 동작 유지하려면 IMG_TAG=main, 특정 버전 고정은 IMG_TAG=v0.11.0.
IMG_TAG="${IMG_TAG:-latest}"
IMG="${IMG:-${REPO}:${IMG_TAG}}"
VOLUME="${VOLUME:-open-webui}"               # /app/backend/data
BACKUP_DIR="${BACKUP_DIR:-/home/user/backup/open-webui}"
UPSTREAM="${UPSTREAM:-http://host.docker.internal:8000/v1}"
API_KEY="${API_KEY:?API 키를 환경변수로 지정하세요}"
TAG="[open-webui]"

exists()  { docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; }
running() { docker ps    --format '{{.Names}}' | grep -qx "$NAME"; }

# 컨테이너 안 package.json 의 version (죽어 있으면 빈 문자열)
cur_version() {
  docker exec "$NAME" sh -lc \
    "sed -n 's/.*\"version\": *\"\\([^\"]*\\)\".*/\\1/p' /app/package.json | head -1" 2>/dev/null || true
}

start() {
  if running; then echo "$TAG 이미 실행 중: $NAME"; status; return 0; fi
  if exists;  then echo "$TAG 중지 상태 컨테이너 제거 후 재생성"; docker rm -f "$NAME" >/dev/null; fi
  echo "$TAG start → :${PORT}, img=${IMG}"
  # WEBUI_AUTH 는 의도적으로 넘기지 않는다 = 인증 켜진 상태(기본값) 유지.
  # WEBUI_SECRET_KEY 도 넘기지 않는다 — 볼륨 안 .webui_secret_key 를 그대로 쓰므로
  # 여기서 주면 오히려 기존 세션이 전부 무효화된다.
  docker run -d --name "$NAME" --restart unless-stopped \
    -p "${PORT}:8080" \
    -e ENABLE_OPENAI_API=True \
    -e "OPENAI_API_BASE_URL=${UPSTREAM}" \
    -e "OPENAI_API_KEY=${API_KEY}" \
    -e SCARF_NO_ANALYTICS=true -e DO_NOT_TRACK=true -e ANONYMIZED_TELEMETRY=false \
    -v "${VOLUME}:/app/backend/data" \
    --add-host=host.docker.internal:host-gateway \
    --label app=open-webui \
    "$IMG" >/dev/null
  echo "$TAG 기동 대기..."; sleep 6
  echo "$TAG version: $(cur_version)"
  status
}

stop()    { echo "$TAG stop"; docker rm -f "$NAME" >/dev/null 2>&1 && echo "$TAG 제거됨" || echo "$TAG 없음"; }
restart() { stop; start; }
logs()    { docker logs -f "$NAME"; }
version() { echo "$TAG running version: $(cur_version)   (image=$(docker inspect "$NAME" --format '{{.Config.Image}}' 2>/dev/null))"; }

status() {
  echo "$TAG ---- status ----"
  docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo "$TAG version : $(cur_version)"
  echo "$TAG upstream: ${UPSTREAM}"
  echo "$TAG volume  : ${VOLUME}  ($(docker run --rm -v "${VOLUME}":/d alpine du -sh /d 2>/dev/null | cut -f1))"
  echo "$TAG local   : http://127.0.0.1:${PORT}/"
}

# 릴리스(깃허브) vs 이미지(ghcr) 를 따로 본다. 릴리스가 먼저 나고 이미지 빌드가
# 나중에 올라오는 구간이 있어서, 태그만 믿고 pull 하면 404 로 헛돈다.
check() {
  local cur latest_rel
  cur="$(cur_version)"
  echo "$TAG 현재 실행 : ${cur:-(중지됨)}"
  latest_rel="$(curl -sS -m 20 https://api.github.com/repos/open-webui/open-webui/releases/latest \
                | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
  echo "$TAG 최신 릴리스: ${latest_rel:-(조회 실패)}"
  if [ -n "$latest_rel" ]; then
    local tok code
    tok="$(curl -sS -m 15 "https://ghcr.io/token?scope=repository:open-webui/open-webui:pull&service=ghcr.io" \
           | sed -n 's/.*"token": *"\([^"]*\)".*/\1/p' | head -1)"
    code="$(curl -sS -m 15 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $tok" \
      -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json' \
      "https://ghcr.io/v2/open-webui/open-webui/manifests/${latest_rel}")"
    if [ "$code" = "200" ]; then
      echo "$TAG ✅ 이미지 발행됨 → IMG_TAG=${latest_rel} $0 update  (또는 latest)"
    else
      echo "$TAG ⏳ ${latest_rel} 이미지 아직 미발행 (HTTP ${code}) — 빌드 올라올 때까지 대기"
    fi
  fi
}

backup() {
  mkdir -p "$BACKUP_DIR"
  local out="${BACKUP_DIR}/${VOLUME}-$(date +%Y%m%d-%H%M%S).tar.gz"
  echo "$TAG backup ${VOLUME} → ${out}"
  # 정지 상태에서 떠야 SQLite 가 일관된다. 돌고 있으면 잠깐 멈췄다 되살린다.
  local was_running=0; running && was_running=1
  [ "$was_running" = 1 ] && { echo "$TAG (일관성 위해 잠시 정지)"; docker stop "$NAME" >/dev/null; }
  docker run --rm -v "${VOLUME}":/data -v "${BACKUP_DIR}":/b alpine \
    tar czf "/b/$(basename "$out")" -C /data .
  [ "$was_running" = 1 ] && { docker start "$NAME" >/dev/null; echo "$TAG (재기동)"; }
  ls -lh "$out"
}

update() {
  local before after out
  before="$(cur_version)"
  echo "$TAG update: ${before:-?} → ${IMG}"
  backup                                    # 마이그레이션은 되돌릴 수 없다. 무조건 먼저.
  out="$(ls -t "${BACKUP_DIR}"/${VOLUME}-*.tar.gz 2>/dev/null | head -1)"
  echo "$TAG pull ${IMG}"
  docker pull "$IMG"
  stop
  start
  after="$(cur_version)"
  echo "$TAG ---- update 완료 ----"
  echo "$TAG  before : ${before:-?}"
  echo "$TAG  after  : ${after:-?}"
  echo "$TAG  backup : ${out}"
  echo "$TAG  롤백   : $0 restore '${out}'  후  IMG_TAG=<이전태그> $0 start"
}

restore() {
  local src="${1:-}"
  [ -n "$src" ] && [ -f "$src" ] || {
    echo "$TAG usage: $0 restore <backup.tar.gz>"
    echo "$TAG 사용 가능한 백업:"; ls -lht "${BACKUP_DIR}"/*.tar.gz 2>/dev/null | head; return 1; }
  echo "$TAG ⚠️  볼륨 ${VOLUME} 의 현재 내용을 지우고 ${src} 로 되돌린다."
  read -r -p "$TAG 계속하려면 'yes' 입력: " ans
  [ "$ans" = "yes" ] || { echo "$TAG 취소됨"; return 1; }
  stop
  docker run --rm -v "${VOLUME}":/data -v "$(cd "$(dirname "$src")" && pwd)":/b alpine \
    sh -c "rm -rf /data/* /data/..?* /data/.[!.]* 2>/dev/null; tar xzf /b/$(basename "$src") -C /data"
  echo "$TAG 복원 완료 — start 로 기동"
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) restart ;;
  status)  status ;;
  logs)    logs ;;
  version) version ;;
  check)   check ;;
  update)  update ;;
  backup)  backup ;;
  restore) shift; restore "${1:-}" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs|version|check|update|backup|restore <file>}"; exit 1 ;;
esac
