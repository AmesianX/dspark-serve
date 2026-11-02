#!/usr/bin/env bash
# ============================================================================
#  레인 디렉터리를 헤드 → 워커(같은 경로)로 밀고 파일별 md5 로 검증한다. (이름의 67 은 우리 워커 주소에서 온 것)
#
#  워커는 git 저장소가 아니라 파일 사본이다. 헤드에서 브랜치를 바꾸거나 커밋해도
#  워커는 그대로라서, 밀지 않으면 두 노드가 다른 코드로 뜬다(랭크 불일치).
#  vllm-glm53-exl3.sh RESTARTALL 이 기동 전에 `--check` 를 돌려 어긋나면 멈춘다.
#
#  사용:  ./sync-lane-67.sh           밀기 + 검증 (커밋 안 된 변경이 있으면 거부, FORCE=1 로 무시)
#         ./sync-lane-67.sh --check   검증만 (읽기 전용)
# ============================================================================
set -uo pipefail
LANE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
[[ -f "$LANE/glm53-serve.env" ]] && . "$LANE/glm53-serve.env"
WORKER="${SYNC_WORKER:-${WORKER_RAIL_IP:?WORKER_RAIL_IP 를 glm53-serve.env 에 설정할 것}}"      # RoCE 레일(200G). LAN 은 100M 로 떨어지는 일이 있음
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
EXCL=(--exclude '__pycache__/' --exclude '*.bak' --exclude '*.bak.*' --exclude '*.bak-*')

manifest() {  # 파일별 md5 목록 (캐시·백업 제외, 정렬)
  cd "$LANE" && find . -type f -not -path '*/__pycache__/*' -not -name '*.bak' -not -name '*.bak.*' -not -name '*.bak-*' -print0 \
    | sort -z | xargs -0 md5sum
}

check() {
  local here there
  here=$(manifest) || return 2
  there=$("${SSH[@]}" "$WORKER" "$(declare -f manifest); LANE=$LANE; manifest") || { echo "⛔ 워커($WORKER) 접속/목록 실패"; return 2; }
  local rev; rev=$(git -C "$LANE" rev-parse --abbrev-ref HEAD 2>/dev/null)@$(git -C "$LANE" rev-parse --short HEAD 2>/dev/null)
  if [[ "$here" == "$there" ]]; then
    echo "[SYNC]   헤드 == 워커($WORKER) ($(wc -l <<<"$here") files, $rev)"
    return 0
  fi
  echo "⛔ 헤드와 워커($WORKER)의 레인이 다르다 ($rev). ./sync-lane-67.sh 로 밀 것:"
  diff <(echo "$here") <(echo "$there") | grep -E '^[<>]' | awk '{print "   " $1 " " $3}' | sort -k2 -u | head -20
  return 1
}

if [[ "${1:-}" == "--check" ]]; then check; exit $?; fi

# 밀기 전 안전장치
if [[ -n "$(git -C "$LANE" status --porcelain -- . 2>/dev/null)" && "${FORCE:-0}" != "1" ]]; then
  echo "⛔ 레인에 커밋 안 된 변경이 있다. 커밋 후 밀 것(실험 기록이 git 에 남아야 되돌릴 수 있다). 무시: FORCE=1"
  git -C "$LANE" status --short -- . | head
  exit 1
fi
hotfix_exists() {  # $1 = local | 워커 주소
  if [[ "$1" == local ]]; then [[ -e "$HOME/glm53-hotfix-b12x" || -L "$HOME/glm53-hotfix-b12x" ]]
  else "${SSH[@]}" "$1" 'test -e ~/glm53-hotfix-b12x || test -L ~/glm53-hotfix-b12x'; fi
}
for h in local "$WORKER"; do
  if hotfix_exists "$h"; then
    echo "⛔ $h 에 ~/glm53-hotfix-b12x 가 있다 — launch.sh 의 overlay/b12x 마운트와 겹쳐 기동 실패. 지울 것."; exit 1
  fi
done

rsync -a --delete "${EXCL[@]}" "$LANE/" "$WORKER:$LANE/" || { echo "⛔ rsync 실패"; exit 1; }
check
