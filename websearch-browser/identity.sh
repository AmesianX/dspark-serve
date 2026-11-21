#!/usr/bin/env bash
# ============================================================================
#  identity.sh — browser-lane 의 "신원" 관리
#
#  신원 = identities/<name>/ 디렉터리 하나 (Chromium user-data-dir).
#  감사 가능하고, 백업 가능하고, 태우면 버리고 새로 만든다.
#
#  ⛔ identities/ 를 git 에 올리지 말 것. 쿠키 = 자격증명이다.
#
#  commands:
#    list                 신원 목록
#    show   <name>        그 신원이 들고 있는 쿠키(호스트/이름/만료) 전부
#    new    <name>        빈 신원 생성
#    rotate <name>        태운 신원을 백업하고 비우기
#    shots                감지된 챌린지 스크린샷 목록
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDS="$ROOT/identities"
STATE="$ROOT/state"
VENV="$ROOT/.venv"
TAG="[identity]"

mkdir -p "$IDS" "$STATE"; chmod 700 "$IDS"

cmd_list() {
  echo "$TAG 신원 목록 ($IDS):"
  local found=0
  for d in "$IDS"/*/; do
    [ -d "$d" ] || continue; found=1
    local n ck size
    n="$(basename "$d")"
    ck="$d/Default/Cookies"
    size="$(du -sh "$d" 2>/dev/null | cut -f1)"
    if [ -f "$ck" ]; then
      local cnt
      cnt="$("$VENV/bin/python" - "$ck" <<'PY' 2>/dev/null || echo "?"
import sqlite3,shutil,sys,tempfile,os
src=sys.argv[1]
with tempfile.TemporaryDirectory() as td:
    p=os.path.join(td,'c'); shutil.copy2(src,p)
    print(sqlite3.connect(f"file:{p}?mode=ro",uri=True).execute("select count(*) from cookies").fetchone()[0])
PY
)"
      printf "  %-16s 쿠키 %-4s 크기 %s\n" "$n" "$cnt" "$size"
    else
      printf "  %-16s (신원 없음 - 쿠키 미생성) 크기 %s\n" "$n" "$size"
    fi
  done
  [ "$found" = 1 ] || echo "  (없음) — './identity.sh new default' 로 만들 것"
}

cmd_show() {
  local n="${1:?신원 이름}"
  local ck="$IDS/$n/Default/Cookies"
  [ -f "$ck" ] || { echo "$TAG $n: 아직 쿠키가 없다(신원 미형성)"; exit 0; }
  echo "$TAG $n 이 들고 있는 신원:"
  "$VENV/bin/python" - "$ck" <<'PY'
import sqlite3,shutil,sys,tempfile,os,datetime
with tempfile.TemporaryDirectory() as td:
    p=os.path.join(td,'c'); shutil.copy2(sys.argv[1],p)
    rows=sqlite3.connect(f"file:{p}?mode=ro",uri=True).execute(
        "select host_key,name,expires_utc,is_httponly,is_secure from cookies order by host_key,name").fetchall()
for h,nm,e,ho,se in rows:
    exp="세션"
    if e:
        try: exp=(datetime.datetime(1601,1,1)+datetime.timedelta(microseconds=e)).strftime('%Y-%m-%d')
        except Exception: exp="?"
    flags="".join(["H" if ho else "-", "S" if se else "-"])
    print(f"  {h:30s} {nm:24s} 만료={exp}  {flags}")
print(f"  총 {len(rows)}개")
PY
}

cmd_new() {
  local n="${1:?신원 이름}"
  mkdir -p "$IDS/$n"; chmod 700 "$IDS/$n"
  echo "$TAG 신원 '$n' 생성됨: $IDS/$n"
  echo "$TAG 1등급(지속 익명) 신원은 레인이 돌면서 저절로 쌓인다."
  echo "$TAG 챌린지를 띄우는 엔진은 사람을 부르지 않고 그냥 재웠다가 버린다."
}

cmd_rotate() {
  local n="${1:?신원 이름}"
  [ -d "$IDS/$n" ] || { echo "$TAG 그런 신원 없음: $n"; exit 1; }
  local bak="$IDS/../identities-burned/$n-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$(dirname "$bak")"; chmod 700 "$(dirname "$bak")"
  mv "$IDS/$n" "$bak"
  mkdir -p "$IDS/$n"; chmod 700 "$IDS/$n"
  echo "$TAG '$n' 을 $bak 로 치우고 새 빈 신원으로 교체했다."
  echo "$TAG ⛔ 백업은 확인 전까지 삭제하지 말 것."
}

cmd_shots() {
  echo "$TAG 감지된 챌린지 스크린샷:"
  ls -lt "$STATE"/challenge-*.png 2>/dev/null | head -20 || echo "  (없음)"
  echo; echo "$TAG 알림 로그 최근 10줄:"
  tail -10 "$STATE/notify.log" 2>/dev/null || echo "  (없음)"
}

case "${1:-}" in
  list)   cmd_list ;;
  show)   shift; cmd_show "$@" ;;
  new)    shift; cmd_new "$@" ;;
  rotate) shift; cmd_rotate "$@" ;;
  shots)  cmd_shots ;;
  *) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
