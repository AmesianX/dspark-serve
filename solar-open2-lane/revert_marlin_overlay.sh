#!/usr/bin/env bash
# b12x 가 검증되면 실행: 내가 임의로 고친 vLLM 원본 오버레이를 제거해
# 레인을 "공식 이미지 + Upstage 포크 파일"만 남는 순정 상태로 되돌린다.
#  대상: overlay/model_executor/layers/quantization/utils/marlin_utils_fp4.py
#  (NVFP4→Marlin repack 파편 회수용. b12x 는 repack 자체가 없어 불필요.)
set -euo pipefail
LANE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
T="$LANE/overlay/model_executor/layers/quantization/utils"
[[ -f "$T/marlin_utils_fp4.py" ]] || { echo "이미 순정 상태"; exit 0; }
rm -f "$T/marlin_utils_fp4.py"
rmdir -p "$T" 2>/dev/null || true
rm -f "$LANE/launch.sh.bak"
rsync -a --delete "$LANE/" 10.0.1.2:"$LANE/"
echo "오버레이 제거 완료 — 남은 오버레이:"
find "$LANE/overlay" -name '*.py' | sed "s|$LANE/overlay/||" | sort
echo "※ RESTARTALL 로 재기동해야 반영된다."
