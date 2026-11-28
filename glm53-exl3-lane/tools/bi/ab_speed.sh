#!/bin/bash
# 같은 세션 A/B: BI 스위치 끔 → 켬. 각각 kmeas + 82K 콜드 프리필 2회
set -u
L=~/work/dspark-serve/glm53-exl3-lane; S=/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad/kvtest
export K=$(bash -c "source $L/glm53-serve.env >/dev/null 2>&1; echo \$VLLM_API_KEY")
for mode in off on; do
  cd $L
  if [ $mode = off ]; then sed -i 's/^: "\${GLM53_BI_/#BIOFF: "${GLM53_BI_/' glm53-serve.env; else sed -i 's/^#BIOFF: "\${GLM53_BI_/: "${GLM53_BI_/' glm53-serve.env; fi
  ./sync-lane-67.sh 2>&1 | tail -1
  timeout 1500 ./vllm-glm53-exl3.sh RESTARTALL 2>&1 | tail -1
  echo "== $mode: BI env head=$(docker exec vllm_glm53 printenv GLM53_BI_DENSE) worker=$(ssh 10.0.1.2 docker exec vllm_glm53 printenv GLM53_BI_DENSE) eager=$(docker inspect vllm_glm53 --format '{{join .Config.Cmd " "}}' | grep -c enforce-eager)"
  python3 tools/kmeas.py $mode
  cd $S && python3 prefill_ab.py $mode 2
done
