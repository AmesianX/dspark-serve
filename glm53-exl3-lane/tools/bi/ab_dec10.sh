#!/bin/bash
L=~/work/dspark-serve-kvoff/glm53-exl3-lane; S=/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad/kvtest
for mode in off on off on; do
  cd $L
  if [ $mode = off ]; then sed -i 's/^: "\${GLM53_BI_/#BIOFF: "${GLM53_BI_/' glm53-serve.env; else sed -i 's/^#BIOFF: "\${GLM53_BI_/: "${GLM53_BI_/' glm53-serve.env; fi
  ./sync-lane-67.sh >/dev/null 2>&1; timeout 1500 ./vllm-glm53-exl3.sh RESTARTALL >/dev/null 2>&1
  echo "== $mode BI=$(docker exec vllm_glm53 printenv GLM53_BI_DENSE)/$(ssh 10.0.1.2 docker exec vllm_glm53 printenv GLM53_BI_DENSE)"
  python3 $S/kmeas8.py $mode
done
