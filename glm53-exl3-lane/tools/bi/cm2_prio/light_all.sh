#!/bin/bash
H=/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad
export K=$(bash -c 'source ~/work/dspark-serve/glm53-exl3-lane/glm53-serve.env >/dev/null 2>&1; echo $VLLM_API_KEY')
echo "##### S1/S2"; python3 $H/cmtest/light.py pre
echo "##### S3 server priority ordering"; python3 $H/kvtest/prio_order.py
echo "##### S4 cm2 wire"; $H/cmtest/run_tests.sh 2>&1 | grep -v "^$"
echo "##### S5 restart -> NVMe"; cd ~/work/dspark-serve/glm53-exl3-lane && timeout 1500 ./vllm-glm53-exl3.sh RESTARTALL 2>&1 | tail -1; python3 $H/cmtest/light.py post
./vllm-glm53-exl3.sh STATUS 2>&1 | grep -E "health|가용"
