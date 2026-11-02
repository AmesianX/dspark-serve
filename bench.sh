#!/usr/bin/env bash
# dspark decode t/s 벤치 — single-stream greedy, N회 반복 (MTP 재튜닝 때와 동일 방법론)
# usage: ./bench.sh [runs=3] [max_tokens=400]
set -uo pipefail
RUNS="${1:-3}"; MT="${2:-400}"
KEY="${KEY:?API 키를 환경변수로 지정하세요}"
MODEL="${MODEL:-DSV4-Flash}"
PROMPT="Explain how a CUDA kernel achieves memory coalescing, with a concrete matrix-transpose example. Be detailed."
total=0
for i in $(seq 1 "$RUNS"); do
  t0=$(date +%s.%N)
  r=$(curl -s -m 300 http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MT,\"temperature\":0}")
  t1=$(date +%s.%N)
  ct=$(jq -r '.usage.completion_tokens // 0' <<<"$r")
  [ "$ct" = "0" ] && { echo "run$i FAILED: $(jq -c '.error // .' <<<"$r" | head -c 200)"; continue; }
  tps=$(awk -v c="$ct" -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", c/(b-a)}')
  echo "run$i: $ct tok, ${tps} t/s"
  total=$(awk -v t="$total" -v x="$tps" 'BEGIN{print t+x}')
done
awk -v t="$total" -v n="$RUNS" 'BEGIN{printf "AVG: %.2f t/s\n", t/n}'
