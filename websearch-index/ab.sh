#!/usr/bin/env bash
TOK=$(cat state/token)
q(){ local e; e=$(.venv/bin/python -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1")
  curl -s --max-time 120 -H "X-Index-Token: $TOK" "http://127.0.0.1:8097/search?q=$e&n=8" | .venv/bin/python fmt.py; }
for Q in "RISC-V vector extension autovectorization" "국내 반도체 수출 규제 영향"; do
  echo "════ $Q"
  echo "1회차:"; q "$Q"
  echo "2회차 (승격 발동):"; q "$Q"
  echo "3회차 (구독 수확 후):"; sleep 6; q "$Q"
done
