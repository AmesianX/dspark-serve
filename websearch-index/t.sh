#!/usr/bin/env bash
TOK=$(cat state/token)
q(){ printf '── %s\n' "$1"
  local e; e=$(.venv/bin/python -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1")
  curl -s --max-time 120 -H "X-Index-Token: $TOK" \
    "http://127.0.0.1:8097/search?q=$e&n=5" | .venv/bin/python fmt.py; }
q "speculative decoding"
q "GB10 unified memory"
echo "(같은 질의 재시도 — 승격 수확 반영)"; sleep 5
q "GB10 unified memory"
