#!/usr/bin/env bash
TOK=$(cat state/token)
q(){ printf '── %s\n' "$1"; local e; e=$(.venv/bin/python -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1")
  curl -s --max-time 120 -H "X-Index-Token: $TOK" "http://127.0.0.1:8097/search?q=$e&n=5" | .venv/bin/python fmt.py; }
q "mocflwsrrhbubkwdev qqxzpl"
q "RISC-V vector extension autovectorization"
q "국내 반도체 수출 규제 영향"
