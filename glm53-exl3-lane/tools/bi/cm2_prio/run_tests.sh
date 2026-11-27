#!/bin/bash
# cm2 priority 하드 테스트: T1 pass / T2 reject→재시도 / T3 llamacpp 백엔드 / T4 서브에이전트 없는 요청
set -u
H=/tmp/claude-1000/-home-user/c68ea8bc-0b33-4cfc-8570-1d18eb6fc008/scratchpad/cmtest; cd $H
export UPSTREAM_KEY=$(bash -c 'source ~/work/dspark-serve-kvoff/glm53-exl3-lane/glm53-serve.env >/dev/null 2>&1; echo $VLLM_API_KEY')
CM=${CM:-$HOME/.local/bin/cm2}
WD=$H/work; rm -rf $WD; mkdir -p $WD/src; for i in 1 2 3 4 5; do echo "line $i" > $WD/src/f$i.txt; done
PROMPT_SUB='Agent 도구로 서브에이전트 2개를 동시에 띄워라. 첫째는 src 디렉터리 파일 개수를 세고, 둘째는 src/f3.txt 내용을 읽어 보고하게 해라. 두 결과를 합쳐 한 줄로 답해라.'
PROMPT_NOSUB='1부터 5까지 더한 값만 숫자로 답해라. 도구는 쓰지 마라.'
run() {  # name mode backend prompt
  local name=$1 mode=$2 backend=$3 prompt=$4
  python3 mkcfg.py $H/cfg_$name $backend >/dev/null
  rm -f $H/$name.jsonl
  python3 prio_proxy.py 18000 $mode $H/$name.jsonl & local pp=$!; sleep 1
  ( cd $WD && CODEMASON_CONFIG_DIR=$H/cfg_$name timeout 900 $CM -p -dsp "$prompt" ) > $H/$name.out 2>&1; local rc=$?
  kill $pp; wait $pp 2>/dev/null
  echo "== $name (mode=$mode backend=$backend) exit=$rc"
  echo "   answer: $(tail -c 300 $H/$name.out | tr '\n' ' ')"
  python3 - $H/$name.jsonl <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
chat = [r for r in rows if r.get("path", "").endswith("/chat/completions")]
c = collections.Counter((str(r.get("priority")), r.get("status")) for r in chat)
print("   chat requests:", len(chat), "| (priority,status) counts:", dict(c))
print("   sequence:", " ".join(f"{r.get('priority')}/{r.get('status')}" for r in chat))
PY
}
#run T1_pass pass vllm "$PROMPT_SUB"
run T2_reject reject vllm "$PROMPT_SUB"

#run T4_nosub pass vllm "$PROMPT_NOSUB"
