#!/usr/bin/env bash
# ============================================================================
#  GLM-5.3-Flash NVFP4 serve command — 컨테이너 안에서 /serve.sh 로 실행됨
#
#  이미지: ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
#          (sm_121 day-0 버그 7건이 이미 패치된 이미지. 우리가 커널을 빌드할
#           필요가 없다. LibertAI 경로는 커널 자가빌드 + 65K 컨텍스트라 폐기.)
#
#  근거 레시피
#   - github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark  (TP2, 262K, MTP)
#   - github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark (DFlash2)
#   - github.com/Libertai/glm53-flash-vllm-gb10 (결함 분석과 MoE 스케일 근거)
#
#  실적: MTP-4 21.8 t/s / DFlash2 K=7 46.9 t/s(수용률 74.1%), KV 581,040 토큰
# ============================================================================
set -euo pipefail

export XDG_CACHE_HOME=/cache/xdg
export TRITON_CACHE_DIR=/cache/triton-cache-rank${NODE_RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache-rank${NODE_RANK:-0}
# TileLang 은 mHC 커널을 기동 때 JIT 컴파일한다. 컴파일러 프로세스가 호스트 RAM 을
# 먹고 GB10 은 그게 곧 GPU 메모리라 첫 기동의 피크가 위험하다. 캐시를 영속화해서
# 두 번째 기동부터는 컴파일을 건너뛴다.
export TILELANG_CACHE_DIR=/cache/tilelang-rank${NODE_RANK:-0}
export TL_CACHE_DIR="$TILELANG_CACHE_DIR"
mkdir -p "$TILELANG_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[glm53] NODE_RANK=${NODE_RANK} GID=${NCCL_IB_GID_INDEX:-?} len=${MAX_MODEL_LEN}" \
     "seqs=${MAX_NUM_SEQS} gpu_mem=${GPU_MEM} moe=${MOE_BACKEND} spec=${SPEC_MODE}"

# --- speculative decoding -----------------------------------------------------
# mtp    : 모델 내장 MTP 층(layer45). 드래프터 다운로드 불필요. 기준선 21.8 t/s.
# dflash : incoai/GLM-5.3-Flash-DFlash2 (게이트 모델, 2.2GB). 46.9 t/s.
#          num_speculative_tokens 는 반드시 7 — 드래프터가 block 8 로 학습됐고
#          그 중 한 자리는 타깃 자신의 검증된 토큰이라 block_size-1 이어야 한다.
#          드래프터 KV 비용 ~0 (MLA 텐서 slot-share 기생). 일반 경로였으면
#          262K 요청에 27.92 GiB 를 더 먹는다.
# off    : speculative 없음. 문제 분리용.
# --- CUDA graph / 스케줄러 (속도 레버) ---------------------------------------
# EAGER=1 : --enforce-eager. 기동이 가장 안전하지만 느릴 수 있다.
# EAGER=0 : CUDA graph 사용. tonyd2wild 플래그 목록엔 enforce-eager 가 없고,
#           그쪽 speculative 없는 기준선이 21.8 t/s 인데 우리는 eager 로 14.07 이었다.
#           그 1.55배 격차의 유력 후보. 단 그래프 캡처가 기동 메모리 봉우리를 키운다.
if [ "${EAGER:-1}" = "1" ]; then EAGER_ARGS="--enforce-eager"; else EAGER_ARGS=""; fi
if [ "${ASYNC_SCHED:-0}" = "1" ]; then ASYNC_ARGS="--async-scheduling"; else ASYNC_ARGS=""; fi

# --- thinking / chat template (tonyd2wild 실제 런처 기준) -----------------------
# 그들은 --default-chat-template-kwargs '{"enable_thinking":false}' 로 생각을 끄고,
# 레포의 chat_template_mm.jinja 를 쓴다. 생각이 켜져 있으면 답 앞에 2,600자 안팎이
# 붙어 체감 지연이 커진다.
# EFFORT: 템플릿 2행이 reasoning_effort in ['low','high'] 아니면 'max' 로 접는다.
# 즉 아무것도 안 보내면 기본이 Max = 생각 폭주. 실사용 최적은 high (low 는
# 억제가 사실상 안 걸려 오히려 폭주 — DSV4 에서 low="" 였던 것과 같은 함정).
EFFORT="${EFFORT:-high}"
if [ "${THINKING:-false}" = "true" ]; then
  THINK_ARGS="--default-chat-template-kwargs {\"enable_thinking\":true,\"reasoning_effort\":\"${EFFORT}\"}"
else
  THINK_ARGS='--default-chat-template-kwargs {"enable_thinking":false}'
fi
if [ -f /patches/chat_template_mm.jinja ]; then
  TEMPLATE_ARGS="--chat-template /patches/chat_template_mm.jinja"
else
  TEMPLATE_ARGS=""
fi

# --- 멀티모달 --------------------------------------------------------------
# 2026-09-01 실측: 기동 중 "Encoder cache ... profiled with 1 video items of the
# maximum feature size" 직후 MemAvailable 이 2GB 로 떨어져 memwatch 가 선제 stop.
# GB10 은 통합메모리라 비전 프로파일링의 피크가 곧 박스 위험이다.
# MM=off  : 프로파일링을 건너뛰고 이미지/비디오 입력도 받지 않는다(텍스트 전용).
# MM=on   : 원래대로. 메모리 여유가 확인된 뒤에만.
# 포럼 실측(2x DGX Spark): --language-model-only 로 텍스트 전용 프로파일을 쓰면
# 약 15.7 GiB 를 아낀다. 통합메모리에선 그게 곧 기동 생존 여부다.
if [ "${MM:-off}" = "off" ]; then
  MM_ARGS="--language-model-only --skip-mm-profiling"
else
  MM_ARGS="--skip-mm-profiling"
fi

case "${SPEC_MODE}" in
  mtp)    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  dflash) SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":7}") ;;
  off)    SPEC_ARGS=() ;;
  *)      echo "ERR: SPEC_MODE=${SPEC_MODE} (mtp|dflash|off)"; exit 1 ;;
esac

exec vllm serve "${MODEL_PATH}" \
  --served-model-name GLM-5.3-Flash \
  --host 0.0.0.0 --port "${PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size "${TP_SIZE}" --pipeline-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEM}" \
  ${KV_BYTES:+--kv-cache-memory-bytes ${KV_BYTES}} \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --moe-backend "${MOE_BACKEND}" \
  --no-enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  ${EAGER_ARGS} ${ASYNC_ARGS} ${MM_ARGS} \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --trust-remote-code --distributed-executor-backend mp \
  ${THINK_ARGS} ${TEMPLATE_ARGS} \
  --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port 25000 \
  ${HEADLESS:+--headless} \
  "${SPEC_ARGS[@]}"

# ---------------------------------------------------------------------------
# 플래그 근거 (지우지 말 것)
#
# --moe-backend         우리 기본값은 flashinfer_cutlass. tonyd2wild 는 marlin 을
#                       쓰지만 그건 RedHatAI W4A4 체크포인트 기준이고, LibertAI 는
#                       marlin 이 FP4→FP16 repack 으로 전문가 메모리를 2배(~182
#                       GiB/노드) 먹어 박스를 wedge 시킨다고 기록했다(Hy3 에서 2회).
#                       우리 vcruz305 체크포인트는 input_scale 텐서 36,288 개가
#                       실재하고(값 0.00079~0.0251, 0 아님) 그 범위가 실제 캘리브
#                       구간 5.58e-04~3.72e-02 안이라, cutlass 경로가 정상 동작할
#                       조건을 이미 만족한다. marlin 은 메모리 위험까지 있어 후순위.
#
# ⚠️ VLLM_GLM53_MOE_INPUT_SCALE 을 절대 켜지 말 것.
#                       그건 input_scale 이 "없는" weight-only 체크포인트용 응급조치다.
#                       플러그인이 조건 없이 param.data.fill_(value) 로 덮어쓰기 때문에
#                       우리처럼 진짜 스케일이 있는 체크포인트에 쓰면 파괴적이다.
#                       게다가 그 상수 1.0 은 저자가 2026-08-30 에 철회했다 —
#                       중앙값 1.58e-03 의 632배라 16원소 블록 단위 underflow 로
#                       "repeats itself time to time" 가 나고 컨텍스트가 길수록 악화.
#
# --block-size          kpool(64) 과 MLA 정렬(128) 의 공배수여야 한다. 틀리면
#                       조용히 망가진다(에러 없이 decode 만 느려지거나 assert 실패).
#                       tonyd2wild 는 262K 에서 2304 를 쓴다.
# --kv-cache-dtype fp8  비우면 auto 가 fp8_ds_mla 로 정규화되고 그 캐시 커널이
#                       pe_dim==64 를 assert → NoPE 모델(qk_rope_head_dim=0)은
#                       영원히 못 만족한다. 이 이미지는 fp8 KV 를 이미 풀었다.
# --no-enable-prefix-   GLM 은 34개 KDA 선형어텐션 층이 순환 상태를 든다.
#   caching             상태 복원이 틀리면 컨텍스트를 잃고 최근 토큰을 베낀다.
# --kv-cache-memory-bytes  ⚠ 우리가 계속 죽던 구간을 없애는 플래그.
#                       기본값(미지정)이면 vLLM 이 프로파일링 forward 를 돌려 KV 크기를
#                       추정하는데, 2026-09-01 계측상 그 순간 6~8GB 가 2초 만에 잡히고
#                       (avail 11.6G -> 9.8G -> 3.8G) memwatch 가 선제 stop 했다.
#                       KV 를 바이트로 못 박으면 그 프로파일링을 건너뛴다.
#                       포럼 2x Spark 값: 비전 포함 9 GiB / 텍스트 전용 10 GiB.
# --no-enable-flashinfer-  ⚠ 이게 없으면 기동 중 프로파일링에서 박스가 죽는다.
#   autotune            autotuner 가 25~40GB 짜리 순간 할당을 하는데 통합메모리
#                       회계에 안 잡힌다(beastllama 레시피 명시). 2026-09-01 실측:
#                       이 플래그 없이 0.85/0.80/0.78 전부 TileLang JIT + 프로파일링
#                       구간에서 MemAvailable 2GB → memwatch 선제 stop.
# --enforce-eager       GB10 에서 CUDA graph 이득 ~1%(대역폭 바운드), 캡처 행 사례 있음.
#                       RTX PRO 6000 의 ~500% 이득은 PCIe collective 레이턴시
#                       바운드라서다 — 아키텍처가 다른 박스 설정을 이식하지 말 것.
# --swap-space          이 이미지(vLLM 0.1.dev20051)에는 없는 플래그다. 실측 확인:
#                       "vllm: error: unrecognized arguments: --swap-space 0".
#                       LibertAI env 예제의 SWAP_SPACE=0 은 다른 빌드 기준.
#
# 헬스체크는 /health 로. /v1/models 는 엔진이 죽어도 200 을 돌려준다(레시피 기록).
# ---------------------------------------------------------------------------
