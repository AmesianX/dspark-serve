#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash EXL3 + DFlash2, TP2 across two DGX Spark (GB10) boxes.
#
# Run the WORKER first (rank 1, ~25 s to join), then the HEAD (rank 0: API
# server + engine). The API serves on the head: http://<head-lan-ip>:8000.
# RELAUNCHING A LIVE CLUSTER: remove the head container BEFORE launching the
# new worker (a fresh worker rendezvouses with the old head's TCP store and
# dies of connection-reset when that head goes away). Then keep the worker->
# head gap under torch's 600 s rendezvous timeout.
# Community-derivative image: ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark
# (vLLM branch glm53-on-infernal + DFlash2 + ring draft-KV + EXL3 fused MoE
# baked; see docs/BUILD.md in the setup repo for full provenance).
#
# Per-box config lives in $HOME/.glm53-serve.env (written by install.sh);
# every value there uses the `: "${VAR:=...}"` idiom, so environment
# variables on the command line always win, e.g.:
#   KV_DTYPE=fp8_e4m3 ./launch-glm53-vllm-tp2.sh 0
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2.sh <0|1>   (0=head/API, 1=worker)}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

LANE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # [ours] 이 레인 디렉터리(양 노드 같은 경로)
GLM53_ENV="${GLM53_ENV:-$LANE_DIR/glm53-serve.env}"  # [ours] 레인 디렉터리의 설정
[[ -f "$GLM53_ENV" ]] && . "$GLM53_ENV"

# ---- topology (from .glm53-serve.env; defaults are the reference kit) ------
HEAD_RAIL_IP="${HEAD_RAIL_IP:-10.200.0.15}"      # head's IP on the 200GbE rail
WORKER_RAIL_IP="${WORKER_RAIL_IP:-10.200.0.33}"  # worker's IP on the same rail
NCCL_IF="${NCCL_IF:-enp1s0f0np0}"                # THIS box's rail interface
NCCL_HCA="${NCCL_HCA:-rocep1s0f0}"               # THIS box's RoCE HCA
NCCL_SUBNET="${NCCL_SUBNET:-10.200.0.0/24}"      # rail subnet for NCCL_IB_ADDR_RANGE
PORT="${PORT:-8000}"
MPORT="${MPORT:-29521}"

# ---- weights ---------------------------------------------------------------
# WEIGHTS_MODE=local : EXL3 weights on THIS box at $MODEL_HOST_PATH (default).
# WEIGHTS_MODE=nfs   : weights live on the WORKER; the worker exports them
#                      (containerized NFS, see install.sh) and the head mounts
#                      a docker NFS volume. This is the topology the reference
#                      kit runs in production (its head box lacks the disk).
#   CAUTION (measured on the reference pair 2026-08-31): weight load drives
#   the head through ALL available swap in BOTH modes (full 32 GiB consumed,
#   0.8 GiB MemFree floor; the worker pegs its 16 GiB). Stock 16 GiB swap
#   OOMs the head at ~90% of shard load (NV_ERR_NO_MEMORY) — grow swap to
#   >=32 GiB on both boxes, or set LOAD_FORMAT=instanttensor (direct I/O
#   sidesteps the page cache entirely). --nfs does NOT pace the load on a
#   fast rail. The drop-caches ritual below is still required.
WEIGHTS_MODE="${WEIGHTS_MODE:-local}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/models/glm53-exl3}"
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/glm53-dflash2-mxfp8}"
NFS_PORT="${NFS_PORT:-12049}"                    # worker's NFS export port (nfs mode)
VOL_NAME="${VOL_NAME:-exl3weights}"
MODEL_PATH="/models/glm53-exl3"

# ---- serving knobs (defaults = the validated production configuration) -----
IMAGE="${IMAGE:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1}"
KIT_VERSION="${KIT_VERSION:-v2.3-tier1}"   # release this launcher shipped with (install.sh stamps it from VERSION)
NAME="${NAME:-vllm_glm53}"
MAX_LEN="${MAX_LEN:-524288}"             # 500k default bank (2026-08-29);
                                         # 131072 was the pre-long-context
                                         # default and still works with
                                         # KV_DTYPE= (bf16)
SPEC="${SPEC:-dflash}"                   # dflash (default) | none; MTP>0 overrides
MTP="${MTP:-0}"                          # fallback: MTP=4 SPEC=none
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"      # trained block size 8 = 1 bonus + 7 masks
EAGER="${EAGER:-0}"                      # 0 = CUDA graphs (validated); 1 = eager
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"  # skip the max-size multimodal
                                         # dummy profile (required at long
                                         # MAX_LEN on GB10 unified memory;
                                         # text profile still runs). 0 restores
                                         # profiling for short-context boots.
BLOCK_SIZE="${BLOCK_SIZE:-2304}"         # KV block; with fp8 KV vLLM auto-bumps
                                         # to 4608 to keep the KDA state page
                                         # equal to the attention page (boot
                                         # log: "Setting attention block size
                                         # to 4608"). 2304 is the bf16 parity
                                         # point; must satisfy
                                         # block %% (index_kpool*64).
KV_DTYPE="${KV_DTYPE-fp8_ds_mla}"        # fp8_ds_mla (GLM_NEXT lane, baked
                                         # default since the v2 image): 528
                                         # B/token packed records, pool
                                         # 1,324,163 @524k = 2.53 banks;
                                         # math_500 91/100, lavd 15 EXACT.
                                         # NOTE ${VAR-} not ${VAR:-}: an
                                         # explicitly EMPTY KV_DTYPE= selects
                                         # bf16; only unset gets the default.
                                         # fp8_e4m3 = the 2026-08-29 lane
                                         # (1,435,070-token pool, 2.74 banks).
                                         # empty = bf16: pool ~520k tokens —
                                         # pair with MAX_LEN=131072.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"   # empty -> 14.4e9 since v2.2: its vLLM
                                         # right-sizes the sparse indexer's
                                         # prefill workspace (~2.5 GiB back per
                                         # rank), so 14.4e9 nets MORE headroom
                                         # than the old 12.4e9 pin (README
                                         # "Memory and context"; 14.9e9 = the
                                         # aggressive option). On v2.1 and
                                         # older images set
                                         # KV_CACHE_MEMORY=12400000000 (5.25
                                         # GiB measured head floor there).
                                         # "auto" -> vLLM budgeting. Do NOT
                                         # raise without re-measuring floors:
                                         # 13.4e9 on v2.1 bought +47k tokens
                                         # and collapsed the floor to 2.26 GiB.
MNBT="${MNBT:-8192}"                     # --max-num-batched-tokens: 112k-prompt
                                         # TTFT 93s->54.7s vs engine default
MM_CACHE_GB="${MM_CACHE_GB:-0}"        # mm processor cache (head-resident)
VISION="${VISION:-0}"                    # 1 = 이미지 입력 수용(비디오는 계속 닫힘).
                                         # 0 = 지금까지의 텍스트 전용(--language-model-only).
                                         # 근거와 상한 산식은 VISION.md
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-4096}"  # 이미지 1장이 만들 수 있는 비전 토큰
                                         # 상한. 반드시 MNBT 밑이어야 한다(아래 가드).
                                         # 체크포인트 기본 8000 = 7,921 토큰이고
                                         # MNBT 9,236 과의 여유가 14% 뿐이라 명시 고정.
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-1}"     # --limit-mm-per-prompt 의 image 개수
VISION_PIXEL_DTYPE="${VISION_PIXEL_DTYPE:-bf16}"  # 전처리 패치 텐서 dtype.
                                         # bf16 = 타워가 어차피 내리는 dtype 으로
                                         # cat 전에 미리 내림 -> 비트 동일(실측)이고
                                         # 전처리 피크가 42.8 -> 23.5 MiB/장(@896², 오프라인).
                                         # fp32 = 업스트림 그대로(A/B). VISION.md §10
VISION_LAZY_DECODE="${VISION_LAZY_DECODE:-0}"  # 1 = 이미지 로더가 헤더만 읽고 디코드를 미룬다.
                                         # 전처리가 한 장씩 디코드->리사이즈->해제(비트 동일).
                                         # 요청의 N장 원본이 동시에 디코드돼 사는 순간이 없어
                                         # 피크가 장수와 무관해진다(48MP x8: OOM -> 0.83 GiB). VISION.md §12
VISION_MAX_IMAGE_PIXELS="${VISION_MAX_IMAGE_PIXELS:-}"  # 원본 한 장 픽셀 상한(디코드 전 헤더로 검사).
                                         # 넘으면 400. 빈 값 = vLLM 기본 178,956,970.
                                         # 디코드는 한 장씩이라 피크는 이 값에 비례. VISION.md §14
GMU="${GMU:-0.85}"                       # gpu-memory-utilization; 0.88+ risks
                                         # unified-memory swap on GB10
MAX_SEQS="${MAX_SEQS:-4}"                # 4 at the 524k default (2.74 banks);
                                         # 6 was the 131k-era value
LOAD_FORMAT="${LOAD_FORMAT-instanttensor}"  # instanttensor (default,
                                         # validated 2026-08-31): safetensors
                                         # via pipelined-prefetch direct I/O,
                                         # bypassing the page cache. Measured
                                         # on the reference pair: model load
                                         # 385.8s -> 43.1s, peak swap 32G ->
                                         # 4G head / 16G -> 2.7G worker, pool
                                         # + decode + prefill at parity.
                                         # Sidesteps the swap OOM
                                         # (NV_ERR_NO_MEMORY) entirely. Note
                                         # ${VAR-}: explicit empty
                                         # LOAD_FORMAT= selects the engine
                                         # default loader (page-cache read;
                                         # needs >=32G swap on the head).
                                         # Expect a ~10-15 min post-boot
                                         # settling window with mildly noisy
                                         # TTFT while load-era pages fault
                                         # back in.
KDA_PREFILL="${KDA_PREFILL:-}"           # KDA prefill kernel for the 34 linear-
                                         # attention layers: triton | flashkda |
                                         # auto (flashkda when supported).
                                         # Empty = engine default. Pass the SAME
                                         # value on both ranks.
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/glm53-vllm-cache}"

if [[ "$KV_CACHE_MEMORY" == "auto" ]]; then
  KV_CACHE_MEMORY=""
elif [[ -z "$KV_CACHE_MEMORY" ]]; then
  KV_CACHE_MEMORY="14400000000"
fi

# Validate numeric knobs BEFORE any teardown: a typo'd value must not
# take down a healthy cluster only for vllm to refuse the relaunch.
_num_err() { echo "config error: $1 (nothing torn down)" >&2; exit 2; }
[[ "$GMU" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || _num_err "GMU='$GMU' must be in (0,1]"
[[ "$MAX_LEN" =~ ^[1-9][0-9]*$ && "$MAX_LEN" -le 1048576 ]] \
  || _num_err "MAX_LEN='$MAX_LEN' must be a positive int <= 1048576"
[[ "$MAX_SEQS" =~ ^[1-9][0-9]*$ && "$MAX_SEQS" -le 256 ]] \
  || _num_err "MAX_SEQS='$MAX_SEQS' must be a positive int <= 256"
[[ -z "$MNBT" || ( "$MNBT" =~ ^[1-9][0-9]*$ && "$MNBT" -le 131072 ) ]] \
  || _num_err "MNBT='$MNBT' must be empty or a positive int <= 131072"
[[ -z "${LONG_PREFILL:-}" || "${LONG_PREFILL:-}" =~ ^[1-9][0-9]*$ ]] \
  || _num_err "LONG_PREFILL='$LONG_PREFILL' must be empty or a positive int"
[[ -z "${MIXED_PREFILL_CAP:-}" || "${MIXED_PREFILL_CAP:-}" =~ ^(-1|0|[1-9][0-9]*)$ ]] \
  || _num_err "MIXED_PREFILL_CAP='$MIXED_PREFILL_CAP' must be empty, -1 (skip), 0 (off), or a positive int"
[[ -z "${MIXED_PREFILL_MAX_DEFER:-}" || "${MIXED_PREFILL_MAX_DEFER:-}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || _num_err "MIXED_PREFILL_MAX_DEFER='$MIXED_PREFILL_MAX_DEFER' must be empty or a non-negative int"
[[ -z "${MIXED_PREFILL_DECODE_WEIGHT:-}" || "${MIXED_PREFILL_DECODE_WEIGHT:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
  || _num_err "MIXED_PREFILL_DECODE_WEIGHT='$MIXED_PREFILL_DECODE_WEIGHT' must be empty or a non-negative number"

case "$NODE_RANK" in
  0) HOST_IP="$HEAD_RAIL_IP"; HEADLESS="" ;;
  1) HOST_IP="$WORKER_RAIL_IP"; HEADLESS="--headless" ;;
esac
ip -o addr show | grep -q "$HOST_IP/" || {
  echo "this host does not own $HOST_IP (rank $NODE_RANK; check HEAD_RAIL_IP/WORKER_RAIL_IP in $GLM53_ENV)" >&2
  exit 2
}

# RoCE GID index for the rail IP. Normally the RoCE v2 entry sits at index 3,
# but the kernel can rebuild the GID table at a different slot after a link
# flap (observed: a peer power-cycle left the v2 GID at index 4 with index 3
# empty, and NCCL pinned to 3 failed QP setup with "remote GID ::"). Derive
# the index from sysfs; NCCL_GID_INDEX overrides.
if [[ -z "${NCCL_GID_INDEX:-}" ]]; then
  _want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' ${HOST_IP//./ })
  for _g in "/sys/class/infiniband/$NCCL_HCA/ports/1/gids/"*; do
    [[ "$(cat "$_g" 2>/dev/null)" == "$_want" ]] || continue
    _idx="${_g##*/}"
    [[ "$(cat "/sys/class/infiniband/$NCCL_HCA/ports/1/gid_attrs/types/$_idx" 2>/dev/null)" == *"v2"* ]] || continue
    NCCL_GID_INDEX="$_idx"; break
  done
  if [[ -z "${NCCL_GID_INDEX:-}" ]]; then
    # Fail fast: a wrong index dies ~60s later in NCCL QP setup with an
    # opaque "remote GID ::". Dump the table so the fix is obvious.
    echo "error: no RoCE v2 GID matching $HOST_IP on $NCCL_HCA; table:" >&2
    for _g in "/sys/class/infiniband/$NCCL_HCA/ports/1/gids/"*; do
      _idx="${_g##*/}"
      echo "  gid $_idx: $(cat "$_g" 2>/dev/null) type=$(cat \
        "/sys/class/infiniband/$NCCL_HCA/ports/1/gid_attrs/types/$_idx" \
        2>/dev/null)" >&2
    done
    echo "override with NCCL_GID_INDEX=<idx> if the table disagrees" >&2
    exit 2
  fi
fi

EXTRA_VOLS=()
# [ours] NSYS=1 이면 nsys launch 로 감싼다(호스트 Nsight 마운트). 프로파일: docker exec vllm_glm53 /opt/nsys/target-linux-sbsa-armv8/nsys start/stop
NSYS_VOL=(); NSYS_CMD=()
if [[ "${NSYS:-0}" == "1" ]]; then NSYS_VOL=(-v /opt/nvidia/nsight-systems/2025.3.2:/opt/nsys:ro); NSYS_CMD=(/opt/nsys/target-linux-sbsa-armv8/nsys launch -t cuda,nvtx --cuda-graph-trace=node); fi
EXTRA_ENVS=()
# Optional kernel-selection override (A/B tests without editing this script),
# e.g. VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel to step the
# MXFP8 draft GEMM ladder down to Marlin W8A16.
[[ -n "${VLLM_DISABLED_KERNELS:-}" ]] && EXTRA_ENVS+=(-e "VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS")
# Admin/dev endpoints (/reset_prefix_cache etc.) for probe windows.
[[ -n "${VLLM_SERVER_DEV_MODE:-}" ]] && EXTRA_ENVS+=(-e "VLLM_SERVER_DEV_MODE=$VLLM_SERVER_DEV_MODE")
# 우리 오버레이: 비-EXL3 dense 층 온라인 블록FP8 (기본 off, 1일 때만)
[[ -n "${VLLM_EXL3_DENSE_FP8:-}" ]] && EXTRA_ENVS+=(-e "VLLM_EXL3_DENSE_FP8=$VLLM_EXL3_DENSE_FP8")
[[ -n "${VLLM_EXL3_LMHEAD_FP8:-}" ]] && EXTRA_ENVS+=(-e "VLLM_EXL3_LMHEAD_FP8=$VLLM_EXL3_LMHEAD_FP8")
[[ -n "${VLLM_EXL3_DENSE_FP8_KERNEL:-}" ]] && EXTRA_ENVS+=(-e "VLLM_EXL3_DENSE_FP8_KERNEL=$VLLM_EXL3_DENSE_FP8_KERNEL")
[[ -n "${VLLM_TEST_FORCE_FP8_MARLIN:-}" ]] && EXTRA_ENVS+=(-e "VLLM_TEST_FORCE_FP8_MARLIN=$VLLM_TEST_FORCE_FP8_MARLIN")
# [ours] 2026-09-03 디코드 커널 A/B: env 파일의 B12X_* 노브를 그대로 컨테이너로 통과
for _v in "${!B12X_@}"; do EXTRA_ENVS+=(-e "$_v=${!_v}"); done
# [ours] MTP 드래프트 INT4 헤드 / eh_proj FP8 (env 파일)
for _v in VLLM_EXL3_INT4_DRAFT_HEAD VLLM_EXL3_MOE_SWIGLU_CLAMP VLLM_EXL3_MTP_EH_PROJ_QUANT VLLM_EXL3_SKINNY_FP32 VLLM_EXL3_PREFILL_BF16 VLLM_EXL3_PREFILL_BF16_MIN_M VLLM_PREFIX_CACHE_RETENTION_INTERVAL; do [[ -n "${!_v:-}" ]] && EXTRA_ENVS+=(-e "$_v=${!_v}"); done
# [ours] 2노드 소형 all-reduce 지연 A/B (env 파일): NCCL_PROTO / NCCL_ALGO
# [ours] 드래프트 전용 축소 어휘 lm_head (MTP). 빈 값이면 끔
for _v in VLLM_GLM53_DRAFT_VOCAB; do [[ -n "${!_v:-}" ]] && EXTRA_ENVS+=(-e "$_v=${!_v}"); done
for _v in NCCL_PROTO NCCL_ALGO; do [[ -n "${!_v:-}" ]] && EXTRA_ENVS+=(-e "$_v=${!_v}"); done
# 우리 추가(Phase0 프로파일): 설정 시에만 통과. /cache 는 이미 호스트 ~/glm53-vllm-cache 에 마운트됨.
[[ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]] && EXTRA_ENVS+=(-e "VLLM_TORCH_PROFILER_DIR=$VLLM_TORCH_PROFILER_DIR")
# NVFP4 KV lane (KV_DTYPE=nvfp4_ds_mla): the rope-less 304 B/token record
# serves ONLY the dynamic per-token-scale mode; the engine refuses to boot
# without this env, so forward it whenever set.
[[ -n "${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-}" ]] && EXTRA_ENVS+=(-e "VLLM_NVFP4_MLA_DYNAMIC_SCALE=$VLLM_NVFP4_MLA_DYNAMIC_SCALE")
# Interim persistent_topk override for pre-fix images (see fork 97f13931e):
# point at a built topk_fix.so inside the container to lift the 1M-declaration
# and drafterless-524k persistent_topk oversubscription.
[[ -n "${GLM53_TOPK_FIX_SO:-}" ]] && EXTRA_ENVS+=(-e "GLM53_TOPK_FIX_SO=$GLM53_TOPK_FIX_SO")
# EXL3 loader override: VLLM_EXL3_STANDARD_FUSED=0 falls back to the
# per-expert parity load path (slow load, ~6 tok/s decode — debugging only).
[[ -n "${VLLM_EXL3_STANDARD_FUSED:-}" ]] && EXTRA_ENVS+=(-e "VLLM_EXL3_STANDARD_FUSED=$VLLM_EXL3_STANDARD_FUSED")
# v2.3 kit defaults (FINDINGS §18-19). On pre-v2.3 images the kit says to set
# VLLM_USE_B12X_FP8_GEMM=0 (no per-M fallback, acceptance loss at 4 streams).
#
# [ours] 2026-10-30 이미지 코드 전수 추적. 원래 주석("DFlash2 드래프터용 GEMM, 0 으로 꺼도 됨")은
# 우리 구성에서 틀리다. VLLM_USE_B12X_FP8_GEMM 이 실제로 하는 일:
#  (1) FP8/MXFP8 선형층의 b12x GEMM 선택 — 우리 레인에선 효과 없음. dense 층은 overlay/exl3.py 가
#      force_kernel=MarlinFP8ScaledMMLinearKernel 로 강제해 kernels/linear/__init__.py:630 에서 먼저
#      반환되고, MXFP8 층(DFlash2 드래프터)은 MTP 구성에 없다.
#  (2) ⚠️ b12x CUDA 그래프 prewarm + 캡처 중 커널 해석 고정 (compilation/b12x_capture.py:14).
#      그 기능을 켜는 OR 조건 6개(SPARSE_INDEXER/MHC/FP8_GEMM/WO_PROJECTION/MOE/MINIMAX_M3_MSA) 중
#      우리 env 에서 켜진 건 이것 하나뿐이다. 켜져 있으면 V2 러너가 FULL 캡처 직전에 같은 상태로
#      eager forward 를 한 번 돌려 b12x 커널(B12X_MLA_SPARSE 어텐션, trellis MoE)을 해석한 뒤
#      캡처한다(v1/worker/gpu/cudagraph_utils.py:519-534 — scratch 뷰 계약상 미해석 launch 가
#      그래프에 녹화되지 않게).
#      → 잔재로 보고 0 으로 끄지 말 것. 끄면 무엇이 깨지는지는 측정하지 않았다.
#  (3) 부수효과: 1 이면 DeepGEMM FP8 워밍업을 건너뛴다(warmup/deep_gemm_warmup.py:365). 우리는 DeepGEMM 미사용.
# VLLM_DFLASH_FP8_DRAFT_HEAD 는 SPEC=dflash 경로(models/qwen3_dflash2.py:330)에서만 읽힌다.
# 우리 env 파일이 0 으로 고정한다(Marlin 패킹 lm_head 와 충돌).
B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
DFLASH_FP8_HEAD="${VLLM_DFLASH_FP8_DRAFT_HEAD:-1}"
EXTRA_ENVS+=(-e "VLLM_USE_B12X_FP8_GEMM=$B12X_FP8_GEMM" -e "VLLM_DFLASH_FP8_DRAFT_HEAD=$DFLASH_FP8_HEAD")
[[ -n "${VLLM_B12X_MXFP8_MAX_M:-}" ]] && EXTRA_ENVS+=(-e "VLLM_B12X_MXFP8_MAX_M=$VLLM_B12X_MXFP8_MAX_M")
# NCCL channel count for the TP2 all-reduces (v2.3 default 8, fixed min=max).
# NCCL_NCHANNELS=0 leaves NCCL's own choice (the v2.2 behaviour).
NCCL_NCHANNELS="${NCCL_NCHANNELS:-8}"
NCCL_CHANNEL_ENVS=()
if [[ "$NCCL_NCHANNELS" != "0" && -n "$NCCL_NCHANNELS" ]]; then
  NCCL_CHANNEL_ENVS=(-e "NCCL_MIN_NCHANNELS=$NCCL_NCHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_NCHANNELS")
fi
if [[ "$WEIGHTS_MODE" == "local" || "$NODE_RANK" == "1" ]]; then
  test -f "$MODEL_HOST_PATH/config.json" || {
    echo "모델이 없다: $MODEL_HOST_PATH" >&2
    echo "  원본 모델(164GiB)을 이 경로에 받을 것 — https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw" >&2
    echo "  받는 명령은 저장소 루트 README.md §5.5 (abliterated 를 이미 가졌다면 §6). 경로를 바꿨다면 glm53-serve.env 의 MODEL_HOST_PATH" >&2
    exit 2
  }
  MODEL_VOL="$MODEL_HOST_PATH:$MODEL_PATH:ro"
else
  # nfs mode, head: mount the worker's containerized export (fsid=0 -> device=:/)
  docker volume inspect "$VOL_NAME" >/dev/null 2>&1 || docker volume create --driver local \
    --opt type=nfs --opt "o=addr=$WORKER_RAIL_IP,ro,vers=4.2,rsize=1048576,port=$NFS_PORT" \
    --opt device=:/ "$VOL_NAME" >/dev/null
  MODEL_VOL="$VOL_NAME:$MODEL_PATH:ro"
fi

# UTF-8 바이트 가드(overlay/vllm_utf8_guard.py). 글자 중간에서 바이트 흐름이
# UTF-8 오토마톤을 벗어나는 것을 샘플러 단계에서 막는다 — 한글 음절 선행 바이트
# 뒤에 라틴 토큰이 붙어 U+FFFD 가 되던 버그(2026-09-14 규명). 0 으로 끄면
# 프로세서도 안 붙고 컨테이너 안 환경변수로도 비활성화된다.
UTF8_GUARD="${UTF8_GUARD:-1}"
UTF8_GUARD_ARGS=()
# 커스텀 로짓 프로세서로 등록하지 않는다: 등록하는 순간 vLLM 이 모델 러너를
# V2→V1 으로 내리고, 이 모델은 그 경로에서 뜨지 못한다(드래프터 config,
# KV 그룹 assert). 대신 V2 샘플러가 가드를 직접 부른다(overlay/gpu_sampler.py
# → v1/worker/gpu/sample/sampler.py, MTP 검증도 이 경로를 탄다). 끄기는 아래 환경변수 하나로.
UTF8_GUARD_ENVS=(-e "VLLM_UTF8_GUARD=$UTF8_GUARD" -e "VLLM_UTF8_GUARD_TOKENIZER=$MODEL_PATH")

SPEC_ARGS=()
if [[ "$MTP" != "0" ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
elif [[ "$SPEC" == "dflash" ]]; then
  { test -f "$DFLASH_DIR/config.json" && test -f "$DFLASH_DIR/model.safetensors"; } || {
    echo "DFlash2 draft weights not found or incomplete at $DFLASH_DIR (run install.sh, or SPEC=none)" >&2
    exit 2
  }
  EXTRA_VOLS+=(-v "$DFLASH_DIR:/models/glm53-dflash2:ro")
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/glm53-dflash2\",\"num_speculative_tokens\":$DFLASH_TOKENS}")
fi
# [ours] 2026-10-01 abliteration 런타임 스위치(overlay/ab_switch.py, README §4.10).
# 원본(brandonmusic tr3-4bpw)을 서빙하고 검열 해제는 여기서 런타임으로 켠다. 어댑터를 물리고 플러그인을
# allowlist 에 넣는다. 빈 값이면 훅이 통째로 비활성 — 원본 그대로 돈다.
# 어댑터는 원본 위에서만 의미가 있다. 이미 abliterated 인 체크포인트에 얹으면
# 편집이 두 번 들어가므로, 대조 실험으로 그쪽을 띄웠다면 스위치를 끄고 간다.
if [[ -n "${GLM53_AB_LORA:-}" && "${MODEL_HOST_PATH:-}" == *Abliterated* ]]; then
  echo "[ab] 이미 abliterated 인 체크포인트라 어댑터를 쓰지 않는다." >&2
  GLM53_AB_LORA=""
fi
if [[ -n "${GLM53_AB_LORA:-}" ]]; then
  test -f "$GLM53_AB_LORA/ab_lora.safetensors" || {
    echo "검열 해제 어댑터가 없다: $GLM53_AB_LORA/ab_lora.safetensors" >&2
    echo "  만드는 법은 저장소 루트 README.md §5.5(2) (tools/fetch_oproj.py → tools/build_ab_lora.py)." >&2
    echo "  어댑터 없이 원본만 띄우려면 glm53-serve.env 에서 GLM53_AB_LORA= (빈 값)" >&2
    exit 1
  }
  EXTRA_VOLS+=(-v "$GLM53_AB_LORA:/ab-lora:ro")
  EXTRA_ENVS+=(-e "GLM53_AB_LORA=/ab-lora" -e "GLM53_AB_SCALE=${GLM53_AB_SCALE:-0}")
  EXTRA_ENVS+=(-e "VLLM_PLUGINS=${VLLM_PLUGINS:-glm53_ab,glm53_ab_worker}")
fi

KDA_ARGS=()
[[ -n "$KDA_PREFILL" ]] && KDA_ARGS=(--kda-prefill-backend "$KDA_PREFILL")
# Prefix-cache hash granularity. Default 512 since v2.2 (2304 in v2.1): fine-
# grained hashing — sub-block prefix hits, producer-tail state reuse, agentic
# warm turns ~3-4x, warm-hit floors on a 512 grid at no measured hash-CPU cost
# (FINDINGS §17). Needs the v2.1+ image — the CoW/backoff fixes are engine-
# fatal-absent on v2 and earlier. Explicit-empty (PREFIX_MATCH_UNIT=) restores
# the coarse 4608 engine default.
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT-512}"
# Old-image guard: v1-dflash2 and v2-glmnext engines predate fine-grained
# prefix reuse (any unit below the 4608-token block asserts in
# copy_kv_cache_blocks_inplace on the first request) and the 14.4 GB KV
# default (their indexer workspace is not right-sized). Refuse the
# combination instead of booting into a crash; the fixes are to drop the
# IMAGE pin (current release) or to set PREFIX_MATCH_UNIT= and
# KV_CACHE_MEMORY=12400000000 for the old image.
if [[ "$IMAGE" == *:v1-dflash2* || "$IMAGE" == *:v2-glmnext* ]]; then
  if [[ -n "$PREFIX_MATCH_UNIT" && "$PREFIX_MATCH_UNIT" -lt 4608 ]]; then
    echo "error: image $IMAGE predates fine-grained prefix reuse; PREFIX_MATCH_UNIT=$PREFIX_MATCH_UNIT would assert at the first request." >&2
    echo "       Either unset IMAGE in .env (use the current release) or set PREFIX_MATCH_UNIT= (coarse) and KV_CACHE_MEMORY=12400000000 for this image." >&2
    exit 2
  fi
  if [[ "${KV_CACHE_MEMORY:-}" == "" || "$KV_CACHE_MEMORY" -gt 12400000000 ]]; then
    echo "warning: image $IMAGE was validated with KV_CACHE_MEMORY=12400000000; using that instead of the 14.4 GB default." >&2
    KV_CACHE_MEMORY=12400000000
  fi
elif [[ -n "${GLM53_TOPK_FIX_SO:-}" ]]; then
  # The persistent_topk retry is baked since v2-glmnext; a v1-era topk_fix.so
  # would be loaded over the image's own op (and can fail the first decode).
  echo "warning: GLM53_TOPK_FIX_SO is set but the fix is baked into $IMAGE; ignoring the v1-era override." >&2
  unset GLM53_TOPK_FIX_SO
fi
[[ -n "$PREFIX_MATCH_UNIT" ]] && KDA_ARGS+=(--prefix-match-unit "$PREFIX_MATCH_UNIT")
# Mixed-prefill decode floor: while anything is decoding, skip (-1) or cap
# (N tokens) peer prefill chunks; solo prefills keep full MNBT chunks.
# Empty = engine default (off). MIXED_PREFILL_MAX_DEFER bounds consecutive
# deferrals before one unrestricted step (anti-starvation; engine default 8).
# Long-prefill chunk cap (--long-prefill-token-threshold): bounds each
# prefill's per-step chunk so a peer prefill fits in the same step's budget.
# Must be a multiple of the 2304 block grid (glm5next rejects < 2304).
# Empty = engine default (off).
LONG_PREFILL="${LONG_PREFILL:-}"
[[ -n "$LONG_PREFILL" ]] \
  && KDA_ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL")
MIXED_PREFILL_CAP="${MIXED_PREFILL_CAP:-}"
[[ -n "$MIXED_PREFILL_CAP" ]] \
  && KDA_ARGS+=(--mixed-prefill-token-cap "$MIXED_PREFILL_CAP")
MIXED_PREFILL_MAX_DEFER="${MIXED_PREFILL_MAX_DEFER:-}"
[[ -n "$MIXED_PREFILL_MAX_DEFER" ]] \
  && KDA_ARGS+=(--mixed-prefill-max-defer-steps "$MIXED_PREFILL_MAX_DEFER")
# Dynamic time-share gate: weight w > 0 admits a mixed prefill chunk only
# after decodes got w*D/P chunk-walls of decode-only time (1.0 = equal
# per-request share). With MIXED_PREFILL_CAP=-1 the gate skips; with a
# positive CAP (> 64; v2.2+) the gate paces WHEN a peer chunk runs and the
# cap sizes it — recommended interactive setting: WEIGHT=1.0 CAP=512
# (0.57 s decode stalls, FINDINGS §17). Supersedes MAX_DEFER.
MIXED_PREFILL_DECODE_WEIGHT="${MIXED_PREFILL_DECODE_WEIGHT:-}"
[[ -n "$MIXED_PREFILL_DECODE_WEIGHT" ]] \
  && KDA_ARGS+=(--mixed-prefill-decode-weight "$MIXED_PREFILL_DECODE_WEIGHT")
KV_ARGS=()
[[ -n "$KV_DTYPE" ]] && KV_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
# Attention backend. B12X_MLA_SPARSE (GLM_NEXT lane, baked default since the
# v2 image) pairs with KV_DTYPE=fp8_ds_mla. Explicitly EMPTY ATTN_BACKEND=
# restores the fork's auto-selection (FLASHINFER_MLA_SPARSE_SM90 on these
# boxes) — pair with a non-GLM_NEXT KV_DTYPE.
ATTN_BACKEND="${ATTN_BACKEND-B12X_MLA_SPARSE}"
[[ -n "$ATTN_BACKEND" ]] && KV_ARGS+=(--attention-backend "$ATTN_BACKEND")
# Layers exempt from KV quantization. The GLM_NEXT fp8_ds_mla lane needs the
# DFlash2 draft ring-KV layers on bf16 (the packed 528 B record is
# target-MLA-only): sliding_window (default with the lane) — the draft ring
# is sliding-window-typed, and the single token avoids the CLI's list
# parsing (a comma-joined index string arrives as one unmatched element).
# Explicitly EMPTY KV_SKIP_LAYERS= disables the exemption (non-lane KV).
# [ours] 2026-10-30: 우리 MTP 구성에는 sliding_window 층이 없다(config.json layer_types 에
# linear_attention / deepseek_sparse_attention 뿐) → 실제로 bf16 으로 빠지는 층은 0 개다.
# 그러나 이 목록이 비어 있지 않으면 platforms/interface.py:650 의 Phase 3 블록 정렬
# (_align_heterogeneous_kv_block_size)이 돌고, 로그 없이 mamba_page_size_padded 를
# block_size x 1152B(FullAttention 공식) 로 다시 쓴다. 이번 기동은 블록 크기를 바꾸지 않았다.
# mamba 배치·KV 토큰 수가 바뀌는지는 미확인 — 빈 값으로 바꾸기 전에 기동 A/B 로
# "GPU KV cache size" 토큰 수(현 1,356,053)를 비교할 것.
KV_SKIP_LAYERS="${KV_SKIP_LAYERS-sliding_window}"
[[ -n "$KV_SKIP_LAYERS" ]] && KV_ARGS+=(--kv-cache-dtype-skip-layers "$KV_SKIP_LAYERS")
[[ -n "$KV_CACHE_MEMORY" ]] && KV_ARGS+=(--kv-cache-memory "$KV_CACHE_MEMORY")
EAGER_ARGS=()
[[ "$EAGER" != "0" ]] && EAGER_ARGS=(--enforce-eager)
# ---------------------------------------------------------------------------
# 멀티모달 게이트. 기본 VISION=0 은 지금까지와 같은 텍스트 전용이다.
#
# 인코더 예산이 "자라는" 구조 (vllm/v1/core/encoder_cache_manager.py):
#     encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_item)
#     encoder_cache_size     = max(encoder_cache_size,           max_tokens_per_item)
# 두 스케줄러 값의 기본값은 모두 MNBT 다. 즉 켜둔 모달리티 중 '한 아이템'이
# MNBT 를 넘으면 예산이 그 아이템 크기까지 들린다. ViT 는 그 예산을 patch 로
# 환산해 한 스텝에 통째로 돌리므로 활성화도 같은 배수로 커진다. 상한을 KV 나
# GMU 로는 못 막는다 — 아이템 크기에서 막아야 한다.
#
#   이미지 1장 = 7,921 토큰 (체크포인트 max_image_tokens=8000) < MNBT 9,236  -> 안 들림
#   비디오 1개 = 32,242 토큰 (_MAX_VIDEO_TOKENS=30000 캡 후)   > MNBT 9,236  -> 3.5배 들림
#
# 그래서 (a) 비디오는 limit 0 으로 닫고 (b) 이미지 아이템 크기를 MNBT 밑으로
# 못 박는다. enable_mm_embeds 는 기본 False 로 둬야 한다: True 면 limit 0 인
# 모달리티도 active 로 들어와 위 max() 에 계속 참여한다(비디오를 닫아도 들린다).
#
# max_pixels 와 max_image_tokens 를 둘 다, 같은 값으로 넘기는 이유: 예산 계산은
# _get_image_max_pixels() -> max_pixels 만 보고, 런타임 전처리 기하는
# images_kwargs 의 max_image_tokens 만 본다. 어긋나면 예산 < 실제가 되어
# 인코더 캐시가 넘친다(upstream vllm#34040 과 같은 모양).
# 구 노브 호환: LANGUAGE_MODEL_ONLY 는 VISION 으로 대체됐다. 조용히 무시하지 않는다.
VISION_ENVS=()
if [[ -n "${LANGUAGE_MODEL_ONLY:-}" ]]; then
  echo "error: LANGUAGE_MODEL_ONLY 는 VISION 으로 대체됐다 (LANGUAGE_MODEL_ONLY=0 -> VISION=1)." >&2
  exit 1
fi
if [[ "$VISION" == "0" ]]; then
  CHAT_TEMPLATE_SRC=chat_template.jinja
  [[ "$SKIP_MM_PROFILING" != "0" ]] && EAGER_ARGS+=(--skip-mm-profiling)
  EAGER_ARGS+=(--language-model-only)
else
  # 텍스트 전용 템플릿은 이미지 아이템을 토큰이 아니라 "처리할 수 없다" 문장으로
  # 치환한다(chat_template.jinja 의 visible_text). 엔진이 받아줘도 그대로면
  # 이미지가 삼켜지므로 비전용 템플릿으로 바꿔 끼운다.
  CHAT_TEMPLATE_SRC=chat_template.vision.jinja
  # 예산 상한은 MNBT 와의 비교로만 증명된다. 숫자가 아니면 비교가 성립하지 않는다.
  [[ "$MNBT" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: VISION=1 에는 MNBT 가 양의 정수여야 한다(지금: '${MNBT}') — 인코더 예산 상한을 검증할 수 없다." >&2
    exit 1
  }
  [[ "$VISION_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: VISION_MAX_TOKENS 가 양의 정수여야 한다(지금: '${VISION_MAX_TOKENS}')." >&2
    exit 1
  }
  if (( VISION_MAX_TOKENS >= MNBT )); then
    echo "error: VISION_MAX_TOKENS=$VISION_MAX_TOKENS 가 MNBT=$MNBT 이상이다." >&2
    echo "       이러면 인코더 예산이 아이템 크기까지 들려 ViT 활성화가 같은 배수로 커진다(OOM 경로)." >&2
    exit 1
  fi
  if (( VISION_MAX_TOKENS * 5 > MNBT * 4 )); then
    echo "warning: VISION_MAX_TOKENS=$VISION_MAX_TOKENS 가 MNBT=$MNBT 의 80% 를 넘는다 — 여유가 얇다." >&2
  fi
  # 비전 토큰 1개가 덮는 픽셀 수 = temporal_patch_size * (patch_size*merge_size)^2
  _VIS_PX_PER_TOK="$(python3 -I -c "
import json
c = json.load(open('$MODEL_HOST_PATH/config.json'))['vision_config']
print(c['temporal_patch_size'] * (c['patch_size'] * c['spatial_merge_size']) ** 2)
")" || { echo "error: $MODEL_HOST_PATH/config.json 의 vision_config 를 읽지 못했다." >&2; exit 1; }
  _VIS_MAX_PX=$(( VISION_MAX_TOKENS * _VIS_PX_PER_TOK ))
  EAGER_ARGS+=(
    --limit-mm-per-prompt "{\"image\": $VISION_MAX_IMAGES, \"video\": 0}"
    --mm-processor-kwargs "{\"max_pixels\": $_VIS_MAX_PX, \"max_image_tokens\": $VISION_MAX_TOKENS}"
  )
  # 상한을 걸었으니 더미 MM 프로파일은 항상 돈다(--skip-mm-profiling 안 붙임): ViT 피크가 안 맞으면
  # 트래픽이 아니라 기동에서 죽는다. 헤드 호스트 전처리는 이 프로파일이 재지 않는다(VISION.md §11).
  case "$VISION_PIXEL_DTYPE" in
    bf16|bfloat16|fp32|float32) ;;
    *) echo "error: VISION_PIXEL_DTYPE='$VISION_PIXEL_DTYPE' 를 모른다 (bf16|fp32)." >&2; exit 1 ;;
  esac
  VISION_ENVS=(-e "GLM53_MM_PIXEL_DTYPE=$VISION_PIXEL_DTYPE")
  case "$VISION_LAZY_DECODE" in
    0|1) ;;
    *) echo "error: VISION_LAZY_DECODE='$VISION_LAZY_DECODE' 는 0|1." >&2; exit 1 ;;
  esac
  VISION_ENVS+=(-e "GLM53_MM_LAZY_DECODE=$VISION_LAZY_DECODE")
  if [[ -n "$VISION_MAX_IMAGE_PIXELS" ]]; then
    [[ "$VISION_MAX_IMAGE_PIXELS" =~ ^[1-9][0-9]*$ ]] || { echo "error: VISION_MAX_IMAGE_PIXELS='$VISION_MAX_IMAGE_PIXELS' 는 양의 정수." >&2; exit 1; }
    VISION_ENVS+=(-e "VLLM_MAX_IMAGE_PIXELS=$VISION_MAX_IMAGE_PIXELS")
  fi
  echo "[VISION] image<=$VISION_MAX_IMAGES, <=$VISION_MAX_TOKENS tok/img (max_pixels=$_VIS_MAX_PX), video=0, MNBT=$MNBT, pixel_dtype=$VISION_PIXEL_DTYPE, lazy_decode=$VISION_LAZY_DECODE, max_px=${VISION_MAX_IMAGE_PIXELS:-default}" >&2
fi
MNBT_ARGS=()
[[ -n "$MNBT" ]] && MNBT_ARGS=(--max-num-batched-tokens "$MNBT")
LOAD_ARGS=()
[[ -n "$LOAD_FORMAT" ]] && LOAD_ARGS=(--load-format "$LOAD_FORMAT")

# Hotfix hooks: any file under $HOME/glm53-hotfix (mirroring the vllm package
# tree) or $HOME/glm53-hotfix-fi (flashinfer tree) is bind-mounted over the
# installed package — fast community debugging without an image rebuild. The
# dirs do not exist in a normal install; REMOVE them (never empty in place)
# once a fix is folded into an image.
HOTFIX_DIR="$HOME/glm53-hotfix"
if [[ -d "$HOTFIX_DIR" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#"$HOTFIX_DIR"/}"
    EXTRA_VOLS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
  done < <(find "$HOTFIX_DIR" -type f -print0)
fi
FI_HOTFIX_DIR="$HOME/glm53-hotfix-fi"
if [[ -d "$FI_HOTFIX_DIR" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#"$FI_HOTFIX_DIR"/}"
    EXTRA_VOLS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/flashinfer/$rel:ro")
  done < <(find "$FI_HOTFIX_DIR" -type f -print0)
fi
# b12x is bumped as a whole package (kernel contracts span many files), so
# this hook binds the directory, not per-file: put a complete b12x/ package
# tree at $HOME/glm53-hotfix-b12x (the package dir itself, containing
# __init__.py). Same rule as the others: REMOVE the dir once folded into an
# image.
B12X_HOTFIX_DIR="$HOME/glm53-hotfix-b12x"
if [[ -d "$B12X_HOTFIX_DIR" ]]; then
  EXTRA_VOLS+=(-v "$B12X_HOTFIX_DIR:/usr/local/lib/python3.12/dist-packages/b12x:ro")
fi
# Overlays are debugging aids that were needed on v1-dflash2 only. Left over
# from an old install they silently shadow the current image's code, so say
# so loudly every launch (the installer refuses to upgrade over them).
_overlay_n=$(( $(find "$HOTFIX_DIR" "$FI_HOTFIX_DIR" -type f 2>/dev/null | wc -l) + $( [[ -d "$B12X_HOTFIX_DIR" ]] && echo 1 || echo 0) ))
if (( _overlay_n > 0 )); then
  echo "WARNING: $_overlay_n hotfix overlay file(s)/tree(s) from \$HOME/glm53-hotfix* are bound over $IMAGE." >&2
  echo "         If they are not a deliberate current-image debugging overlay, move them aside:" >&2
  echo "         mv ~/glm53-hotfix ~/glm53-hotfix.retired-\$(date +%F) (same for -fi / -b12x) and relaunch." >&2
fi

# Persist JIT compile caches (Triton, FlashInfer, b12x CuTeDSL, vLLM
# torch.compile) across container recreates — hash-keyed, so stale entries
# are inert. Without this every relaunch re-JITs from scratch.
mkdir -p "$CACHE_HOST_PATH" \
  "$CACHE_HOST_PATH/jit/triton" "$CACHE_HOST_PATH/jit/flashinfer" \
  "$CACHE_HOST_PATH/jit/b12x" "$CACHE_HOST_PATH/jit/vllm" \
  "$CACHE_HOST_PATH/jit/tilelang" "$CACHE_HOST_PATH/jit/inductor" \
  "$CACHE_HOST_PATH/jit/nv-compute"
docker rm -f "$NAME" 2>/dev/null || true

# Memory preflight. The defaults (KV 14.4 GB, GMU 0.85) were validated on
# lightly loaded headless boxes with < 6 GB of system memory in use before
# launch; GB10 unified memory swap-wedges rather than failing cleanly when
# that headroom is gone. Refuse to launch above MEM_USED_MAX_GB (default 6)
# and name the consumers; MEM_USED_MAX_GB=0 disables the check.
MEM_USED_MAX_GB="${MEM_USED_MAX_GB:-6}"
if [[ "$MEM_USED_MAX_GB" != "0" ]]; then
  # A just-removed serving container (~110 GB) takes several seconds to hand
  # its memory back; poll the free(1) "used" column (total - free -
  # buffers/cache) for up to 60 s before judging.
  _used_mb=0
  # Memory from a container that was just removed is returned over tens of
  # seconds; poll for up to 120 s before deciding.
  for _i in $(seq 1 60); do
    _used_mb=$(free -m | awk 'NR==2{print $3}')
    (( _used_mb <= MEM_USED_MAX_GB * 1024 )) && break
    sleep 2
  done
  if (( _used_mb > MEM_USED_MAX_GB * 1024 )); then
    echo "error: $((_used_mb/1024)).$(( (_used_mb%1024)*10/1024 )) GB of system memory is in use before launch (limit ${MEM_USED_MAX_GB} GB)." >&2
    echo "       The validated memory defaults assume a lightly loaded box. Stop other services" >&2
    echo "       (a docker pull/load or a desktop session in progress inflates this number)," >&2
    echo "       lower KV_CACHE_MEMORY (~90k pool tokens per GB), or raise/disable the check:" >&2
    echo "       MEM_USED_MAX_GB=<gb> ./launch-glm53-vllm-tp2.sh $NODE_RANK   (0 = skip)" >&2
    echo "       top consumers:" >&2
    ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<=7 {printf "         %6d MB  %s\n", $1/1024, $2}' >&2
    exit 2
  fi
  echo "memory preflight: $((_used_mb/1024)) GB in use (limit ${MEM_USED_MAX_GB} GB) — ok"
fi

# GB10 pre-launch ritual — a hot page cache at model-load time wedges the box
# into swap (unified-memory starvation). Root via privileged docker; falls
# back to passwordless sudo where available.
sync
docker run --rm --privileged alpine sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" >/dev/null 2>&1 \
  || { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; }

# [ours] 2026-11-13 overlay/server_utils.py — --api-key upstream only guarded
# ("/v1","/v2","/inference","/cohere"); /invocations /metrics /tokenize /docs /ab
# all answered with no key. The overlay guards EVERY path and method.
docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_VOL" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/jit/triton:/root/.triton" \
  -v "$CACHE_HOST_PATH/jit/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_HOST_PATH/jit/b12x:/root/.cache/b12x" \
  -v "$CACHE_HOST_PATH/jit/tilelang:/root/.tilelang" \
  -v "$CACHE_HOST_PATH/jit/inductor:/tmp/torchinductor_root" \
  -v "$CACHE_HOST_PATH/jit/nv-compute:/root/.nv/ComputeCache" \
  "${NSYS_VOL[@]}" \
  -v "$LANE_DIR/overlay/exl3.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$LANE_DIR/overlay/glm5next/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro" \
  -v "$LANE_DIR/overlay/glm5next/kda.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py:ro" \
  -v "$LANE_DIR/overlay/glm5next/mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/mtp.py:ro" \
  -v "$LANE_DIR/overlay/glm5next/attention.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/attention.py:ro" \
  -v "$LANE_DIR/$CHAT_TEMPLATE_SRC:/lane/chat_template.jinja:ro" \
  -v "$LANE_DIR/draft_vocab_32k.json:/lane/draft_vocab_32k.json:ro" \
  -v "$LANE_DIR/overlay/vllm_utf8_guard.py:/usr/local/lib/python3.12/dist-packages/vllm_utf8_guard.py:ro" \
  -v "$LANE_DIR/overlay/gpu_sampler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/sampler.py:ro" \
  -v "$LANE_DIR/overlay/b12x:/usr/local/lib/python3.12/dist-packages/b12x:ro" \
  -v "$LANE_DIR/overlay/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -v "$LANE_DIR/overlay/b12x_mla_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py:ro" \
  -v "$LANE_DIR/overlay/sparse_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_utils.py:ro" \
  -v "$LANE_DIR/overlay/mla.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla.py:ro" \
  -v "$LANE_DIR/overlay/ab_switch.py:/usr/local/lib/python3.12/dist-packages/ab_switch.py:ro" \
  -v "$LANE_DIR/overlay/ab_plugin/glm53_ab_plugin.py:/usr/local/lib/python3.12/dist-packages/glm53_ab_plugin.py:ro" \
  -v "$LANE_DIR/overlay/ab_plugin/dist-info/glm53_ab_plugin-0.1.0.dist-info:/usr/local/lib/python3.12/dist-packages/glm53_ab_plugin-0.1.0.dist-info:ro" \
  -v "$LANE_DIR/overlay/scatter_states.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/scatter_states.py:ro" \
  -v "$LANE_DIR/overlay/server_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/server_utils.py:ro" \
  -v "$LANE_DIR/overlay/v2runner/model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro" \
  -v "$LANE_DIR/overlay/v2runner/spec_ar/speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro" \
  -v "$LANE_DIR/overlay/v2runner/spec_mmtp/speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py:ro" \
  -v "$LANE_DIR/overlay/processors/glm5next.py:/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processors/glm5next.py:ro" \
  -v "$LANE_DIR/overlay/multimodal_media/image.py:/usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/image.py:ro" \
  -v "$LANE_DIR/overlay/glm5next/ops/kpool_compress.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py:ro" \
  -v "$LANE_DIR/overlay/v1core/sched/scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro" \
  -v "$LANE_DIR/overlay/v1core/single_type_kv_cache_manager.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py:ro" \
  -v "$CACHE_HOST_PATH/jit/vllm:/root/.cache/vllm" \
  "${EXTRA_VOLS[@]}" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_API_KEY="${VLLM_API_KEY:?VLLM_API_KEY 필수}" \
  "${UTF8_GUARD_ENVS[@]}" \
  "${VISION_ENVS[@]}" \
  "${EXTRA_ENVS[@]}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_IB_GID_INDEX="$NCCL_GID_INDEX" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_SUBNET" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF" \
  -e TP_SOCKET_IFNAME="$NCCL_IF" -e MN_IF_NAME="$NCCL_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "${NCCL_CHANNEL_ENVS[@]}" \
  "$IMAGE" \
    "${NSYS_CMD[@]}" \
    vllm serve "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code --quantization exl3 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" --block-size "$BLOCK_SIZE" --mm-processor-cache-gb "$MM_CACHE_GB" \
    "${LOAD_ARGS[@]}" "${MNBT_ARGS[@]}" "${SPEC_ARGS[@]}" "${KV_ARGS[@]}" \
    "${KDA_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    "${UTF8_GUARD_ARGS[@]}" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "high"}' \
    --chat-template /lane/chat_template.jinja \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_RAIL_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP gid=$NCCL_GID_INDEX image=$IMAGE weights=$WEIGHTS_MODE kv=${KV_CACHE_MEMORY:-auto}${KV_DTYPE:+/$KV_DTYPE}${ATTN_BACKEND:+ attn=$ATTN_BACKEND} spec=${SPEC}${MTP:+ mtp=$MTP} mnbt=${MNBT:-default}${KDA_PREFILL:+ kda=$KDA_PREFILL}"

# Once-daily kit update check (head only): a plain GET of the one-line LATEST
# file in the kit repository, 3 s timeout, nothing sent about this machine,
# never affects the launch. Disable with GLM53_NO_UPDATE_CHECK=1 (.env or
# environment); GLM53_UPDATE_URL overrides the source (tests use file://).
kit_version_key() { # "v2.3-tier1" -> "2 3 0" (numeric prefix; suffix ignored)
  local v="${1#v}"; v="${v%%-*}"
  local a="${v%%.*}" rest="${v#*.}" b c
  [[ "$rest" == "$v" ]] && rest=0
  b="${rest%%.*}"; c="${rest#*.}"; [[ "$c" == "$rest" ]] && c=0
  printf '%d %d %d' "${a:-0}" "${b:-0}" "${c:-0}" 2>/dev/null || printf '0 0 0'
}
kit_version_newer() { # kit_version_newer <remote> <local>
  local r l; r=$(kit_version_key "$1"); l=$(kit_version_key "$2")
  [[ "$r" == "0 0 0" ]] && return 1
  local ra rb rc la lb lc; read -r ra rb rc <<<"$r"; read -r la lb lc <<<"$l"
  (( ra != la )) && (( ra > la )) && return 0
  (( ra == la && rb != lb )) && (( rb > lb )) && return 0
  (( ra == la && rb == lb && rc > lc )) && return 0
  return 1
}
kit_update_check() {
  [[ "${GLM53_NO_UPDATE_CHECK:-0}" == "0" ]] || return 0
  [[ "$NODE_RANK" == "0" ]] || return 0
  local stamp="$HOME/.cache/glm53-kit/update-check" latest
  local url="${GLM53_UPDATE_URL:-https://raw.githubusercontent.com/Entrpi/glm-5.3-flash-exl3-2x-spark/main/LATEST}"
  mkdir -p "${stamp%/*}" 2>/dev/null || return 0
  if [[ -f "$stamp" ]] && (( $(date +%s) - $(stat -c %Y "$stamp" 2>/dev/null || echo 0) < 86400 )); then return 0; fi
  : > "$stamp"
  latest=$(curl -fsS --max-time 3 "$url" 2>/dev/null | head -1 | tr -d '\r') || return 0
  [[ -n "$latest" ]] || return 0
  if kit_version_newer "$latest" "$KIT_VERSION"; then
    echo "kit update available: $latest (installed: $KIT_VERSION). Upgrade: git pull in the kit checkout, then ./install.sh"
    echo "  (this once-daily check reads one line from the kit repository and sends nothing; disable with GLM53_NO_UPDATE_CHECK=1)"
  fi
  return 0
}
kit_update_check
