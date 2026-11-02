# HANDOFF — 개발자용: 패치 원리와 함정 (GLM-5.3-Flash EXL3, 2×DGX Spark)

설치·업그레이드는 저장소 루트 [`README.md`](../README.md), 운영 명령과 실측은 [`README.md`](README.md).
이 문서는 **무엇을 왜 어떻게 바꿨는지**와 **다시 밟지 말 함정**을 남긴다. 각 오버레이의 정확한 변경은
`overlay/<파일>.diff`(이미지 원본 `.orig` 대비)에 있다. 최종 갱신 2026-11-01.

## 0. 현재 상태
원본 `brandonmusic/GLM-5.3-Flash-tr3-4bpw`(EXL3 4bpw) 하나를 Entrpi v2.3-tier1 vLLM 이미지로 2노드 TP2 서빙.
검열 해제는 가중치가 아니라 런타임 `/ab` 어댑터(`AB-SWITCH.md`). 우리 패치: dense 층 FP8 Marlin W8A16 +
프리필 BF16 이중경로 + 32K 드래프트 어휘 + 라우터 GEMM 중복 제거 + SwiGLU clamp + temperature 0 결정성 +
UTF-8 가드 + **'!' 붕괴 수정(vllm#51865 백포트)** + **kpool 캐시 오염 수정(vllm#57477 백포트)**
+ **이미지 수용(`VISION=1`, 한 요청 5장 상한 — `VISION.md` §0)**.

⚠️ 이미지 관련은 `VISION.md` §0 을 먼저 읽을 것. 이미 실패한 시도와 측정 오염 조건이 거기 있다.

## 1. 출처 (Provenance)
| 구성요소 | 출처 | 비고 |
|---|---|---|
| 서빙 킷 | https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark commit 63f254f (v2.3-tier1) | 킷의 `install.sh` 는 쓰지 않는다. 런처 `scripts/launch-glm53-vllm-tp2.sh` 를 복사해 수정(`launch.sh.diff`) |
| 도커 이미지 | `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` (image id `sha256:606b5f3c…`, registry digest `sha256:44b2dbaf…`) | vLLM 포크 `Entrpi/vllm-glm-5.3-flash-spark` @ f223ff9f2(상류 기점 2026-08-10 무렵) + b12x(sparkinfer-glmrt glm-next-backport 2f53ce3) + exllamav3 c5d9c657 |
| 서빙 모델 | HF `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ a5fee929 | 라우팅 전문가만 EXL3 4bpw, 나머지 BF16. `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw` |
| abliterated | HF `lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated` @ c8f58e6a | 원본과 o_proj 45개만 다르다. 어댑터의 재료로만 쓴다 |
| `/ab` 어댑터 | `tools/build_ab_lora.py` 로 생성(29MB) | `~/Models/glm53-ab-lora` |
| 드래프터(미사용) | HF `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` rev 62f758c0 | `SPEC=dflash` 일 때만. MTP-4 에 져서 안 쓴다 |

오버레이를 새 이미지로 옮길 때: 새 이미지에서 원본을 꺼내(`cid=$(docker create <IMAGE>); docker cp $cid:<경로> X.orig`)
기존 `.diff` 를 다시 적용하고 충돌을 손으로 푼다. 원본 경로는 launch.sh 의 마운트 목적지 그대로다.

## 2. 재구성
루트 README §5(새 설치) / §6(09-05 키트에서 업그레이드). 이미지 재빌드는 필요 없다(오버레이 방식).

## 3. 패치 원리 (무엇을 왜 어떻게)
### 3.1 launch.sh (킷 런처 대비, `launch.sh.diff`)
- 레인 위치(`LANE_DIR`)를 스크립트 자리에서 계산하고 설정 파일도 레인 디렉터리의 것을 쓴다(경로 하드코딩 없음). `--language-model-only`(기본 VISION=0. 모든 모달리티 limit 을 0 으로 만들어 비전 타워 웨이트 적재와 기동 더미 MM 프로파일을 둘 다 건너뛴다 — 과거 주석 "비전 타워 warmup OOM" 이 맞다. ⛔ 10-30 에 "입력 거부 게이트일 뿐, 타워는 어차피 로드된다"고 고쳐 썼던 것은 거짓이었다. VISION.md §1·§2). API 키. 오버레이 bind-mount(:ro, 목록은 launch.sh 의 `-v "$LANE_DIR/...` 줄). env 노브 통과(`B12X_*`, `VLLM_EXL3_*`, NCCL_PROTO/ALGO 등). `NSYS=1` 이면 nsys launch 래핑. 이미 abliterated 인 모델을 가리키면 `/ab` 어댑터를 자동으로 끈다(편집 이중 적용 방지).
- 컨테이너 안 경로: vllm = `/usr/local/lib/python3.12/dist-packages/vllm`, b12x = `.../dist-packages/b12x`. b12x 패키지 전체는 `overlay/b12x` 를 직접 마운트(09-27~). 킷의 `$HOME/glm53-hotfix-b12x` 훅 코드는 원본대로 남아 있으므로 **그 심링크가 남아 있으면 같은 자리에 이중 마운트되어 기동이 실패**한다(09-05 키트에서 올라올 때 반드시 지울 것).
### 3.2 overlay/exl3.py — dense 층 FP8 Marlin (핵심, `overlay/exl3.py.diff`)
- 배경: EXL3 체크포인트는 routed experts 만 4bpw 이고 attention/KDA/MLA 투영·shared expert·lm_head·dense MLP 는 BF16. nsys 실측에서 디코드 스텝의 56% 가 이 BF16 GEMM(스텝당 17.6GiB 읽기).
- 방법: `Exl3Config.get_quant_method` 에서 EXL3 가 아닌 `LinearBase`/`ParallelLMHead` 를 vLLM 포크의 `Fp8PerBlockOnlineLinearMethod`(로드 시 128x128 블록 FP8 e4m3 양자화)로 보내되, 커널을 `MarlinFP8ScaledMMLinearKernel` 로 강제(`force_kernel`). Marlin = **가중치만 FP8, 활성화 BF16(W8A16)**. sm_121 에서 Marlin 은 `VLLM_TEST_FORCE_FP8_MARLIN=1` 필요. lm_head 는 `ParallelLMHead` 라 attr 이름이 embedding_dim/num_embeddings 임에 주의.
- 금지: 포크 기본 `B12xFp8BlockScaledMMKernel`(활성화 MXFP8 양자화, W8A8)은 작화 유발(탐침 가짜버그 4~6/8) → 쓰지 말 것.
- 결과: 스텝 117→87ms, 산문 25.3→36.
### 3.3 overlay/glm5next/model.py, kda.py
- MLA/KDA 투영이 `quant_config=None` 을 강제하던 부분을 `VLLM_EXL3_DENSE_FP8=1` 일 때 quant_config 통과로. (kda.py 는 `saved_quant_config` 트릭으로 부모 __init__ 호출)
### 3.4 프리필 BF16 이중경로 (exl3.py 내)
- 문제: Marlin 은 M 이 크면(프리필 M=8192) cuBLAS BF16 보다 2.7배 느림(in_proj 31 vs 11ms) → 프리필 -12%.
- 방법: `_Fp8BlockOnlineMarlin.apply` 에서 `M > VLLM_EXL3_PREFILL_BF16_MIN_M(1024)` 이면 Marlin 패킹 FP8 을 **역변환**해 BF16 으로 풀고 `F.linear`. 역변환은 vLLM `marlin_utils_test.marlin_weights` 의 정확한 역: 16x16 타일 재배열 → 1024단위 `get_weight_perm(8)` 순열 → 4바이트 팩. 스케일은 `get_scale_perms` 64단위 역순열 + `fp8_fused_exponent_bias`(bf16=2^120) 제거. Triton 커널 1패스(1024바이트 청크 = 출력 16×64 타일). 최대 층 2.25ms, 상주 메모리 0. 단독 검증 5형상 비트일치. lm_head 는 제외.
### 3.5 overlay/glm5next/mtp.py
- `eh_proj` nn.Linear(BF16 cuBLAS gemv 291µs×4/스텝) → `ReplicatedLinear(quant_config)` 로 같은 Marlin FP8 경로. 드래프트 INT4 헤드 프록시 코드는 **비활성**(유저 규칙: FP8 밑 정밀도 삭감 금지, 실측 이득도 없음).
### 3.5.1 드래프트 전용 축소 어휘 lm_head (09-27, exl3.py + mtp.py)
- exl3.py `_build_subset_draft_head`: FP8 패킹 전 BF16 lm_head 에서 이 랭크 조각의 서브셋 행(draft_vocab_32k.json)만 뽑아 FP8 W8A16(g128) Marlin 헤드를 만든다. **본 모델 lm_head 에만**(prefix `lm_head`). V2 `load_eagle_model`(spec_decode/eagle/utils.py)이 MTP 의 `shared_head.head` 를 본 모델 lm_head 로 갈아끼우므로 드래프트도 이 모듈을 쓴다. MTP 자체 헤드(`…layers.<N>.head`)는 웨이트 없이 만들어졌다 버려진다 — 거기 붙이면 헤드가 안 쓰인다(09-27에 실제로 밟음, 경고 로그로 발견). 서브셋 헤드는 속성으로만 붙어서 verify 경로(quant_method)는 불변. 패딩 행은 실제 행 복사(0 행 → FP8 스케일 0 → NaN).
- `_SubsetDraftHead.apply`: 압축 로짓을 이 랭크 조각 폭의 -inf 텐서에 써넣어 기존 vocab-parallel gather/argmax 를 그대로 탄다. apply 의 layer 인자는 프록시라 출력 폭은 헤드 객체가 들고 있다.
- mtp.py `compute_logits` → `_draft_head()` 프록시. **V2 speculator 는 드래프트를 compute_logits+argmax 로 뽑는다(get_top_tokens 아님)** — 여기 안 끼우면 헤드는 만들어져도 안 쓰인다. env 가 켜졌는데 서브셋 헤드가 없으면 경고 1회.
- verify 는 본 모델 전체 어휘 헤드 → 출력 무손실. 확률 드래프트도 서브셋 분포로 뽑고 같은 로짓이 q 로 가므로 무손실.

### 3.5.2 routed 전문가 SwiGLU clamp (09-27, exl3.py + b12x mixed_trellis.py)
- `Exl3MoEMethod._swiglu_clamp_limit()`: env `VLLM_EXL3_MOE_SWIGLU_CLAMP=1` 이면 `self.moe.swiglu_limit`(GLM-5.3 = 10). 상류(이미지 원본)는 이 값을 어느 경로에도 안 넘겼다.
- 전달처: b12x `Caps(swiglu_limit=…)`(실사용, 커널 `_clamp_swiglu_inputs`: gate≤L, up∈[-L,L], FP32) / `compile_mixed_trellis(swiglu_limit=…)` → `W4A16FusedMoeKernel` / exllamav3 `exl3_moe(_fused)` 의 유일한 float 인자 `act_limit`(0.0=끔) / torch 대체 경로 clamp.
- 대조 결과 다른 누락 없음: 라우팅 스케일·정규화·점수·bias 는 라우터, dense·shared 는 vLLM MLP 가 clamp.

### 3.6 overlay/glm5next/attention.py
- 인덱서 fp32 `weights_proj`: cuBLAS 가 [M,4096]x[4096,64] fp32 NN 에서 M=5 일 때 1블록 커널(90~130µs) 선택 → `(W_nk @ x.T).T` 로 19µs. `.contiguous()` 는 M=1 에서 no-op(stride(0)=1) 이라 DeepGEMM 이 거부 → `clone(memory_format=contiguous_format)` 필수.
### 3.7 overlay/b12x (`overlay/b12x.diff`)
- `moe/_shared/kernels/w4a16/kernel.py` m8(TC-decode) 경로 SM당 CTA 수 env 훅 하나. 타일 override 와 함께 디코드 -3% 였으나 프리필 -10% 라 **env 비움(비활성)**. 마운트는 유지(무해).

### 3.8 temperature 0 결정성 (09-29~30, README §4.9)
- 원인 셋은 전부 **업스트림 코드**(우리 패치가 만든 것 아님), 모두 atomic 이 합산 순서를 정하던 것:
  ① b12x dense split-K bf16 atomicAdd → env `B12X_DENSE_SPLITK_TURBO=0`
  ② vLLM `sparse_utils.py` 인덱스 압축(COMPACT_TO_FRONT) 타일 자리선점 — GLM 행 폭 2112 가 업스트림의 2의 거듭제곱 전용 무원자 경로를 벗어남 → 4096 패딩 + 열 마스크, 행=프로그램 1개
  ③ vLLM top-k(`persistent_topk`, `top_k_per_row_prefill`) — 풀 512(=2,048 토큰) 초과 시 atomic 순서 + 경계 동점 선착순 → `_canonicalize_topk_`(k번째 값 t 기준 >t 전부 + ==t 낮은 인덱스부터, torch.topk 규칙)
- 흐름: 인덱서 점수 → top-k(③) → 논리→물리 슬롯 압축(②) → Sparse MLA 커널(무죄). 커널은 받은 목록 순서대로 더한다.

## 4. 실측 원장·실험 기록
README.md §4~§5 및 메모리 `project_glm53_decode_kernel_truth`, `project_glm53_dense_fp8_overlay`. 핵심: 디코드는 바이트 바운드(MoE 커널 M=1 에서 천장 73%, verify 는 천장 근처, Marlin 83%). 커널 상한 ≈ -12ms(산문 42/코드 50). +10 t/s 는 드래프터 한국어 수용률(MTP 헤드 자기증류, FastMTP 레시피)뿐. 폐기: b12x W8A8, INT4 헤드, split-K 플래그, 타일 override, NCCL LL, DFlash2(산문), MTP k 스윕, 라우터 ll_bf16 dotprod(스텝 +2~3ms), verify top-p Triton(소음), CPU C-state 끄기(무효).
09-27 추가 채택: 라우터 중복 제거, 32K 드래프트 어휘(스텝 -3ms), SwiGLU clamp(정확성). 남은 레버: 32K 헤드 TP 랭크 균형(~0.8ms), 드래프트 그래프 시작 공백(~1.5ms, 원인 미상), 드래프터 수용률(파인튜닝).

## 5. 함정 (다시 밟지 말 것)
- **temp 0 결정성 회귀 검사**: 이미지·오버레이를 바꾸면 짧은 출력만 보지 말 것 — ③은 컨텍스트 2,048 토큰을 넘어야 드러난다. 생각 켬 코드 요청(2,200 토큰 출력) 4회 해시 동일을 확인.
- top-k 를 **정렬만** 하는 건 답이 아니다(경계 동점이면 집합이 바뀜). 디버깅 탐침은 커밋 금지 — 끝나면 `git diff <운영 기준 커밋>` 으로 운영본+수정뿐임을 증명.
- 드래프트 헤드 교체는 MTP `compute_logits` 에 해야 한다(V2). get_top_tokens 에만 걸면 조용히 무효(09-27 첫 측정 스텝 변화 0). 서브셋 헤드는 본 모델 lm_head 에 붙일 것(MTP shared_head.head 는 로드 시 본 모델 lm_head 로 교체된다).
- 이미지 교체 시: mtp.py `_DraftHeadProxy` 는 tp_size/shard_indices/quant_method 만 덮고 나머지는 원래 lm_head 로 위임한다. upstream logits_processor 가 헤드에서 읽는 속성이 바뀌면 위임으로 대부분 흡수되지만, quant_method.apply 시그니처가 바뀌면 `_SubsetDraftHead.apply` 를 맞출 것. 기동 로그에서 'subset draft lm_head on …lm_head' 1줄 + 경고 0줄 확인.
- nsys 커널 합계로 절감을 추정하지 말 것: 멀티스트림·PDL 로 실제 스텝과 안 맞는다(라우터 커널 44→23us 인데 스텝 +2~3ms). 스텝 실측만.
- 2노드는 RESTARTALL 만. 워커 단독 Up = 좀비. 실험용 자체 재기동 스크립트를 쓰면 memwatch 가 안 켜진다 → 래퍼만 쓸 것.
- ssh 로 환경변수는 안 넘어간다. 노브는 env 파일에 넣고 launch.sh 통과 목록에 등록하고 양노드 rsync+md5.
- `pkill -f <스크립트명>` 은 자기 셸을 죽인다(exit 144). pid 파일/`$!` 로 죽일 것.
- 마운트된 오버레이 파일을 서빙 중 편집하면 컨테이너가 즉시 본다(다음 import 부터). 편집은 재기동 사이에.
- `.contiguous()` no-op 함정(M=1), DeepGEMM 은 stride(-1)==1·stride(0)==H 요구.
- Marlin 패킹 lm_head 는 DFlash `quantize_draft_head` 와 충돌 → `VLLM_DFLASH_FP8_DRAFT_HEAD=0`.
- 단독 GPU 테스트는 STOPALL 후에만(통합메모리 OOM = 박스 전체). 테스트 컨테이너에 JIT 캐시(~/glm53-vllm-cache/jit/triton) 마운트해도 무해하나 캐시 키 충돌은 없음.
- nsys: 컨테이너 2025.3.2 로 캡처, 호스트 기본 nsys(2024.6) 는 리포트를 못 읽음 → `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys stats`.
- cuBLAS 스키니 GEMM 은 "cold" 조건에서 재라(L2 상주 벤치는 3배 낙관).
- 킷 자동 업데이트 체크(launch.sh kit_update_check)는 무해하나 `GLM53_NO_UPDATE_CHECK=1` 로 끌 수 있음.
### 3.9 '!' 붕괴 — 프리필 꼬리가 디코드 그래프로 재생 (10-02, `overlay/v2runner/`)
- 증상: 추론 채널에 token 0('!') 연타. **프롬프트 길이 ≡ 5 (mod 512)** 이면 100% 재현(내용·캐시 무관).
- 원인: 프리필 청크가 512 경계에서 끊겨 마지막 조각이 정확히 1+MTP(=5) 토큰이 되면, V2 러너가 배치 모양만 보고
  (`get_uniform_token_count`) spec-verify FULL cudagraph 를 재생. 그 그래프는 프롬프트의 KDA 순환 상태를 만들지 않아
  로짓 NaN → 샘플러가 0 을 뽑고 이후 디코드도 전부 NaN.
- 수정: 상류 vllm#51865(08-12, 포크 기점 이후라 누락) 백포트 — 프리필 중인 요청이 있으면 uniform decode 가 아니다.
  러너 + speculator 2개(`model_runner.py`, `spec_ar/`, `spec_mmtp/`). 회귀 검사 `tools/prefill_tail_nan_check.py`.
- 디버깅 교훈: 층별 NaN 프로브가 조용한데 출력층 입력만 NaN 이면 그 스텝이 **그래프 재생**(파이썬 미실행)이다.
  logprobs 를 요청하면 NaN 일 때 서버가 HTTP 400 "Out of range float values are not JSON compliant: nan" 을 낸다 — 가장 싼 검출기.

### 3.10 kpool 꼬리 캐시 stride (10-02, `overlay/glm5next/ops/kpool_compress.py`)
- 상류 vllm#57477 백포트. 프리필 tail-seed 커널이 꼬리 캐시를 덴스 stride(1024)로 써서, 실제 패딩 stride(76032)의
  자리 대신 **다른 요청의 인덱서 캐시 블록**을 덮어썼다. 검사 `tools/kpool_cache_probe.py`(같은 캐시 프리픽스의
  첫 토큰 logprob 이 다른 요청 처리 뒤에도 변하지 않아야 한다).

### 3.11 `/ab` 런타임 스위치 — `AB-SWITCH.md`. ### 3.12 UTF-8 가드 — `UTF8-GUARD.md`. ### 3.13 이미지 수용 — `VISION.md`.
