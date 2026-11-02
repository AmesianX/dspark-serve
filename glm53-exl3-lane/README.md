# GLM-5.3-Flash EXL3 lane (2×DGX Spark GB10) — 운영 문서

설치·업그레이드는 저장소 루트 [`README.md`](../README.md). 패치 원리·함정은 [`HANDOFF.md`](HANDOFF.md).
이 디렉터리는 **양 노드(헤드/워커) 같은 경로·같은 내용**이어야 한다(`sync-lane-67.sh` 가 맞춘다). 최종 갱신 2026-11-01.

### 문서 지도 — 뭘 찾을 때 어디를 보는가

| 찾는 것 | 문서 | 상태 |
|---|---|---|
| 기동·정지·상태 확인, 현 설정값, 실측 t/s | **이 문서** | 현행 |
| 이미지(비전) — 설정·안전범위·측정법·**이미 실패한 시도** | **`VISION.md` §0 필독** | 현행 |
| 왜 이렇게 패치했나, 다시 밟지 말 함정 | `HANDOFF.md` | 현행 |
| `/ab` 검열 해제 스위치 사용·원리 | `AB-SWITCH.md` | 현행 |
| UTF-8 바이트 가드 설계·검증 | `UTF8-GUARD.md` | 현행 |
| 왜 abliterated 대신 원본을 서빙하는가 | `ORIGINAL-VS-ABLITERATED.md` | 배경(9~10월) |
| 날짜순 작업 이력 | `HISTORY.md` | 이력 |
| 설치·업그레이드·오프라인 재구성 | 루트 [`README.md`](../README.md), `../glm53-materials/HANDOFF.md` | 현행 / 9월 스냅숏 |

⚠️ **이미지 관련으로 뭘 재거나 고치기 전에 `VISION.md` §0 을 먼저 읽을 것.** 이미 해보고 실패한
시도(`.data` 조기 해제, 한도 4 로 내리기, malloc 노브, 세마포어)와 측정 오염 조건이 거기 있다.

## 1. 운영 (헤드에서 실행)
```
./vllm-glm53-exl3.sh RESTARTALL   # 유일한 기동 명령(양노드 정리→memwatch→워커→헤드→READY)
./vllm-glm53-exl3.sh STATUS       # 컨테이너/health/모델·/ab 상태/가용메모리/memwatch
./vllm-glm53-exl3.sh LOGS | LOGSW # 헤드/워커 로그 follow
./vllm-glm53-exl3.sh STOPALL      # 양노드 정지(+memwatch 해제)
```
- ⚠️ 2노드는 무조건 RESTARTALL. 워커만 살아 있으면 랑데부 대기 좀비(601초 낭비). `START` 단독은 워커가 이미 대기 중일 때만.
- 워커는 git 이 아니라 파일 사본. 헤드에서 커밋·브랜치 전환 후엔 `./sync-lane-67.sh` 로 밀어야 RESTARTALL 이 뜬다(기동 전 자동 검사).
- 실험은 브랜치에서(`git switch -c exp/...`), 되돌림은 `git switch main` → `./sync-lane-67.sh` → RESTARTALL. 이미지 태그 v2.3-tier1 은 rmi/retag 금지.
- 재부팅 후에도 수동 RESTARTALL(systemd 유닛 없음). READY까지 약 210~255초, 첫 요청은 JIT 워밍으로 느림.
- 서빙 중 헤드에서 무거운 작업 금지(통합메모리). memwatch 가 가용 3GB 밑이면 컨테이너를 선제 정지한다.
- API: `http://<헤드 LAN IP>:8000/v1`, 모델명 `glm-5.3-flash`, 키는 env 의 `VLLM_API_KEY`. 검열 해제 스위치 `POST /ab {"scale":0|1}`(`AB-SWITCH.md`).

## 2. 파일
| 파일 | 역할 |
|---|---|
| `vllm-glm53-exl3.sh` | 조작 래퍼(양노드 제어, memwatch 내장). 워커 ssh 대상·키는 env 에서 읽는다 |
| `launch.sh` (+`launch.sh.diff`) | Entrpi 킷 런처 사본 + 우리 수정: 레인 경로 자동, 비전 게이트 `VISION`(1=이미지 수용 — 라이브 기본, `VISION.md` §0. 0 이면 `--language-model-only`), API 키, 오버레이 bind-mount, env 노브 통과, `/ab` 어댑터·플러그인, `NSYS=1` 프로파일 |
| `glm53-serve.env` | 설정. ① 사이트 값(설치 때 고침) ② 모델·어댑터 경로 ③ 검증된 운영 설정 |
| `sync-lane-67.sh` | 레인을 워커로 rsync --delete + 파일별 md5 대조(`--check` 는 RESTARTALL 이 자동 실행). 옛 `~/glm53-hotfix-b12x` 가 남아 있으면 거부 |
| `chat_template.jinja` (+.orig/.diff) | 모델 템플릿 사본 + zai-org 09-04 수정 백포트. `--chat-template` 로 주입 |
| `draft_vocab_32k.json` | MTP 드래프트 전용 어휘 32,768 토큰 id |
| `overlay/exl3.py` | dense 층 FP8(Marlin W8A16)·lm_head FP8·프리필 BF16 이중경로·32K 드래프트 헤드·SwiGLU clamp |
| `overlay/glm5next/{model,kda,mtp,attention}.py` | 투영 양자화 통과, MTP eh_proj FP8·드래프트 헤드 프록시, 인덱서 fp32 전치 레이아웃, `/ab` 훅(KDA) |
| `overlay/glm5next/ops/kpool_compress.py` | kpool 꼬리 캐시 stride 수정(vllm#57477 백포트) |
| `overlay/v2runner/` | '!' 붕괴 수정(vllm#51865 백포트): 러너 + speculator 2개 |
| `overlay/mla.py`, `ab_switch.py`, `ab_plugin/` | `/ab` 런타임 검열 해제 스위치(MLA 훅, 훅 본체, HTTP 라우트·워커 전파) |
| `overlay/sparse_utils.py`, `sparse_attn_indexer_kpool.py`, `b12x_mla_sparse.py`, `scatter_states.py` | temperature 0 결정성(§4.9), 09-28 감사 수정, null 블록 처리 |
| `overlay/vllm_utf8_guard.py`, `gpu_sampler.py` | UTF-8 가드(`UTF8-GUARD.md`) |
| `overlay/b12x/` (+`b12x.orig`, `b12x.diff`) | b12x 패키지 전체. 우리 diff = m8 경로 env 훅(기본 비활성) |
| `tools/fetch_oproj.py` | o_proj 46개(3.89GB)만 HTTP Range 로 받기(원본/abliterated) |
| `tools/build_ab_lora.py` | `/ab` 어댑터 생성(원본 − abliterated 의 rank-24 근사, 29MB) |
| `tools/build_original_from_abliterated.py` | abliterated 설치에서 원본 만들기(하드링크) |
| `tools/prefill_tail_nan_check.py` | '!' 붕괴 회귀 검사(나머지 5 mod 512 길이들) |
| `tools/kpool_cache_probe.py` | kpool 캐시 오염 검사 |
| `tools/kmeas.py`, `prefill_bench.py` | 디코드 t/s·수용률·스텝 / 프리필 t/s |
| `tools/probe_long.py`, `tfcap.py`, `lpcap.py`, `probe8.py`, `refusal_probe.py` | 작화 탐침, 정밀도 게이트, 거부율 측정 |
| `tools/agent_sim.py` | cm2 요청 모양 그대로의 트래픽 재현기('!' 재현에 썼다) |
| `tools/draft_vocab_*.py`, `router_gemm_bench.py`, `probe_head.rs` | 드래프트 어휘 재생성, 라우터 GEMM 벤치(서버 정지 후), 탐침 보조 |
| `tools/_lane.py` | 도구 공용: 레인 경로·URL·API 키(env 에서 읽음) |

## 3. 확정 구성 (glm53-serve.env)
- 이미지 `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`, 모델 **원본** `brandonmusic/GLM-5.3-Flash-tr3-4bpw`,
  `/ab` 어댑터 `~/Models/glm53-ab-lora`(기본 꺼짐), TP2, max-model-len 524288, KV 8GB/노드(fp8_ds_mla), max-num-seqs 4.
- 스펙 디코딩 MTP-4 (`SPEC=none MTP=4`). DFlash2 는 옵션이지만 MTP-4 에 져서 미사용(§5).
- 정밀도: MoE 라우팅 전문가 W4A16(EXL3 trellis), dense·shared expert·lm_head·MTP eh_proj **W8A16 Marlin**(가중치 FP8, 활성화 BF16), KV FP8.
  **활성화 양자화(W8A8/W4A8)·INT4 는 금지**(작화 유발).
- 프리필: M>1024 dense 호출은 FP8 을 BF16 으로 풀어 cuBLAS(`VLLM_EXL3_PREFILL_BF16=1`), 디코드는 Marlin.
- UTF-8 바이트 가드(`VLLM_UTF8_GUARD=1`), MoE 라우터 GEMM 중복 제거, 드래프트 전용 32K 어휘(`VLLM_GLM53_DRAFT_VOCAB`),
  routed 전문가 SwiGLU clamp(`VLLM_EXL3_MOE_SWIGLU_CLAMP=1`), temperature 0 결정성(`B12X_DENSE_SPLITK_TURBO=0` + 오버레이 2개),
  '!' 붕괴 수정(`overlay/v2runner/`), kpool 캐시 수정(`overlay/glm5next/ops/`).
- 프리픽스 캐시: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=18432`(mamba 스냅샷 솎기, 근거는 env 주석).

## 4. 실측 원장

**최신(2026-10-05, 현 main, MTP-4, `/ab` 꺼짐)** — kmeas(temp 0, 생각 끔, 300토큰×3): hash-map **34.9** / 한국어 산문 **38.9** / 코드 **41.8** t/s,
수용 1.85 / 2.08 / 2.39, 스텝 79~81ms. (아래는 그 시점 기록이다.)
**최신(09-27, exp/step70: 라우터 중복 제거 + 32K 드래프트 어휘 + SwiGLU clamp, MTP-4)** — kmeas(temp 0, 생각 끔, 300토큰×3)
| 항목 | 값 |
|---|---|
| 디코드 영어 / 한국어 산문 / 코드 | 34.5 / 39.7 / 44.3 t/s (수용 1.70 / 2.05 / 2.54) |
| 스텝 | 77~79ms (09-09 82~84.5). nsys 분해(09-27, 32K 전): MoE verify 29.0 · Dense Marlin 20.0 · lm_head×5 8.1 · NCCL 4.7 · BF16 GEMV 3.7 · 소형 6.2 · idle 4.0 |
| 프리필 | 12k 1,348~1,362 / 24k 1,327 / 75k cold 1,328 t/s |

**09-09 기준선**
| 항목 | 값 |
|---|---|
| 디코드 한국어 산문 / 코드 / 영문 산문 | 34.5 / 44.1 / 34.8 t/s (MTP-4, 수용 1.82 / 2.66 / 1.98) |
| 스텝 | 82.0~84.5ms = MoE 33.5 + Marlin 29.1 + NCCL 5.5 + 소형커널 ~12 + idle ~4 |
| 프리필 | warm 12k 1,341 / 24k 1,311 t/s, 첫 요청 cold 851 |
| 로딩 | READY 225초 |
| 비교 | 09-03(DOCA 3.4.0): 디코드 36/43/35, 프리필 1,333/1,312 — 동일 수준, 회귀 없음 |
| 비교 | 전부 BF16(09-02): 디코드 25.6/31.1, 프리필 1,395/1,078. FP8 단독: 36.2/44.1, 1,225/982 |
- 디코드는 바이트 바운드: MoE 커널은 M=1 에서 천장(244GB/s)의 73%, verify(M=5)는 전문가 30+개 읽기로 천장 근처. Marlin 전체 83%.
- RDMA 경로 확증(09-09): 디코드 벤치 구간에 mlx5_0 port_xmit/rcv +8.7GiB → NCCL TCP 폴백 아님. `NCCL_IB_GID_INDEX=3` 은 rdma-core 교체 후에도 양노드에서 RoCE v2/IPv4(10.0.1.1·.2)로 일치.
- 커널로 남은 상한 ≈ -12ms(산문 42/코드 50). 그 이상은 드래프터 수용률(한국어 학습)뿐. 상세 = 메모리 `project_glm53_decode_kernel_truth`.

## 4.9 ✅ 해결 — temperature 0 비결정성 (2026-09-29~30)

**증상**: 운영 설정(MTP-4)에서 같은 프롬프트·temperature 0 인데 매번 다른 답(6 회 6 종류).
투기 디코딩은 타깃이 검증하므로 무손실이어야 하는데 아니었다. 첫 갈림은 logprob 값의
~1e-4 흔들림이고, 그게 KV 에 쌓여 수백 토큰 뒤 near-tie argmax 를 뒤집는다.

**원인 셋 — 전부 "atomic 이 순서(와 동점 선택)를 정한다"는 같은 계열.** 셋 다 **업스트림 코드**의 결함이다(b12x / vLLM; 우리 패치가 만든 것 아님 — 같은 이미지로 GLM-5.3 을 서빙하면 누구나 겪는다). 업스트림 최신 main 에서 고쳐졌는지는 미확인. ROCm 레인(NONAME)도 ③ 동일 결함을 단위테스트로 확인(①② 는 그쪽 커널에 없음). 부동소수 덧셈은
결합법칙이 안 서므로 더하는 순서가 실행마다 바뀌면 값이 흔들린다.

1. **dense GEMM split-K** (`99298bd`): 부분합을 bf16 atomicAdd(`red.global.add.bf16x2`,
   `_lib/dense_gemm.py` `scatter_add_bf16x2`)로 누적. → 커널에 이미 있던 인덱스 슬롯 +
   고정순서 합산(`B12X_DENSE_SPLITK_TURBO=0`, 기본값). 이것만으로 MTP=1 은 결정적.
2. **sparse 인덱스 압축** (`overlay/sparse_utils.py`): 논리 토큰 → 물리 슬롯 변환에서 유효한
   것을 앞으로 모을 때(`COMPACT_TO_FRONT`) 64 열 타일들이 `tl.atomic_add` 로 자리를 선점.
   꼬리 타일이 먼저 이기면 `[T40, T0..T39]`, 늦으면 `[T0..T40]` — 같은 집합, 다른 순서.
   업스트림(#50365)은 행 폭이 2 의 거듭제곱(2048)일 때만 원자연산 없는 단일 타일을 쓰는데
   GLM-5.3 kpool 행은 2048 + 꼬리 풀(64 정렬) = **2112** 폭이라 원자연산 경로로 떨어졌다.
   → 폭을 4096 으로 패딩 + 열 마스크, 행 하나 = 프로그램 하나. 순서 = 열 순서.
3. **인덱서 top-k** (`overlay/sparse_attn_indexer_kpool.py` `_canonicalize_topk_`): 행의 풀이
   k(512 = 2,048 토큰)를 넘으면 persistent_topk(디코드)·top_k_per_row_prefill(프리필)이
   출력 자리를 shared-memory atomicAdd 로 나눠 **순서**가 바뀌고, 경계에 정확히 같은 값이
   걸리면 선착순이라 **집합**까지 바뀐다. 실측: 3 회 모두 컨텍스트 2,051 까지 logprob 비트
   동일, 2,052 에서 첫 갈림. 단위테스트: 원본 20 회 중 순서 16/20 행·집합 6/20 행(동점 행)이 흔들림.
   → 빠른 커널로 k 번째 값 t 만 얻고(동점과 무관하게 결정적), Triton 커널이 행을 인덱스
   오름차순으로 훑어 `> t` 전부 + `== t` 는 작은 인덱스부터 남은 자리만큼 다시 고른다.
   torch.topk 와 같은 규칙(threshold + stable gather), 원자연산 없음. 풀 ≤ 512 인 행은
   원래도 오름차순이라 출력이 바뀌지 않는다(단위테스트 4/4 비트 동일).

**찾은 방법**(재발 시 같은 순서로): ① 층별 체크섬을 GPU 동기화 없이 찍어 첫 갈림 층 특정
(항상 sparse 층) ② 디코드 커널만 두 번 → 0/40 차이(커널 무죄), 어텐션 경로 전체 두 번 →
6/24 차이(상류가 범인) ③ 커널 입력을 요청 간 해시 대조 → KV 레코드 집합은 같고 순서만 다름
④ 물리 슬롯을 블록 테이블로 역산 → 꼬리 토큰이 앞/뒤로 바뀜(원인 2) ⑤ 긴 출력에서만
남는 갈림을 logprob 로 토큰 단위 위치 측정 → 정확히 512 풀 경계(원인 3).

**검증** (운영 설정, CUDA graph + MTP-4, 탐침 없음):

| 조건 | 결과 |
|---|---|
| effort low, 3 종 × 5 회 × (풀 프리필 / 캐시 적중), ≤1,000 토큰 | **30/30 동일**, 풀 프리필과 캐시 적중끼리도 동일 |
| 생각 켬, 코드 2,201 토큰 / 산문 3,500 토큰 (컨텍스트 2,048 초과) | 4/4 · 4/4 동일 (원인 3 수정 전 4/4 전부 다름) |
| 12,393 토큰 프롬프트 + 600 토큰 | 3/3 동일 |
| MTP 끔(MTP=0): 짧은 배터리 + 생각 켬 코드 2,061 / 산문 3,500 토큰 | 30/30, 3/3, 3/3 동일 |
| 대조군: 원인 2 오버레이만 빼고 동일 조건 | 산문 5/5, 코드 5/5 전부 다름 |

**비용**: 재선택 커널 호출당 9 / 20 / 127 µs(풀 2K / 16K / 131K, 5 행). 디코드 스텝
산문 78.5 / 코드 79.2 / hash-map 80.3ms — 09-29 기준 77.5 / 79.3 / 80.8 과 같은 범위.
프리필 12k 1,352 · 25k 1,332 t/s(09-03 기록 12k 1,333). kmeas t/s 는 산문 39.4 / 코드 42.5 /
hash-map 35.2 로 09-29 기록(40.6/43.5/36.4)보다 낮은데 스텝이 같으므로 전부 **수락률**
차이다(2.147 → 2.082 등). 예전 값은 매번 갈리던 경로들의 표본이고, 지금은 경로가 하나로
고정돼 그 경로의 수락률이 나온다(재측정해도 소수점까지 같다).

**아닌 것으로 확인된 것**(반복하지 말 것): TC-decode FC2 float atomic, `route_pack.py`
행 자리 atomic, CUDA graph, 인덱서 logits(유효 구간은 비트 동일), sparse MLA 커널 자체
(split-K 병합 포함). top-k 를 **정렬만** 하는 것도 답이 아니다 — 동점이면 집합이 바뀐다.

## 4.10 abliteration 런타임 스위치 (`/ab`, 2026-10-01)

원본을 서빙하면서 검열 해제를 요청 단위로 켰다 끈다. `o_proj` 가 4bpw 체크포인트에서도
BF16 이라 두 체크포인트 차이를 rank-24 어댑터(29MB)로 뽑아 출력에 더하는 방식.
재기동·재프리필·그래프 재캡처 없음. 상세·실측·이식 메모는 `AB-SWITCH.md`.

기동은 그냥 `./vllm-glm53-exl3.sh RESTARTALL`(원본 + 어댑터가 기본), 조작은
`POST /ab {"scale":0|1}` 또는 codemason `/ab off|on|auto`. 실측: 거부 5/16 → 0/16,
작화는 꺼진 상태 4/60(abliterated 통째 서빙은 18~22/60), 속도 손실 스텝당 1~2ms,
temperature 0 결정성 양쪽 모두 유지(12/12). `STATUS` 가 스위치 상태를 표시한다.

## 5. 실험 기록 (그 시점 조건의 기록 — 당시 타깃은 abliterated 였던 경우가 있다)
### 09-27 — 드래프트 전용 32K 어휘 lm_head (채택, 09-29 main 병합 완료)
- MTP 드래프트 argmax 만 32,768 토큰 서브셋 FP8 W8A16(g128) 헤드로(`overlay/exl3.py` _SubsetDraftHead, `draft_vocab_32k.json`, env `VLLM_GLM53_DRAFT_VOCAB`). verify 는 전체 어휘 → 출력 무손실.
- kmeas 스텝 82.9/80.6/81.9 → 79.2/76.4~77.3/78.5~79.8ms, t/s 32.2/38.0/43.6 → 33.8~34.0/39.2~39.6/44.7~45.2, 수용률 변화 없음. 프리필 12k 1,348/1,362 · 24k 1,327.
- 세트: 필수 4,444(특수·1바이트·UTF-8조각·한글·KS X 1001 한 글자 한자) + 코퍼스 빈도 + BPE 병합순위. 중국어 전용 한자·일본 신자체·가나·키릴/아랍/태국/데바나가리 제외. 재생성 `tools/draft_vocab_corpus_gen.py` → `tools/draft_vocab_build.py`.
- ⚠️ V2 speculator 는 드래프트를 `compute_logits`+argmax 로 뽑는다. 드래프트 헤드 교체는 MTP `compute_logits` 에 해야 실제로 쓰인다(옛 INT4 드래프트 헤드는 get_top_tokens 에만 걸려 있었음).
- **routed 전문가 SwiGLU clamp 누락(상류 Entrpi 버그) 수정** — `VLLM_EXL3_MOE_SWIGLU_CLAMP=1`(코드 기본 0). 모델 `swiglu_limit=10` 이 vLLM `moe_config` 에 있는데 EXL3 메서드가 커널을 직접 불러 넘기지 않았다(이미지 원본 exl3.py 부터 누락, MiaAI 킷은 적용). 4경로 모두: b12x fused Trellis `Caps.swiglu_limit`(실사용) / mixed Trellis `compile_mixed_trellis(swiglu_limit)` / exllamav3 `act_limit` / torch 대체. clamp ON 실측: 괭·과학자 통과, kmeas 34.5/39.7/44.3 · 79.1/77.1/79.1ms(변화 없음). 업스트림 대조: 라우팅 스케일 2.5 는 라우터가 1회(runner 1.0), renormalize·sigmoid·bias 는 라우터, dense 3층·shared expert 는 vLLM `SiluAndMulWithClamp` 로 원래 clamp — 추가 누락 없음. ⚠️ 긴 컨텍스트 외계어+'!' 연타(09-02 codemason 21.5K, NaN 흔적)와의 인과는 **미확정**: 실사용 b12x 경로 중간값은 BF16/FP32 라 fp16 오버플로 경로가 아니다.
- **MTP k 재측정(32K 헤드 적용 후, kmeas t/s hash-map/산문/코드 · 스텝):** MTP2 **35.8**/39.5/42.2 · 61.5~63.1ms / MTP4 33.9/**39.8**/**44.3** · 77.7~79.4ms / MTP5 30.4/35.2/43.3 · 84.5~88.0ms / MTP6 31.3/35.4/39.1 · 93.1~96.3ms / MTP7 미측정(중단). 드래프트가 싸지자 영어는 MTP2 가 +5.6% 앞서지만 산문 동률·코드 -4.7% → **MTP4 유지(유저 결정)**.
- 기각(재론 금지): 라우터 ll_bf16 dotprod(`VLLM_SM12X_LL_ROUTER`, 커널은 빠른데 스텝 +2~3ms) / verify top-p Triton(`VLLM_TOPKP_TRITON_MIN_ROWS`, 소음) / CPU C-state 끄기(무효). 코드는 env 게이트로 보존, 기본 upstream.

채택: dense FP8 Marlin W8A16 / lm_head FP8 / eh_proj FP8 / fp32 인덱서 레이아웃 / 프리필 BF16 이중경로.
폐기(재론 금지, 근거는 메모리): b12x 블록FP8 W8A8(작화) / INT4 드래프트 헤드(수용률 -4%, 정밀도 규칙) / b12x split-K 플래그(무효) / b12x 타일 64x128+CTA3(디코드 -3% 이지만 프리필 -10%) / NCCL_PROTO=LL(무효) / DFlash2(산문 29 vs 36) / MTP k 스윕(09-02 완료, MTP=4).

### 09-11 상류 재조사 + 드래프터 재검
- 상류 상태: lovesenko 웨이트(`c8f58e6a`)·entrpi 이미지(`v2.3-tier1`, digest `44b2dba…`) 모두 갱신 없음 = 우리가 헤드. 부모 양자화 brandonmusic tr3-4bpw 는 09-04·09-10 커밋이 있으나 LICENSE/카드뿐(가중치 무변경).
- **채택**: zai-org 09-04 템플릿 수정 백포트 3건 — visible_text 의 `content=None` 이 리터럴 "None" 을 렌더하던 버그, `'<tool_call>' + tc.name` → `~`, can_sort 루프 단락 가드. 10가지 메시지 형태로 구·신 렌더 대조: `content:null` 케이스만 다르고 나머지 전부 바이트 동일. 다만 vLLM OpenAI 계층이 `content:null` 을 `""` 로 정규화하므로 **현 API 경로에서 실제 피해는 없었다**(`/tokenize` 로 확인). 상류 정합성 목적의 예방 패치.
- 상류 템플릿 통짜 교체는 **금지**: 최신 zai 것은 (a) 미디어 토큰을 emit 하고(당시 우리는 `--language-model-only`. ⛔ 2026-10-30 부터 `VISION=1` 로 이미지를 받는다 — `VISION.md`), (b) `clear_thinking` 기본값이 false 로 바뀌어 과거 생각을 전부 보존한다. 둘 다 우리 의도와 반대. 백포트는 항상 최소 훅으로.
- **폐기(재론 금지)** — 새 DFlash2 드래프터 2종, 모두 MTP-4 에 패배(같은 날 같은 하네스):

| 드래프터 | 영문 | 한국어 산문 | 한국어 코드 | 스텝 |
|---|---:|---:|---:|---:|
| MTP-4 (현행) | **36.1** | **37.4** | **43.6** | 82ms |
| cfontes/Coder40-95 TR3-v3 (재학습, K=7) | 31.1 | 27.3 | 38.2 | 93~97ms |
| incoai 08-31 신규 체크포인트 (K=7) | 30.5 | 29.9 | 40.6 | 93~96ms |

- cfontes 챔피언 설정(K=2)은 이 이미지에서 쓸 수 없다. `qwen3_dflash2.py:143` 이
  `1 + num_speculative_tokens` 를 드래프터 config 의 `block_size`(8)와 대조해 `ValueError`.
  ⚠️ **이건 이미지의 임의 제약이 아니라 학습 조건 가드다.** 소스 주석: "The convolutions'
  block modulus is baked in at training time; a mismatch drafts silently garbage-quality
  tokens." **풀지 말 것** — 풀면 에러 대신 조용히 망가진 드래프트를 쓰게 된다.
  (K 가 자유로운 포크도 있다. 그건 이 가드가 아직 안 들어온 버전일 수 있으므로, 그쪽
  K=2 수치를 그대로 인용하지 말 것.)
- cfontes 카드의 **"+68%" 는 벤더 DFlash 드래프터 대비**(18.9→31.7)이지 MTP 대비가 아니다.
  챔피언 31.7 t/s(4노드 TP4) 자체가 우리 2노드 MTP-4 산문 37.4 보다 낮다.
- **AMD 레인 교차검증(09-29)**: 원본(o_proj 복원) + K 자유 포크에서 재측정 — 산문 accept len
  1.78(카드의 1.83 과 근사)로 **수락률은 정상**, 사이클 119ms 대 MTP2 82ms 로 드래프터가
  37ms 를 더해 패배. **범인은 수락률이 아니라 드래프터 비용**이라는 결론이 양쪽에서 일치.
  DFlash2 계열이 이 모델에서 MTP 를 이긴 기록은 아직 없다.
- 드래프터 자체는 나쁘지 않다(incoai 신규는 코드 수용 2.90 으로 MTP 의 2.56 보다 높다). 지는 이유는 스텝 시간: BF16 드래프터가 스텝을 82 → 96ms 로 올려 수용률 이득을 먹는다.
- ⚠️ **위 표의 조건(09-11)**: 타깃이 abliterated 였고, 드래프터는 BF16 그대로였다(dense FP8
  온라인 양자화는 09-27 도입). 둘 다 지금은 다르다. 그리고 **분야별 수락률을 t/s 와 따로
  남기지 않아**(코드 2.90 하나뿐) 산문이 낮은 이유가 수락률인지 스텝인지 구분이 안 된다 —
  다시 잰다면 분야별 accept/step 을 반드시 분리해서 남길 것.
- 재계산(09-29): FP8 로 드래프터를 깎아도 2.34GB → 1.2GB, 244GB/s 천장에서 약 4.4ms.
  스텝 96 → 91~92ms 로 MTP-4 의 82ms 에 못 미친다. 산문 29.9 대 37.4 를 5ms 로 못 메운다.
  → **MTP-4 유지 결정.** 뒤집으려면 수락률이 아니라 스텝이 내려와야 한다.
- 참고: `VLLM_EXL3_DENSE_FP8=1` 은 비-EXL3 dense 선형층을 로드 시 FP8 Marlin 으로 바꾸는데
  드래프터를 제외하는 코드가 없다. `SPEC=dflash` 와 같이 켜면 드래프터도 FP8 로 갈 것으로
  보인다(미검증). MXFP8 본(`~/Models/glm53-dflash2-mxfp8`)은 HF 기성품이고 **한 번도 안 쟀다**.

## 6. 변경 절차
1. 레인 파일 편집(이미지 원본 직접 수정 금지, 오버레이는 `.orig`/`.diff` 유지) → 2. **git 커밋** → 3. `./sync-lane-67.sh`
→ 4. RESTARTALL(기동 전에 양 노드 대조) → 5. `tools/prefill_tail_nan_check.py`, `tools/kmeas.py`, `tools/prefill_bench.py`,
정밀도 관련이면 `tools/probe_long.py` 로 게이트 → 6. 이 문서와 HISTORY 갱신.
- env 노브를 새로 쓰면 launch.sh 의 통과 목록(`for _v in ...`)에 추가해야 컨테이너에 들어간다. ssh 로는 환경변수가 안 넘어간다.
- 프로파일: `NSYS=1 ./launch.sh 0`(헤드) 후 `docker exec vllm_glm53 /opt/nsys/target-linux-sbsa-armv8/nsys start -o /cache/prof/X.nsys-rep` / `stop`.
  리포트는 Nsight Systems 2025.3.2 의 `nsys stats`(구버전 호스트 nsys 는 못 읽음).
- 단독 GPU·변환 작업은 STOPALL 후에만(통합메모리).

## 7. 복구
- 컨테이너는 매 기동마다 새로 만드므로 지워져도 RESTARTALL.
- 이미지가 지워지면 `docker pull ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`(양 노드, image id `sha256:606b5f3c…` 확인).
- 필요 자산(양 노드 같은 경로): 이 디렉터리, `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw`, `~/Models/glm53-ab-lora`,
  `~/glm53-vllm-cache`(JIT 캐시, 없으면 자동 재생성). 모델·어댑터를 다시 만드는 법은 루트 README §5.5.
