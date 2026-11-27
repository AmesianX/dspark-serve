**한국어** | [English](BATCH_INVARIANCE.en.md)

# 온도 0 결정성 — 동시 배치에서 출력이 바뀌는 버그 (배치 불변성)

브랜치 `exp/glm53-kv-offload`(11-27 main 병합). **main에도 원래 있던 버그**. 2026-11-26 착수 → **2026-11-27 해결**.

## ✅ 결과 요약 (2026-11-27, 운영 설정·CUDA 그래프 켬)
- 운영 스위치 하나: env `GLM53_BATCH_INVARIANT=1` → launch.sh가 `GLM53_BI_{DENSE,MHC,MLA,MOE,ROUTER,TOPK,INDEXER}=1`로 풀어 양 노드에 전달.
- 결정성(출력 텍스트 완전 일치): 공동 프리필 e13 3/3(⛔정정 11-27: 접두부를 미리 데운 뒤라 꼬리 프리필만 공동 — 콜드 공동 프리필은 2단계 실험이 검증) · ABCD 동시 e9 8/8 · e7 혼자반복/디코드 중 콜드 프리필 3/3/ABCD 동시 8/8 — 전부 통과.
- 속도(같은 세션 A/B): 디코드 10프롬프트×2 평균 t/s **36.6 → 36.7(동일)**, 스텝 +0.7ms(대부분 출력 문장이 달라져 생기는 전문가 구성 차이 — 프롬프트별로 0~+3ms 들쭉날쭉),
  82K 프리필 1,356~1,373 → **1,340~1,348 t/s(−1.5%)**.
- 커널/수치를 또 바꾸면 `KV_OFFLOAD_HASH_SEED`를 갈 것(현재 `glm53-kv-offload-bi-1`). NVMe가 같은 키의 옛 내용을 계속 쓴다.

### 최종 해법 표
| 지점 | 원인 | 해법 |
|---|---|---|
| dense FP8 Marlin | M≤16 호출만 행 불변, M>1024는 cuBLAS | 디코드 행 Marlin ≤16행 단위(단일 스트림 = 기존 단일 호출), 프리필 행 persistent matmul(dequant BF16). lm_head·역할 미상 >16행은 16행 분할 |
| mHC(vLLM TileLang) | M<8/8~16/>16마다 split·경로 변경 | 디코드 행 = M<8 소형 설정 고정, 프리필 행 = 큰 경로 split 1 고정(`overlay/mhc_tilelang.py`) |
| MLA 희소 어텐션 | 프리필 섞이면 스텝 전체 extend | 혼합 스텝에서 디코드 행 = 디코드 커널, 프리필 행 = extend 분리 호출 |
| MoE trellis | m≤32 trellis / m>32 프리필 플랜 | 디코드 행 trellis 32행 단위, 프리필 행 항상 프리필 플랜 |
| MoE 라우터 GateLinear | GB10은 전용 커널 없음 → 전부 cuBLAS, M·**뷰 위치** 의존 | 디코드 행 고정설정 Triton(`exl3.bi_linear`), 프리필 행 persistent |
| grouped_topk | M>1024 구현 전환 | 1024행 단위 호출(+clone) |
| 인덱서 wk·kpool gate(bf16), 헤드가중치(fp32) | cuBLAS M·뷰 위치 의존 → **K 캐시에 저장**돼 늦게 터짐 | bf16: 디코드 bi_linear / 프리필 persistent. fp32: 디코드 `bi_rowdot_fp32`(10µs, 원래 37µs) / 프리필 tf32x3 |
| 역할 판정 | — | 모델 러너의 호스트 `is_prefilling_np`에서 계산(동기화 0). **CUDA 그래프 캡처 중에는 전 행 = 디코드** |

### 기각한 대안
- cuBLASLt split-K 끄기(vLLM 배치불변 모드의 Hopper/Blackwell 방식, `CUBLASLT_WORKSPACE_SIZE=1`): bf16 작은 GEMM 디코드 2배 느림(12→25µs), 라우터 뷰 위치 의존 잔존(`tools/bi/cublaslt_bi.py`).
- cuBLAS 고정 형상 호출(16/1024행 패딩): 서버 안에서 뷰 위치(포인터 정렬)에 따라 여전히 다름(bi29).
- "스텝당 프리필 1개" 스케줄 제한: 불필요해짐.

## 0. 증상과 원칙

- 온도 0으로 같은 요청을 **혼자** 반복하면 매번 비트 동일. **다른 요청과 동시에** 돌면 출력이 바뀐다.
- 원칙(유저): 온도 0은 언제나 같은 출력이어야 한다. 아니면 "랜덤 출력 기계". 서비스에서 허용 불가.
- 제약(유저): **단일 디코드·프리필 속도를 깎으면 안 된다.** 프리필 1,300+ t/s 유지(기준선 1,377~1,381).
  결정성 얻자고 1,300 → 800 같은 손실은 "미친짓".

## 1. 근본 원인 한 줄

여러 커널이 **"이번 스텝의 전체 행 수(M)" 또는 "이번 스텝에 프리필이 섞였는가"** 로 분할 방식/커널을 고른다.
M은 그 순간 같이 도는 다른 사용자가 정한다 → 같은 토큰의 fp32 합산 순서가 바뀜 → 마지막 비트 차이
→ **KDA 재귀 상태가 그 1비트를 계속 끌고 가며 증폭** → 몇십 토큰 뒤 그리디 선택이 바뀐다.
품질 문제 아님(두 경로 다 수학적으로 같은 계산). 순수 결정성 문제.

고치는 원칙: 커널 선택을 **행 수가 아니라 그 행 자신의 역할(디코드 행 / 프리필 행)** 로 하고, 역할마다 **고정 설정**.
업계의 batch-invariant inference와 같은 방향. 커널을 새로 쓰지 않고 기존 빠른 커널을 고정 크기로 부른다.

## 2. 범인 목록 (확정 순서대로)

| # | 지점 | 무엇이 M/배치 구성에 따라 바뀌나 | 증거 | 고정 스위치 |
|---|---|---|---|---|
| 1 | dense GEMM: FP8 Marlin W8A16 (`overlay/exl3.py` `_Fp8BlockOnlineMarlin.apply`) | Marlin 내부 블록·K분할이 M 의존. **M≤16 호출이면 행별 결과가 M=1과 동일**, 17↑ 다름. M>1024는 BF16 cuBLAS(별개 커널) | `tools/bi/marlin_bi*.py` | `GLM53_BI_DENSE_MARLIN16`(진단) |
| 2 | mHC: **vLLM TileLang** `kernels/mhc/tilelang.py` | M<8: split 8·tile 2 / 8~16: split 4·tile 3 / >16: deep_gemm TF32+`compute_num_split(M)` | 0층 attn_out 동일·mlp_in 다름 | `GLM53_BI_MHC` |
| 3 | MLA 희소 어텐션 `b12x_mla_sparse.py` | **프리필이 섞이면 스텝 전체를 extend 커널로** (디코드 전용은 split-K 디코드 커널) | 3층 attn_out 7.8e-3, 혼합 스텝에서만 | `GLM53_BI_MLA` (디코드 행=디코드 커널, 프리필 행=extend) |
| 4 | MoE trellis (`exl3.py _apply_rank_sliced`) | m≤32 trellis 플랜 / m>32 프리필 플랜 | (라우터 고친 뒤 확인) | `GLM53_BI_MOE` (디코드 행=trellis 32행 단위, 프리필 행=프리필 플랜) |
| 5 | **MoE 라우터 게이트** GateLinear | **GB10(sm_121)은 전용 라우터 커널 조건(Hopper/SM100) 밖 → 모든 M이 cuBLAS F.linear**. cuBLAS는 M=2~16끼리 동일, M=1·17↑ 다름 | MoE 내부 탐침 3a_router | `GLM53_BI_ROUTER` (디코드 행 cuBLAS 2~16행, 프리필 행 persistent) |
| 6 | `ops.grouped_topk` (라우팅 top-k) | **M>1024에서 구현 전환**. ≤1024는 행별 M=1과 동일. ⚠️ 출력 버퍼를 재사용할 수 있어 쪼개 부를 땐 clone 필수 | `tools/bi/topk_bi*.py` | `GLM53_BI_TOPK` (1024행 단위) |
| 7 | **인덱서 bf16 cuBLAS** (`attention.py` `wk_weights_proj`, `index_kpool_compress_gate`) | M=4부터 결과 바뀜. 결과가 **인덱서 K 캐시에 저장**되어, 혼합 스텝이 끝난 뒤의 디코드 스텝에서 토큰 선택이 바뀜(늦게 나타나는 갈라짐) | 혼합 후 M=5 스텝 3층 attn_out | `GLM53_BI_INDEXER` (`exl3.bi_fixed_rows`: 디코드 16행·프리필 1024행 고정 형상) |
| - | 인덱서 fp32 헤드가중치 GEMM | 무죄(모든 M 동일) | `idx_bi.py` | 불필요 |
| - | KDA 디코드 재귀 `fused_recurrent_kda`, FlashKDA 프리필 | 무죄. FlashKDA를 시퀀스별로 쪼개도 결과 동일(→ `GLM53_BI_KDA`는 효과 없음) | bi20 | (불필요) |
| ? | lm_head, MTP 드래프트 | 종단 시험에서 문제로 드러난 적 없음 | | |

**함정: b12x에도 mHC 구현(`overlay/b12x/norm/mhc/`)과 `B12X_MHC_*` 환경변수가 있지만 이 모델은 vLLM TileLang mHC를 쓴다.**

### 2b. 커널이 아니라 "캐시 출처"가 다른 경우 (2026-11-27 발견)

ABCD 동시 시험에서 B·C·D만 틀린 원인 추적: 같은 위치(147,968)에서 읽은 **KDA 초기 상태가 실행마다 다른 버전**이었다
(탐침 `1b_c3_init_state` 체크섬 9e-4 차이). 4개×149K는 GPU 캐시에 다 안 들어가 일부가 **NVMe에서 복원**되는데,
NVMe에는 **예전 설정(수정 전 커널·다른 프리필 경계)으로 계산된 상태**가 있고, 같은 키가 이미 있으면 다시 저장하지 않으므로
옛 버전이 계속 쓰인다. → **수치 경로(커널)를 바꾸면 NVMe 키 공간(`KV_OFFLOAD_HASH_SEED`)도 같이 바꿔야 한다.**
더 근본적으로는 같은 접두부라도 "어떤 청크 경계로 계산됐나"에 따라 상태 비트가 달라진다(§7-4).

## 3. 진단 스위치 (전부 기본 꺼짐, env 파일 + sync-lane-67.sh 로만 — 명령줄 env는 워커에 안 감)

| 스위치 | 효과 |
|---|---|
| `GLM53_BI_PROBE=/cache/bi` + `GLM53_BI_PROBE_MINPOS=<pos>` | rank 0, pos≥MINPOS 행의 층별/하위지점 활성값 덤프. 키=(pos, 입력 임베딩). `m_total` 기록 |
| `GLM53_BI_DEBUG=1` | `[BI-DBG]` 로그: 역할(d,m), 프리필 스텝의 요청별 qlens·seq_lens·prefilling |
| `GLM53_BI_DENSE_PREFILL` / `GLM53_BI_DENSE_MARLIN16` | dense: M>1024 persistent / 전 행 Marlin 16행(진단, 프리필 매우 느림) |
| `GLM53_BI_MHC` `GLM53_BI_MLA` `GLM53_BI_MOE` `GLM53_BI_ROUTER` `GLM53_BI_TOPK` `GLM53_BI_INDEXER` `GLM53_BI_KDA` | §2 표 참조 |

탐침 지점: 층 `1_attn_in` `2_attn_out` `3_mlp_in` `4_mlp_out`, KDA `1b_a_in_proj` `1b_b_f_b_proj` `1b_c2_conv` `1b_c3_init_state`
`1b_c_g_b_proj` `1b_d_recurrent` `1b_e_o_norm` `1b_f_o_proj`, MoE `3a_router` `3b_topk_w` `3c_topk_id` `3d_routed` `3e_shared`.

역할(role) 전달: MLA 메타데이터 빌더가 `is_prefilling`으로 "프롬프트를 끝낸 요청의 행(앞쪽 블록)" 수를 계산(`bi_decode_tokens`)
→ 모델 forward가 `exl3.bi_set_role(d, m)` → MoE/라우터/인덱서가 읽음. 판정 기준은 query 길이가 아니라 is_prefilling.

## 4. 재현 절차 (도구: `tools/bi/`)

```bash
cd tools/bi && export K=<VLLM_API_KEY>
# 탐침 켜고 재기동 (EAGER=1 필요 — env 파일에 임시로)
GLM53_BI_PROBE=/cache/bi GLM53_BI_PROBE_MINPOS=148962 ../../vllm-glm53-exl3.sh RESTARTALL
docker exec vllm_glm53 rm -rf /cache/bi /cache/bi_alone /cache/bi_batch
python3 bi_run.py warm                       # X(148,962토큰)를 GPU 프리픽스 캐시에(탐침 범위 밖)
docker exec vllm_glm53 rm -rf /cache/bi
python3 bi_run.py alone 60                   # X 혼자 60토큰
docker exec vllm_glm53 bash -c 'mkdir -p /cache/bi_alone && mv /cache/bi/* /cache/bi_alone/'
python3 bi_run.py mix                        # X 디코드 중 콜드 프롬프트 2개 프리필 유입 (또는 batch: 다른 3개 디코드 중 X 합류)
docker exec vllm_glm53 bash -c 'mkdir -p /cache/bi_batch && mv /cache/bi/* /cache/bi_batch/'
docker exec -i vllm_glm53 python3 - /cache/bi_alone /cache/bi_batch < bi_compare.py     # 요약
docker exec -i vllm_glm53 python3 - /cache/bi_alone /cache/bi_batch < bi_detail_all.py  # 행별 첫 갈라짐+m_total
```

종단(출력 텍스트) 시험: `python3 e7_det.py 200` — 혼자 반복 / 디코드 중 콜드 프리필 3회 / ABCD 동시 2회.
커널 단독 시험(서버 내린 상태, GPU 프로세스 1개 원칙): `docker run --rm --gpus all ... python3 /w/marlin_bi2.py`.
`p60k.txt`=프롬프트 원문(vLLM 소스, ~82K 토큰). 결정성 시험은 **같은 프롬프트**가 필수라 리포에 보존.

## 5. 실측 기록

- Marlin(N=12800,K=4096) GEMM ms: M=5/12/16 → 0.24 (16행 분할도 0.24, 동일) / M=20·32 → 0.24 vs 분할 0.47~0.49 /
  M=994 → Marlin 3.79, 16분할 15.1, persistent 2.63, cuBLAS 1.51. persistent는 M=1~2048 전부 행 불변.
- 프리필 82K 콜드: 기준(cuBLAS) 1,381/1,377 t/s → dense persistent 1,339/1,362/1,364 (−1.5%). `logs/prefill_ab_dense_bi.out`
- 탐침 진행(`logs/bi*.out`):
  - bi3/bi4: 꼬리 프리필 행이 0층 GEMM에서 갈라짐(PREFILL_BF16 꺼도 동일 → Marlin 자체)
  - bi6: 그 순전파 m_total 혼자 994 / 동시 1003 → 둘 다 Marlin, M만 다름 = Marlin M 의존 확정
  - bi7: 스위치가 워커에 안 감(아래 함정) → rank 0 쪽은 d_recurrent까지 동일해짐
  - bi8: 양 노드 dense 고정 → 남은 갈라짐은 0층 mlp_in(=mHC)
  - bi9: b12x mHC 노브(효과 없음)
  - **bi10: dense+mHC 고정 → 97/97행 최종 노름까지 비트 동일** (시나리오: X 꼬리 프리필 중 타 요청 디코드)
  - e7(dense+mHC): 혼자 반복 ✅, 디코드 중 콜드 프리필 ✗×3, ABCD 동시 ✗×2
  - bi11 (mix): 콜드 유입 전 30행 동일, 혼합 스텝부터 3층 attn_out 갈라짐 = MLA 커널 전환
  - bi12: MLA 분리 → 3층 attn_out 동일, 3층 mlp_out(MoE)으로 이동
  - bi13: MoE 분리만으론 불변(디버그로 역할 전달은 정상 확인) → MoE 내부 탐침 추가
  - bi15: 첫 갈라짐 = `3a_router`(라우터 로짓). ll_bf16 경로 가정이 틀림(GB10은 cuBLAS만)
  - bi16: 라우터 고정 → `3b_topk_w`(3e-7) = grouped_topk M>1024 전환
  - bi17: top-k 고정 → 혼합 스텝 행 전부 동일. 혼합 끝난 뒤 M=5 디코드에서 3층 attn_out = 인덱서 캐시 키
  - **bi18: 인덱서 고정 → mix 시나리오 93/93 최종 노름까지 동일**
  - e7 (7곳 고정): 혼자 반복 ✅, **디코드 중 콜드 프리필 3/3 ✅**, ABCD 동시 A만 ✅
  - e8 (A+B 짝): 3/3 ✅ / e9 (ABCD + qlens 로그): 꼬리를 혼자 프리필한 요청만 맞음
  - bi19 (co: A가 B·D와 공동 프리필): 0층 d_recurrent / bi20: FlashKDA 시퀀스별 호출해도 동일(무죄)
  - bi21: `1b_c3_init_state` 체크섬 다름 → 캐시에서 읽은 KDA 초기 상태 자체가 다른 버전 (§2b)
- 라우터 마이크로벤치(4096→288): cuBLAS M=5 12.4µs / persistent 101µs(디코드 부적합) / M=9236 cuBLAS 454 vs persistent 667µs
- 인덱서 마이크로벤치: wk_proj·gate bf16 M=5·16 모두 ~12µs, M=1024 ~40µs, M=9236 ~400µs (1024 고정 10회 ≈ 단일 호출)

## 6. 함정 모음 (다시 밟지 말 것)

1. **명령줄 env는 워커(.67)에 안 간다.** `VAR=1 ./vllm-glm53-exl3.sh RESTARTALL`은 헤드만. TP2라 o_proj 합에서
   워커 쪽 차이가 섞여 엉뚱한 결론이 난다. 양 노드에 필요한 스위치는 **env 파일 + sync-lane-67.sh**. 확인:
   `ssh 10.0.1.2 docker exec vllm_glm53 printenv <VAR>`.
2. 탐침 키는 (pos, 임베딩). 드래프트가 거부돼 같은 (pos,token)이 두 번 계산되면 `occ=2` — 비교 무효, 무시.
3. 비전 레인은 `input_ids=None`(inputs_embeds) → 행 식별은 임베딩으로.
4. 모델 config의 EXL3 범위는 `glm53_routed_experts_only` — 라우티드 전문가만 trellis, **나머지 dense는 FP8 Marlin 오버레이**.
5. 단일 GPU 원칙: 커널 단독 시험은 서버 STOPALL 후에만.
6. "혼자 반복 동일"은 결정성 증거가 아니다(같은 M이면 항상 같음). 반드시 배치 구성을 바꿔 비교.
7. 커밋 안 된 변경이 있으면 `RESTARTALL`이 "⛔ 기동 중단: 레인 불일치"로 **재기동을 건너뛰고 옛 서버가 계속 돈다**.
   로그 파일 복사도 커밋해야 한다. 재기동 후 `printenv`로 스위치 확인 습관.
8. 디버그 로그 예산(`_bi_dbg` 120줄)은 기동 워밍업 더미런(m=9236)과 순수 프리필 스텝이 먼저 써버린다 → 필터 필수.
9. vLLM `GateLinear`의 ll_bf16/dsv3/fp32 전용 경로는 Hopper/SM100 전용. GB10(sm_121)은 전부 cuBLAS.
10. 키 충돌: 같은 본문·같은 길이 프롬프트를 여러 개 동시에 탐침하면 (pos,임베딩) 키가 겹친다. 헤더 길이가 달라 위치가 어긋난
    프롬프트(E5-A/B/D: 148962/148955/148957)를 쓰면 구분된다.
11. NVMe 오프로드 캐시는 키가 같으면 옛 내용을 계속 쓴다. 수치 경로를 바꾼 실험은 `KV_OFFLOAD_HASH_SEED`를 바꿔 격리.

## 6-b. ⚠️ 빈틈: 짧은 동일 프롬프트의 동시 프리필 (2026-11-27 발견, 미해결)

/ab 분리 검증 중 발견. `GLM53_BI_*` 7개는 양 노드 모두 켜진 상태.

| 조건 | 결과 |
|---|---|
| ~30토큰 같은 프롬프트 3개 동시 전송 | 2/3 이 **첫 토큰부터** 다름 (max\|Δlp\| 1.364, 텍스트도 다름) |
| 같은 프롬프트 2개 동시 | 2/2 다름 |
| 이웃 2요청이 디코드 중일 때 다른 요청이 끼어듦 (0.4s·2.0s 간격) | 4회 모두 이웃 160토큰 logprob 비트 동일 |

2단계(17K 프롬프트, 4608 정렬 청크)의 공동 프리필 통과와 모순은 아니다 — 짧은 프롬프트 여러 개가
한 스텝에 같이 프리필되는 경우는 시험한 적이 없다. 재현: `tools/bi/ab_iso_probe.py diff`, `plain3`.
원인 미조사(후보: 짧은 varlen 프리필 행이 여럿일 때의 KDA 청크 경계 또는 MLA/인덱서 프리필 경로).

## 6-c. ✅ 해결: 긴 문맥 디코드에서 인덱서 top-k 가 틀린 문맥을 고르던 버그 (2026-11-27 발견·수정)

**증상.** 18K 넘는 프롬프트에서 temperature 0 콜드끼리(동시 요청 없음, 캐시 무관) 출력이 토큰 4~9 에서 갈렸다.
22.7K 는 동일, 38K·40K·41K 는 갈림. 11-27 최종 검증에서 발견했는데, 처음엔 이걸 18K 로 시험을 줄여 숨겼다(실수 — 아래 교훈).

**추적 (하나씩 끄기 → 위치 → 행 → 커널).**
1. 배치 불변 OFF / NVMe 오프로드 OFF(커넥터·salt 코드 포함) / `/ab` 훅 OFF / 새 캐시 오버레이(block_pool·kv_cache_manager) 제거 — 넷 다 재현. 그날 변경과 무관.
2. `prompt_logprobs` 로 38,264 위치 전부 2회 비교 → **완전히 동일**. 프리필은 결정적이고, **디코드**에서 갈린다.
3. BI 탐침(`GLM53_BI_PROBE`, EAGER): 두 실행은 디코드 스텝 구성·MTP 초안 토큰까지 같았는데, **첫 디코드 스텝(MTP 검증 5행)의 행 1·2 에서 층 7 어텐션 출력**이 달랐다(입력은 비트 동일).
4. 디코드 인덱서 입출력 덤프(진단 전용, 이후 제거): 유효 길이 안 logits 는 동일한데 `pool_topk` 가 행 1·2 에서 다르고,
   **정답 top-512 와 겹침이 432~451/512**. 행 2 는 정렬이 깨지고 최고점 묶음(0번, +93.3)이 빠져 있었다.

**원인 (두 겹).**
- 빠른 디코드 top-k(`persistent_topk`)가 MTP 검증 행 일부에서 정확한 top-512 가 아닌 집합을 돌려준다(정답 기준 −129.8 보다 낮은
  −130.1 묶음이 섞임). 유효 길이 밖 logits 는 `clean_logits=False` 라 버퍼에 남은 값(+1~+30)이다.
- 정규화 커널(`_canonical_topk_kernel`)이 그 집합의 **최솟값을 기준값으로 믿었다.** 기준이 너무 낮으면 "기준 초과"가 K 를 넘고,
  **쓰기에 경계 검사가 없어 다음 행의 K 칸을 덮어썼다.** 행 2 의 깨진 정렬·빠진 최고점 묶음이 이 덮어쓰기 흔적이다.
- 결과: 긴 문맥에서 일부 행이 **틀린 문맥(최대 ~80/512 묶음)** 으로 디코드했고, 남은 버퍼 값이 실행마다 달라 비결정적이었다.
  결정성 문제이기 전에 **답의 품질 문제**다.

**수정 (`overlay/sparse_attn_indexer_kpool.py`).**
- 정규화 커널이 기준값을 **검증**한다: `n_above < K ≤ n_above + n_at` 이면 정확한 K번째 값. 실패한 행만 float 를 순서 보존
  int32 키로 바꿔 **32회 이분 탐색**으로 정확한 K번째 값을 찾는다 — 유일하므로 정답이고 실행마다 같다. 정상 행은 추가 비용 0.
- 쓰기에 `pos < K` 경계 — 다음 행 덮어쓰기 원천 차단.
- top-k 전에 각 행 길이 밖 logits 를 −inf 로(`_mask_logits_past_row_len_`), 두 디코드 top-k 경로 모두.

**검증.**
- 수정 후 덤프: 840행 전부 정답 top-512 와 일치, 두 실행 84/84 스텝·층 동일.
- 운영 설정(CUDA 그래프 켬): 18K·38K·78K 콜드×3 비트 동일. 44.7K F1(적중)·F2(동시 처리) 비트 동일(KV_OFFLOAD_TESTPLAN.md §6).
- 속도: kmeas 34.9 / 39.0 / 44.8 t/s(수정 전과 같음). 38K 문맥 디코드 34.0 t/s(혼자 돈 회차, 수정 전과의 A/B 는 아님).

**교훈.** 결정성 시험은 출력 텍스트가 아니라 **logprob 까지**, 그리고 **18K 를 넘는 길이**로 할 것. 이 버그는 BI 작업 때 텍스트만
비교한 시험과 18K 이하 검증 사이에 숨어 있었다. 실패한 길이를 시험에서 빼면 결함이 "통과"로 숨는다.

## 7. 남은 일 (2026-11-27 기준 1~5 전부 완료)

1. (진행 중) 새 해시 시드로 ABCD 종단 시험 → 커널 수준 배치 불변 완성 여부 판정.
2. 최종 구현(진단 스위치 → 운영 경로): 역할 기반 분기, 속도 손실 없는 형태로
   - dense: 디코드 행 → Marlin 16행 단위(동시 5명까지 16행 이하라 지금과 동일 커널·속도), 프리필 행 → persistent matmul(−1.5%)
   - mHC: 디코드 행 → M<8 설정 고정, 프리필 행 → 비용 실측 후 결정
   - MLA·MoE·라우터·top-k·인덱서: 현재 진단 구현을 정리해 운영 경로로
   - CUDA 그래프(디코드 전용 스텝)에서 동작 확인
3. 운영 설정에서 검증: kmeas 단일 디코드(≈34.2/38.3/41.2), 프리필 1,300+, e7·e9 전 항목 동일.
4. ✅ 2단계(캐시 출처) 2026-11-27 통과 — 수정 불필요. 콜드 프리필 8개 부하 조건·캐시 이어쓰기·통째 콜드·분기점 부분 적중 전부 일치.
   이유: align 모드 스케줄러가 청크 끝을 상태 격자(4608)에, 적중을 prefix-match-unit(512) 배수에 맞춰 모든 분할점이 KDA 청크(64) 배수.
   절차·조건은 `KV_OFFLOAD_TESTPLAN.md` §3, 스크립트 `tools/bi/s2_prov.py`.
5. 끝나면 env 파일의 `[BI 진단 임시 — 지울 것]` 줄 전부 제거(EAGER·해시 시드 포함), 운영 설정으로 재기동.
