# GLM-5.3-Flash EXL3 레인 — 작업 이력

날짜순 기록. 수치는 그 시점 실측, 해시는 `AmesianX/dspark-serve` 커밋. 상세는 README §5(실험 기록)·HANDOFF·메모리.

## 10-30 ~ 11-01 — 이미지(비전) 수용 main 반영, 한도 5, `.data` 조기 해제 실패
- `exp/glm53-vision` 을 **main 에 FF 머지**(`15047c4`, 27커밋/3,475줄). 브랜치는 로컬·리모트 모두 삭제.
  되돌릴 지점은 `c45c10d`. 라이브가 비전 템플릿으로 뜨며 이미지 인식 확인(도형·색·7자리 숫자·표 전부 정답).
- **장수별 실측**(clean n=2~4, 1792²): 1장 낙폭 **0.00GB**·창 0.0s / 2장 0.44GB·26.2s / 5장 0.78GB·37.3s /
  8장 1.00~1.26GB·창 31.2s(TTFT 30.4 = 창의 97%). 여유 = 헤드 기준선 5.3~6.0 − memwatch 바 3.0.
- **`VISION_MAX_IMAGES` 8 → 5**(`a334496`). 8장 동시 3개는 3.06GB 로 정지하지만 5장은 2.06GB 실측,
  **3/3 성공·바 위 +0.69**. 6장은 0.0초에 400. 유저 실사용(보통 1장, 많아야 2장, 심각할 때 4~5장) 손실 없음.
- ⛔ **`.data`(pixel_values) 조기 해제는 실패**(태그 `glm53-mm-data-release-failed` 로 박제, 브랜치는 삭제. main 미반영).
  1단계(KV 저장 후) 효과 0, 2단계(인코딩 직후+worker 사본) 0.06GB. 이유: 낙폭 1.03GB 중 `AnonPages` 는
  +0.29GB 뿐이고 0.74GB(71%)가 프로세스 anon 밖(CUDA staging 추정, 미규명). 프로세스 RSS 는 요청 중
  변화 0. 파이썬에서 참조를 끊는 접근의 상한이 0.29GB 였다. **산술을 코드보다 먼저 해야 했다.**
- 측정 오염 2건 자가 정정: 창 "38.9초"는 회복 샘플링 6초로 창 끝이 잘린 것(실제 31.2초), 기동 직후
  첫 측정은 ViT 3.3GiB 묶임으로 낙폭 3.4GB 로 오독. 프로토콜은 `VISION.md` §0·§15.
- 문서: `VISION.md` §0(이미지 작업 전 필독)·§15 신설, 레인 `README.md` 에 문서 지도, 낡은 기록 교정
  (루트 README 의 "비전 타워 제외", 삭제된 노브 `VISION_PROFILE`).

## 10-04 ~ 10-05 — 이식성 정리·원본 명명·README·설치 리허설
- 경로·사이트 값 분리: 스크립트가 레인 위치와 `$HOME` 으로 경로를 계산, 사이트 값(레일 IP·워커 ssh·API 키)은 env ① 블록 한 곳. 도구도 키를 env 에서 읽는다(`tools/_lane.py`).
- 서빙 모델 이름을 실체대로: `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw`(원본 저장소와 샤드 sha256·부속 파일 일치 확인). `/ab` mode 는 `original|abliterated`.
- 도구: `fetch_oproj.py`(o_proj 3.89GB 만 Range 다운로드, 스트리밍·최대 메모리 0.3GB), `build_original_from_abliterated.py`, `build_ab_lora.py`(o_proj 파일 입력 허용). 어댑터를 두 o_proj 파일만으로 다시 만들면 운영본과 바이트 동일.
- 미마운트 오버레이 8종·옛 env 백업 삭제(git 이력에만). HANDOFF 부록(낡은 diff 사본) 제거.
- 저장소 루트 README 신규: 구조, 차이점, `/ab`, '!' 버그, 새 설치(§5), 09-05 키트 업그레이드(§6).
- **설치 리허설 2종 실행**: (a) 새 경로에 clone 해 README §5 그대로 → 기동·검증 통과(산문 38.9 / 코드 41.8 t/s). (b) 09-05 키트를 실제로 설치해 abliterated 로 띄운 뒤 §6 그대로 → 원본·어댑터 바이트 동일, '!' 검사 PASS, `/ab` 양방향 정상. 리허설에서 고친 것: 이미지 확인(docker load 이미지는 digest 없음 → image id), git 사용자 정보 없을 때 커밋 실패, 어댑터 빌드 소요(약 10분) 안내, 동기화 출력의 사이트 IP 고정 문구, 모델·어댑터 없음 오류에 받는 곳 안내.

## 10-02 — '!' 붕괴 원인 규명·수정 (fix/glm53-prefill-tail-decode-graph)
- 증상: 추론 채널에 '!'(token 0)만 반복, `finish=repetition`. cm2 실요청 형태(시스템 프롬프트+도구 18개+한국어 과제)에서 첫 턴 100% 재현.
- 원인: **프롬프트 길이 ≡ 5 (mod 512)** 이면 마지막 프리필 조각이 정확히 5토큰(=1+MTP 4)이 되고, V2 러너가 배치 모양만 보고 spec-verify FULL 그래프를 재생 → KDA 순환상태가 안 만들어져 로짓 NaN, 이후 디코드 전부 NaN. 프리픽스 캐시·FlashInfer·컨텍스트 길이 무관.
- 수정: 상류 vllm#51865(08-12, 포크 기점 08-10 이후라 누락) 백포트 — 프리필 중인 요청이 있으면 uniform decode 아님. `overlay/v2runner/`(러너+speculator 2개).
- 검증: `tools/prefill_tail_nan_check.py` 수정 전 7/7 NaN(FAIL) → 수정 후 PASS. 재현 프롬프트 정상 응답.
- 같이 반영: 상류 vllm#57477 백포트(`overlay/glm5next/ops/kpool_compress.py`) — 프리필 tail-seed 커널이 패딩된 인덱서 stride(실측 76032)를 무시하고 덴스(1024)로 써서 **다른 요청의 인덱서 캐시 블록을 덮어쓰던** 버그. A/B(`tools/kpool_cache_probe.py`): 버그판은 같은 캐시 프리픽스의 첫 토큰 logprob 이 다른 요청 처리 후 바뀜(−0.222→−0.122), 수정판은 3라운드 불변. '!' 원인은 아님.
- 도구: `tools/agent_sim.py`(cm2 요청 모양 그대로의 트래픽 재현기 — CODEMASON_DEBUG 로 뜬 요청 본문을 템플릿으로 씀).

## 09-02 ~ 09-03 — 레인 성립
- Entrpi 킷 v2.3-tier1 이미지로 2노드 TP2 서빙 시작(lovesenko tr3-4bpw EXL3 abliterated). NVFP4 레인은 은퇴(attic/).
- MTP k 스윕 → MTP-4 확정.
- dense 층 온라인 FP8(Marlin W8A16): 스텝 117 → 86~89ms(디코드 +40%). b12x 블록 FP8(W8A8)은 작화 유발로 금지.
- 프리필 BF16 이중경로(M>1024): 프리필 12k 1,225 → 1,333 t/s 회복.
- 09-02 codemason 세션 `d5774cad`(21.5K ctx)에서 첫 토큰부터 외계어 + '!' 연타(NaN 로짓 흔적) 1건 — 09-27 기록 조사로 발견.

## 09-09 — DOCA 3.5.0 재측정
- 디코드 34.5/44.1/34.8 t/s(산문/코드/영문), 스텝 82~84.5ms, 프리필 1,341/1,311 — 회귀 없음. RDMA 경로 확증.

## 09-11 — 상류 재조사
- 상류 갱신 없음(우리가 헤드). zai-org 09-04 템플릿 수정 3건 백포트. 새 DFlash2 드래프터 2종 MTP-4 에 패배.

## 09-12 ~ 09-14 — 버전 관리·UTF-8 가드
- `1c35d89` 레인을 git 으로 추적 시작. `320da97` 보존 주석 정리.
- `544a6ad` UTF-8 바이트 가드: 글자 중간 스크립트 전환으로 U+FFFD 가 나던 것을 샘플러에서 차단(7/8 → 0/8, 속도 무손실).

## 09-27 — 스텝 단축·정확성 수정 (main + exp/step70)
**main**
- `948cd83` MoE 라우터 GEMM 중복 제거: 모델과 MoERunner 가 게이트를 두 번 계산(상류 버그, NONAME 세션이 먼저 발견). 디코드 36.83 → 38.44 t/s(+4.4%).
- `d46b184`·`2747dce`·`06178e5` 다른 세션이 남긴 변경(키 교체, DSV4 512K, attic 정리)을 그대로 커밋. 캐시·materials(172G)·상류 킷 클론은 ignore.
- `48bca07` b12x 오버레이를 홈 심링크 훅 대신 launch.sh 에서 직접 마운트(런타임 동일, md5 c89d8c0c).
- `a0dd9ff`·`f0ee751` `sync-lane-67.sh` + RESTARTALL 기동 전 .66/.67 대조(불일치면 서버 건드리기 전에 중단). 브랜치 왕복 롤백 실측 확인.

**exp/step70**
- nsys 재분해(스텝 76.9ms): MoE verify 29.0(대역폭 벽) · Dense 20.0 · lm_head×5 8.1 · NCCL 4.7 · BF16 GEMV 3.7 · 소형 6.2 · idle 4.0.
- `a79a091`·`7f8a865`·`257eacb` 라우터 ll_bf16 dotprod(커널 44→23us, 스텝 +2~3ms → 기각), verify top-p Triton(소음 → 기각). GB10 에서 cuteDSL 8-CTA 클러스터 split-K = launch failure. CPU C-state 끄기 무효(원복).
- `278ae82`·`93ba399`·`98830d3` 드래프트 전용 32K 어휘 lm_head: 스텝 82.9/80.6/81.9 → 79.2/76.4/78.5ms, 수용률 변화 없음. 함정: V2 드래프트는 `compute_logits` 경로.
- `66b41d1`·`a217a24`·`e8de017`·`1d580f2` 감사 수정: 서브셋 헤드는 본 모델 lm_head 에(MTP shared_head.head 는 로드 시 교체됨), 패딩 실제 행, 기각 오버레이 미마운트, 프록시 속성 위임, 재생성 도구(`draft_vocab_*`).
- MTP k 재측정(32K 후): MTP2 35.8/39.5/42.2 · MTP4 33.9/39.8/44.3 · MTP5·6 열세 → MTP-4 유지(유저 결정).
- 프리필 12k 1,348~1,362 / 24k 1,327 — 목표(1,300+) 유지.
- `5200a48`·`6f26fda` routed 전문가 SwiGLU clamp: 상류 exl3.py 가 `swiglu_limit=10` 을 어느 MoE 경로에도 안 넘김(MiaAI 킷은 적용). 4경로 수정, env 게이트(코드 기본 끔, 레인 env 켬). 속도 변화 없음. 실사용 b12x 경로는 BF16/FP32 중간값이라 오버플로 경로 아님 → 외계어 증상과 인과 **미확정**.
- 업스트림 대조: 라우팅 스케일 2.5 라우터 1회, renorm·sigmoid·bias 라우터, dense·shared clamp 정상 — 다른 누락 없음.
- OpenWebUI: DB 에 남은 옛 키(sk-dsv4) → 새 키로 교체, `glm-5.3-flash` 모델 기본값 temperature 1.0 / top_p 0.95(공식 권장), vision 끔. `enable_thinking:false` 시 추론 텍스트가 content 로 새는 템플릿·파서 불일치 확인(codemason·OpenWebUI 는 미해당, 측정 도구만) — 미수정.
- NONAME(AMD 레인)에 32K 드래프트 어휘·라우터 중복·SwiGLU clamp 이식 방법 전달(32K 이식 완료, 스텝 -2.1ms).
- NONAME SwiGLU clamp 반영·검증 완료(같은 env 규칙): 헤드·워커 clamp ON, 괭·과학자 8회 U+FFFD 0, 스텝 70.0ms 불변. temp 0 수용률 1.371→1.320 = 생성 텍스트가 바뀜 → 이상치 clamp 가 실사용에서 실제로 걸린다는 증거. 그쪽 mid_kernel 은 gate/up 원시 GEMM 출력을 FP16(Hadamard 공간)에 저장 → clamp 로 못 막는 별개 오버플로 경로(우리 b12x 경로는 BF16/FP32 라 해당 없음).
- 상태: 09-29 main 병합 완료(패스트포워드, 484fc09). exp/step70 브랜치는 삭제 — 이 문서의
  다른 곳에 남은 `exp/step70` 표기는 당시 작업 브랜치를 가리키는 기록이며, 내용은 모두 main 에 있다.

## 09-28~29 — 품질 감사와 작화 원인 규명

- 상류 감사 7건 수정(thinking budget 이 UTF-8 가드를 우회하던 문제, DCP CKV-gather 의
  656B writer 를 GLM 528B 로, kpool 청크 정렬 가드, scatter_states 의 null 블록 쓰기 차단,
  persistent_topk 폭을 풀 단위로, UTF-8 가드 캐시 키에 토크나이저 해시, V1 사본 인자 순서).
- **작화 원인 = abliteration.** o_proj 46 개만 원본으로 되돌려 가짜 대상 함정 60건에서
  22 → 4. 속도 손해 없음(산문 38.3 → 40.6 t/s). dense 정밀도·EXL3 4bpw·KV fp8·비결정성은
  전부 무죄. 44층만 복원은 효과 없음(22 → 25). 근거와 재현 절차는 `OPROJ-RESTORE.md`.
- 체크포인트를 `glm53-tr3-4bpw-oproj-restored` 로 전환(`glm53-serve.env`). 거부 성능이
  같이 돌아오므로, 펜테스트 용도는 codemason 의 운영자 지시어(`~/.codemason/directive.md`)로
  세션 단위로 지정한다 — 실측상 에이전트 모드는 지시어 없이도 도구를 실행한다.

## 09-29~30 — temperature 0 비결정성 해결

- 원인 셋, 전부 atomic 이 합산 순서를 정하던 것: dense split-K bf16 atomicAdd(`99298bd`),
  sparse 인덱스 압축의 타일 자리선점(`overlay/sparse_utils.py`, GLM 행 폭 2112 가 업스트림의
  2 의 거듭제곱 전용 무원자 경로를 벗어남), 인덱서 top-k 의 atomic 순서·동점 선착순(풀 512 =
  2,048 토큰 초과 시, `_canonicalize_topk_`). 상세 README §4.9.
- 검증(MTP-4): 짧은 배터리 30/30, 2,048 토큰 넘는 생각 켬 출력 8/8, 12k 프롬프트 3/3 동일.
  MTP=0 도 30/30·6/6. 대조군
  (압축 수정만 제거) 10/10 전부 다름. 스텝·프리필(12k 1,352 / 25k 1,332) 불변.
- 추적용 탐침은 커밋하지 않았고, `99298bd` 에 섞여 들어간 디버그 노브(`B12X_TC_DECODE_MAX_M`)는
  `1a5942e` 에서 484fc09 원본으로 되돌렸다.
- 세 원인 모두 업스트림(b12x/vLLM) 결함. ROCm 레인도 ③ 동일 확인(①② 해당 없음). 09-30 양노드 재부팅(원인 미조사) 후 재기동·결정성 재확인, `cc300f8` 푸시.
