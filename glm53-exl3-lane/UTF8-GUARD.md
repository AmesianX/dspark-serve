# UTF-8 바이트 가드 — 설계·패치·검증 문서

작성 2026-09-14 · 레인 `glm53-exl3-lane` · vLLM `0.26.1rc1.dev1006+gf223ff9f2.d20260902`
이미지 `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`

---

## 1. 증상

GLM-5.3-Flash 응답에서 **한글 이름 한 글자가 U+FFFD(`�`)로 깨져** 나온다. 재현 문장은
"양자역학에 기여한 과학자 10인" 류의 나열이고, 깨지는 자리는 거의 항상 같다:

```
4. **�erner 하이젠베르크(Werner Heisenberg) - 행렬역학 창시, 불확정성 원리
     ^^^ "베르너" 가 들어갈 자리
```

기간이 길다. 대화형에서 "하이젠베르크 앞 이름이 항상 깨진다"는 관찰이 수개월 있었다.

## 2. 원인 규명 (한 걸음씩 배제)

### 2.1 클라이언트 무죄
codemason(파이썬·러스트) 양쪽에서 동일 증상. 녹화한 정상 SSE 스트림을 **1바이트씩**
쪼개 클라이언트에 재생 → FFFD 0개. 스트림 디코더는 `codecs` 증분 디코더로 미완성
바이트를 붙들고 있어 경계 분할에 안전하다.

### 2.2 서버가 보낸 바이트에 이미 깨져 있음
투명 프록시(`scratchpad/proxy.py` 방식)로 원본 SSE를 그대로 떠보면:

```
data: {... "delta":{"content":" **\xef\xbf\xbderner"} ...}
```

`EF BF BD` = U+FFFD 가 **서버 응답 바이트 안에** 있다. 클라이언트가 만든 게 아니다.

### 2.3 디토크나이저 버그도 아님
`vllm/v1/engine/detokenizer.py` 는 `tokenizers.decoders.DecodeStream` 을 쓴다. 미완성
바이트를 붙들고 있다가 **완성이 불가능해졌을 때** U+FFFD 를 낸다. 즉 정직하게 동작했다.
(알려진 vLLM #27300 은 logprobs 출력 한정이며 PR #34875 로 이미 수정, 우리 빌드에 포함.
구버전 문제가 아니다.)

### 2.4 진짜 원인 — 모델이 글자 한복판에서 스크립트를 갈아탐
`logprobs`(top 8)로 그 자리를 뜬 결과:

```
1번 토큰  선택 ''  bytes[]  logprob -0.358      ← 0xEB 단독 바이트 토큰(id 167)
   후보:  '루'[235,163,168] -1.23 · '르'[235,165,180] -5.42
          '버'[235,178,132] -7.61 · ' Werner'[32,87,...] -7.98
2번 토큰  선택 'erner' -0.578                   ← 라틴으로 갈아탐
   후보:  'erner' -0.578  vs  '베'[235,178,160] -0.828     (≈ 5:4)
```

모델은 "베르너"를 쓰려고 **한글 음절의 선행 바이트 0xEB 를 뽑았다(정상)**. 그 다음
스텝에서 한글을 완성하는 `'베'` 와 라틴 `'erner'` 가 거의 반반이었고 `'erner'` 가 뽑혔다.
그 순간 앞의 0xEB 는 **영원히 글자가 될 수 없다**.

**결함의 소재**: 첫 선택이 아니라, *연속 바이트를 기다리는 상태에서 라틴 토큰이 후보로
남아 있었다는 것*. 바이트 흐름이 UTF-8 오토마톤 밖으로 나가는 걸 어느 층도 막지 않았다.

### 2.5 온도는 원인이 아니라 빈도 조절기
codemason 경유, 두 프로파일을 **번갈아** 8쌍(시간대 편향 제거), U+FFFD 포함 응답 수:

| 팔 | 온도 | top_k | 깨짐 |
|---|---|---|---|
| A | 1.0 (벤더) | 없음 | **7/8** |
| B | 0.6 | 40 | **1/8** |
| D | 0.6 | 없음 | **2/6** |

부호검정 p≈0.016. 온도가 큰 축, top_k도 일부 기여. **그러나 0.6도 깨진다** — 증상 빈도만
낮출 뿐 결함은 그대로다. 그래서 온도는 벤더 권고 1.0으로 되돌리고 결함을 직접 고친다.

---

## 3. 설계

### 3.1 원리
요청마다 **UTF-8 디코더 상태**(Björn Höhrmann DFA)를 실제 생성 토큰 위로 추적한다.
글자가 미완성인 동안에는 **그 글자를 완성할 수 없는 모든 토큰을 -inf 로 마스킹**한다.
글자 경계(ACCEPT)에서는 아무것도 마스킹하지 않는다.

내용 제약이 아니라 **형식(well-formedness) 제약**이다. 경계에서는 전 어휘가 열려 있고,
중간에서는 그 글자를 완성하는 조각만 열린다. 도달 가능한 텍스트는 하나도 금지되지 않는다.

### 3.2 표
- 상태: `0(ACCEPT), 24, 36, 48, 60, 72, 84, 96` + `12(REJECT)` — 12배수 인코딩
- `bad[state_row, token]` bool — 이 상태에서 이 토큰을 막아야 하는가
- `next_state[state_row, token]` uint8 — 이 토큰 뒤의 상태
- 크기: 8 × 154,856 × (1+1) ≈ **2.4MB**, 기동 시 1회 생성 후 `/root/.cache/vllm/utf8_guard/` 캐시
- 토큰→바이트: `xgrammar.TokenizerInfo.from_huggingface(...).decoded_vocab`
  (실패 시 GPT-2 byte-level 알파벳 폴백)

### 3.3 비용
경계 상태인 행은 마스킹 대상이 아니므로 **평상시 텐서 복사 0**. 미완성 상태인 행이 있을
때만 `bad[rows]` gather + `masked_fill_`. 실측 어휘 기준 경계에서 막히는 토큰은 256개
(연속 바이트로만 시작하는 것들), 글자 중간에서 허용되는 토큰은 상태별 최대 211개.

---

## 4. MTP(speculative decoding) 문제 — 이 패치의 핵심

스톡 vLLM은 두 군데서 커스텀 로짓 프로세서를 막는다.

### 4.1 아예 거부
`vllm/v1/sample/logits_processor/__init__.py:build_logitsprocs`

```python
if vllm_config.speculative_config:
    if custom_logitsprocs:
        raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
```

→ spec decode가 켜져 있으면 커스텀 프로세서를 **기동 단계에서 거부**한다.

### 4.2 거부하는 이유(구멍)
`vllm/v1/sample/rejection_sampler.py:apply_logits_processors`

```python
for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
    if isinstance(processor, MinTokensLogitsProcessor):
        logits = processor.apply_with_spec_decode(logits, metadata.num_draft_tokens)
```

초안 검증 로짓에는 `MinTokens` **하나만** 적용한다. 나머지는 조용히 건너뛴다. MTP에서는
토큰 대부분이 이 경로에서 결정되므로, 거부가 없었다면 가드가 사실상 무력했을 것이다.

### 4.3 우리 해법 — 가드를 깨지 않고 정밀화
1. **구멍을 메운다**: `apply_with_spec_decode_tokens(logits, num_draft_tokens, spec_token_ids)`
   를 노출한 프로세서를 초안 위치별로 호출한다. 행 배치는 요청-메이저이고, 요청 r의 j번째
   행은 *그 요청의 초안 0..j-1이 이미 문맥에 들어간 상태*로 샘플링되므로, 진입 상태는
   커밋된 상태에서 초안 접두사를 따라 전진시켜 구한다.
2. **거부를 능력 검사로 바꾼다**: 그 메서드를 가진 프로세서만 통과시키고, 없는 프로세서는
   **기존과 똑같은 에러로 거부**한다. 업스트림 가드의 의도(무력한 프로세서를 받지 않는다)는
   그대로 유지된다.

---

## 5. 파일 (최종 구성)

| 파일 | 역할 |
|---|---|
| `overlay/vllm_utf8_guard.py` | 가드 본체(신규). 컨테이너 `site-packages/vllm_utf8_guard.py` 로 마운트 |
| `overlay/gpu_sampler.py` | **V2 러너 샘플러** `vllm/v1/worker/gpu/sample/sampler.py` 에 가드 호출 1줄(+`.orig`, `.diff`) |
| `launch.sh` | 마운트 2줄 + `-e VLLM_UTF8_GUARD` / `-e VLLM_UTF8_GUARD_TOKENIZER=$MODEL_PATH` |

**켜고 끄기**: `UTF8_GUARD=0 ./vllm-glm53-exl3.sh RESTARTALL` (컨테이너 안 `VLLM_UTF8_GUARD=0` 과 동일 효과, 오버레이 원복 불필요)

### 5.1 왜 이 자리인가 — 버려진 경로 셋
1. **커스텀 LogitsProcessor 등록** → vLLM 이 모델 러너를 V2→V1 으로 강등시킨다
   (`Model Runner V2 does not yet support custom logits processors`). 이 모델은 V1 경로에서
   뜨지 못한다: 드래프터가 `Glm5NextConfig.image_token_index` 에서 죽고, 그걸 별칭으로
   메우면 다음은 `_get_kv_cache_groups_glm5_next`의 `assert len(mla_pages) == 1` 에서 죽는다.
   **상류 결함이며 우리 문제와 무관한 다른 경로**다. → 폐기.
2. **`v1/sample/sampler.py`, `v1/sample/rejection_sampler.py` 패치** → 이 레인은 V2 러너를
   쓰므로 그 파일들은 **호출되지 않는다**(가드 초기화 로그가 아예 안 찍혀 발각). → 폐기.
   (그 흔적이던 `logitsproc_init.py`·`glm5_next_config.py`·V1 `sampler.py`·`rejection_sampler.py`·
   `gpu_input_batch.py` 오버레이는 2026-10-04 저장소에서 지웠다. git 이력에만 있다.)
3. **채택**: V2 샘플러의 `apply_sampling_params()` 한 곳. 일반 디코드와 **MTP 초안 검증**
   (`gpu/spec_decode/rejection_sampler.py` 가 초안 토큰을 `input_ids` 자리에 넣어 같은 함수를
   부른다)이 **모두 이 함수를 지난다**. 즉 한 줄로 두 경로가 덮인다.

### 5.2 상태를 어디서 얻나
V2 샘플러는 각 행의 **직전 토큰**을 `input_ids` 로 이미 GPU에 들고 있다. 토큰 하나에서
"경계에서 출발했을 때 도달 상태"를 미리 표로 만들어 두면, 매 스텝 **게더 1회 + masked_fill 1회**
로 끝난다. 호스트 동기화(D2H 복사)가 없어 스트림이 멈추지 않는다.
- 요청별 상태 추적·배치 재정렬 대응이 전부 불필요해졌다.
- 한 글자가 **3토큰 이상**으로 쪼개진 경우 중간 조각 다음 행은 상태를 한 토큰에서 읽을 수
  없어 "허용 행"으로 보낸다(마스킹 안 함). 관측된 실패는 2토큰 형태이며, 이 구멍은 남아 있다.
- 엔진 설정에 의존하지 않는다: 어휘 크기는 **로짓 너비**에서, 토크나이저 경로는
  `VLLM_UTF8_GUARD_TOKENIZER` 에서 받는다(샘플링 시점엔 `get_current_vllm_config()` 가 무효).

## 6. 검증

### 6.1 배포 전 (서버 무접촉)
- **DFA 정확성**: 실제 텍스트(한글·라틴·일본어·이모지·기호)의 **모든 부분열**을 파이썬
  `bytes.decode('utf-8')` 판정과 대조 → 전부 일치. 랜덤 바이트열 6만개 → 전부 일치.
  과잉인코딩(`C0 AF`, `E0 80 80`)·서로게이트(`ED A0 80`)·범위초과(`F4 90 80 80`)·고립
  연속바이트(`80`)·5바이트 시퀀스 → 전부 거부.
- **대기 상태**: `EB` → 상태 36, `EB B2` → 상태 24 (ACCEPT도 REJECT도 아님)
- **버그 케이스**: 상태 36에서 `'e'` → REJECT, 이어지는 `B2` → 정상 진행
- **실제 어휘(154,856)**: 한글·영문·이모지 혼합 문장이 한 토큰도 막히지 않고 끝에서 경계
  복귀. 토큰 167(`0xEB`) 뒤 상태 36에서 토큰 41138(`'erner'`) **차단 확인**

### 6.2 배포 후 실측 (2026-09-14 02:4x, 2노드 라이브)

**깨짐률** — A팔(벤더 temp 1.0, codemason 경유, 같은 프롬프트) 8회:

| | 깨짐 |
|---|---|
| 패치 전 | **7/8** (`**\ufffderner 하이젠베르크`) |
| 패치 후 | **0/8** (`**베르너 하이젠베르크`) |

8회 전부 이름이 온전하게 나왔다. 부호검정 기준으로도 우연일 확률은 1% 미만이다.

**속도** — 같은 질의 3회 평균:

| | t/s |
|---|---|
| 패치 전(당일 실측) | 37.9 |
| 패치 후 | **36.23** (35.58 / 36.62 / 36.48) |

오차 범위. 설계대로 스텝당 비용이 게더 1회라 측정에 잡히지 않는다.

**기동**: API READY 210초, KV 847,533 토큰 — 평소와 동일.

## 7. 위험·유지보수

- **이미지 교체 시**: 두 파일(`vllm_utf8_guard.py`, `gpu_sampler.py`)은 이 빌드의 소스에 맞춘 것이다. 이미지가 바뀌면
  `.orig` 와 새 이미지의 원본을 diff 해 재이식할 것. 그냥 얹으면 조용히 구버전 코드가 된다.
- **샘플러는 핫패스**다. 변경은 최소(게더 1회 + masked_fill 1회)로 유지하고, 문제가 보이면
  `UTF8_GUARD=0` 으로 즉시 무력화.
- **V1 런너 강등**: 위 경고대로 커스텀 LP는 현재 V2 런너와 공존하지 않는다. t/s 손실이
  크면 (a) 가드를 끄거나 (b) V2 런너용 경로를 따로 파야 한다.
- **상류 기여 후보**: 4.3의 두 패치는 모델·레인 비의존이다. "spec decode에서 커스텀 LP를
  능력 기반으로 허용"은 vLLM 본체에 낼 만하다.

## 8. 재현 도구

- 프록시 캡처: 서버↔클라 사이에 끼워 원본 SSE 바이트 저장
- `logprobs` 탐침: `logprobs:true, top_logprobs:8` 로 깨진 델타의 선택/후보 토큰 바이트 확인
- 하네스: `CODEMASON_CONFIG_DIR` 로 설정 디렉터리를 복제해 프로파일별 N회 실행 후
  U+FFFD 포함 응답 수 집계 (실제 config.json 무접촉)

---

## 9. 상태: 적용 완료

가드 ON 으로 양 노드 라이브. 되돌리려면 `UTF8_GUARD=0 ./vllm-glm53-exl3.sh RESTARTALL`.

**다음에 손볼 것**
- 3토큰 이상으로 쪼개진 글자(§5.2)를 덮으려면 요청별 꼬리 상태를 들고 있어야 한다.
  지금 구조에 얹을 수는 있으나, 관측된 실패가 아니므로 미룬다.
- 딥식이 레인 이식: 가드 본체는 모델 비의존(토크나이저에서 표 생성)이라 재사용 가능하다.
  단 ① `gpu_sampler.py` 는 **그 이미지의 원본에서 다시 떠서** diff 재적용
  ② 토크나이저가 `<0xEB>` 식 byte-fallback 표기면 어휘 추출기를 보강해야 한다
  (현재는 GPT-2 바이트레벨 알파벳 기준, 미해독 토큰은 "바이트 없음"으로 떨어져 **조용히
  무력해질 수 있음** — 이식 시 기동 로그의 미해독 토큰 경고를 반드시 확인할 것).
- 상류 기여 후보: "spec decode 에서 커스텀 LP를 능력 기반으로 허용"(§4.3)은 모델 비의존이다.
