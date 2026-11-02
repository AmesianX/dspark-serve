# 원본 vs abliterated — 왜 원본을 서빙하는가

서빙 모델은 `brandonmusic/GLM-5.3-Flash-tr3-4bpw`(원본)다. 9월 초에는 lovesenko 의 abliterated 판을
통째로 서빙했다. 9월 29일 원본으로 바꿨고, 검열 해제는 10월 1일부터 런타임 스위치(`AB-SWITCH.md`)로 한다.
이 디렉터리를 예전에 "o_proj 복원본(restored)"이라 불렀는데, abliterated 에서 o_proj 를 원본으로 되돌려
만든 것이라 그렇게 불렀을 뿐 **결과물은 원본 저장소와 파일 단위로 같다**(아래 "원본과 같다는 근거").

## 왜 바꿨나

`lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated` 는 없는 대상을 실재하는 것처럼
설명하는 작화가 심했다. 2026-09-28~29 규명 결과 **원인은 abliteration** 이다.

가짜 대상 함정 20문항 × 3회 = 60건, 서버 기본값(effort high):

| 조건 | 작화 |
|---|---|
| abliterated, dense FP8 (운영) | 26/60 |
| abliterated, dense BF16 | 22/60 |
| **원본(o_proj 전체 복원)** | **4/60** |
| 44층만 복원 | 25/60 |
| 공식 Z.AI fp8 (OpenRouter) | 8/60 |

무죄로 갈린 것: dense 정밀도(FP8/BF16, AMD INT8 포함), 라우팅 전문가 EXL3 4bpw,
KV fp8_ds_mla, 서빙 비결정성. 원본에서도 이것들은 그대로 켜져 있다.

**44층만 복원은 효과 없다**(22 → 25). 편집이 44층에 집중돼 있지만(행 84% 변경,
다른 층 0.2~4%) 작화는 45개 층 편집이 누적된 결과다. 부분 복원으로 거부 성능과
작화를 절충하는 선택지는 없다.

속도는 손해 없음. 오히려 초안 수락률이 올라 산문이 빨라졌다.

| 항목 | abliterated | 원본 |
|---|---|---|
| 한국어 산문 | 38.3 t/s (accept 2.061) | 40.6 t/s (2.147) |
| 한국어 코드 | 43.5 t/s (2.443) | 43.5 t/s (2.448) |
| hash-map | 35.0 t/s (1.857) | 36.4 t/s (1.941) |

**트레이드오프: 거부 성능이 같이 돌아온다.** abliteration 의 원래 목적이 사라진다.

**남은 한계**: 원본도 완전하지 않다. 희소 어휘 구석에서는 여전히 지어낸다
(괭테스트의 "괭이밥(돌멩이를 부르는 말)" 등). 공식 Z.AI 도 8/60 이다.

## 근거

모델 카드: "The abliteration edits only the BF16 `o_proj` tensors; all
TR3-quantized expert tensors are byte-identical to the base quantization."
베이스 = `brandonmusic/GLM-5.3-Flash-tr3-4bpw`.

직접 검증한 것:
- `weight_map` 150,226 개 텐서가 베이스와 **완전 동일**
- o_proj 46 개 중 **45 개가 다름**, 상대변화 `‖Δ‖_F/‖W‖_F` 평균 0.10
- **45층(MTP)은 바이트 동일** — MTP 헤드는 편집 안 됐다. 카드의 "45 decoder
  o_proj" 가 정확한 표현이고, 46 번째는 원본 그대로다
- 행 노름 변화 중앙값 0.003~0.004 → 카드의 `preserving row magnitudes` 는 사실
  (단 44층만 중앙값 0.046, 행의 84% 가 1% 초과 변경)

## 원본과 같다는 근거 (2026-10-04)

- 원본 저장소(`brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `a5fee929`)가 공개한 LFS sha256 과 대조: o_proj 를 덮어쓴
  샤드 2개, 하드링크로 가져온 샤드 1개 **일치**. config·토크나이저·채팅 템플릿·인덱스·양자화 설정 9개 파일 **일치**.
  다른 것은 lovesenko 의 모델 카드 문서(README/LICENSE/PROVENANCE/NOTICES)뿐 — 서빙과 무관.
- 서빙 중이던 체크포인트에서 o_proj 를 뽑으면 원본에서 받은 파일과 md5 가 같다(`5d1f22fd…`).

## 만드는 방법

- **처음 설치**: 원본을 그냥 받는다(`hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw`). 루트 README §5.5.
- **abliterated 를 이미 가진 경우**: 원본 o_proj 46개(3.89GB)만 HTTP Range 로 받아 덮는다. 루트 README §6.

   ```
   python3 tools/fetch_oproj.py --source original --out ~/Models/glm53-oproj/o_proj_original.safetensors   # md5 5d1f22fd…
   python3 tools/build_original_from_abliterated.py --src <abliterated 스냅샷> \
       --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw
   ```

   영향받는 샤드 44 개만 복사 후 같은 오프셋에 원본 바이트를 덮어쓴다. 나머지 76 개 샤드와 config/tokenizer 는
   **하드링크**. 실사용 추가 용량은 약 72GB. 웨이트는 노드마다 로컬이어야 하므로 양 노드에서 각각 만든다.

## 함정

- **심링크 금지.** 호스트 절대경로를 가리키면 컨테이너 안에서 깨진다
  (`Invalid repository ID or local directory specified`). 하드링크를 써야 한다.
  디렉터리는 하드링크가 안 되는데 로딩에 불필요하므로 건너뛴다.
- **인덱스만 고치는 방법은 안 된다.** `instanttensor_weights_iterator` 는 우선순위
  접두사에 걸리는 텐서만 인덱스 기반으로 읽고, 나머지는
  `safetensors_weights_iterator` 로 **파일별 전체 텐서**를 읽는다. 인덱스를 고쳐도
  원본 샤드의 o_proj 가 또 나와서 로드 순서에 따라 결과가 달라진다.
- 하드링크라 abliterated 스냅샷을 지워도 원본은 살아 있다. 다만 원본을 **제자리 수정**
  하면 양쪽이 같이 바뀐다.

## 판정 방식 주의

작화 판정을 키워드로 하면 오분류가 난다. 실제로 겪은 것:
`"없습니다"` 는 잡는데 `"없고"`/`"없으며"` 를 놓쳤고, `"아닙니다"` 가 다른 문맥
("Вan? 아닙니다 — 정확히는 ...")으로 들어가 작화를 정답으로 셌다. `"표준"` 도 위험.
**원문을 저장하고 눈으로 확인할 것.** 위 표의 숫자도 키워드 판정이며, 원본 4건과
abliterated 22건은 원문을 읽어 확인했다(원본의 진짜 작화는 1~2건).
