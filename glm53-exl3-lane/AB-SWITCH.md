# 런타임 abliteration 스위치 (`/ab`)

원본(거부 능력 있음, `brandonmusic/GLM-5.3-Flash-tr3-4bpw`) 가중치를 서빙하면서, abliteration 편집을 **요청 단위로 켰다
껐다** 한다. 재기동도 재프리필도 없다.

```bash
curl -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"scale":1}' http://localhost:8000/ab     # 검열 해제 ON
curl -X POST ... -d '{"scale":0}' http://localhost:8000/ab   # OFF(기본)
curl -H "Authorization: Bearer $KEY" http://localhost:8000/ab # 현재 상태
# → {"scale":0.0,"workers":[0.0,0.0],"consistent":true,"mode":"original"}
```

## 왜 이 구조인가

abliteration 은 거부를 없애면서 **"없다"고 말하는 능력도 같이 없앤다**. 요청 거절,
없는 대상 부정, 틀린 전제 반박이 모델 안에서 같은 방향을 쓰기 때문이다. 실측:
가짜 대상 60건 함정에서 원본 4건, abliterated 18~22건. 링크에 없는 결과물을 있다고
하거나 없는 버그를 우기는 증상이 여기서 나온다.

그래서 **끄는 것이 기본**이고, 켜는 구간을 "요청을 승낙하는 그 한 턴"으로 좁힌다.
도구 결과·링크·diff 를 판단하는 구간은 전부 꺼진 채로 돈다. 판단이 필요 없는 구간으로
손상을 밀어내는 것이 설계의 핵심이다.

## 구현

`o_proj` 는 4bpw EXL3 체크포인트에서도 **BF16 그대로**다(양자화된 것은 MoE 전문가뿐).
그래서 두 체크포인트의 차이를 저랭크로 뽑아 서빙 중에 더할 수 있다:

    o_proj(x) + scale * (x @ A.T) @ B.T      # W_abl = W_original + B @ A

| 항목 | 값 |
|---|---|
| 편집된 층 | 45개 (`layers.0~44`). **45층=MTP 는 바이트 동일** |
| ΔW rank | 층당 19~20 (에너지 99.9%). rank 24 로 자르면 잔차 0.5~2% |
| 어댑터 크기 | **29MB** BF16 (통째 사본은 8~9GB) |
| 재현 정확도 | scale=1 에서 상대오차 0.17%, cos 0.999999. **scale=0 은 정확히 0** |
| 층 분포 | KDA 34층(in 8192) + sparse MLA 11층(in 16384) |

44층만 ΔW/W 가 28%(다른 층 8~12%)이고 첫 특잇값이 85%를 차지한다 — 거부는 여기
몰려 있다. 단 **작화는 45층 전체에 퍼져 있어** 44층만 복원하면 안 잡힌다(22→25).

### 파일

| 파일 | 역할 |
|---|---|
| `tools/build_ab_lora.py` | 두 체크포인트(또는 `tools/fetch_oproj.py` 로 받은 o_proj 파일) 차이 → rank-24 어댑터 추출 |
| `overlay/ab_switch.py` | 훅 본체. scale 은 GPU 텐서(그래프 재캡처 불필요) |
| `overlay/mla.py` | sparse MLA 11층 훅 (`mla.py:405` 의 `o_proj` 출구) |
| `overlay/glm5next/kda.py` | KDA 34층 훅 (`kda.py:344`) |
| `overlay/ab_plugin/` | `/ab` 라우트 + 워커 전파(`collective_rpc`) |

플러그인은 vLLM 공식 확장점을 쓴다. 서버 코드를 고치지 않는다. TP 라 스케일이 모든
워커에 닿아야 해서 두 짝으로 나뉜다 — `vllm.general_plugins` 가 워커 RPC 를 설치하고
`vllm.endpoint_plugins` 가 HTTP 로 노출한다. 둘 다 `VLLM_PLUGINS` 에 있어야 뜬다.
컨테이너에 pip install 을 못 하므로 `dist-info/` 를 만들어 마운트해 엔트리포인트를
발견시킨다.

### 기동

```bash
./vllm-glm53-exl3.sh RESTARTALL     # 원본 + 어댑터가 기본값(glm53-serve.env)
```

`GLM53_AB_LORA` 가 비면 훅이 통째로 비활성 — 원본 그대로다. 어댑터는 원본 위에서만
의미가 있고, 이미 abliterated 인 체크포인트를 띄우면 `launch.sh` 가 스위치를 꺼서
편집이 두 번 들어가는 것을 막는다.

## 실측 (2026-10-01, 2노드 TP, MTP-4)

| 항목 | scale=0 | scale=1 | 어댑터 없던 기준 |
|---|---|---|---|
| 디코드 산문 / 코드 / hash-map | 39.2 / 41.7 / 35.4 | 37.2 / 42.5 / 34.2 | 39.4 / 42.5 / 35.2 |
| 스텝 | 78.9~80.9ms | 79.1~82.0ms | 78.5~80.3ms |
| 프리필 12k / 25k | 1,332 / 1,319 | — | 1,352 / 1,332 |
| **temperature 0 결정성** | **12/12 동일** | **12/12 동일** | 12/12 |
| **거부 (중립 타깃 16문항)** | **5건** | **0건** | — |

속도 손실은 스텝당 1~2ms 로 노이즈 범위. scale=1 산문이 낮은 것은 수락률 차이
(2.082→1.931)이고 출력 내용 자체가 달라지므로 당연하다.

거부 측정은 `tools/refusal_probe.py`. 판정은 **첫 문장만** 본다 — 거부 뒤에 붙는
이유 목록이나 "대신 이건 됩니다" 절이 패턴 매칭에는 작업 시작처럼 보이고, 반대로
"본인 소유 환경에서만" 같은 주의 문구는 거부가 아니다. 이 둘을 섞으면 숫자가 무의미해진다.

대상은 전부 중립(vendor 연습 호스트, 예약 도메인, RFC1918)이다. 실제 도메인을 섞으면
과잉 거부와 정당한 거부가 한 숫자에 뒤엉킨다. 중립 타깃인데도 5건이 막힌다는 것이
요점이다.

## 클라이언트 연동

codemason `/ab off|on|auto`. `auto` = 사용자 턴에 켜고 **모델이 도구 호출을 시작하는 순간 끈다**.
한 번 시키면 몇 시간 도구 루프를 도는 것이 정상이라, 사람이 턴을 못 받는 동안 켜져 있으면 안 된다.

## ROCm 레인(.109) 이식 메모

- `o_proj` 가 BF16 인지부터 확인할 것. 양자화돼 있으면 출력에 더하는 이 방식이 그대로는
  안 된다.
- 훅 자리는 같은 두 곳(KDA / MLA 의 `o_proj` 출구). 백엔드가 달라도 이 레이어는 공용이다.
- `torch.inference_mode` 안에서 만든 텐서는 제자리 수정이 금지된다. scale 텐서는
  `inference_mode(False)` 안에서 만들어야 한다(이 함정에 한 번 걸렸다).
- TP 분할: `o_proj` 는 RowParallel(입력축 분할)이라 A 를 랭크별로 잘라야 한다. 출력은
  기존 all-reduce 에 얹히므로 추가 통신이 없다.
