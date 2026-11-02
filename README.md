# GLM-5.3-Flash on 2× DGX Spark — dspark-serve

> ## ⚠️ 고지 · Disclaimer
>
> 이 저장소는 **있는 그대로(as-is)** 공개됩니다. 어떤 보증도 하지 않습니다.
>
> - 이 코드를 사용해 발생하는 **모든 결과와 책임은 사용자 본인**에게 있습니다.
>   데이터 손실, 장비 손상, 비용 발생, 서비스 중단, 법적 분쟁 — 무엇이든
>   작성자는 책임지지 않습니다.
> - 운영 환경에 그대로 올리라고 만든 물건이 아닙니다. **본인 환경에서 직접
>   검증한 뒤** 쓰세요.
> - 관할 **법령과 제3자 약관(모델·API·서비스 라이선스 포함)을 지킬 책임은
>   전적으로 사용자**에게 있습니다.
> - 문서에 적힌 주소·설정·수치는 작성자의 환경 기준입니다. 그대로 복사해 쓰지 마세요.
> - 이 구성에는 모델의 응답 거부 동작을 요청 단위로 바꾸는 스위치가 포함돼
  있습니다. **연구·평가 목적**으로 공개하며, 그 기능으로 생성된 출력과
  그 사용에 대한 책임은 전적으로 사용자에게 있습니다.
>
> *Provided **as-is**, without warranty of any kind. You bear sole responsibility
> for any use and for compliance with applicable laws and third-party terms.
> See [LICENSE](LICENSE).*

---

DGX Spark(GB10) **두 대**를 200GbE 레일로 묶어 **GLM-5.3-Flash**(4bpw EXL3)를 vLLM으로 서빙하는 구성이다.
컨텍스트 512K, 동시 4요청, OpenAI 호환 API. 원본 모델을 서빙하면서 **검열 해제를 요청 단위로 실시간 전환**한다.

이 저장소의 본체는 [`glm53-exl3-lane/`](glm53-exl3-lane/) 디렉터리다. 나머지 최상위 파일은 같은 박스에서
돌렸던 다른 레인(DeepSeek-V4, Qwen 등)과 보조 스크립트로, 이 문서의 설치와는 무관하다.

| 문서 | 내용 |
|---|---|
| **이 README** | 구조, 우리 구성의 차이점, **새로 설치하기**, **구버전(09-05 키트)에서 업그레이드하기** |
| [`glm53-exl3-lane/README.md`](glm53-exl3-lane/README.md) | 운영 명령, 설정 노브, 실측 원장, 변경 절차 |
| [`glm53-exl3-lane/HANDOFF.md`](glm53-exl3-lane/HANDOFF.md) | 패치별 원리와 함정(개발자용) |
| [`glm53-exl3-lane/AB-SWITCH.md`](glm53-exl3-lane/AB-SWITCH.md) | 실시간 검열 해제 스위치 상세 |
| [`glm53-exl3-lane/ORIGINAL-VS-ABLITERATED.md`](glm53-exl3-lane/ORIGINAL-VS-ABLITERATED.md) | 왜 abliterated 대신 원본을 서빙하는가 |
| [`glm53-exl3-lane/UTF8-GUARD.md`](glm53-exl3-lane/UTF8-GUARD.md) | 한글 깨짐(U+FFFD) 방지 가드 |
| [`glm53-exl3-lane/HISTORY.md`](glm53-exl3-lane/HISTORY.md) | 날짜순 작업 이력 |

---

## 1. 구조

```
                         OpenAI 호환 API  http://<헤드 LAN IP>:8000/v1   모델명 glm-5.3-flash
                                   │
  ┌────────── 헤드 (rank 0) ───────┴──┐     200GbE RoCE 레일      ┌────────── 워커 (rank 1) ──────────┐
  │ docker: vllm_glm53                │◄──── NCCL (TP=2) ────────►│ docker: vllm_glm53 (--headless)   │
  │  vLLM API 서버 + 엔진 + 텐서 절반  │   mlx5_0 / enp1s0f0np0    │  텐서 나머지 절반                  │
  │  메모리 감시기(memwatch)            │                           │  메모리 감시기(memwatch)            │
  └───────────────────────────────────┘                           └───────────────────────────────────┘
        ↑ 같은 이미지, 같은 레인 디렉터리, 같은 모델 경로 (양 노드 동일 — 동기화 스크립트가 보장)
```

- **이미지**: [`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
  (digest `sha256:44b2dbaf…`). Entrpi의 vLLM 포크(`f223ff9f2`) + b12x CuTeDSL 커널 + EXL3 fused MoE.
- **모델**: [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — EXL3,
  라우팅 전문가만 4bpw, 나머지는 BF16. 노드마다 로컬에 한 벌씩(약 164GiB).
- **우리 코드는 이미지를 다시 빌드하지 않는다.** 이미지 안의 파일 몇 개를 레인의 `overlay/` 파일로
  읽기 전용 bind-mount 해서 갈아끼운다. 각 오버레이 옆에 `.orig`(이미지 원본)와 `.diff`가 있어
  무엇을 바꿨는지 그대로 보인다. 마운트를 빼면 순정 이미지 동작으로 돌아간다.
- **조작은 스크립트 하나**: 헤드에서 `glm53-exl3-lane/vllm-glm53-exl3.sh RESTARTALL`.
  양 노드 정지 → 메모리 감시기 → 워커 기동 → 헤드 기동 → READY 대기까지 한 번에 한다.
- 디코딩: MTP-4 스펙 디코딩(모델 내장 MTP 헤드). KV 캐시 FP8(`fp8_ds_mla`), 프리픽스 캐시 사용.

## 2. 우리 구성이 순정 킷과 다른 점

전부 실측으로 채택했고, 근거와 수치는 [레인 README](glm53-exl3-lane/README.md)와 [HISTORY](glm53-exl3-lane/HISTORY.md)에 있다.

| 구분 | 무엇을 | 효과 |
|---|---|---|
| 속도 | dense 층·lm_head·MTP 투영을 **FP8 가중치 + BF16 활성화(Marlin W8A16)** 로 로드 시 양자화 | 디코드 +40% (스텝 117→80ms 대) |
| 속도 | 프리필(M>1024)은 FP8을 BF16으로 풀어 cuBLAS — **이중 경로** | FP8로 잃은 프리필 회복(12k ≈ 1,350 t/s) |
| 속도 | MTP 드래프트 전용 **32K 어휘 lm_head**(검증은 전체 어휘) | 스텝 −3ms, 출력 무손실 |
| 속도 | MoE 라우터 GEMM 이중 계산 제거(상류 버그) | 디코드 +4.4% |
| 정확성 | **'!' 붕괴 수정** — 아래 §4 | 붕괴 0건 |
| 정확성 | **kpool 캐시 오염 수정**(상류 vllm#57477 백포트) — 프리필이 다른 요청의 인덱서 캐시를 덮어쓰던 버그 | 캐시 적중 결과가 다른 요청에 흔들리지 않음 |
| 정확성 | **temperature 0 결정성** — 상류 커널 3곳의 atomic 순서 문제 수정 | 같은 입력 → 같은 출력 |
| 정확성 | routed 전문가 SwiGLU clamp(`swiglu_limit=10`) 누락 보완(상류 누락) | 레퍼런스 동작과 일치 |
| 정확성 | **UTF-8 가드** — 한글 음절 한복판에서 라틴 토큰으로 갈아타 U+FFFD가 나던 것을 샘플러에서 차단 | 7/8 → 0/8, 속도 손실 없음 |
| 기능 | **실시간 검열 해제 스위치 `/ab`** — 아래 §3 | 재기동 없이 요청 단위 전환 |
| 운영 | 비전 게이트 `VISION`(1=이미지 수용, 라이브 기본. 0 이면 `--language-model-only` 로 타워 제외 — GB10 워밍업 OOM 방지), zai-org 채팅 템플릿 수정 백포트 | 안정 기동 |
| 운영 | 양 노드 동기화 검사(`sync-lane-67.sh`), 메모리 감시기, 경로·사이트 값 분리 | 2노드 불일치·OOM 사고 예방 |

**금지된 것**(작화를 일으키거나 이득이 없어 기각): 활성화까지 양자화하는 W8A8(b12x 블록 FP8), INT4 드래프트 헤드,
DFlash2 드래프터(산문에서 MTP-4보다 느림). 이유는 레인 README §5.

## 3. 실시간 검열 해제 스위치 (`/ab`) — 우리만의 구조

abliteration(검열 제거) 모델을 통째로 서빙하면 거부만 사라지는 게 아니라 **"없는 것은 없다"고 말하는 능력도 같이
사라진다.** 가짜 대상 60건 함정에서 원본은 4건, abliterated는 18~22건을 지어냈다. 그래서 우리는 **원본을 서빙하고,
검열 해제는 필요한 순간에만 켠다.**

원리: abliteration이 건드린 것은 `self_attn.o_proj` 45개뿐이고, 이 텐서는 4bpw 체크포인트에서도 BF16이다.
그래서 원본과 abliterated의 차이를 **rank-24 저랭크 어댑터(29MB)** 로 뽑아 서빙 중에 출력에 더한다.

```
o_proj(x) + scale · (x·Aᵀ)·Bᵀ        scale = 0 → 원본,  scale = 1 → abliterated (상대오차 0.17%)
```

`scale`은 GPU 텐서라 CUDA 그래프를 다시 캡처하지 않고, 프리필도 다시 하지 않는다. 양 노드 TP 워커에 동시에 반영된다.

```bash
curl -H "Authorization: Bearer $KEY" http://<헤드>:8000/ab                     # 상태
# {"scale":0.0,"workers":[0.0,0.0],"consistent":true,"mode":"original"}
curl -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"scale":1}' http://<헤드>:8000/ab                                     # 검열 해제 ON
curl -X POST ... -d '{"scale":0}' http://<헤드>:8000/ab                         # OFF (기본)
```

실측: 중립 타깃 16문항 거부 5건 → 0건, 속도 손실 스텝당 1~2ms, temperature 0 결정성 양쪽 모두 유지.
codemason 클라이언트는 `/ab auto`로 "요청을 승낙하는 턴"에만 켜고 도구 호출이 시작되면 끈다. 상세는 [AB-SWITCH.md](glm53-exl3-lane/AB-SWITCH.md).

## 4. 마지막 버그: '!' 붕괴 (2026-10-02 해결)

**증상**: 긴 에이전트 세션에서 추론 채널에 `!`만 수백 개 나오고 답이 없다(`finish_reason=repetition`).
GLM-5.3-Flash 사용자들이 vLLM·SGLang 양쪽에서 보고했지만 상류에도 확정된 원인이 없던 문제다.

**재현**: codemason 실제 요청 모양(시스템 프롬프트 + 도구 18개 + 한국어 과제 한 줄)이 첫 턴에서 100% 붕괴했다.
잘라 가며 찾아보니 **프롬프트 길이가 512로 나눠 나머지 5일 때만** 터졌다. 내용·캐시와는 무관했다.

**원인**: 프리필 청크는 512 경계에서 끊기므로 마지막 조각은 `길이 mod 512` 토큰이다. 그게 정확히
5(= 1 + MTP 추측 4개)면 vLLM V2 러너가 배치 **모양만 보고** 디코드 검증용 CUDA 그래프를 재생했다.
그 그래프는 프롬프트의 KDA 순환 상태를 만들지 않으므로 출력 점수가 NaN이 되고, 샘플러는 NaN에서 token 0(`!`)을 뽑는다.
이후 디코드도 전부 NaN이라 `!`가 끝없이 이어진다.

**수정**: 상류 vLLM이 08-12에 고친 [#51865](https://github.com/vllm-project/vllm/pull/51865)("요청이 전부 디코딩 중일 때만
uniform decode")가 이미지의 포크(08-10 분기)에 빠져 있었다. 이를 백포트했다(`overlay/v2runner/`).

**검증**: `tools/prefill_tail_nan_check.py` — 수정 전 7/7 NaN, 수정 후 14/14 정상. 속도 변화 없음.
수정 후 codemason 장기 실험 약 250요청 붕괴 0건.

---

## 5. 새로 설치하기 (DGX Spark 2대, 처음부터)

> 모든 명령은 **같은 사용자 이름, 같은 경로**로 양 노드에 적용된다고 가정한다(레인·모델·캐시 경로가 양쪽에서 같아야 한다).
> 아래에서 `헤드`는 API를 띄울 박스, `워커`는 나머지 한 대다. 명령은 특별한 말이 없으면 **헤드**에서 친다.
> 예시 주소는 레일 `10.0.1.1`(헤드) / `10.0.1.2`(워커)다. 명령 속 `10.0.1.2` 와 `<워커>` 는 자기 값으로 바꿔 친다.
>
> 이 절차는 2026-10-05에 저장소를 새 경로에 clone해 처음부터 그대로 따라 하는 방식으로 검증했다
> (어댑터 바이트 동일, '!' 회귀 검사 PASS, 디코드 산문 38.9 / 코드 41.8 t/s).

### 5.0 준비물

| 항목 | 우리 환경(검증됨) | 비고 |
|---|---|---|
| 하드웨어 | DGX Spark(GB10, 메모리 128GB) **2대** | 한 대로는 안 뜬다(모델 + KV가 한 대를 넘는다) |
| 연결 | 두 대의 ConnectX-7 QSFP 포트를 **직결 케이블**로 연결 | 첫 포트 `enp1s0f0np0` / RoCE 장치 `mlx5_0` |
| OS | DGX OS (Ubuntu 24.04.5), 커널 7.0.0-1019-nvidia | |
| 드라이버 | NVIDIA 580.178.04 | |
| Docker | 29.6.2 + nvidia-container-toolkit 1.20.1 | DGX OS 기본 구성 |
| 디스크 | 노드당 여유 **약 200GB** | 모델 164GiB + 어댑터 작업 파일 |
| 기타 | 헤드→워커 **비밀번호 없는 ssh**, `sudo` 비밀번호 없음(권장) | sudo는 기동 전 페이지 캐시 비우기에 쓴다(없어도 동작) |

swap은 필요 없다(로더가 direct I/O로 읽는다).

### 5.1 레일 네트워크 (양 노드)

두 대를 잇는 포트에 고정 IP와 MTU 9000을 준다. 예: `/etc/netplan/40-cx7.yaml`

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp1s0f0np0:            # 레일 포트 (ip -br a 로 이름 확인)
      addresses: ["10.0.1.1/24"]   # 워커는 10.0.1.2/24
      dhcp4: false
      mtu: 9000
```

```bash
sudo netplan apply
ping -c 3 10.0.1.2                    # 헤드에서 워커 레일 IP로
rdma link                             # mlx5_0/1 ... state ACTIVE ... netdev enp1s0f0np0 확인
```

헤드→워커 ssh 키 등록(헤드에서 한 번):

```bash
ssh-keygen -t ed25519            # 키가 없을 때만
ssh-copy-id <사용자>@<워커 LAN IP>
ssh-copy-id <사용자>@10.0.1.2      # 레일 주소로도(동기화 스크립트가 레일로 민다)
```

### 5.2 도커 이미지 (양 노드)

```bash
docker pull ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1      # 약 7GB 압축 / 18GB 설치
docker image inspect ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1 --format '{{.Id}}'
# → sha256:606b5f3c61b4900db05555b72a7b20a9455a8bdd09a5c74e288839c40cc3703a 이어야 한다
#   (pull 로 받았으면 레지스트리 digest 도 sha256:44b2dbaf… 로 보인다. docker load 로 옮긴 이미지는 digest 가 없다)
```

### 5.3 저장소 (헤드에만 clone)

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/AmesianX/dspark-serve.git
cd ~/work/dspark-serve/glm53-exl3-lane
```

워커에는 clone하지 않는다. 레인 디렉터리는 §5.6의 동기화 스크립트가 같은 경로로 밀어 준다. 워커에는 부모 디렉터리만 만든다.

```bash
ssh <워커> 'mkdir -p ~/work/dspark-serve'
```

### 5.4 사이트 값 설정

`glm53-exl3-lane/glm53-serve.env` 맨 위 **① 사이트 값** 블록을 자기 환경으로 고친다. 그 아래 ②·③은 건드리지 않는다.

| 변수 | 넣을 값 | 확인 방법 |
|---|---|---|
| `HEAD_RAIL_IP` / `WORKER_RAIL_IP` | 레일 IP(예: 10.0.1.1 / 10.0.1.2) | `ip -br a` |
| `WORKER_SSH` | 헤드에서 워커로 접속할 ssh 대상(예: `me@192.168.1.67`) | `ssh <값> true` 가 비밀번호 없이 끝나야 함 |
| `NCCL_IF` | 레일 인터페이스명 | `ip -br a` |
| `NCCL_HCA` | 레일의 RoCE 장치명 | `rdma link` 에서 레일 netdev 옆의 이름 |
| `NCCL_SUBNET` | 레일 서브넷(예: 10.0.1.0/24) | |
| `VLLM_API_KEY` | **새로 정한 임의의 문자열** | 클라이언트가 `Authorization: Bearer <키>` 로 보낸다 |

RoCE GID 인덱스는 런처가 sysfs에서 자동으로 찾는다. 직접 정하지 말 것.

고친 설정은 커밋해 둔다(워커로 밀 때 동기화 스크립트가 커밋되지 않은 변경을 거부한다). 새 박스에는 git 사용자
정보가 없어 커밋이 실패하므로 이 저장소에 한 번 정해 둔다(값은 아무거나 상관없다).

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
git config user.name "$USER"; git config user.email "$USER@$(hostname)"   # 이 저장소에서 한 번만
git commit -am "site config"
```

### 5.5 모델과 어댑터 준비

**(1) 원본 모델 — 양 노드에 같은 경로로** (164GiB)
[huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) 를
`~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw` 에 받는다. 이 경로에 모델이 없으면 기동이 멈추고 어디서 받는지 알려 준다.

호스트에 파이썬 패키지를 설치하지 않는다(DGX OS의 시스템 python은 `pip install`을 막는다). 서빙 이미지 안에
`hf`·torch가 다 들어 있으므로 다운로드와 빌드는 이미지로 한다. 아래 두 줄로 줄여 쓴다.

```bash
IMG=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1
run() { docker run --rm --user $(id -u):$(id -g) -e HF_HOME=/m/.hf -v $HOME/Models:/m -v $PWD/tools:/tools --entrypoint "$@"; }
```

```bash
mkdir -p ~/Models
cd ~/work/dspark-serve/glm53-exl3-lane
run hf $IMG download brandonmusic/GLM-5.3-Flash-tr3-4bpw \
   --revision a5fee929cf4888b1824323e33e8a19b60129e025 \
   --local-dir /m/brandonmusic--GLM-5.3-Flash-tr3-4bpw
# 워커로는 레일로 복사하는 게 가장 빠르다(워커에서 같은 명령으로 받아도 된다)
rsync -a --info=progress2 ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw 10.0.1.2:Models/
```

`--local-dir` 로 **실제 파일**을 받는다. HF 캐시의 심링크 구조를 쓰면 컨테이너 안에서 링크가 깨져 로딩에 실패한다.

**(2) 검열 해제 어댑터** (헤드에서 만들어 워커로 복사, 29MB)

abliterated 모델([huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated))에서
필요한 건 o_proj 46개(3.89GB)뿐이다. 그것만 HTTP Range로 받아 원본과의 차이로 어댑터를 만든다(175GB를 다 받지 않는다).

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
python3 tools/fetch_oproj.py --source abliterated \
   --out ~/Models/glm53-oproj/o_proj_abliterated.safetensors          # 끝에 md5 ... OK 가 나와야 한다

# 어댑터 빌드(torch 필요 → 이미지 안에서). 약 10분. 서버가 떠 있지 않을 때 할 것(메모리를 수 GB 쓴다).
run python3 $IMG /tools/build_ab_lora.py --base /m/brandonmusic--GLM-5.3-Flash-tr3-4bpw \
   --abl /m/glm53-oproj/o_proj_abliterated.safetensors --out /m/glm53-ab-lora
rsync -a ~/Models/glm53-ab-lora 10.0.1.2:Models/
```

### 5.6 워커 동기화와 기동

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
./sync-lane-67.sh            # 레인 디렉터리를 워커의 같은 경로로 밀고 파일별 md5 대조
./vllm-glm53-exl3.sh RESTARTALL
```

READY까지 약 3~4분. 마지막에 `✅ API READY ... 원본 + /ab off` 가 찍히면 성공이다.
처음 기동은 JIT 컴파일 캐시(`~/glm53-vllm-cache`)를 만드느라 첫 요청이 느리다.

> `sync-lane-67.sh` 는 커밋되지 않은 변경이 있으면 거부한다(§5.4에서 커밋했으면 그대로 된다).
> 레인을 고칠 때마다 커밋 → `sync-lane-67.sh` → `RESTARTALL`. RESTARTALL은 기동 전에 양 노드가 같은지 자동으로 검사한다.

### 5.7 동작 확인

```bash
./vllm-glm53-exl3.sh STATUS                       # 양 노드 컨테이너, health=200, 모델/스위치 상태, 가용 메모리
python3 tools/prefill_tail_nan_check.py           # '!' 붕괴 회귀 검사 → PASS
python3 tools/kmeas.py                            # 디코드 속도(산문/코드/영문)
curl -s -H "Authorization: Bearer <키>" http://localhost:8000/ab     # {"scale":0.0, ... "mode":"original"}
```

간단한 대화:

```bash
curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer <키>" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"안녕? 한 문장으로 자기소개 해줘."}],"max_tokens":400}'
```

---

## 6. 구버전(09-05 인수인계 키트)에서 업그레이드하기

2026-09-05에 전달한 `glm53-flash-2xspark-kit` 으로 설치해 쓰고 있다면, 그때는 **abliterated 모델이 기본**이었다.
그 뒤로 바뀐 것이 많아서(위 §2 대부분, 원본 + 실시간 스위치 구조, 경로 정리) **변경분을 따라가지 말고 레인을 통째로
바꾼다.** 이미지(v2.3-tier1)와 이미 받아 둔 abliterated 모델은 그대로 쓴다.

**바뀌는 것**: 레인 디렉터리 전체 · 서빙 모델(abliterated → 원본) · 어댑터 추가 · `~/glm53-hotfix-b12x` 심링크 제거.
**그대로인 것**: 도커 이미지 · 레일 네트워크 · ssh · 모델 다운로드(추가 다운로드는 3.89GB 하나).

```bash
# ── 헤드에서 ─────────────────────────────────────────────────────────────
~/work/dspark-serve/glm53-exl3-lane/vllm-glm53-exl3.sh STOPALL     # (1) 옛 레인으로 서버 정지

# (2) 옛 b12x 심링크 제거 — 양 노드. 새 레인은 b12x 를 직접 마운트하므로 남아 있으면 중복 마운트로 기동 실패
rm -f ~/glm53-hotfix-b12x; ssh <워커> 'rm -f ~/glm53-hotfix-b12x'

# (3) 옛 설치 디렉터리를 통째로 보관(사이트 값을 옮겨 적으려고) — 양 노드
mv ~/work/dspark-serve ~/work/dspark-serve.kit0905
ssh <워커> 'mv ~/work/dspark-serve ~/work/dspark-serve.kit0905 && mkdir -p ~/work/dspark-serve'

# (4) 새 저장소 받기 (§5.3과 같다)
cd ~/work && git clone https://github.com/AmesianX/dspark-serve.git
cd ~/work/dspark-serve/glm53-exl3-lane
IMG=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1
run() { docker run --rm --user $(id -u):$(id -g) -e HF_HOME=/m/.hf -v $HOME/Models:/m -v $PWD/tools:/tools --entrypoint "$@"; }
# 09-05 키트 안내(HF_HOME=~/Models/hfcache)대로 받았다면 abliterated 는 여기 있다. 다른 곳이면 이 줄만 고친다(~/Models 아래 상대경로).
ABL=hfcache/hub/models--lovesenko--GLM-5.3-Flash-tr3-4bpw-Abliterated/snapshots/c8f58e6aa9117c73607d692978b22f091d80450c

# (5) 사이트 값 옮기기: 옛 ~/work/dspark-serve.kit0905/glm53-exl3-lane/ 의 glm53-serve.env 와
#     vllm-glm53-exl3.sh 맨 위(WORKER, KEY) 값을 새 glm53-serve.env ① 블록에 적는다(§5.4 표).
#     옛 WORKER → WORKER_SSH, 옛 KEY/VLLM_API_KEY → VLLM_API_KEY. 클라이언트가 쓰던 키를 그대로 두려면 같은 값으로.
git config user.name "$USER"; git config user.email "$USER@$(hostname)"   # 커밋용, 한 번만

# (6) 이미 가진 abliterated 에서 원본 만들기 — 원본 o_proj 3.89GB 만 받고, 나머지는 하드링크(추가 디스크 약 72GB)
python3 tools/fetch_oproj.py --source original --out ~/Models/glm53-oproj/o_proj_original.safetensors   # md5 ... OK
python3 tools/build_original_from_abliterated.py --src ~/Models/$ABL \
   --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw
#     워커에도 똑같이: o_proj 파일을 보내고 거기서 빌드(워커의 abliterated 경로를 확인할 것)
ssh <워커> 'mkdir -p ~/Models/glm53-oproj'
rsync -a ~/Models/glm53-oproj/o_proj_original.safetensors 10.0.1.2:Models/glm53-oproj/
rsync -a tools/build_original_from_abliterated.py 10.0.1.2:/tmp/
ssh <워커> "python3 /tmp/build_original_from_abliterated.py --src ~/Models/$ABL \
   --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw"

# (7) 어댑터 만들기 — 원본과 abliterated 가 둘 다 있으니 abliterated 스냅샷을 그대로 쓴다(이미지 안에서, 약 10분)
run python3 $IMG /tools/build_ab_lora.py --base /m/brandonmusic--GLM-5.3-Flash-tr3-4bpw \
   --abl /m/$ABL --out /m/glm53-ab-lora
rsync -a ~/Models/glm53-ab-lora 10.0.1.2:Models/

# (8) 커밋(바뀐 게 있을 때만) → 동기화 → 기동 → 확인 (§5.6, §5.7)
git diff --quiet || git commit -qam "site config"
./sync-lane-67.sh && ./vllm-glm53-exl3.sh RESTARTALL       # 끝에 "✅ API READY ... 원본 + /ab off"
python3 tools/prefill_tail_nan_check.py                     # PASS
curl -s -H "Authorization: Bearer <키>" http://localhost:8000/ab   # "mode":"original"
```

`--dst` 가 원본 이름(`brandonmusic--…`)인 이유: 결과물이 원본 저장소와 **파일 단위로 같다**(샤드 sha256, config·토크나이저 일치를
확인했다). 이 절차는 2026-10-05에 09-05 키트를 실제로 설치해 abliterated로 띄운 상태에서 그대로 따라 해 검증했다
(원본 샤드·어댑터 바이트 동일, '!' 회귀 검사 PASS, `/ab` 양방향 정상). 워커의 원본 빌드는 1분 남짓, 어댑터는 10분 정도 걸린다. 업그레이드가 끝나고 잘 돌면 옛 설치(`~/work/dspark-serve.kit0905`)는 지워도 된다. abliterated 스냅샷은 지우면 안 된다 —
원본의 76개 샤드가 그 파일과 하드링크를 공유한다(지워도 원본 쪽은 남지만, 어댑터를 다시 만들 때 필요하다).

---

## 7. 자주 하는 실수

- **2노드는 무조건 `RESTARTALL`.** 워커만 살아 있으면 헤드를 600초 기다리다 죽는 좀비가 된다.
- **서버가 떠 있을 때 헤드에서 무거운 작업 금지.** GB10은 CPU와 GPU가 메모리를 공유한다. 가용이 3GB 밑으로 떨어지면
  메모리 감시기가 박스 보호를 위해 컨테이너를 내린다. 빌드·변환 작업은 `STOPALL` 후에.
- **모델·어댑터에 심링크 금지.** 컨테이너 안에서 호스트 경로를 가리키는 링크는 깨진다. 실파일 또는 하드링크.
- **레인을 고쳤으면 커밋 → `sync-lane-67.sh` → `RESTARTALL`.** 워커는 git이 아니라 파일 사본이다.
- 환경변수는 ssh로 넘어가지 않는다. 노브는 `glm53-serve.env` 에 넣는다(새 노브는 `launch.sh` 의 통과 목록에도 등록).
- 기동이 `this host does not own <IP>` 로 멈추면 레일 IP 설정이, `no RoCE v2 GID matching` 이면 `NCCL_HCA` 가 틀린 것이다.

## 8. 출처와 감사

- 서빙 킷·이미지: [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark), vLLM 포크 [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark)
- 모델: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) → EXL3 4bpw [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) → abliteration [lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated)
- 커널: b12x(sparkinfer-glmrt), [exllamav3](https://github.com/turboderp-org/exllamav3), [vLLM](https://github.com/vllm-project/vllm)
