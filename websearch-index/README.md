**한국어** | [English](README.en.md)

# websearch-index — 쓸수록 **덜** 막히는 검색 서버

## 논지

기존 검색서버는 "질의 → 외부엔진 중계기"다. 그래서 **쓸수록 더 막힌다** — 질의마다
차단 확률과 레이트리밋이 쌓인다. SearXNG 가 막히는 근본 이유도 이것이다.

여기서는 뒤집는다.

```
질의 ─┬─ 0층  로컬 색인 (SQLite FTS5)          2.7ms   차단이라는 개념이 없음
      │       상시 수집기가 채움 (피드 + 승격된 구독)
      ├─ 1층  기계용 공개 창구                  0.6~0.9s  키 없음·차단 없음
      │       GoogleNews RSS · HN · GitHub · StackExchange · Wikipedia
      ├─ 2층  익명 SearXNG (기존)               1.4s    가끔 막힘
      └─ 3층  신원 브라우저 (~/work/websearch-browser)  8s  자주 막힘·최후수단
```

핵심은 층을 나눈 게 아니라 **승격 두 개**다.

1. **질의 → 구독 승격.** 같은 주제를 두 번 물으면 Google News RSS 구독으로 등록한다.
   그때부터 그 주제는 영구히 0층에서 나온다. **검색할수록 무차단 영역이 넓어진다.**
2. **1층 결과의 영구 적재.** 밖에서 받아온 건 전부 색인에 넣는다. 두 번째부터는
   그 창구를 아예 안 때린다.

**3층 사용률이 KPI다. 내려가면 성공.** `./ws.sh stats` 의 `layer_mix_24h`.

## 결정적 발견 (실측 2026-11-18)

**구글은 `google.com/search` 를 막지 `news.google.com/rss/search` 는 안 막는다.**
같은 회사, 같은 색인 계열인데 한쪽 문은 기계용으로 열려 있다. 키도 필요 없다.

| 질의 | 결과 | 시간 |
|---|---|---|
| `vllm site:github.com` | 100건, 전부 GitHub — **`site:` 연산자가 실제로 동작** | 0.65s |
| `AI when:1h` | 100건, 수 분 전 기사 — **`when:` 도 동작** | 0.65s |
| `vLLM 추론 최적화` (한글) | 7건, 국내 매체 | 0.38s |
| `mocflwsrrhbubkwdevzz qqxz` (난수) | **0건** ← 정직성 통과 | 0.42s |

그리고 논지를 그대로 증명하는 한 쌍:

| arXiv 접근법 | 결과 |
|---|---|
| 실시간 API (`export.arxiv.org/api/query`) | 5건 / **12.9초** / 연속호출 시 거절 |
| RSS 피드 (`rss.arxiv.org/rss/cs.LG`) | **200건 / 0.1초** / 제한 없음 |

→ 그래서 arXiv 는 1층에서 빼고 0층 수집기로 내렸다.

⛔ **쓰지 않기로 한 것**: Reddit `.json`(403), CommonCrawl CDX(504), Bing News RSS(빈 응답),
Brave Search API(무료 2,000건/월 티어 **2026-02 폐지** → 카드 + 월 $5 크레딧), Mojeek API(유료).

## 구성

| 파일 | 역할 |
|---|---|
| `core.py` | 0층. SQLite FTS5 스키마, 적재, 검색, **적합도 관문** |
| `channels.py` | 1층. 기계용 공개 창구 어댑터 + 호스트별 스로틀 |
| `collect.py` | 상시 수집기. 고정 피드(`sources.json`) + 승격된 구독 |
| `serve.py` | HTTP. `/search` `/stats` `/subs` `/health` |
| `ws.sh` | 제어. `start stop restart status collect search stats subs logs` |

---

## 기동 — systemd 가 소유한다 (2026-11-21)

⛔ **`./ws.sh start` 로 띄우지 말 것.** 2026-11-19 재부팅에 죽은 뒤 자동시작이
없어서 **2.5일간 조용히 멈춰 있었다.** 그동안 SearXNG 의 `idx` 엔진은 계속
실패했고, 수집기가 멈췄으니 "쓸수록 덜 막힌다"의 전제도 같이 멈췄다.
유저 규칙: 조용히 멈추면 그건 전부 결함이다.

⚠️ 유닛 안의 경로는 **우리 환경 절대경로**(`/home/user/work/dspark-serve/websearch-index`)
다. 다른 박스면 `WorkingDirectory`·`ExecStart`·`EnvironmentFile`·`StandardOutput`·
`StandardError` 다섯 줄을 자기 경로로 바꿔야 한다. systemd 는 경로가 틀리면 조용히 넘어가지
않고 `status=200/CHDIR` 로 실패하니 `systemctl --user status` 에 바로 드러난다.

```bash
# 설치 (유닛은 이 리포의 systemd/ 에 사본이 있다 — 리포 밖에 두면 또 잃는다)
sed "s|/home/user/work/dspark-serve/websearch-index|$PWD|g" \
  systemd/websearch-index.service > ~/.config/systemd/user/websearch-index.service
sed "s|/home/user/work/dspark-serve/websearch-index|$PWD|g" \
  systemd/websearch-collect.service > ~/.config/systemd/user/websearch-collect.service
printf 'INDEX_TOKEN=%s\nINDEX_PORT=8097\n' "$(cat state/token)" > state/env
chmod 600 state/env                      # 토큰이 들어가므로

sudo loginctl enable-linger $USER        # 로그인 없이도 가동 (이게 없으면 재부팅 후 안 뜸)
systemctl --user daemon-reload
systemctl --user enable --now websearch-index websearch-collect

# 확인
systemctl --user is-active websearch-index websearch-collect   # active active
systemctl --user is-enabled websearch-index websearch-collect  # enabled enabled
```

`Restart=always` 가 실제로 도는지 한 번 죽여서 봤다 — `kill -9` 후 약 6초에
부활하고 검색이 그대로 됐다(`NRestarts=1`).

메모리 캡은 `ws.sh` 와 같은 `MemoryMax=600M` / `MemorySwapMax=0` 이다. 이 박스는
vLLM 이 통합메모리를 점유하고 memwatch 가 `MemAvailable<3GB` 에서 서빙
컨테이너를 선제 종료하므로, 캡 없이 띄우면 검색이 서버를 죽일 수 있다.

**소유자가 둘이면 안 된다.** systemd 가 든 프로세스는 `state/*.pid` 를 쓰지
않으므로 `ws.sh start` 는 중복 기동을 시도하고 8097 바인드에서 깨진다.
`ws.sh` 는 그 경우를 감지해 거부한다. 제어는 `systemctl --user`.
`ws.sh` 의 `search` `stats` `subs` `collect` `logs` 는 그대로 쓴다.

---

## 참고값 — 우리 환경 실측 (2026-11-21)

재구성한 쪽이 "제대로 선 것인지" 비교할 기준이다. 숫자가 자릿수로 다르면 뭔가 틀렸다.

| 항목 | 우리 값 |
|---|---|
| 포트 / 바인딩 | `8097`, `127.0.0.1` + `172.18.0.1`(도커 브리지)만. **LAN·공인 IP 에서 안 닿음** |
| 색인 문서 | 4,318건 / `index.db` 4.8MB |
| 문서 출처별 | 고정피드 3,959 · 승격구독 257 · 1층적재 102 |
| 고정 피드 | 27개 (`sources.json`) |
| 승격된 구독 | 7개 |
| 0층 지연 | **6.7~7.4ms** (20건 반환) |
| 메모리 캡 | `MemoryMax=600M` / `MemorySwapMax=0` |

```
$ ./ws.sh search vllm            → 20건   0.0067s
$ ./ws.sh search "GLM 5.3"       → 16건   0.0070s
$ ./ws.sh search <난수 문자열>   →  0건   0.0074s   ← 정직성 통과. 0 이 아니면 적합도 관문이 깨진 것
```

**난수 질의가 0건이 아니면 거기서 멈춰라.** FTS 가 아무 문서나 집어오고 있다는 뜻이고,
그 상태로는 모델에 쓰레기가 들어간다. ⛔ `core.py` 의 적합도 관문을 제거하지 말 것.

### 1층 창구와 스로틀 (`channels.py`)

```
googlenews_ko · googlenews_en · hackernews · github · stackexchange · wikipedia_ko
```

⛔ `_MIN_GAP` 간격을 줄이지 말 것 — 공짜 창구를 망가뜨린다.

| 호스트 | 최소 간격 | 근거 |
|---|---|---|
| `export.arxiv.org` | 3.0s | arXiv 공식 요구 |
| `api.stackexchange.com` | 1.0s | 무인증 일일 300건 |
| `api.github.com` | 1.0s | 무인증 분당 10회 |

⛔ `arxiv()` 는 1층(`LIVE`)에 넣지 않는다. 실측 — 실시간 API 12.9초/5건 vs RSS 0.1초/200건.
arXiv 는 0층 수집기로 내렸다. 이 프로젝트의 논지가 그대로 드러나는 자리다.

### KPI

`./ws.sh stats` 의 `layer_mix_24h`. **3층(신원 브라우저) 비율이 내려가면 성공이다.**
`layer0_hit_rate_24h` 는 0층이 단독으로 답한 비율이다.

복귀 직후(표본 10질의)는 `0: 10%` / `0L: 90%` 였다 — 색인이 아직 얇아서 대부분 1층까지
내려갔다. 수집기가 돌면서 0층 비율이 올라간다. **며칠 돌린 뒤에 보는 숫자다.**

## SearXNG 연동

```yaml
- name: local index     # shortcut idx  — 기본셋에 들어있음
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=0
- name: index live      # shortcut xidx — bang 전용(밖으로 나감)
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=1
```

`local index` 를 **기본셋에 넣어도 되는 이유**: `live=0` 이라 외부로 한 번도 안 나가고
2.7ms 다. 그리고 **부수효과가 본체다** — 평소 검색 질의가 전부 기록되어 승격 엔진을
공짜로 먹여살린다. 에이전트가 자주 묻는 주제 쪽으로 0층이 저절로 자란다.

## ⛔ 재발 방지

1. **적합도 관문을 빼지 말 것** (`core.COVER_MIN`). 실측 사고: OR 폴백이 흔한 토큰
   하나만 맞아도 통과시켜 `국내 반도체 수출 규제 영향` → `수능 연기`(영향),
   `샤오미18 국내 출시`(국내) 가 0층 상위로 나왔다. **0층이 쓰레기를 자신있게
   내놓으면 1층으로 넘어가지도 않고 에이전트가 틀린 답을 한다 — 차단당하는 것보다
   나쁘다.** 질의 토큰의 60% 이상이 문서에 실제로 나타나야 한다.
2. **한글은 prefix `*` 필수.** FTS5 unicode61 은 조사를 못 뗀다. `추론` 은 `추론을` 에
   매치되지 않고 `추론*` 은 된다 (`core.make_match`). trigram 토크나이저는 2글자
   질의를 못 받아 부적합.
3. **정직성 가드**: 난수 질의에 0건이 나와야 정상. `./ws.sh search <난수>`.
4. **SearXNG 연동엔 `enable_http: true` + `enable_http2: false` 둘 다** 필요.
   안 하면 둘 다 "unexpected crash" 로만 보인다 (파이썬 http.server 는 H2 불가).
5. **NPM 캐시가 실패 응답도 30분 캐시**한다. 디버깅 전 반드시 비울 것:
   `docker exec nginx-proxy-manager sh -c 'rm -rf /data/cache/searxng/*'`
6. **메모리 하드캡 필수** (`ws.sh` 의 `MemoryMax=600M`). GB10 은 통합메모리라
   OOM 이 박스 전체를 죽인다.
7. **공짜로 열어준 문을 망가뜨리지 말 것.** `channels._MIN_GAP` 의 호스트별 간격을
   줄이지 않는다 (arXiv 는 공식적으로 3초 간격 요구, StackExchange 무인증
   일일 할당량 300건, GitHub 무인증 분당 10회).
