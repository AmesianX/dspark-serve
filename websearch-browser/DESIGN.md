**한국어** | [English](DESIGN.en.md)

# browser-lane — 신원 있는 검색 레인

> ⛔ **2026-11-27: SearXNG bang 연결(`!gbr` `!bbr` `!qbr`) 제거.** 3층(오너 신원)으로 갈지를 호출자가
> 정하게 한 잘못된 설계였다 — 모델은 문법을 몰라 아무도 못 썼고(실제 경로 미검증), SearXNG 키를 가진
> 누구나 오너 신원으로 검색하게 하는 구멍이었다. 이제 클라이언트 → 3층 경로는 없다. 레인 자체는
> 그대로 떠 있고(127.0.0.1·172.18.0.1, `X-Lane-Token`), **승격은 서버가 판단한다(11-27 구현):**
> SearXNG 플러그인 `edge/searxng/plugins/browser_escalate.py` 가 `post_search` 에서 전 엔진 결과를 보고
> 5건 미만이거나(엔진 절반 이상 막힘 + 10건 미만) 일 때만 레인을 직접 부른다. 첫 페이지·general·기본
> 엔진셋만, 10분 6회 예산, 질의 캐시 30분, 타임아웃 15초. 질의에 무엇을 넣어도 승격을 강제할 수 없다.
> 실측: 0건 질의 → 레인 호출(콜드 4.8s), 27건 질의 → 미호출, 강제 임계 e2e → 32건 중 10건이 레인.
> 아래 bang 관련 서술은 당시 기록.


## 왜 필요한가 (2026-11-18 실측)

봇탐지가 보는 축은 "AI냐"가 아니라 **"신원을 들고 다니냐"** 였다.

- SearXNG 는 `searx/network/client.py:93` 에서 `discard_cookies: True` 를 **하드코딩**한다.
  설정값이 아니고 `_CLIENT_KWARGS`(verify/max_redirects/impersonate/curl_options/enable_http3)
  에도 없어서 끌 수 없다. 즉 질의마다 쿠키 0개 = 매번 생판 처음 온 익명 방문자.
  한 IP 에서 낯선 방문자가 계속 오는 꼴 → 데이터센터 프록시와 구분 불가.
- 반대로 영속 프로필을 쓴 Chromium 은 같은 IP 에서 google 유기적 결과 9건을 받았고
  난수 질의엔 0건이었다(정직성 통과). 프로필에 쌓인 것:
  `NID`(만료 2027-05) `AEC` `SNID` `SEARCH_SAMESITE` `_GRECAPTCHA`(챌린지 통과 증서).
- 그래서 역할을 쪼갠다. SearXNG 의 익명성은 그 도구의 존재 이유이므로 **깨지 않는다.**

| 레인 | 담당 엔진 | 성질 |
|---|---|---|
| SearXNG (익명, 기존) | google cse · duckduckgo web · yahoo · mwmbl · wikipedia | 신원 안 따짐 |
| **browser-lane (식별, 신규)** | google · brave · qwant | 신원 요구 |

browser-lane 은 SearXNG 의 `json_engine` 으로 물린다 → 업스트림 코드 무수정, 우리 것만 게이트.

## 신원 모델

**신원 = 디렉터리 하나.** `identities/<name>/` = Chromium user-data-dir.
감사 가능(쿠키 이름·만료를 그대로 조회), 백업 가능, 폐기·교체 가능. 태우면 새로 만든다.

### 등급

| 등급 | 내용 | 개인정보 비용 | 수명 |
|---|---|---|---|
| 0 | 쿠키 0 (현 SearXNG) | 없음 | — |
| **1 지속 익명** ← 기본 | 쿠키 유지, 로그인 안 함. 동의·clearance·reCAPTCHA 증서만 누적 | **없음** | google ~6개월 / CF clearance 수시간~수일 |
| 2 계정 로그인 | 실제 계정 로그인 | 질의가 계정에 귀속 + ToS 위반 위험 | 길다 |
| 3 공식 신원 | API 키 / Web Bot Auth | 돈 / 표준 성숙 대기 | 영구 |

**기본은 1등급.** 2등급은 권하지 않는다 — 통과율은 오르지만 형 계정에 모든 질의가 붙고
자동질의는 대개 약관 위반이다. 쓸 거면 전용 계정으로, 그것도 위험을 알고.
참고: `google cse` 가 이 난리를 다 통과한 이유는 그게 이미 **3등급**이기 때문이다.

### 신원을 주는 방법 — 사람을 부르지 않는다 ⛔

~~사람이 VNC 로 붙어 챌린지를 한 번 통과해 준다(vouching)~~ → **2026-11-18 폐기.**
유저 판정: "VNC 사용하는 건 너무 더러운 방식이야." 맞는 지적이다. 사람을 화면
앞에 앉혀 캡차를 누르게 하는 건 해법이 아니라 기술부채고, 운영이 사람 가용성에
묶이면 무인 에이전트가 성립하지 않는다. `vouch_browser.py` 삭제, `identity.sh` 의
x11vnc/Xvfb 경로 전부 제거됨.

**대신 이렇게 한다:**

1등급 신원은 **저절로 쌓인다**(google NID/_GRECAPTCHA 는 사람 개입 없이 붙었다).
그걸로 되는 엔진은 되고, 안 되는 엔진은 **그냥 포기한다.**

    [레인] 챌린지 감지
       → 스크린샷 state/challenge-<engine>-<ts>.png (사후 확인용)
       → 상태 suspended, 유예 1h → 2h → 4h … 최대 24h (챌린지마다 2배)
       → 유예가 지나면 자동 복귀. 사람 호출 없음.
    [사람] 할 일 없음. 알림은 통보일 뿐 조치 요구가 아니다.

챌린지를 띄우는 엔진은 "우리를 원하지 않는 엔진"일 뿐이다. 계속 두드리면 IP 평판만
탄다. **뚫는 대신 그 수요를 0층/1층으로 옮기는 게 정답이다** — `~/work/websearch-index/`
참조. brave 의 경우 유료 API 외에 합법 경로가 없다는 것도 확인했다(무료 2,000건/월
티어는 2026-02 폐지, 현재 카드 등록 + 월 $5 크레딧).

### 격리 규칙

- `identities/` 는 0700, git 에 절대 올리지 않는다 (쿠키 = 자격증명)
- 레인은 `127.0.0.1` + `172.18.0.1`(docker bridge) 에만 바인딩. LAN·공인망 노출 금지
- 헤더 토큰 필수 (`X-Lane-Token`)
- 엔진군별로 신원을 분리할 수 있다 (brave 가 타면 google 신원은 무사)

## 자원 제약

이 박스는 vLLM 이 117GB 를 쥐고 있고 가용 ~4GB 다. **통합메모리라 OOM = 박스 전체 사망.**
그래서:
- 브라우저는 **상주하지 않는다.** 유휴 180초면 프로세스 종료, 다음 질의에 재기동.
  프로필은 디스크에 남으므로 신원은 유지된다.
- 메모리 하드캡으로 띄운다 (`systemd-run -p MemoryMax=...`). 캡을 넘으면 레인만 죽고
  박스는 산다.
- 동시 네비게이션 1개 (브라우저 하나니까 자연히 직렬). 그게 레이트 보호도 된다.

## 정직성 가드

bing 교훈 — 조용히 쓰레기를 반환하는 엔진이 최악이다. 레인은 `/selftest` 에서
난수 질의를 던져 **0건이 나오는지** 확인한다. 0건이 아니면 그 엔진을 믿지 않는다.

## 구현하면서 밟은 함정 (전부 실측, 재발 방지용)

1. **챌린지 탐지를 본문 문자열로 하면 안 된다.**
   처음엔 본문 전체에서 `/sorry/index` `recaptcha/api2` `enablejs` 를 찾았다. 그랬더니
   **유기적 결과 9건이 멀쩡히 뜬 정상 구글 페이지**가 챌린지로 오판되어 엔진이
   영구 차단되고 알림까지 발송됐다. 정상 페이지에도 그 문자열들이 있다.
   → **먼저 추출하고, 결과가 나오면 그걸로 끝.** 0건일 때만 챌린지를 따진다.
   표식도 최종 URL / 제목 / 본문 앞 1,500자로 좁혔다. 산출물이 문자열 매칭보다 정확하다.

2. **Chromium 프로필은 단일 쓰기 자원.** 같은 프로필로 두 번째 브라우저를 띄우면
   "기존 브라우저 세션에서 여는 중입니다" 하고 즉사한다. 레인이 유일한 소유자여야
   한다 (이 제약 때문에 보증 브라우저가 레인을 내렸다 올려야 했고, 그 복잡도가
   vouch 를 폐기한 이유 중 하나다).

3. **SearXNG → 레인 연결에 두 개를 꺼야 한다.** 둘 다 "unexpected crash" 로만 보여서
   원인이 안 보인다:
   - `enable_http: true` — 업스트림 기본은 HTTPS 전용(`InvalidSchema`)
   - `enable_http2: false` — SearXNG 는 평문에도 HTTP/2 prior-knowledge 로 붙는데
     레인은 파이썬 http.server 라 H2 를 못 한다. 실측: `--http1.1` 200,
     `--http2-prior-knowledge` 000 / exit=16.
   둘 다 엔진의 `network:` 블록에 넣는다 (`curl_options` 로 HTTP_VERSION 을 넣는 방법은
   안 먹혔다 — http_version 세션 인자가 이긴다).

4. **NPM 캐시가 실패 응답도 30분 캐시한다.** 디버깅 중엔 반드시 비울 것:
   `docker exec nginx-proxy-manager sh -c 'rm -rf /data/cache/searxng/*'`
   안 비우면 고쳐놓고도 계속 옛날 실패를 보게 된다 (한참 헤맸다).

5. **브라우저 엔진은 기본 검색에 넣지 않는다.** SearXNG 는 모든 엔진을 기다리므로
   5~8초짜리 엔진을 기본셋에 넣으면 **모든 검색이 그만큼 느려진다**. `disabled: true`
   로 두고 bang 으로만 쓴다: `!gbr` `!bbr` `!qbr`. 기본 검색은 1.4초를 유지한다.

6. **google 은 결과 URL 을 완전히 난독화했다** (2026-11-18). `href` 도 `ping` 도
   `/goto?url=CAEScgHEIkXX...` 불투명 토큰이고 DOM 어디에도 실제 URL 이 없다.
   `cite` 는 호스트만 신뢰 가능(경로는 `…` 로 잘림).
   → 제목·요약·호스트는 정확하다. 링크는 fetch 시점에 풀린다(실제로 읽는 1~3개만
   비용을 낸다). 다만 **다른 엔진 결과와 URL 중복제거가 안 된다.**

## 실측 상태 (2026-11-18)

| 엔진 | 결과 | 비고 |
|---|---|---|
| google | ✅ 유기적 결과 10건, 난수 질의 0건(정직) | URL 래핑 |
| brave | ⛔ HTTP 429 + "봇이 아닌지 확인" | 자동 백오프 후 재시도. 안 되면 버린다 |
| qwant | ⛔ HTTP 200 / 제목 "Service unavailable" | 챌린지 아님, 그냥 거부 |

신원 `default` 는 자동으로 쿠키 9개를 쌓았다 (google NID/AEC/SNID/_GRECAPTCHA 등).
brave 쿠키는 0개다. **그래서 brave 는 버린다** — 사람을 불러 통과시키지 않는다.

## 기동 — systemd 가 소유한다 (2026-11-21)

### 전제

| 항목 | 우리 값 |
|---|---|
| `xvfb-run` | `/usr/bin/xvfb-run` (`apt install xvfb`). **headful + xvfb 로 돈다** — headless-shell 은 UA 에 `HeadlessChrome` 이 박힌다 |
| Chromium | playwright 가 `~/.cache/ms-playwright/` 에 받아 둔 것. **리포 밖**이다 → `.venv/bin/playwright install chromium` |
| venv | `.venv` (playwright) |
| 토큰 | `state/token` (없으면 `serve.sh` 가 `openssl rand -hex 32` 로 만든다) |

```bash
# 유닛은 이 디렉터리의 systemd/ 에 사본이 있다 — 리포 밖에 두면 또 잃는다.
# ⚠️ 유닛 안 경로는 우리 환경 절대경로다. 다른 박스면 $PWD 로 치환해서 넣는다.
printf 'LANE_TOKEN=%s\nLANE_PORT=8099\nLANE_IDENTITY=default\n' "$(cat state/token)" > state/env
chmod 600 state/env

sed "s|/home/user/work/dspark-serve/websearch-browser|$PWD|g" \
  systemd/websearch-browser.service > ~/.config/systemd/user/websearch-browser.service
sudo loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user enable --now websearch-browser

systemctl --user is-active websearch-browser     # active
curl -s http://127.0.0.1:8099/health | python3 -m json.tool
```

`MemoryMax=1500M` / `MemorySwapMax=0` 은 유닛에 들어 있다 — 위 "자원 제약" 때문에
**빼면 안 된다.** `serve.sh` 는 수동 기동용으로 남겨 두지만, systemd 가 들고 있는
동안에는 쓰지 않는다(소유자가 둘이면 8099 바인드에서 깨진다).

### 참고값 — 우리 환경 실측 (2026-11-21)

재구성한 쪽이 제대로 선 것인지 비교할 기준이다.

| 항목 | 우리 값 |
|---|---|
| 포트 / 바인딩 | `8099`, `127.0.0.1` + `172.18.0.1`(도커 브리지)만. LAN·공인 IP 에서 **안 닿음**(다른 기계에서 확인) |
| 유휴 상태 점유 | **124MB** (파이썬 서버만. `browser_up: false`) |
| 메모리 캡 | `MemoryMax=1500M` / swap 0 |
| 유휴 종료 | `idle_exit_s: 180` |
| 신원 | `identities/default` — google 쿠키 9개 자동 축적, brave 0개 |
| 자동복구 | `kill -9` → 약 10초 후 부활 (실측, `NRestarts=1`) |

```bash
# SearXNG 컨테이너에서 닿는지 — 3층 엔진은 이 주소로 물린다
docker exec searxng sh -c 'wget -qO- http://172.18.0.1:8099/health'
# {"ok": true, "identity": "default", "browser_up": false, "idle_exit_s": 180.0, ...}
```

⛔ SearXNG 템플릿에서 `google browser`·`brave browser`·`qwant browser` 세 엔진은
`disabled: true` 다. **bang 전용**(`!gbr`)으로 쓴다 — 기본셋에 넣으면 모든 질의가 8초
경로를 타고, 질의 수가 늘어 차단만 앞당긴다. 이 레인은 적게 쓰는 게 목표다.

---

## 이 레인의 위치 (2026-11-18 재정의)

이건 **3층**이고, **적게 쓰는 게 목표인 층**이다. 0층(로컬 색인)·1층(기계용 공개창구)
은 같은 리포의 [`../websearch-index/`](../websearch-index/README.md) 가 맡는다. 3층 사용률이 KPI 이고, 내려가면 성공이다.
