# edge — 외부 노출면 (NPM + searxng)

> 이 문서는 **이 구성을 대신 세우는 에이전트**를 상대로 쓴 것이다. 상대 환경을
> 알 수 없으므로 값을 일일이 지정하지 않는다. 지켜야 할 불변식과 판단 근거,
> 그리고 끝났는지 스스로 확인할 방법을 적는다. 설정 파일은 이 디렉터리에 있다.

**깨면 안 되는 것 네 가지.** 나머지는 환경에 맞춰 바꿔도 된다.

1. 키 없는 요청은 거부한다. 예외는 ACME 챌린지 경로 하나뿐이다.
2. ACME 챌린지(`/.well-known/acme-challenge/`)는 반드시 통과시킨다.
3. searxng 는 호스트 포트를 퍼블리시하지 않는다.
4. NPM 관리 UI 는 인터넷에서 닿지 않게 둔다.

**사람한테 요청해야 하는 것** — 에이전트가 못 한다.

- 도메인 A 레코드를 공인 IP 로 (DNS)
- 공유기에서 TCP 80·443 포워딩 (ACME 는 **80** 으로 들어온다)
- NPM 첫 관리자 계정 생성 (브라우저 setup wizard. 아래 ⛔ 참고)
- API 키 값을 클라이언트 설정에 넣기

공유기는 **필요한 포트만** 이 박스로 포워딩한다: TCP 80·443, ssh 포트, WireGuard 를
쓰면 그 UDP 포트. 그 전에는 DMZ 였는데, `192.168.1.66:8888` 로 띄운 searxng 가
공인 IP 의 8888 로 그대로 응답했다. **LAN IP 바인딩은 보안이 아니다** — 포워딩 규칙
한 줄이면 다시 열린다. 그래서 서비스 쪽에서도 호스트 포트를 퍼블리시하지 않고
양쪽에서 막는다.

nft/iptables 는 쓰지 않는다. docker 가 DNAT 로 목적지 포트를 먼저 바꿔서 filter 훅의
`tcp dport` 규칙은 애초에 매칭되지 않고, 재구성 때 빠뜨리기도 쉽다.

| 포트 | 바인딩 |
|---|---|
| 80 / 443 | `0.0.0.0` — ACME 와 모든 vhost |
| 81 (NPM 관리 UI) | **`127.0.0.1` 만** — SSH 터널로만 |
| searxng | **퍼블리시 없음** — NPM 이 `nginx-network` 로 `searxng:8080` 직결 |

---

## Nginx Proxy Manager

도커로 띄운다. `/data` 와 `/etc/letsencrypt` 를 호스트에 바인드해 두면 vhost·인증서·
설정이 거기 남는다 — 그 디렉터리가 살아 있으면 재구성은 compose 한 줄로 끝난다.

```bash
docker network create nginx-network 2>/dev/null || true
docker compose -f edge/nginx/docker-compose.yml up -d
```

관리 UI(81)는 루프백에만 묶여 있으니 SSH 터널로 들어가서, 처음이면 setup wizard 로
관리자를 만든다. ⛔ `INITIAL_ADMIN_EMAIL`/`PASSWORD` 환경변수로 자동화하려 하지 말 것 —
NPM 2.13.2 의 `backend/setup.js:40` 이 정의되지 않은 식별자 `email` 을 참조해서
그 경로를 타면 터진다.

```bash
ssh -p <ssh포트> -L 8081:127.0.0.1:81 <user>@<공인IP>   # → http://localhost:8081
```

⛔ 관리 UI 를 vhost 로 공개하지 말 것. 81 을 루프백에 묶어도 **컨테이너 안에서는
여전히 81 로 떠 있어서**, `127.0.0.1:81` 을 가리키는 vhost 를 만들면 호스트 바인딩과
무관하게 다시 뚫린다. 실제로 그런 vhost 가 access list 없이 관리 로그인을 전 세계에
노출하고 있었다(2026-11-13 삭제).

**vhost 는 두 개 건다.** 도메인 A 레코드가 공인 IP 를 가리켜야 인증서가 나온다.
SSL 탭에서 인증서 요청 + Force SSL.

- `searxng.<도메인>` → `http://searxng:8080` (컨테이너 이름. 호스트 포트가 없으니까)
  Advanced 에 **키 게이트**를 넣는다 → `edge/npm/vhost-searxng-advanced.conf`.
  `openssl rand -hex 32` 로 키를 만들어 치환한다. searxng 자체에는 인증이 없으니
  이 블록이 유일한 자물쇠다.
- `llm.<도메인>` → `http://<호스트 LAN IP>:8443` — 암호화 게이트. Advanced 는 비운다.
  게이트가 AES 키로 자체 인증한다. 게이트는 호스트 프로세스라 컨테이너 이름으로
  못 부르므로 LAN IP 를 쓴다. ⛔ vLLM(`:8000`)에 직접 물리지 말 것 — 베어러 토큰이
  매 요청 네트워크를 건넌다. 게이트 설치는 저장소 루트 README §5.8.

Advanced 에 넣을 내용은 `edge/npm/vhost-searxng-advanced.conf` 에 주석까지 달아
뒀으니 그대로 붙여넣고 키만 치환하면 된다. 반복해서 세울 일이 많으면 NPM 의 REST API
(`/api/tokens` → `/api/nginx/proxy-hosts`)로 자동화할 수도 있다.

---

## searxng

도커로 띄운다. **호스트 포트는 열지 않는다** — NPM 만 도커 네트워크로 닿으면 된다.

```bash
./searxng-serve.sh start      # settings.yml 이 없으면 템플릿에서 만든다
./searxng-serve.sh test
```

설정은 `edge/searxng/settings.yml.example` 을 복사해서 `secret_key` 만 새로 만들면
된다. 내용은 기본값 위에 몇 개만 덮어쓰는 수준인데, **두 가지는 반드시 맞춰야 한다.**

- `search.formats` 에 **`json` 을 넣을 것.** 없으면 JSON API 가 막혀서 클라이언트가
  SearXNG 를 못 쓰고 계속 ddgs 로 폴백한다. 조용히 품질만 떨어지는 유형이다.
- `server.limiter` 를 **끌 것.** 켜져 있으면 프로그램적 요청이 429 로 막힌다.
  인증은 앞단 NPM 의 키 게이트가 하므로 여기서 또 막을 이유가 없다.

엔진은 취향대로 고르되, 템플릿은 `bing` 을 꺼 둔다. 검색을 하지 않고 광고·트렌딩
링크를 **조용히 200 으로** 반환하는 상태로 확인됐다(난수 질의로 검증). duckduckgo 나
qwant 는 막히면 예외를 던져 unresponsive 로 표시되는데, bing 은 실패가 드러나지
않아서 최상단을 차지한 채 그대로 컨텍스트로 들어갔다.

---

## 클라이언트

검색은 **클라이언트에서** 돈다. vLLM 이 대신 검색하지 않는다 —
`codemason/tools/web_search_tool.py::_searxng_search` 가 stdlib urllib 로 직접 HTTP 를
때리고 `X-API-Key` 를 붙인다.

```
cm2 (어느 기계든)
  ├─ 모델   ──► 443 llm.<도메인>     ──► :8443 AES 게이트 ──► vLLM
  └─ 검색   ──► 443 searxng.<도메인> ──► NPM ──► searxng:8080
                   (X-API-Key 없으면 403 → ddgs 폴백)
```

`~/.codemason/config.json` 의 `searxng_url`·`searxng_api_key` 를 맞춘다. 설정은
머신마다 따로 산다.

---

## 조용히 깨지는 것들

- **키 게이트에서 ACME 면제를 빼면** 인증서 발급도 갱신도 실패한다. nginx 의 server
  레벨 `if` 는 location 선택보다 먼저 돌아서 `location ^~ /.well-known/acme-challenge/`
  조차 못 이긴다. 실제로 인증서가 만료돼 검색이 25일간 ddgs 로 폴백했다.
- **키가 어긋나면** 검색이 죽지 않고 품질만 떨어진다. 바꿀 땐 vhost 와 클라이언트를
  같이 바꾼다.
- **`./searxng-serve.sh test` 에서 `키 없이 → 403` 이 안 나오면** 게이트가 빠진 것이다.
  그 상태는 무인증 공개 검색 프록시다. 남이 이 박스 IP 로 검색엔진을 긁어서 IP 밴까지
  유발할 수 있다.

## ⚠️ NPM 의 상태는 git 에 없다

vhost·인증서·Access List 는 전부 `/home/user/work/nginx/data/database.sqlite` 안에
있고 키와 인증서가 들어 있어 커밋하지 않는다. 그 디렉터리를 잃으면 위 vhost 셋팅을
다시 해야 한다.

호스트 경로(`/home/user/...`)는 `edge/nginx/docker-compose.yml`,
`edge/searxng/docker-compose.yml`, `searxng-serve.sh` 세 곳에 있다.

---

## 끝났는지 확인하는 법

에이전트는 아래가 전부 맞을 때까지 끝난 게 아니다.

```bash
./searxng-serve.sh test
```

기대값 세 줄:

```
[1/2] 내부 직결  → ✅ results=<0 아님>      NPM 이 컨테이너 이름으로 닿는다
[2/2] 공개 경로  → ✅ results=<0 아님>      TLS + 키가 맞는다
                 → ✅ 키 없이 403          게이트가 살아 있다
```

**`키 없이 403` 이 아니면 거기서 멈춰라.** 그 상태는 무인증 공개 검색 프록시다.

박스 밖(다른 기계)에서:

```bash
for p in 80 443 8888 81; do nc -z -w3 <공인IP> $p && echo "$p OPEN" || echo "$p 닫힘"; done
# 80·443 만 OPEN 이어야 한다
```

인증서 갱신까지 보려면 `certbot renew --dry-run --no-random-sleep-on-renew`.
플래그를 빼면 최대 5분 가만히 있는다(비대화형 지터) — 멎은 걸로 오해하기 쉽다.

## 에이전트가 틀리기 쉬운 지점

- **포트가 안 닿는다고 `-p` 로 퍼블리시하지 말 것.** NPM 과 같은 도커 네트워크에
  올리고 컨테이너 이름으로 부르면 된다. 퍼블리시는 그 자체가 구멍이다.
- **nft/iptables 로 막으려 하지 말 것.** docker 가 DNAT 로 목적지 포트를 먼저
  바꿔서 filter 훅의 `tcp dport` 규칙은 매칭되지 않는다. 포트를 안 여는 게 답이다.
- **ACME 가 403 이면 게이트 조건을 의심할 것.** nginx `if` 에는 `&&` 가 없어서
  플래그 방식으로 써야 하고, server 레벨 `if` 는 location 선택보다 먼저 돈다.
- **모델 vhost 를 vLLM(`:8000`)에 직접 물리지 말 것.** 게이트(`:8443`)를 거친다.
- **검색이 되는데 결과가 부실하면** 폴백 중일 가능성이 높다. 키 불일치나 JSON
  포맷 누락은 에러를 내지 않고 조용히 ddgs 단일 소스로 떨어진다.

