# edge — 외부 노출면 (NPM + searxng)

`.66` 은 공유기 **DMZ** 안에 있다. 모든 포트가 인터넷에서 직접 닿는다.
따라서 **LAN IP 로 바인딩하는 것은 보안이 아니다** — `192.168.1.66:8888` 로 띄운
searxng 가 `http://220.85.97.203:8888` 로 그대로 응답했다(2026-11-13 실측).

노출을 줄이는 수단은 하나다: **호스트 포트를 퍼블리시하지 않는 것.**
nft/iptables 규칙은 쓰지 않는다 — docker 가 DNAT 로 목적지 포트를 먼저 바꿔서
filter 훅의 `tcp dport` 규칙이 애초에 매칭되지 않고, 재구성 때 빠뜨리기 쉽다.

## 포트 정책

| 포트 | 바인딩 | 근거 |
|---|---|---|
| 80 | `0.0.0.0` | ACME HTTP-01 + 443 리다이렉트. 필수 |
| 443 | `0.0.0.0` | 모든 vhost 입구 |
| 81 (NPM 관리 UI) | **`127.0.0.1` 만** | SSH 터널로만 접근 |
| searxng | **퍼블리시 없음** | NPM 이 `nginx-network` 로 `searxng:8080` 직결 |

### NPM 관리 UI 접속

```bash
ssh -p 2222 -L 8081:127.0.0.1:81 user@<공인IP>
# → http://localhost:8081
```

⛔ 관리 UI 를 vhost 로 다시 공개하지 말 것. 과거 vhost `proxy.systemboan.net` →
`http://192.168.1.66:81` 이 access list 없이 관리 로그인을 전 세계에 노출하고
있었다(2026-11-13 삭제). 81 을 루프백에 묶어도 **컨테이너 내부에서는 여전히
81 로 떠 있으므로**, `127.0.0.1:81` 을 가리키는 vhost 를 만들면 호스트 바인딩과
무관하게 다시 뚫린다.

## 재구성 순서

```bash
# 0) 공용 네트워크
docker network create nginx-network 2>/dev/null || true

# 1) NPM — /data, /etc/letsencrypt 는 호스트 바인드. 기존 디렉터리가 있으면
#    DB·인증서가 그대로 살아난다(=vhost 와 키 게이트도 복원됨).
docker compose -f edge/nginx/docker-compose.yml up -d

# 2) searxng — 설정이 없으면 템플릿에서 만든다 (secret_key 는 새로 생성)
mkdir -p /home/user/searxng
[ -f /home/user/searxng/settings.yml ] || {
  sed "s|__GENERATE_WITH_openssl_rand_hex_32__|$(openssl rand -hex 32)|" \
    edge/searxng/settings.yml.example > /home/user/searxng/settings.yml
}
../searxng-serve.sh start     # 또는: docker compose -f edge/searxng/docker-compose.yml up -d

# 3) 확인
./searxng-serve.sh test
```

## ⚠️ NPM 의 상태는 git 에 없다

vhost 목록·인증서·Access List 는 전부 `/home/user/work/nginx/data/database.sqlite`
안에 있고, 키와 인증서가 들어 있어 커밋하지 않는다. **그 디렉터리를 잃으면
아래가 같이 사라진다:**

- searxng 의 `x-api-key` 게이트 (= searxng 의 유일한 인증)
- ACME 챌린지 면제 설정 (= 90일 후 인증서 만료 → 검색이 조용히 ddgs 로 폴백)
- 모든 vhost 와 Let's Encrypt 인증서

백지에서 복원할 때 **반드시** 아래를 다시 넣어야 한다:

| vhost | 타겟 | Advanced |
|---|---|---|
| `searxng.systemboan.net` | `http://searxng:8080` | `edge/npm/vhost-searxng-advanced.conf` (키 치환) |
| `llm.systemboan.net` | `http://192.168.1.66:8443` | 없음 (AES 게이트가 자체 인증) |

`edge/npm/vhost-searxng-advanced.conf` 의 `__SEARXNG_API_KEY__` 를 실제 키로
바꿔 넣는다. 그 값은 클라이언트의 `~/.codemason/config.json` 의
`searxng_api_key` 와 같아야 한다.

## codemason 검색 경로 (중요)

검색은 **클라이언트에서** 돈다. vLLM 이 대신 검색하지 않는다 —
`codemason/tools/web_search_tool.py::_searxng_search` 가 stdlib urllib 로
직접 HTTP 를 때리고 `X-API-Key` 헤더를 붙인다.

```
cm2 (어느 기계든)
  ├─ 모델   ──► 443 llm.systemboan.net     ──► :8443 AES 게이트 ──► vLLM
  └─ 검색   ──► 443 searxng.systemboan.net ──► NPM ──► searxng:8080
                   (X-API-Key 없으면 403 → ddgs 폴백)
```

그래서 `searxng.systemboan.net` 의 443 은 계속 공개돼 있어야 한다. 키 게이트가
그 공개를 떠받친다. 키가 틀리면 검색이 죽지 않고 **조용히 품질만 떨어진다**
(ddgs 단일 소스) — 그게 2026-10-18~11-12 에 25일간 벌어진 일이다.
