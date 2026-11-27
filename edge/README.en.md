[한국어](README.md) | **English**

# edge — externally exposed surface (NPM + searxng)

> This document is written for **an agent that builds this setup on someone's behalf**. Since the target
> environment is unknown, values are not specified one by one. It states the invariants that must hold, the reasoning behind decisions,
> and how to check by yourself whether you are done. The configuration files are in this directory.

**Four things that must not be broken.** Everything else may be changed to fit the environment.

1. Requests without a key are rejected. The only exception is the ACME challenge path.
2. The ACME challenge (`/.well-known/acme-challenge/`) must always be let through.
3. searxng does not publish a host port.
4. The NPM admin UI is kept unreachable from the internet.

**Things you must ask a human for** — the agent cannot do them.

- Domain A record pointing to the public IP (DNS)
- TCP 80·443 forwarding on the router (ACME comes in on **80**)
- Creating the first NPM admin account (browser setup wizard. See ⛔ below)
- Putting the API key value into the client configuration

The router forwards **only the ports that are needed** to this box: TCP 80·443, the ssh port, and, if you
use WireGuard, its UDP port. Before that it was a DMZ, and searxng brought up on `192.168.1.66:8888` answered
as-is on port 8888 of the public IP. **Binding to a LAN IP is not security** — a single forwarding rule
opens it again. So the service side also does not publish host ports, and it is blocked
from both sides.

nft/iptables are not used. docker rewrites the destination port with DNAT first, so `tcp dport` rules in the filter hook
never match in the first place, and they are easy to forget during a rebuild.

| Port | Binding |
|---|---|
| 80 / 443 | `0.0.0.0` — ACME and all vhosts |
| 81 (NPM admin UI) | **`127.0.0.1` only** — via SSH tunnel only |
| searxng | **not published** — NPM connects directly to `searxng:8080` over `nginx-network` |

---

## Nginx Proxy Manager

Run it with docker. If `/data` and `/etc/letsencrypt` are bind-mounted to the host, vhosts, certificates, and
settings stay there — as long as that directory survives, a rebuild is one compose line.

```bash
docker network create nginx-network 2>/dev/null || true
docker compose -f edge/nginx/docker-compose.yml up -d
```

The admin UI (81) is bound to loopback only, so go in through an SSH tunnel and, the first time, create the
admin with the setup wizard. ⛔ Do not try to automate it with the `INITIAL_ADMIN_EMAIL`/`PASSWORD` environment variables —
NPM 2.13.2's `backend/setup.js:40` references the undefined identifier `email`, so
taking that path crashes.

```bash
ssh -p <ssh-port> -L 8081:127.0.0.1:81 <user>@<public-IP>   # → http://localhost:8081
```

⛔ Do not publish the admin UI as a vhost. Even with 81 bound to loopback, **inside the container
it is still listening on 81**, so if you create a vhost pointing to `127.0.0.1:81`, it is exposed again regardless of the
host binding. Such a vhost actually was exposing the admin login to the whole world without an access list
(deleted 2026-11-13).

**Set up two vhosts.** The domain A record must point to the public IP for a certificate to be issued.
In the SSL tab, request a certificate + Force SSL.

- `searxng.<domain>` → `http://searxng:8080` (the container name, since there is no host port)
  Put the **key gate** in Advanced → `edge/npm/vhost-searxng-advanced.conf`.
  Generate a key with `openssl rand -hex 32` and substitute it. searxng itself has no authentication, so
  this block is the only lock.
- `llm.<domain>` → `http://<host LAN IP>:8443` — the encrypted gateway. Leave Advanced empty.
  The gateway authenticates on its own with the AES key. The gateway is a host process and cannot be reached by a container name,
  so use the LAN IP. ⛔ Do not point it directly at vLLM (`:8000`) — the bearer token would cross
  the network on every request. Gateway installation: repository root README §5.8.

What goes into Advanced is in `edge/npm/vhost-searxng-advanced.conf`, with comments,
so paste it as-is and substitute only the key. If you set this up repeatedly, you can also automate it with NPM's REST API
(`/api/tokens` → `/api/nginx/proxy-hosts`).

⛔ **That Advanced block is paired with `edge/npm/http.conf`. Put that in first.** The block's
`proxy_cache sxng;` references a zone defined at the http level. If you paste it without the zone, nginx
**rejects the config check with nginx: [emerg] "proxy_cache" zone "sxng" is unknown and the whole front end fails to come up** — the admin
UI dies with it, which makes it easy to panic.

```bash
# The NPM data directory is the /data bind in edge/nginx/docker-compose.yml
#   (our environment: /home/user/work/nginx/data → container /data)
mkdir -p <NPM data>/nginx/custom
cp edge/npm/http.conf <NPM data>/nginx/custom/http.conf

docker exec nginx-proxy-manager nginx -t        # must say syntax is ok
docker exec nginx-proxy-manager nginx -s reload
```

If you will not use the cache, match them the other way — delete the vhost's `location /search { … }` block and
`http.conf` is not needed either. Having only one of the two is the only combination that breaks.
For the detailed rationale and the check / purge procedure, see "/search result cache" below.

---

## searxng

Run it with docker. **Do not open a host port** — it only needs to be reachable by NPM over the docker network.

```bash
./searxng-serve.sh start      # creates settings.yml from the template if it does not exist
./searxng-serve.sh test
```

For settings, copy `edge/searxng/settings.yml.example` and generate only a new `secret_key`.
The content only overrides a few things on top of the defaults, but **two things must be set correctly.**

- **Include `json` in `search.formats`.** Without it the JSON API is blocked, so the client
  cannot use SearXNG and keeps falling back to ddgs. This is the kind of failure where only quality silently drops.
- **Turn off `server.limiter`.** If it is on, programmatic requests get blocked with 429.
  Authentication is done by the key gate in NPM in front, so there is no reason to block again here.

Choose engines as you like, but the template turns `bing` off. It was confirmed to be in a state where it does not search and
**silently returns ads and trending links with 200** (verified with random queries). duckduckgo and
qwant throw an exception when blocked and get marked unresponsive, but bing's failure does not show,
so it took the top spot and went straight into the context.

---

## /search result cache (2026-11-18)

The real constraint behind engine blocking was not the "shape" of requests but **the number of queries per IP**. Measured — from the same IP
a real browser got 9 organic google results, but after a few more queries it showed a 429 `/sorry` page with
our IP written on it. brave locked us out for several hours after 15 queries in 10 minutes. So **reducing the number of queries
is the only free lever**. We do not go in the direction of disguising fingerprints to increase throughput.

**The two files are a pair. Put in only one and nginx does not come up.**

| | Where it goes |
|---|---|
| `edge/npm/http.conf` (zone definition) | container `/data/nginx/custom/http.conf` |
| `location /search` in `edge/npm/vhost-searxng-advanced.conf` | Advanced of the corresponding NPM vhost |

If you put in only the vhost without the zone definition, the config check rejects it and **the whole front end does not come up** (measured:
`nginx: [emerg] "proxy_cache" zone "sxng" is unknown in /etc/nginx/nginx.conf:85`). If you will not use the cache, also remove the vhost's `location /search` block.

```bash
# Install (the host path is the data volume in edge/nginx/docker-compose.yml)
cp edge/npm/http.conf <NPM data>/nginx/custom/http.conf
docker exec nginx-proxy-manager nginx -t && docker exec nginx-proxy-manager nginx -s reload

# Check it works — the second call must be a HIT
curl -sI -H "X-API-Key: <key>" "https://searxng.<domain>/search?q=test&format=json" | grep -i x-cache
curl -sI -H "X-API-Key: <key>" "https://searxng.<domain>/search?q=test&format=json" | grep -i x-cache

# Purge the cache
docker exec nginx-proxy-manager sh -c 'rm -rf /data/cache/searxng/*'
docker exec nginx-proxy-manager nginx -s reload
```

Only GET/HEAD are cached — POST searches from the web UI are not covered. `proxy_cache_lock`
merges upstream requests into one when an agent fires the same query in parallel (8 parallel → 1 upstream).

---

## Layer 0 — local index (`websearch-index/`)

The root cause of search getting blocked is that it is "query → relay to external engines", so **the more you use it, the more you get blocked**.
So queries are split into layers. Detailed design, measurements, and startup:
[`../websearch-index/README.en.md`](../websearch-index/README.en.md).

```
Layer 0  local index (SQLite FTS5)         6.7~7.4ms  no concept of blocking
Layer 1  public machine-facing endpoints   0.6~0.9s   no key, no blocking
Layer 2  anonymous SearXNG (this dir)      1.4s       sometimes blocked
Layer 3  identity browser                  8s         often blocked, last resort
```

It is attached to SearXNG as a `json_engine`. **Upstream code is not touched.**

```yaml
- name: local index     # shortcut idx  — default set
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=0
- name: index live      # shortcut xidx — bang only (goes outside)
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=1
```

Layer 3 (identity browser, [`../websearch-browser/DESIGN.en.md`](../websearch-browser/DESIGN.en.md)) is **not an engine.**
Whether to use the owner's browser identity is decided by the server — it cannot be specified through the query. It used to be attached as a `!gbr` bang-only
engine, but that left the decision to the caller (the model did not know the syntax, so nobody could use it) and anyone holding the SearXNG key
could search with the owner's identity (bangs work even with `disabled: true`). Removed 2026-11-27.

Now the plugin [`searxng/plugins/browser_escalate.py`](searxng/plugins/browser_escalate.py) looks at all engines' results
in `post_search` and calls the lane (`172.18.0.1:8099`) directly only when there are **fewer than 5 results, or at least half the engines are unresponsive + fewer than 10 results**.
First page · general · default engine set only, budget of 6 per 10 minutes,
query cache 30 minutes, timeout 15 seconds (cold Chromium). How to set it up:

```bash
# 1) lane token into the searxng config directory (outside the repo, readable only by searxng uid 977)
cat ~/work/dspark-serve/websearch-browser/state/token | docker exec -u 0 -i searxng \
  sh -c 'cat > /etc/searxng/browser_lane_token && chown 977:977 /etc/searxng/browser_lane_token && chmod 600 /etc/searxng/browser_lane_token'
# 2) the plugin mount is already in docker-compose.yml → docker compose up -d
# 3) the plugins: block in settings.yml (it is in the template)
```

⛔ Two pitfalls (both measured 11-27):
- The `plugins:` block **replaces** the default list rather than **merging** with it. If you do not also list the default plugins (calculator, tracker URL remover, etc.),
  they are all turned off. The template has the default list copied into it.
- Ordinary results returned by a plugin come in under the engine name `plugin: <id>`, but the score counters in `ResultContainer.close()`
  exist only for engines registered at startup, so **KeyError → that whole search fails**. The plugin avoids this by registering
  the counter (`_ensure_score_counter`). Upstream plugins only return answers, so nobody had stepped on it.

⛔ **In the template these two engines are `disabled: true`.** Lower `disabled` on `local index` only on the box
where `websearch-index` is running. If you turn it on without the service, every query goes unresponsive, but
SearXNG does not surface engine failures as errors, so **only quality silently drops** — the same kind as
"Things that break silently" below.

`172.18.0.1` is the docker bridge gateway. The index service binds only to `127.0.0.1` and that address,
so it is not reachable from the LAN or public IP. Check from the searxng container:

```bash
docker exec searxng sh -c 'wget -qO- http://172.18.0.1:8097/health'
# {"ok": true, "port": 8097, "docs": 4269}
```

---

## Client

Search runs **on the client**. vLLM does not search on its behalf —
`codemason/tools/web_search_tool.py::_searxng_search` hits HTTP directly with stdlib urllib
and attaches `X-API-Key`.

```
cm2 (any machine)
  ├─ model  ──► 443 llm.<domain>     ──► :8443 AES gateway ──► vLLM
  └─ search ──► 443 searxng.<domain> ──► NPM ──► searxng:8080
                   (no X-API-Key → 403 → ddgs fallback)
```

Set `searxng_url` and `searxng_api_key` in `~/.codemason/config.json`. The configuration
lives separately on each machine.

---

## Things that break silently

- **If you remove the ACME exemption from the key gate**, certificate issuance and renewal both fail. nginx's server-level
  `if` runs before location selection, so not even `location ^~ /.well-known/acme-challenge/` can win over it.
  A certificate actually expired and search fell back to ddgs for 25 days.
- **If the keys don't match**, search does not die; only quality drops. When changing it, change the vhost and the client
  together.
- **If `./searxng-serve.sh test` does not show `키 없이 → 403`** ("without key → 403"), the gate is missing.
  In that state it is an unauthenticated public search proxy. Others could scrape search engines through this box's IP and even
  get it IP-banned.

## ⚠️ NPM's state is not in git

vhosts, certificates, and Access Lists are all inside `/home/user/work/nginx/data/database.sqlite`,
which contains keys and certificates, so it is not committed. If you lose that directory, you have to redo the vhost setup above.

`nginx/custom/http.conf` (the cache zone) lives in the same directory but has no secrets, so this repository keeps a copy
as `edge/npm/http.conf` — when restoring, just copy it as-is.

Host paths (`/home/user/...`) appear in three places: `edge/nginx/docker-compose.yml`,
`edge/searxng/docker-compose.yml`, and `searxng-serve.sh`.

---

## How to check you are done

The agent is not done until all of the following are true.

```bash
./searxng-serve.sh test
```

Expected three lines:

```
[1/2] 내부 직결  → ✅ results=<not 0>      ("internal direct") NPM reaches it by container name
[2/2] 공개 경로  → ✅ results=<not 0>      ("public path") TLS + key are correct
                 → ✅ 키 없이 403          ("without key 403") the gate is alive
```

**If it is not `키 없이 403` ("without key 403"), stop there.** In that state it is an unauthenticated public search proxy.

From outside the box (another machine):

```bash
for p in 80 443 8888 81; do nc -z -w3 <public-IP> $p && echo "$p OPEN" || echo "$p 닫힘"; done
# only 80 and 443 should be OPEN ("닫힘" = closed)
```

To check certificate renewal too: `certbot renew --dry-run --no-random-sleep-on-renew`.
Without the flag it sits silent for up to 5 minutes (non-interactive jitter) — easy to mistake for a hang.

## Where agents tend to go wrong

- **Do not publish with `-p` because a port is unreachable.** Put it on the same docker network as NPM
  and call it by container name. Publishing is itself a hole.
- **Do not try to block with nft/iptables.** docker rewrites the destination port with DNAT first,
  so `tcp dport` rules in the filter hook do not match. Not opening the port is the answer.
- **If ACME gets 403, suspect the gate condition.** nginx `if` has no `&&`, so it has to be written
  with flags, and a server-level `if` runs before location selection.
- **Do not point the model vhost directly at vLLM (`:8000`).** It goes through the gateway (`:8443`).
- **If search works but results are thin**, it is most likely falling back. A key mismatch or a missing JSON
  format does not raise an error; it silently drops to ddgs as the single source.
