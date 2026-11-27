[한국어](README.md) | **English**

# websearch-index — a search server that gets blocked **less** the more you use it

## Thesis

A conventional search server is a "query → external engine relay". That is why it **gets blocked more the more you use it** — every query adds to the block probability and the rate limits. This is also the root reason SearXNG gets blocked.

Here we invert that.

```
query ┬─ layer 0  local index (SQLite FTS5)       2.7ms   no notion of "blocked"
      │       filled by an always-on collector (feeds + promoted subscriptions)
      ├─ layer 1  machine-facing public endpoints  0.6~0.9s  no key, no blocking
      │       GoogleNews RSS · HN · GitHub · StackExchange · Wikipedia
      ├─ layer 2  anonymous SearXNG (existing)     1.4s    occasionally blocked
      └─ layer 3  identity browser (~/work/websearch-browser)  8s  often blocked, last resort
```

The key is not the layering itself but **two promotions**.

1. **Query → subscription promotion.** Ask about the same topic twice and it is registered as a Google News RSS subscription.
   From then on that topic is served from layer 0 permanently. **The more you search, the larger the block-free area becomes.**
2. **Permanent storage of layer-1 results.** Everything fetched from outside goes into the index. From the second time on,
   that endpoint is not hit at all.

**The layer-3 usage rate is the KPI. Going down = success.** `layer_mix_24h` in `./ws.sh stats`.

## Decisive finding (measured 2026-11-18)

**Google blocks `google.com/search` but does not block `news.google.com/rss/search`.**
Same company, same index family, yet one door is open for machines. No key needed either.

| Query | Result | Time |
|---|---|---|
| `vllm site:github.com` | 100 results, all GitHub — **the `site:` operator actually works** | 0.65s |
| `AI when:1h` | 100 results, articles from minutes ago — **`when:` works too** | 0.65s |
| `vLLM 추론 최적화` (Korean) | 7 results, domestic outlets | 0.38s |
| `mocflwsrrhbubkwdevzz qqxz` (random) | **0 results** ← passes the honesty check | 0.42s |

And a pair that proves the thesis directly:

| arXiv approach | Result |
|---|---|
| Live API (`export.arxiv.org/api/query`) | 5 results / **12.9 s** / refused on consecutive calls |
| RSS feed (`rss.arxiv.org/rss/cs.LG`) | **200 results / 0.1 s** / no limit |

→ So arXiv was removed from layer 1 and moved down to the layer-0 collector.

⛔ **Decided not to use**: Reddit `.json` (403), CommonCrawl CDX (504), Bing News RSS (empty response),
Brave Search API (free 2,000/month tier **discontinued 2026-02** → card + $5/month credit), Mojeek API (paid).

## Layout

| File | Role |
|---|---|
| `core.py` | Layer 0. SQLite FTS5 schema, ingestion, search, **relevance gate** |
| `channels.py` | Layer 1. Adapters for machine-facing public endpoints + per-host throttle |
| `collect.py` | Always-on collector. Fixed feeds (`sources.json`) + promoted subscriptions |
| `serve.py` | HTTP. `/search` `/stats` `/subs` `/health` |
| `ws.sh` | Control. `start stop restart status collect search stats subs logs` |

---

## Startup — systemd owns it (2026-11-21)

⛔ **Do not start it with `./ws.sh start`.** After it died in the 2026-11-19 reboot, there was no autostart,
so it **sat silently stopped for 2.5 days.** During that time SearXNG's `idx` engine kept
failing, and since the collector had stopped, the premise of "blocked less the more you use it" stopped too.
User rule: anything that stops silently is a defect, full stop.

⚠️ Paths inside the units are **absolute paths for our environment** (`/home/user/work/dspark-serve/websearch-index`).
On another box you must change five lines — `WorkingDirectory`, `ExecStart`, `EnvironmentFile`, `StandardOutput`,
`StandardError` — to your own path. If a path is wrong, systemd does not silently pass over it;
it fails with `status=200/CHDIR`, which shows up immediately in `systemctl --user status`.

```bash
# Install (copies of the units live in this repo's systemd/ — keep them outside the repo and they get lost again)
sed "s|/home/user/work/dspark-serve/websearch-index|$PWD|g" \
  systemd/websearch-index.service > ~/.config/systemd/user/websearch-index.service
sed "s|/home/user/work/dspark-serve/websearch-index|$PWD|g" \
  systemd/websearch-collect.service > ~/.config/systemd/user/websearch-collect.service
printf 'INDEX_TOKEN=%s\nINDEX_PORT=8097\n' "$(cat state/token)" > state/env
chmod 600 state/env                      # because it contains the token

sudo loginctl enable-linger $USER        # run without a login session (without this it does not come up after reboot)
systemctl --user daemon-reload
systemctl --user enable --now websearch-index websearch-collect

# Verify
systemctl --user is-active websearch-index websearch-collect   # active active
systemctl --user is-enabled websearch-index websearch-collect  # enabled enabled
```

We checked that `Restart=always` actually works by killing it once — after `kill -9` it came back
in about 6 seconds and search worked as before (`NRestarts=1`).

The memory cap is the same as in `ws.sh`: `MemoryMax=600M` / `MemorySwapMax=0`. On this box
vLLM occupies the unified memory and memwatch preemptively kills the serving
container at `MemAvailable<3GB`, so launching without a cap lets search kill the server.

**There must not be two owners.** A process held by systemd does not write `state/*.pid`,
so `ws.sh start` would attempt a duplicate launch and break on the 8097 bind.
`ws.sh` detects that case and refuses. Control is via `systemctl --user`.
`search` `stats` `subs` `collect` `logs` in `ws.sh` are used as before.

---

## Reference values — measured in our environment (2026-11-21)

The baseline for the rebuilding side to check whether "it came up correctly". If a number differs by an order of magnitude, something is wrong.

| Item | Our value |
|---|---|
| Port / binding | `8097`, only `127.0.0.1` + `172.18.0.1` (docker bridge). **Not reachable from LAN or public IP** |
| Indexed documents | 4,318 / `index.db` 4.8MB |
| Documents by source | fixed feeds 3,959 · promoted subscriptions 257 · layer-1 stored 102 |
| Fixed feeds | 27 (`sources.json`) |
| Promoted subscriptions | 7 |
| Layer-0 latency | **6.7~7.4ms** (20 results returned) |
| Memory cap | `MemoryMax=600M` / `MemorySwapMax=0` |

```
$ ./ws.sh search vllm            → 20건   0.0067s
$ ./ws.sh search "GLM 5.3"       → 16건   0.0070s
$ ./ws.sh search <random string> →  0건   0.0074s   ← passes the honesty check. If not 0, the relevance gate is broken
```

(`건` = results, as printed by the tool.)

**If the random query does not return 0 results, stop there.** It means FTS is picking up arbitrary documents,
and in that state garbage goes into the model. ⛔ Do not remove the relevance gate in `core.py`.

### Layer-1 endpoints and throttle (`channels.py`)

```
googlenews_ko · googlenews_en · hackernews · github · stackexchange · wikipedia_ko
```

⛔ Do not reduce the `_MIN_GAP` intervals — it ruins free endpoints.

| Host | Minimum gap | Basis |
|---|---|---|
| `export.arxiv.org` | 3.0s | arXiv's official requirement |
| `api.stackexchange.com` | 1.0s | 300/day unauthenticated |
| `api.github.com` | 1.0s | 10/minute unauthenticated |

⛔ `arxiv()` is not put in layer 1 (`LIVE`). Measured — live API 12.9 s/5 results vs RSS 0.1 s/200 results.
arXiv was moved down to the layer-0 collector. This is where the project's thesis shows most plainly.

### KPI

`layer_mix_24h` in `./ws.sh stats`. **Success = the share of layer 3 (identity browser) going down.**
`layer0_hit_rate_24h` is the share answered by layer 0 alone.

Right after it was brought back (sample of 10 queries) it was `0: 10%` / `0L: 90%` — the index was still thin, so most queries went
down to layer 1. As the collector runs, the layer-0 share rises. **This is a number to look at after a few days of running.**

## SearXNG integration

```yaml
- name: local index     # shortcut idx  — included in the default set
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=0
- name: index live      # shortcut xidx — bang only (goes outside)
  search_url: http://172.18.0.1:8097/search?q={query}&n=20&live=1
```

**Why it is fine to put `local index` in the default set**: with `live=0` it never goes outside, and it takes
2.7ms. And **the side effect is the main point** — every ordinary search query gets recorded and feeds the promotion engine
for free. Layer 0 grows on its own toward the topics the agent asks about often.

## ⛔ Preventing recurrence

1. **Do not remove the relevance gate** (`core.COVER_MIN`). Actual incident: the OR fallback let a document through if it matched just one
   common token, so for `국내 반도체 수출 규제 영향` ("impact of domestic semiconductor export regulation"), `수능 연기` ("CSAT postponed"; matched 영향 "impact")
   and `샤오미18 국내 출시` ("Xiaomi 18 domestic launch"; matched 국내 "domestic") came out at the top of layer 0. **When layer 0 confidently
   returns garbage, it does not even fall through to layer 1, and the agent gives a wrong answer — worse than
   being blocked.** At least 60% of the query tokens must actually appear in the document.
2. **Korean requires the prefix `*`.** FTS5 unicode61 cannot strip particles. `추론` does not match
   `추론을`, but `추론*` does (`core.make_match`). The trigram tokenizer cannot take 2-character
   queries, so it is unsuitable.
3. **Honesty guard**: a random query must return 0 results. `./ws.sh search <random>`.
4. **SearXNG integration needs both `enable_http: true` + `enable_http2: false`.**
   Without them, both show up only as "unexpected crash" (Python http.server cannot do H2).
5. **The NPM cache also caches failure responses for 30 minutes.** Always clear it before debugging:
   `docker exec nginx-proxy-manager sh -c 'rm -rf /data/cache/searxng/*'`
6. **A hard memory cap is mandatory** (`MemoryMax=600M` in `ws.sh`). GB10 has unified memory, so
   an OOM kills the whole box.
7. **Do not ruin doors that were opened for free.** Do not reduce the per-host gaps in `channels._MIN_GAP`
   (arXiv officially requires a 3-second gap; StackExchange unauthenticated
   daily quota 300; GitHub unauthenticated 10/minute).
