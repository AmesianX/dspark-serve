[한국어](DESIGN.md) | **English**

# browser-lane — the search lane with an identity

> ⛔ **2026-11-27: SearXNG bang hookup (`!gbr` `!bbr` `!qbr`) removed.** It was a wrong design that let the caller decide whether to go
> to layer 3 (the owner's identity) — the model did not know the syntax so nobody could use it (the actual path was never verified), and it was a hole that let
> anyone holding the SearXNG key search under the owner's identity. There is now no client → layer-3 path. The lane itself
> is still up (127.0.0.1 · 172.18.0.1, `X-Lane-Token`), and **escalation is decided by the server (implemented 11-27):**
> the SearXNG plugin `edge/searxng/plugins/browser_escalate.py` looks at all engines' results in `post_search` and calls the lane directly
> only when there are fewer than 5 results or (at least half of the engines are blocked + fewer than 10 results). First page, general category, default
> engine set only; budget of 6 per 10 minutes, query cache 30 minutes, timeout 15 seconds. Nothing put in the query can force escalation.
> Measured: 0-result query → lane called (cold 4.8s), 27-result query → not called, forced-threshold e2e → 10 of 32 results came from the lane.
> The bang-related text below is the record from that time.


## Why it is needed (measured 2026-11-18)

The axis bot detection looks at was not "is it AI" but **"does it carry an identity"**.

- SearXNG **hardcodes** `discard_cookies: True` at `searx/network/client.py:93`.
  It is not a setting and is not in `_CLIENT_KWARGS` (verify/max_redirects/impersonate/curl_options/enable_http3)
  either, so it cannot be turned off. That is, 0 cookies per query = a completely new anonymous visitor every time.
  A stream of unfamiliar visitors from one IP → indistinguishable from a datacenter proxy.
- Conversely, Chromium with a persistent profile got 9 Google organic results from the same IP,
  and 0 results for a random query (passes the honesty check). What accumulated in the profile:
  `NID` (expires 2027-05) `AEC` `SNID` `SEARCH_SAMESITE` `_GRECAPTCHA` (the challenge-passed certificate).
- So the roles are split. SearXNG's anonymity is that tool's reason to exist, so **we do not break it.**

| Lane | Engines handled | Nature |
|---|---|---|
| SearXNG (anonymous, existing) | google cse · duckduckgo web · yahoo · mwmbl · wikipedia | identity not required |
| **browser-lane (identified, new)** | google · brave · qwant | identity required |

browser-lane is hooked in as a SearXNG `json_engine` → no changes to upstream code, only our own part is gated.

## Identity model

**Identity = one directory.** `identities/<name>/` = a Chromium user-data-dir.
Auditable (cookie names and expiries can be queried directly), backupable, disposable and replaceable. If it gets burned, make a new one.

### Tiers

| Tier | Content | Privacy cost | Lifetime |
|---|---|---|---|
| 0 | 0 cookies (current SearXNG) | none | — |
| **1 persistent anonymous** ← default | keep cookies, no login. Only consent, clearance and reCAPTCHA certificates accumulate | **none** | google ~6 months / CF clearance hours to days |
| 2 account login | log in with a real account | queries are attributed to the account + risk of ToS violation | long |
| 3 official identity | API key / Web Bot Auth | money / waiting for the standard to mature | permanent |

**The default is tier 1.** Tier 2 is not recommended — the pass rate goes up, but every query gets attached to your account,
and automated queries usually violate the terms. If you use it, use a dedicated account, and only knowing the risk.
Note: the reason `google cse` got through all this trouble is that it is already **tier 3**.

### How identity is obtained — no human is called in ⛔

~~A human attaches via VNC and passes the challenge once (vouching)~~ → **discarded 2026-11-18.**
User verdict: "Using VNC is far too dirty an approach." That is correct. Sitting a person in front of a screen
to click captchas is technical debt, not a solution, and if operation is tied to human
availability an unattended agent is not possible. `vouch_browser.py` deleted, all x11vnc/Xvfb paths in
`identity.sh` removed.

**Instead, this is what we do:**

A tier-1 identity **accumulates on its own** (google NID/_GRECAPTCHA attached with no human intervention).
Engines that work with that work; engines that do not are **simply given up.**

    [lane] challenge detected
       → screenshot state/challenge-<engine>-<ts>.png (for after-the-fact review)
       → state suspended, backoff 1h → 2h → 4h … max 24h (doubles per challenge)
       → automatically resumes when the backoff expires. No human is called.
    [human] nothing to do. Notifications are informational only, not a request for action.

An engine that throws challenges is simply "an engine that does not want us". Keep knocking and only the IP reputation
burns. **The right answer is to move that demand to layer 0/layer 1 instead of breaking through** — see
`~/work/websearch-index/`. For brave we also confirmed there is no legitimate path other than the paid API (the free 2,000/month
tier was discontinued 2026-02; currently card registration + $5/month credit).

### Isolation rules

- `identities/` is 0700 and is never committed to git (cookies = credentials)
- The lane binds only to `127.0.0.1` + `172.18.0.1` (docker bridge). No exposure to LAN or public network
- Header token required (`X-Lane-Token`)
- Identities can be separated per engine group (if brave gets burned, the google identity is unaffected)

## Resource constraints

On this box vLLM holds 117GB and ~4GB is available. **Unified memory, so OOM = the whole box dies.**
Therefore:
- The browser **is not resident.** After 180 seconds idle the process exits; it is relaunched on the next query.
  The profile stays on disk, so the identity is preserved.
- It is launched under a hard memory cap (`systemd-run -p MemoryMax=...`). If the cap is exceeded only the lane dies
  and the box survives.
- 1 concurrent navigation (one browser, so naturally serial). That also serves as rate protection.

## Honesty guard

Lesson from bing — an engine that silently returns garbage is the worst. The lane fires a random query in `/selftest`
and checks that **0 results come back**. If not 0, that engine is not trusted.

## Pitfalls hit during implementation (all measured, for preventing recurrence)

1. **Challenge detection must not be done by body string matching.**
   At first we searched the whole body for `/sorry/index` `recaptcha/api2` `enablejs`. As a result
   **a normal Google page showing 9 organic results just fine** was misjudged as a challenge, and the engine was
   permanently blocked and a notification was even sent. Normal pages also contain those strings.
   → **Extract first; if results come out, that is the end of it.** Only consider a challenge when there are 0 results.
   The markers were also narrowed to final URL / title / first 1,500 characters of the body. The output is more accurate than string matching.

2. **A Chromium profile is a single-writer resource.** Launching a second browser with the same profile
   dies immediately with "Opening in existing browser session". The lane must be the sole
   owner (because of this constraint the vouching browser had to take the lane down and bring it back up, and that complexity
   is one of the reasons vouching was discarded).

3. **Two things must be turned off for the SearXNG → lane connection.** Both show up only as "unexpected crash",
   so the cause is invisible:
   - `enable_http: true` — upstream default is HTTPS only (`InvalidSchema`)
   - `enable_http2: false` — SearXNG connects with HTTP/2 prior-knowledge even over plaintext, but
     the lane is Python http.server and cannot do H2. Measured: `--http1.1` 200,
     `--http2-prior-knowledge` 000 / exit=16.
   Both go in the engine's `network:` block (putting HTTP_VERSION via `curl_options`
   did not work — the http_version session argument wins).

4. **The NPM cache also caches failure responses for 30 minutes.** Always clear it while debugging:
   `docker exec nginx-proxy-manager sh -c 'rm -rf /data/cache/searxng/*'`
   If you do not, you keep seeing the old failure even after fixing it (we wasted a long time on this).

5. **Browser engines are not put in the default search.** SearXNG waits for all engines, so
   putting a 5–8 second engine in the default set **slows every search by that much**. Keep them `disabled: true`
   and use them only via bang: `!gbr` `!bbr` `!qbr`. Default search stays at 1.4 seconds.

6. **Google has fully obfuscated result URLs** (2026-11-18). Both `href` and `ping` are
   the opaque token `/goto?url=CAEScgHEIkXX...`, and the real URL is nowhere in the DOM.
   For `cite` only the host is trustworthy (the path is truncated with `…`).
   → Title, snippet and host are accurate. Links are resolved at fetch time (only the 1–3 actually read
   incur the cost). However, **URL deduplication against other engines' results does not work.**

## Measured status (2026-11-18)

| Engine | Result | Notes |
|---|---|---|
| google | ✅ 10 organic results, 0 results for random query (honest) | URL wrapping |
| brave | ⛔ HTTP 429 + "verify you are not a bot" | retry after automatic backoff. If it still fails, drop it |
| qwant | ⛔ HTTP 200 / title "Service unavailable" | not a challenge, just a refusal |

Identity `default` automatically accumulated 9 cookies (google NID/AEC/SNID/_GRECAPTCHA, etc.).
brave cookies: 0. **So brave is dropped** — we do not call in a human to get it through.

## Startup — systemd owns it (2026-11-21)

### Prerequisites

| Item | Our value |
|---|---|
| `xvfb-run` | `/usr/bin/xvfb-run` (`apt install xvfb`). **Runs headful + xvfb** — headless-shell has `HeadlessChrome` baked into the UA |
| Chromium | the one playwright downloaded into `~/.cache/ms-playwright/`. It is **outside the repo** → `.venv/bin/playwright install chromium` |
| venv | `.venv` (playwright) |
| Token | `state/token` (if missing, `serve.sh` creates it with `openssl rand -hex 32`) |

```bash
# Copies of the units live in this directory's systemd/ — keep them outside the repo and they get lost again.
# ⚠️ Paths inside the units are absolute paths for our environment. On another box, substitute $PWD when installing.
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

`MemoryMax=1500M` / `MemorySwapMax=0` are in the unit — because of "Resource constraints" above,
**they must not be removed.** `serve.sh` is kept for manual launches, but it is not used while systemd
holds the lane (with two owners it breaks on the 8099 bind).

### Reference values — measured in our environment (2026-11-21)

The baseline for the rebuilding side to check whether it came up correctly.

| Item | Our value |
|---|---|
| Port / binding | `8099`, only `127.0.0.1` + `172.18.0.1` (docker bridge). **Not reachable** from LAN or public IP (verified from another machine) |
| Idle footprint | **124MB** (Python server only. `browser_up: false`) |
| Memory cap | `MemoryMax=1500M` / swap 0 |
| Idle exit | `idle_exit_s: 180` |
| Identity | `identities/default` — 9 google cookies accumulated automatically, 0 brave |
| Auto-recovery | `kill -9` → back in about 10 seconds (measured, `NRestarts=1`) |

```bash
# Reachable from the SearXNG container? — layer-3 engines are hooked in at this address
docker exec searxng sh -c 'wget -qO- http://172.18.0.1:8099/health'
# {"ok": true, "identity": "default", "browser_up": false, "idle_exit_s": 180.0, ...}
```

⛔ In the SearXNG template the three engines `google browser`, `brave browser`, `qwant browser` are
`disabled: true`. They are used **bang-only** (`!gbr`) — put them in the default set and every query takes the 8-second
path, and the higher query volume only brings blocking sooner. The goal of this lane is to be used as little as possible.

---

## Where this lane sits (redefined 2026-11-18)

This is **layer 3**, **the layer whose goal is to be used as little as possible**. Layer 0 (local index) and layer 1 (machine-facing public endpoints)
are handled by [`../websearch-index/`](../websearch-index/README.en.md) in the same repo. The layer-3 usage rate is the KPI; going down = success.
