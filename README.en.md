[한국어](README.md) | **English**

# GLM-5.3-Flash on 2× DGX Spark — dspark-serve

This setup ties **two** DGX Spark (GB10) machines together over a 200GbE rail and serves **GLM-5.3-Flash** (4bpw EXL3) with vLLM.
512K context, 4 concurrent requests, OpenAI-compatible API. It serves the original model and **switches uncensoring on/off in real time, per request**.

The core of this repository is the [`glm53-exl3-lane/`](glm53-exl3-lane/) directory. The other top-level files are other lanes
that ran on the same boxes (DeepSeek-V4, Qwen, etc.) and helper scripts; they are not part of the installation in this document.

| Document | Contents |
|---|---|
| **This README** | Structure, how our setup differs, **fresh installation**, **upgrading from the old version (09-05 kit)** |
| [`glm53-exl3-lane/README.md`](glm53-exl3-lane/README.en.md) | Operating commands, configuration knobs, measurement ledger, change procedure |
| [`glm53-exl3-lane/HANDOFF.md`](glm53-exl3-lane/HANDOFF.en.md) | How each patch works and its pitfalls (for developers) |
| [`glm53-exl3-lane/AB-SWITCH.md`](glm53-exl3-lane/AB-SWITCH.en.md) | Details of the real-time uncensoring switch |
| [`glm53-exl3-lane/ORIGINAL-VS-ABLITERATED.md`](glm53-exl3-lane/ORIGINAL-VS-ABLITERATED.en.md) | Why we serve the original instead of the abliterated model |
| [`glm53-exl3-lane/UTF8-GUARD.md`](glm53-exl3-lane/UTF8-GUARD.en.md) | Guard against broken Korean text (U+FFFD) |
| [`glm53-exl3-lane/HISTORY.md`](glm53-exl3-lane/HISTORY.en.md) | Chronological work history |

---

## 1. Structure

```
                         OpenAI-compatible API  http://<head LAN IP>:8000/v1   model name glm-5.3-flash
                                   │
  ┌────────── head (rank 0) ───────┴──┐     200GbE RoCE rail      ┌────────── worker (rank 1) ────────┐
  │ docker: vllm_glm53                │◄──── NCCL (TP=2) ────────►│ docker: vllm_glm53 (--headless)   │
  │  vLLM API server + engine +       │   mlx5_0 / enp1s0f0np0    │  other half of the tensors        │
  │  half of the tensors              │                           │  memory watchdog (memwatch)       │
  │  memory watchdog (memwatch)       │                           │                                   │
  └───────────────────────────────────┘                           └───────────────────────────────────┘
        ↑ same image, same lane directory, same model path (identical on both nodes — guaranteed by the sync script)
```

- **Image**: [`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
  (digest `sha256:44b2dbaf…`). Entrpi's vLLM fork (`f223ff9f2`) + b12x CuTeDSL kernels + EXL3 fused MoE.
- **Model**: [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — EXL3,
  only the routed experts are 4bpw, the rest is BF16. One local copy per node (about 164GiB).
- **Our code does not rebuild the image.** It replaces a few files inside the image by read-only bind-mounting
  files from the lane's `overlay/`. Next to each overlay there is a `.orig` (the image original) and a `.diff`, so
  exactly what was changed is visible. Remove the mounts and you are back to stock image behavior.
- **One script does everything**: on the head, `glm53-exl3-lane/vllm-glm53-exl3.sh RESTARTALL`.
  It does stop both nodes → memory watchdog → start worker → start head → wait for READY, in one go.
- Decoding: MTP-4 speculative decoding (the model's built-in MTP head). KV cache FP8 (`fp8_ds_mla`), prefix caching enabled.
- **The externally exposed surface is `edge/`**: Nginx Proxy Manager (443/80) + searxng. `.66` sits in the router's DMZ, so
  binding to a LAN IP is not security — not publishing a port is the only thing that blocks access.
  Port policy, rebuild order, and reaching the NPM admin UI through an SSH tunnel: [`edge/README.md`](edge/README.en.md).
- **Search layer 0 is `websearch-index/`**: a local index (SQLite FTS5, 6.7~7.4ms) placed in front of SearXNG.
  Merely relaying external engines gets you blocked more the more you use it, so everything fetched is stored permanently and topics
  asked about twice are promoted to subscriptions, so that **the unblocked area grows the more you use it**. Design, reference values, startup:
  [`websearch-index/README.md`](websearch-index/README.en.md).

## 2. How our setup differs from the stock kit

Everything was adopted based on measurements; the rationale and numbers are in the [lane README](glm53-exl3-lane/README.en.md) and [HISTORY](glm53-exl3-lane/HISTORY.en.md).

| Category | What | Effect |
|---|---|---|
| Speed | Dense layers, lm_head and MTP projections are quantized at load time to **FP8 weights + BF16 activations (Marlin W8A16)** | Decode +40% (step 117→~80ms) |
| Speed | Prefill (M>1024) dequantizes FP8 to BF16 and uses cuBLAS — **dual path** | Recovers the prefill lost to FP8 (12k ≈ 1,350 t/s) |
| Speed | **32K-vocabulary lm_head** for the MTP draft only (verification uses the full vocabulary) | Step −3ms, lossless output |
| Speed | Removed duplicate computation of the MoE router GEMM (upstream bug) | Decode +4.4% |
| Correctness | **'!' collapse fix** — §4 below | 0 collapses |
| Correctness | **kpool cache corruption fix** (backport of upstream vllm#57477) — bug where a prefill overwrote another request's indexer cache | Cache-hit results no longer swayed by other requests |
| Correctness | **temperature 0 determinism** — fixed atomic-ordering issues in 3 upstream kernels | Same input → same output |
| Correctness | Added the missing routed-expert SwiGLU clamp (`swiglu_limit=10`) (missing upstream) | Matches reference behavior |
| Correctness | **UTF-8 guard** — blocks in the sampler the switch to a Latin token in the middle of a Korean syllable that produced U+FFFD | 7/8 → 0/8, no speed loss |
| Feature | **Real-time uncensoring switch `/ab`** — §3 below. Per-request only (no global switch, client not trusted) | Per-request switching without restart, 0 effect on other people's requests (co-batching and the prefix cache alike) |
| Feature | **NVMe KV cold-tier offload** — prefix KV evicted from GPU is kept on 200GB of NVMe per node (O_DIRECT, slot-header integrity check, recompute on failure) | Effectively unlimited prefix cache, **cache survives restarts**. Details: [`KV_OFFLOAD.md`](glm53-exl3-lane/KV_OFFLOAD.en.md) |
| Correctness | **Concurrent batch invariance (temp 0)** — same output regardless of batch composition, load, history or restarts (`GLM53_BATCH_INVARIANT=1`) | Same t/s, prefill −1.5%. Details: [`BATCH_INVARIANCE.md`](glm53-exl3-lane/BATCH_INVARIANCE.en.md) |
| Operations | **priority scheduling** — codemason sub-agent requests get priority 1 | The main conversation goes first in the queue |
| Operations | Vision gate `VISION` (1 = accept images, live default. 0 excludes the tower via `--language-model-only` — avoids GB10 warm-up OOM), backport of the zai-org chat template fix | Stable startup |
| Operations | Two-node sync check (`sync-lane-67.sh`), memory watchdog, separation of path and site values | Prevents two-node mismatch and OOM incidents |

**Rejected** (cause hallucination or bring no gain): W8A8 that also quantizes activations (b12x block FP8), INT4 draft head,
the DFlash2 drafter (slower than MTP-4 on prose). Reasons in lane README §5.

## 3. Real-time uncensoring switch (`/ab`) — our own design

Serving an abliteration (censorship-removed) model wholesale does not only remove refusals; **the ability to say "that does not exist"
disappears along with it.** On a 60-item fake-target trap, the original invented 4 and the abliterated model 18~22. So we **serve the original and
turn uncensoring on only at the moment it is needed.**

How it works: abliteration only touched the 45 `self_attn.o_proj` tensors, and these tensors are BF16 even in the 4bpw checkpoint.
So the difference between original and abliterated is extracted as a **rank-24 low-rank adapter (29MB)** and added to the output while serving.

```
o_proj(x) + scale · (x·Aᵀ)·Bᵀ        scale = 0 → original,  scale = 1 → abliterated (relative error 0.17%)
```

`scale` is a GPU tensor, so CUDA graphs are not recaptured and nothing is re-prefilled. It is applied to the TP workers on both nodes at the same time.

```bash
curl -H "Authorization: Bearer $KEY" http://<head>:8000/ab                     # status (read-only)
# Turning it on is only for your own request: in the request body, "vllm_xargs":{"ab_scale":1}
# POST /ab returns 410 — the global switch changed the model for everyone else too, so it was removed (2026-11-27)
```

Measured: on 16 neutral-target questions, refusals 5 → 0; speed cost 1~2ms per step; temperature 0 determinism preserved in both modes.
The codemason client, with `/ab auto`, puts `ab_scale=1` only on the first request of a user turn (it does not change server state). Details in [AB-SWITCH.md](glm53-exl3-lane/AB-SWITCH.en.md).

## 4. The last bug: '!' collapse (fixed 2026-10-02)

**Symptom**: in long agent sessions the reasoning channel emits hundreds of `!` and no answer (`finish_reason=repetition`).
GLM-5.3-Flash users reported it on both vLLM and SGLang, but upstream had no confirmed cause either.

**Reproduction**: the shape of a real codemason request (system prompt + 18 tools + a one-line Korean task) collapsed 100% of the time on the first turn.
Bisecting it showed it blew up **only when the prompt length divided by 512 leaves remainder 5**. Content and cache were irrelevant.

**Cause**: prefill chunks are cut at 512 boundaries, so the last piece is `length mod 512` tokens. When that is exactly
5 (= 1 + 4 MTP speculative tokens), the vLLM V2 runner replayed the CUDA graph for decode verification **based only on the batch shape**.
That graph does not build the prompt's KDA recurrent state, so the output scores become NaN, and the sampler picks token 0 (`!`) from NaN.
All subsequent decode steps are NaN as well, so `!` continues forever.

**Fix**: upstream vLLM fixed this on 08-12 in [#51865](https://github.com/vllm-project/vllm/pull/51865) ("uniform decode only
when all requests are decoding"), but the image's fork (branched 08-10) did not have it. We backported it (`overlay/v2runner/`).

**Verification**: `tools/prefill_tail_nan_check.py` — before the fix 7/7 NaN, after the fix 14/14 normal. No speed change.
After the fix, about 250 requests in a long codemason experiment with 0 collapses.

---

## 5. Fresh installation (2× DGX Spark, from scratch)

> All commands assume **the same user name and the same paths** on both nodes (lane, model and cache paths must be identical on both sides).
> Below, `head` is the box that runs the API and `worker` is the other one. Unless stated otherwise, commands are run on the **head**.
> Example addresses are rail `10.0.1.1` (head) / `10.0.1.2` (worker). Replace `10.0.1.2` and `<worker>` in the commands with your own values.
>
> This procedure was re-verified on **2026-11-27** by cloning the public repository (`c02117c`) into a new path and following it as written
> (including NVMe offload, batch invariance, priority and `/ab` zero trust — the same configuration as the current production server):
> - `glm53-serve.env` built from the template + only the ① site values is **byte-identical** to the production env
> - Started with no offload store present → created fresh on both nodes, READY in 195 seconds
> - `STATUS` feature line `NVMe오프로드=on priority=on 배치불변=on MTP동적=on /ab=요청별만` (NVMe offload=on, batch invariance=on, dynamic MTP=on, /ab=per-request only), '!' regression check PASS
> - Decode prose 39.2 / code 44.7 / hash-map 35.0 t/s (step 83ms)
>
> Skipped: §5.1~5.2 (boxes with rail and image already set up), §5.5 model download (already present at the same path —
> adapter byte-identity was verified on 10-05). Previous verification: 2026-10-05 (prose 38.9 / code 41.8 t/s).

### 5.0 Requirements

| Item | Our environment (verified) | Notes |
|---|---|---|
| Hardware | **2×** DGX Spark (GB10, 128GB memory) | Will not start on one (model + KV exceed a single box) |
| Connection | The ConnectX-7 QSFP ports of the two boxes connected with a **direct cable** | First port `enp1s0f0np0` / RoCE device `mlx5_0` |
| OS | DGX OS (Ubuntu 24.04.5), kernel 7.0.0-1019-nvidia | |
| Driver | NVIDIA 580.178.04 | |
| Docker | 29.6.2 + nvidia-container-toolkit 1.20.1 | DGX OS default configuration |
| Disk | **About 500GB** free per node, ext4 (ours is ext4 on NVMe) | Model 164GiB + adapter working files + **NVMe KV offload store 200GB + 100GB headroom**. The launcher measures free space at `KV_OFFLOAD_DIR` (default `~/glm53-kv-offload`) and **refuses to start** if it is insufficient. The store uses O_DIRECT, so tmpfs/overlay will not work. If the disk is small, reduce `KV_OFFLOAD_BYTES` in `glm53-serve.env` |
| Other | **Passwordless ssh** from head→worker, passwordless `sudo` (recommended) | sudo is used to drop the page cache before startup (works without it) |

No swap needed (the loader reads with direct I/O).

### 5.1 Rail network (both nodes)

Give the port connecting the two boxes a static IP and MTU 9000. Example: `/etc/netplan/40-cx7.yaml`

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp1s0f0np0:            # rail port (check the name with ip -br a)
      addresses: ["10.0.1.1/24"]   # worker uses 10.0.1.2/24
      dhcp4: false
      mtu: 9000
```

```bash
sudo netplan apply
ping -c 3 10.0.1.2                    # from the head to the worker rail IP
rdma link                             # check for mlx5_0/1 ... state ACTIVE ... netdev enp1s0f0np0
```

Register the head→worker ssh key (once, on the head):

```bash
ssh-keygen -t ed25519            # only if you have no key
ssh-copy-id <user>@<worker LAN IP>
ssh-copy-id <user>@10.0.1.2      # also to the rail address (the sync script pushes over the rail)
```

### 5.2 Docker image (both nodes)

```bash
docker pull ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1      # about 7GB compressed / 18GB installed
docker image inspect ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1 --format '{{.Id}}'
# → must be sha256:606b5f3c61b4900db05555b72a7b20a9455a8bdd09a5c74e288839c40cc3703a
#   (if you got it with pull, the registry digest also shows as sha256:44b2dbaf…. An image moved with docker load has no digest)
```

### 5.3 Repository (clone on the head only)

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/AmesianX/dspark-serve.git
cd ~/work/dspark-serve/glm53-exl3-lane
```

Do not clone on the worker. The sync script in §5.6 pushes the lane directory to the same path. On the worker, only create the parent directory.

```bash
ssh <worker> 'mkdir -p ~/work/dspark-serve'
```

### 5.4 Site values

The config file **contains a key, so it is not tracked in git** (this repository is public). Copy the template and use that.

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
cp glm53-serve.env.example glm53-serve.env
chmod 600 glm53-serve.env
```

Edit the **① site values** block at the top of the copied `glm53-serve.env` for your environment. Do not touch ② and ③ below it.

| Variable | Value | How to check |
|---|---|---|
| `HEAD_RAIL_IP` / `WORKER_RAIL_IP` | Rail IPs (e.g. 10.0.1.1 / 10.0.1.2) | `ip -br a` |
| `WORKER_SSH` | ssh target for connecting from head to worker (e.g. `me@192.168.1.67`) | `ssh <value> true` must finish without a password |
| `NCCL_IF` | Rail interface name | `ip -br a` |
| `NCCL_HCA` | RoCE device name of the rail | In `rdma link`, the name next to the rail netdev |
| `NCCL_SUBNET` | Rail subnet (e.g. 10.0.1.0/24) | |
| `VLLM_API_KEY` | **An arbitrary new string you choose** | Clients send it as `Authorization: Bearer <key>` |

The launcher finds the RoCE GID index automatically from sysfs. Do not set it yourself.

`glm53-serve.env` itself is in `.gitignore`, so it is not committed — rsync copies it to the worker
as-is (`sync-lane-67.sh` only checks `git status --porcelain`, and ignored files do not
show up there).

If you changed **other** files in the lane, commit them (when pushing to the worker, the sync script refuses uncommitted
changes — experiment history must be in git so it can be reverted). A new box has no git user
information, so commits fail; set it once for this repository (any values will do).

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
git config user.name "$USER"; git config user.email "$USER@$(hostname)"   # once, in this repository only
git commit -am "site config"
```

### 5.5 Model and adapter

**(1) Original model — on both nodes at the same path** (164GiB)
Download [huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) to
`~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw`. If the model is not at this path, startup stops and tells you where to get it.

Do not install Python packages on the host (DGX OS's system python blocks `pip install`). The serving image
contains `hf` and torch, so downloads and builds are done with the image. The two lines below are shorthands.

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
# copying to the worker over the rail is fastest (downloading on the worker with the same command also works)
rsync -a --info=progress2 ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw 10.0.1.2:Models/
```

Use `--local-dir` to get **real files**. If you use the HF cache's symlink layout, the links break inside the container and loading fails.

**(2) Uncensoring adapter** (built on the head and copied to the worker, 29MB)

From the abliterated model ([huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated))
only the 46 o_proj tensors (3.89GB) are needed. Only those are fetched with HTTP Range requests, and the adapter is built from the difference to the original (the full 175GB is not downloaded).

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
python3 tools/fetch_oproj.py --source abliterated \
   --out ~/Models/glm53-oproj/o_proj_abliterated.safetensors          # must end with md5 ... OK

# build the adapter (needs torch → inside the image). About 10 minutes. Do it while the server is not running (uses several GB of memory).
run python3 $IMG /tools/build_ab_lora.py --base /m/brandonmusic--GLM-5.3-Flash-tr3-4bpw \
   --abl /m/glm53-oproj/o_proj_abliterated.safetensors --out /m/glm53-ab-lora
rsync -a ~/Models/glm53-ab-lora 10.0.1.2:Models/
```

### 5.6 Worker sync and startup

```bash
cd ~/work/dspark-serve/glm53-exl3-lane
./sync-lane-67.sh            # push the lane directory to the same path on the worker and compare md5 per file
./vllm-glm53-exl3.sh RESTARTALL
```

About 3~4 minutes to READY. Success is when `✅ API READY ... 원본 + /ab off` is printed at the end (`원본` = "original": the original model is served with `/ab` off).
On the first startup the first request is slow because the JIT compile cache (`~/glm53-vllm-cache`) is being built.

> `sync-lane-67.sh` refuses if there are uncommitted changes (if you committed in §5.4 it just works).
> Every time you change the lane: commit → `sync-lane-67.sh` → `RESTARTALL`. RESTARTALL automatically checks that both nodes are identical before startup.

### 5.7 Verifying it works

```bash
./vllm-glm53-exl3.sh STATUS                       # containers on both nodes, health=200, model/switch state, features, available memory
#   the "기능:" (features) line must be exactly this to match our current server (if any is OFF, glm53-serve.env differs from the template):
#   기능: NVMe오프로드=on(…) priority=on 배치불변=on MTP동적=on /ab=요청별만
python3 tools/prefill_tail_nan_check.py           # '!' collapse regression check → PASS
python3 tools/kmeas.py                            # decode speed (prose/code/English)
curl -s -H "Authorization: Bearer <key>" http://localhost:8000/ab     # {"scale":0.0, ... "mode":"original", "global_write":false}
ls -la ~/glm53-kv-offload/; ssh <worker> 'ls -la ~/glm53-kv-offload/'   # per-rank offload files appear on both nodes
```

The `STATUS` output is in Korean. The expected line reads, in English: `features: NVMe offload=on(…) priority=on batch invariance=on dynamic MTP=on /ab=per-request only`.

A simple chat:

```bash
curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"안녕? 한 문장으로 자기소개 해줘."}],"max_tokens":400}'
```

(The prompt is Korean on purpose and means "Hi? Introduce yourself in one sentence." — it also exercises the UTF-8 guard. Any prompt works.)

### 5.8 Encrypted gateway `:8443` (only for remote use)

`encrypt_gateway.py` sits in front of vLLM and exchanges **request/response bodies** with AES-256-GCM.
Even over plain HTTP the body is ciphertext, and the key never travels over the network
(possession itself is the credential — successful decryption proves the key is held, and then the gateway **attaches** the vLLM
bearer token **on the client's behalf**). So the client does not need to hold the vLLM key,
and does not leak a bearer token over plain HTTP.

```bash
cd ~/work/dspark-serve
cp gateway.env.example gateway.env && chmod 600 gateway.env
python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"  # → LLAMACPP_AES_KEY
python3 -c "import secrets;print(secrets.token_hex(32))"                        # → GATEWAY_CACHE_SALT_SECRET (server-only)
./setup-systemd.sh install        # register with systemd + start (refuses to start without the salt secret)
./run-gateway.sh status
```

Put the same key in the client (`~/.codemason/config.json`):

| Key | Value |
|---|---|
| `model_aes_key` | **Identical** to `LLAMACPP_AES_KEY` in `gateway.env` |
| `model_base_url` | `http://<public address>:8443` (or a TLS vhost in front of it) |

(The old names `llamacpp_aes_key` / `llamacpp_base_url` are still read, but the current names are the ones above.)

**Per-session cache isolation (2026-11-27).** The gateway **forces** a `cache_salt` into every decrypted generation
request (a client-sent value is overwritten). codemason sends a random per-install session id
(`~/.codemason/cache_session`) **inside the encrypted body**; the gateway derives the salt with the server-only
`GATEWAY_CACHE_SALT_SECRET` and removes the id (without one, the client address is used). So nobody can reuse another
user's prefix or time it. When codemason opts in with `vllm_xargs.cache_share_system_prefix=1`, everything **up to the
last system block before the first user turn** (tool schema, static instructions) is shared in one cache entry — the
lane decides that boundary from the tokens (`overlay/v1core/kv_cache_utils.py`). Measured (cm2 → gateway → vLLM, 18.7K
prompt): a new session's first request hits 9,216 shared tokens in 11.1s (cold 19.0s), the same session's next request
hits 17,920, and another session's private part hits 0 even with identical tokens. Tests:
`glm53-exl3-lane/tools/test_gateway_session_salt.py`, `test_cache_session_salt.py`.
- ⛔ Do not derive the salt secret from the AES key. The AES key is something clients hold, and any value put in a
  browser/WASM build is public. Whoever can compute salts can aim at someone else's cache namespace.

- `gateway.env` is in `.gitignore`. **Do not commit it** — until 2026-11-13
  this key was committed in plaintext in `run-gateway.sh`, and a key pushed to a public repo
  defeats the entire gateway.
- The gateway does not serve `/docs` or `/openapi.json` (`FastAPI(docs_url=None, ...)`).
  Previously those paths returned 200 without authentication.
- ⛔ Do not inject `Authorization` in `_passthrough`. If you inject it there, unencrypted
  requests bypass vLLM authentication outright.

---

### 5.9 Search (almost certainly what you will use most)

Once you stand up an LLM server, **the feature that actually runs the most is search**. Things the model does not know, recent
information and link checks all go through search. So if you have come this far, setting up search is the most valuable of
the remaining tasks.

```bash
# 1) searxng + NPM (key gate, port policy, ACME) — details in edge/README.md
./searxng-serve.sh start
./searxng-serve.sh test        # all three lines must be ✅. Stop if it is not "키 없이 403" (403 without key)

# 2) put the address and key into the client (each machine has its own)
#    searxng_url and searxng_api_key in ~/.codemason/config.json

# 3) layer-0 local index — the part that makes you get blocked less the more you use it
cd websearch-index && ./ws.sh collect       # first collection
cp systemd/websearch-*.service ~/.config/systemd/user/
sudo loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user enable --now websearch-index websearch-collect
./ws.sh search vllm                        # about 20 hits / 7ms means it is set up correctly
```

`searxng-serve.sh test` prints in Korean; the line to look for is `✅ 키 없이 → 403 (거부 기본값 정상)` ("without key → 403 (deny-by-default OK)"). If it prints `🚨 키 없이 → ...` instead, the gate is missing.

**Search fails silently.** If the key mismatches, `json` is missing from `formats`, or the certificate
expires, it **falls back to ddgs as the single source without any error**, and only result quality drops. It actually once fell back
for 25 days because of an expired certificate. So keep running `test` periodically even after setup.

- External surface, port policy, vhost, cache: [`edge/README.md`](edge/README.en.md)
- Layer-0 design, reference values, KPIs: [`websearch-index/README.md`](websearch-index/README.en.md)
- Layer 3 (identity browser, **promoted automatically by the server** — clients cannot request it): [`edge/README.md`](edge/README.en.md) · [`websearch-browser/DESIGN.md`](websearch-browser/DESIGN.en.md)

---

## 6. Upgrading from the old version (09-05 handoff kit)

If you installed and are using the `glm53-flash-2xspark-kit` delivered on 2026-09-05, back then the **abliterated model was the default**.
So much has changed since (most of §2 above, the original + real-time switch design, path cleanup) that you should **not follow the changes incrementally; replace the lane
wholesale.** The image (v2.3-tier1) and the abliterated model you already downloaded are reused as-is.

**What changes**: the entire lane directory · served model (abliterated → original) · adapter added · `~/glm53-hotfix-b12x` symlink removed.
**What stays**: Docker image · rail network · ssh · model download (the only additional download is one 3.89GB file).

```bash
# ── on the head ──────────────────────────────────────────────────────────
~/work/dspark-serve/glm53-exl3-lane/vllm-glm53-exl3.sh STOPALL     # (1) stop the server with the old lane

# (2) remove the old b12x symlink — both nodes. The new lane mounts b12x directly, so if it remains, the duplicate mount makes startup fail
rm -f ~/glm53-hotfix-b12x; ssh <worker> 'rm -f ~/glm53-hotfix-b12x'

# (3) keep the old install directory whole (to copy the site values over) — both nodes
mv ~/work/dspark-serve ~/work/dspark-serve.kit0905
ssh <worker> 'mv ~/work/dspark-serve ~/work/dspark-serve.kit0905 && mkdir -p ~/work/dspark-serve'

# (4) get the new repository (same as §5.3)
cd ~/work && git clone https://github.com/AmesianX/dspark-serve.git
cd ~/work/dspark-serve/glm53-exl3-lane
IMG=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1
run() { docker run --rm --user $(id -u):$(id -g) -e HF_HOME=/m/.hf -v $HOME/Models:/m -v $PWD/tools:/tools --entrypoint "$@"; }
# if you downloaded per the 09-05 kit instructions (HF_HOME=~/Models/hfcache), the abliterated model is here. If it is elsewhere, change only this line (path relative to ~/Models).
ABL=hfcache/hub/models--lovesenko--GLM-5.3-Flash-tr3-4bpw-Abliterated/snapshots/c8f58e6aa9117c73607d692978b22f091d80450c

# (5) carry over site values: write the values from the old ~/work/dspark-serve.kit0905/glm53-exl3-lane/ glm53-serve.env and
#     the top of vllm-glm53-exl3.sh (WORKER, KEY) into the ① block of the new glm53-serve.env (§5.4 table).
#     old WORKER → WORKER_SSH, old KEY/VLLM_API_KEY → VLLM_API_KEY. To keep the key your clients were using, use the same value.
git config user.name "$USER"; git config user.email "$USER@$(hostname)"   # for commits, once only

# (6) build the original from the abliterated model you already have — download only the original o_proj 3.89GB, hardlink the rest (about 72GB extra disk)
python3 tools/fetch_oproj.py --source original --out ~/Models/glm53-oproj/o_proj_original.safetensors   # md5 ... OK
python3 tools/build_original_from_abliterated.py --src ~/Models/$ABL \
   --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw
#     same on the worker: send the o_proj file and build there (check the worker's abliterated path)
ssh <worker> 'mkdir -p ~/Models/glm53-oproj'
rsync -a ~/Models/glm53-oproj/o_proj_original.safetensors 10.0.1.2:Models/glm53-oproj/
rsync -a tools/build_original_from_abliterated.py 10.0.1.2:/tmp/
ssh <worker> "python3 /tmp/build_original_from_abliterated.py --src ~/Models/$ABL \
   --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw"

# (7) build the adapter — both original and abliterated are present, so use the abliterated snapshot directly (inside the image, about 10 minutes)
run python3 $IMG /tools/build_ab_lora.py --base /m/brandonmusic--GLM-5.3-Flash-tr3-4bpw \
   --abl /m/$ABL --out /m/glm53-ab-lora
rsync -a ~/Models/glm53-ab-lora 10.0.1.2:Models/

# (8) commit (only if something changed) → sync → start → verify (§5.6, §5.7)
git diff --quiet || git commit -qam "site config"
./sync-lane-67.sh && ./vllm-glm53-exl3.sh RESTARTALL       # ends with "✅ API READY ... 원본 + /ab off" (원본 = original)
python3 tools/prefill_tail_nan_check.py                     # PASS
curl -s -H "Authorization: Bearer <key>" http://localhost:8000/ab   # "mode":"original"
```

Why `--dst` uses the original's name (`brandonmusic--…`): the result is **identical file by file** to the original repository (shard sha256, config and tokenizer
were confirmed to match). This procedure was verified on 2026-10-05 by actually installing the 09-05 kit, starting it with the abliterated model, and following these steps as written
(original shards and adapter byte-identical, '!' regression check PASS, `/ab` works in both directions). Building the original on the worker takes a little over 1 minute, the adapter about 10 minutes. Once the upgrade is done and running well, the old install (`~/work/dspark-serve.kit0905`) can be deleted. The abliterated snapshot must not be deleted —
76 shards of the original share hardlinks with those files (deleting it leaves the original intact, but it is needed to rebuild the adapter).

---

## 7. Common mistakes

- **Two nodes always means `RESTARTALL`.** If only the worker is alive, it becomes a zombie that waits 600 seconds for the head and then dies.
- **No heavy work on the head while the server is up.** On GB10 the CPU and GPU share memory. If available memory drops below 3GB,
  the memory watchdog takes the containers down to protect the box. Do build/conversion work after `STOPALL`.
- **No symlinks for the model or adapter.** Links pointing to host paths break inside the container. Use real files or hardlinks.
- **If you changed the lane: commit → `sync-lane-67.sh` → `RESTARTALL`.** The worker is a file copy, not git.
- Environment variables are not passed over ssh. Put knobs in `glm53-serve.env` (a new knob must also be added to the pass-through list in `launch.sh`).
- If startup stops with `this host does not own <IP>`, the rail IP setting is wrong; if with `no RoCE v2 GID matching`, `NCCL_HCA` is wrong.

## 8. Sources and acknowledgements

- Serving kit and image: [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark), vLLM fork [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark)
- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) → EXL3 4bpw [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) → abliteration [lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated)
- Kernels: b12x (sparkinfer-glmrt), [exllamav3](https://github.com/turboderp-org/exllamav3), [vLLM](https://github.com/vllm-project/vllm)
