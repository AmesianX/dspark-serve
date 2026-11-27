[한국어](KV_OFFLOAD.md) | **English**

# KV NVMe offload — design

Status (2026-11-27): **implemented and verified, merged into main, on by default on the production server (`KV_OFFLOAD=1`).** Stress E1~E6, light S1~S5, and stage 2 (cache provenance A·B·C) all pass —
the canonical procedures and results are in `KV_OFFLOAD_TESTPLAN.md`. The experiment branch `exp/glm53-kv-offload` was merged into main on 11-27 and deleted.

## Goal
Turn the multi-minute re-prefills caused by the main conversation and subagents pushing each other out of the GPU KV pool
into restores of a few seconds. The GPU pool is responsible only for "requests currently generating"; the KV of finished conversations
goes down to each node's NVMe and comes back up when the conversation returns. It survives restarts.

## Memory allocation (per node, splitting the former 12e9 slot)
| Purpose | Size |
|---|---|
| GPU KV pool (active) | 10e9 (≈1.6M raw tokens) |
| Extra headroom above the memwatch 3GB guard | 2e9 |
| Offload RAM | bounce buffers only, 8 IO threads × 1 slot each (≈28MiB) ≈ 230MiB |
| NVMe store | one 200GB preallocated file per rank (≈30 million tokens) |

## Structure
- New spec `NvmeOffloadingSpec` (our own implementation. MiaAI #232 is AGPL, so none of its code is used; only its design was consulted).
- No CPU tier. The scheduler-side manager reuses the existing `CPUOffloadingManager` (LRU) and
  only changes the interpretation to "block id = NVMe slot number". The scheduler is unmodified.
- Worker: slot = fixed size (sum of the group's maximum pages + 4KiB header, 4KiB aligned), file offset = slot × size.
  GPU page → pinned bounce → O_DIRECT pwrite / O_DIRECT pread → bounce → GPU.
  Because it bypasses the page cache, the dirty-page hard-lock risk (MiaAI #57 ⑥) structurally does not exist.
- No data movement between nodes. The head scheduler picks the slot and broadcasts the same instruction to all ranks;
  each rank writes its own share to its own NVMe. Completion = ack from all world_size ranks.

## Integrity (what we fixed that MiaAI #232 left unresolved)
1. Slot header: magic · fingerprint · key · group · length · crc32 · write time. Every rank verifies all of it on read.
   If rank1's data is missing or stale, that rank reports the failure → the block is recomputed (re-prefill instead of contamination).
2. Failures are not hidden: the upstream connector does `assert transfer_result.success` (no failure support).
   Failed jobs are propagated up to the scheduler: a store failure becomes `complete_store(success=False)`, a load failure becomes
   key invalidation + `kv_load_failure_policy=recompute`.
3. Fingerprint = model path · checkpoint size · KV dtype · TP · block size · group layout · this file's version.
   If it differs, the whole store is treated as empty.
4. The kpool tail group (1-block circular scratch, `participates_in_prefix_caching=False`) is excluded from offload.

## Eviction policy (⛔ corrected 2026-11-27: the upstream LRU description was stale — the actual implementation is SegmentLRUPolicy)
1. Unit: chunk (4608 tokens) × group. The slot count is fixed by the preallocated file size — the disk does not grow at runtime.
2. Order: `SegmentLRUPolicy` (glm53_nvme.py). Recency (tick) increases by 1 every scheduler step; depth is the chunk index relative to the anchor group (0).
   Eviction = (tick ascending, depth descending) → the least recently used conversation loses entries **from its tail first**, and the head (shared system prompt, parent's leading part) stays.
   Why upstream LRU was dropped (E2): sliding groups refresh only the last window, so KDA state boundaries were missed, and refreshing on completion made the tail
   newer than the head, so a reused conversation got 0 restored.
3. Reuse rule: touching the anchor key also refreshes the other group keys (KDA state etc.) with the same block hash. Completion (mark_evictable) does not count as use.
4. Protection: slots being loaded or stored (ref_cnt>0) are never evicted. Slots that fail verification are immediately invalidated and returned (`invalidate`; deferred if in progress).
5. Restart: rank0 scans the slot headers and restores the index only for those whose fingerprint matches (assigns negative ticks in write_seq order → reconstructs LRU order).
   rank1 verifies headers at load time. If the fingerprint differs, all slots are empty. No TTL.
6. ⚠️ For the same key, the old content keeps being used (no re-store). If you change a kernel or numeric path, change `KV_OFFLOAD_HASH_SEED` (→ VLLM_PREFIX_HASH_SEED)
   to switch to a new key space (BATCH_INVARIANCE.md §2b).

## Concurrency (proposal — to be confirmed by measurement)
- Number of connections (sessions) ≠ number of concurrent generations. NVMe absorbs the sessions. Concurrent generations are set by the per-user decode floor.
- Floor: target 25 t/s per user, 20 t/s as the last line.
- Estimate (before measurement, assuming the experts of different streams barely overlap): n=1 38.9 · n=2 ≈29 · n=3 ≈23 · n=4 ≈20.
  → to keep 25, 2 concurrent generations; for 20, 3~4. The current max-num-seqs 4 sits at the floor boundary when full.
- The rest wait in the queue. Main conversation first (vLLM `--scheduling-policy priority` + the client passes priority).

## Concurrent generation results (measured 2026-11-26, 4 code prompts, temp0, 300tok, pure decode average per user)
| Concurrent | fixed MTP4 | MTP_PER_BATCH [[1,1,4],[2,2,3],[3,4,2]] |
|---|---|---|
| 1 | 43.7 | 43.5 |
| 2 | 31.5 | 32.8 |
| 3 | 24.8 | 26.5 |
| 4 | 21.1 (min 20.3) | 23.1 (min 22.4) |
- Cause of the collapse: +~27ms per step per stream, ~80% of which is reading per-stream unique MoE experts (each conversation uses different experts). τ is unchanged.
- When enabling it, `VLLM_USE_V2_MODEL_RUNNER=1` is required (without it the FULL graph drops to PIECEWISE, single stream 28.4).
- Remaining candidate: confidence-based row allocation (DSpark Alg.1, built into the fork) — start by measuring MTP probability calibration.

## Production settings (glm53-serve.env — gitignored because of secrets; the same values are in glm53-serve.env.example. This is the canonical copy)
```
: "${KV_CACHE_MEMORY:=10000000000}"     # active 10e9 (+ 2e9 headroom above memwatch)
: "${KV_OFFLOAD:=1}"
: "${KV_OFFLOAD_DIR:=$HOME/glm53-kv-offload}"
: "${KV_OFFLOAD_BYTES:=200000000000}"
: "${KV_OFFLOAD_THREADS:=8}"
: "${MTP_PER_BATCH:=[[1,1,4],[2,2,3],[3,4,2]]}"   # VLLM_USE_V2_MODEL_RUNNER=1 is added by launch.sh
: "${SCHED_POLICY:=priority}"            # codemason subagents = priority 1
: "${GLM53_BATCH_INVARIANT:=1}"          # temperature-0 batch invariance (BATCH_INVARIANCE.md)
: "${KV_OFFLOAD_HASH_SEED:=glm53-kv-offload-bi-1}"  # change it when the numeric path changes
```
Debug options (GLM53_MTP_CONF_PROBE, NSYS) are off by default — do not turn them on.

## Stress experiments (2026-11-26, all including restart/flood/corruption, outputs compared bit-for-bit at temp0)
| # | Test | Result |
|---|---|---|
| Order | Late-arriving main vs earlier-arriving subagent with slots full | Main gets a slot first |
| E1 | 82K parent + 82K×18 flood (1.48M) — no restart | NVMe restore 7.0s (cold 59.7), match |
| E2 | Store 100 slots, LRU | Upstream LRU restored 0 for the reused conversation → with SegmentLRUPolicy C1 7.0s · C2 leading part 46.9s |
| E3 | 446,746-token parent, restart | 10.1s (cold 357.8), match |
| E4 | 133K×3 subagents concurrently → parent / +707K flood → parent | 1.95s (GPU) / 9.7s (NVMe), match |
| E5 | 149K×4, concurrent restore after restart | Restored KV correct (sequential re-query 4/4 match). Character differences in concurrent decode are batch non-invariance (same thing happens with concurrent GPU hits without offload) |
| E6 | rank0 body corruption / header deletion | crc detected → recompute (match) / that chunk excluded from the index → partial restore |
| Memory | Floor across all experiments | head ≥6.0GB, worker ≥8.75GB (memwatch 3GB) |

## 2026-11-27 light re-verification (after batch invariance + cm2 priority install, production settings) — all pass
| # | Test | Result |
|---|---|---|
| S1 | 19,295-token cold → same question again | TTFT 15.09s → **1.26s** (GPU hit), output identical, decode 37.1 t/s |
| S2 | Same question + 1 decode · 1 cold prefill concurrently | Output identical |
| S3 | 4 slots filled, earlier subagent (priority 1) vs later main | Main first token 15.0s < subagent 18.1s |
| S4 | cm2 v3.0.0 (exp/request-priority bc89cd3) request bodies | Main = no priority, all subagent requests `priority: 1`; on 400 rejection, retries without it and omits it from then on, task completes; with no subagents, omitted everywhere |
| S5 | S1 question after RESTARTALL | TTFT **5.36s** (NVMe restore, cold 15.09s), output identical to before restart |
Tools: `tools/bi/cm2_prio/` (logging proxy · isolated config `CODEMASON_CONFIG_DIR` · test scripts · records).
cm2 install: `~/.local/share/codemason/codemason` (md5 350856d5…), previous one `codemason.bak-20261127-pre-priority`.


## Isolation — when several people use the same server (summarized 2026-11-27)
- There is no direct leak or contamination path: the cache key is a hash chain of tokens from the very start, so to hit you must already have sent that content,
  and the same tokens give the same computed result (corruption → crc · fingerprint → recompute). The model attends only to its own request's tokens — there is no path to read someone else's KV.
- What remains = a **timing side channel**: TTFT (hit ~1s vs cold ~13s) can confirm "someone sent this prefix". A known template + a short secret
  allows candidate guessing. NVMe persistence with no TTL → the window during which this can be confirmed is effectively unlimited. KV stays on disk as plaintext tensors.
- Countermeasure (orthodox, official vLLM): the gateway enforces a **random per-session `cache_salt`** (no identity needed; a 128-bit random value bound to a cookie/session token) and
  overwrites any salt the client sends. Cross-session hits 0, within-session hits unchanged. Cost = one cold prefill of the system prompt per new session.
  The salt is mixed into the hash from the first block, so "sharing only a common system prompt" is impossible. The salt also goes into the offload key (in the stage 2 experiments, changing only the salt was confirmed to be cold every time).
- Supplementary: store disk encryption (LUKS), per-session deletion / retention period (not implemented). Not applicable to single-user operation (current).
- For an anonymous shared console (virtual machine), **environment remnants** (codemason conversation records `~/.codemason/projects/…`, work files, shell history) are a bigger leak than the cache —
  discarding the environment at session end comes first.
