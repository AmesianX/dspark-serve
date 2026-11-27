[한국어](KV_OFFLOAD_TESTPLAN.md) | **English**

# KV NVMe offload — verification test plan (canonical, for porting)

This document is the procedure for **verifying against the same criteria** after implementing offload on another machine or another stack.
The design is in `KV_OFFLOAD.md`; temperature-0 batch invariance is in `BATCH_INVARIANCE.md`.
All reference numbers are values from **our environment (GB10 ×2 TP2, GLM-5.3-Flash EXL3, vLLM 0.26.1rc1.dev1006 f223ff9f2)**.
Another machine must first measure its own baseline (cold TTFT) and judge by **ratios and match/no-match** — do not use our numbers as pass thresholds.

Scripts: `tools/kvtest/` (E1~E6 · S3), `tools/bi/cm2_prio/light.py` (S1 · S2 · S5), `tools/bi/s2_prov.py` (stage 2),
`tools/bi/e7_det.py` · `e9_conc4.py` (batch invariance). Shared clients `harness.py`/`client.py`, body text `p60k.txt` (code, ~82K tokens).
Provide the API key only via `export K=...` (do not hard-code it in scripts).

## 0. Common rules (break them and the results are meaningless)

1. **The verdict is exact match of the output text** (temp 0, `enable_thinking=false`, fixed max_tokens). "Similar" is a failure.
2. **Without batch invariance, concurrent tests are wrong regardless of offload.** If concurrent tests (E5 · S2 · stage 2 A) show differences,
   first run concurrent GPU hits without offload and check whether the same difference appears → if it does, that is a kernel batch non-invariance problem (BATCH_INVARIANCE.md).
   We too initially got character differences in E5 because of this (sequential re-query matched 4/4 = the restored KV was correct).
3. **The hash seed must be fixed** (`VLLM_PREFIX_HASH_SEED`, same value on both nodes). Without it, NONE_HASH changes every boot and hits after restart are 0.
4. **If you change a kernel or numeric path, change the seed.** The old content for the same key keeps being used, so you end up comparing against "KV computed with the old kernel"
   (BATCH_INVARIANCE.md §2b — the trap that cost us a day).
5. Give each conversation a **unique first line** (`[session tag uuid5]`) to split the prefix hash chains so they don't hit each other.
   To measure the same input cold several times, use a new `cache_salt` per request.
6. Discard the first request right after startup (JIT · warm-up). With 2 nodes, always restart both together.
7. Record the **memory floor** (lowest MemAvailable on both nodes) for every experiment — harness's `MemFloor`.
8. One experiment at a time. Only when there is no other load on the server.

## 1. Stage 1 — stress experiments E1~E6 (2026-11-26, all pass)

| # | Purpose | Procedure | Pass criterion | Our result |
|---|---|---|---|---|
| E1 | GPU eviction → NVMe → restore without restart | Parent P (82K) cold → re-query (GPU hit) → flood the GPU pool with 18 other 82K conversations (1.48M tokens) → re-query P | After the flood P is restored from NVMe (TTFT at ~10% of cold), output = cold | 7.0s (cold 59.7s), match |
| E2 | Eviction order with the store full | **Temporarily shrink** the store **to 100 slots** (`KV_OFFLOAD_BYTES`=100×slot size). C1 · C2 · C3 cold → re-query C1 (use) → C4 cold → **restart** → re-query in order C4 · C1 · C3 · C2 | **Only the tail** of C2 (least recently used) is dropped and its leading part is restored, the rest are restored whole, all outputs = cold | C1 7.0s · C3 7.0s · C4 7.6s, C2 46.9s (leading part only), all match |
| E3 | Large parent survives restart | 446K-token parent cold → restart → re-query | Restore time ≪ cold, output = cold | 10.1s (cold 357.8s), match |
| E4 | Parent after subagent congestion | (continuing from E3) 3 × 133K subagents concurrently (`priority:1`, 300 tokens) → parent → 707K of other traffic → parent | Both times output = cold | 1.95s (GPU) / 9.7s (NVMe), match |
| E5 | Concurrent restore after restart | 149K conversations A · B · C · D cold → restart → re-query all 4 concurrently | All 4 restored + output = cold (with batch invariance). Without it, confirm KV correctness with sequential re-queries | Restore correct (sequential 4/4 match) |
| E6 | Corruption detection | Before the E5 restart, corrupt the rank0 store directly: B = flip 1 byte in one chunk's body, C = zero one chunk's 4KiB header (`corrupt_r0.py`) | B: crc mismatch → load failure reported → **recompute**, output match (contaminated output forbidden). C: excluded from the index from that chunk on → only the leading part restored, output match | Both as expected. Log `load job failed on some rank … recomputes, 34 chunks invalidated` |
| Memory | Floor across all experiments | MemFloor | Above the guard (our memwatch 3GB) | head ≥6.0GB, worker ≥8.75GB |

Additional check: after each step, print per-rank valid slot count and group distribution with `count.py` and confirm **both ranks are the same**
(with E2 full, both sides `100 by group: {0:59, 2:14, 3:14, 4:13}`).

## 2. Light re-verification S1~S5 (2026-11-27, production settings, all pass)

A 15-minute bundle to run whenever settings change after the big experiments (`light_all.sh`).

| # | Procedure | Pass criterion | Our result |
|---|---|---|---|
| S1 | 19K cold → same question again | Repeat TTFT ≪ cold, output identical, record t/s | 15.09s → 1.26s, identical, 37.1 t/s |
| S2 | Same question + (1 decode + 1 cold prefill) concurrently | Output identical (requires batch invariance) | Identical |
| S3 | Fill the 4 slots (`max-num-seqs`) with `priority:1` long generations → earlier subagent (priority 1) vs main 1 second later (no priority) (`prio_order.py`) | Main's first token comes before the subagent's | Main 15.0s < subagent 18.1s |
| S4 | Capture the client (cm2) request bodies with a logging proxy: a task with 2 subagents / proxy rejects priority with 400 / a task without subagents | Only subagent requests have `priority:1`, main has none · on 400, retry without it and omit it from then on · task completes | All as expected |
| S5 | Restart both nodes → S1 question | NVMe restore (TTFT less than cold and greater than a GPU hit), output identical to before restart | 5.36s, identical |

## 3. Stage 2 — cache provenance experiments (2026-11-27, A · B · C all pass) `tools/bi/s2_prov.py A|B|C`

Question: **Do the KV/KDA state bits differ depending on what load and what history the same prefix was computed under?**
In hybrid models (with recurrent state such as KDA/GDN), state bits can differ if prefill chunk boundaries differ,
and once that version is stored in the GPU cache or NVMe, that conversation keeps using state of "a different provenance" from then on.

Method: cold-prefill the same input X (17K) with a **new `cache_salt`** per request (input identical, cache miss), changing only the load condition.

**A. Cold prefill per load condition — is X's output the same as alone (R1)?**

| Condition | Description | Result |
|---|---|---|
| R1 · R2 | Alone ×2 | Baseline · repeat match |
| D1 | X while 1 decode is running | Match |
| D3 | X while 3 decodes are running (`max-num-seqs` full) | Match |
| P2 | Concurrent with cold prefill Y | Match |
| P3L | X **third**, after cold Y · Z (the position that receives the budget remainder) | Match |
| P3F | X first, Y · Z following | Match |
| MX | 1 decode + cold Y + X | Match |

**B. Continuing from that cache — follow-up input X+tail (fixed sentence)**

| Condition | Description | Result |
|---|---|---|
| hit-* | X+tail with each salt from A → hits the X cache stored under each condition (TTFT 1.05s) | All 8 match each other |
| cold1 · 2 | X+tail with a new salt, **fully cold** (12.98s) | Matches hit |
| junc | First send W, which shares the first 61% of X and diverges off the grid, then X+tail (partial hit up to the divergence point, 6.56s) | Match |

**C. After restart (NVMe provenance)** — restart both nodes → re-query X and X+tail with the same salts

| Condition | Result |
|---|---|
| X with the 8 salts from A | NVMe restore TTFT 6.4s (cold 13.0s), all 8 match the pre-restart output |
| X+tail with the same salts | 1.1s (restored up to the chunks stored by B), all 8 match pre-restart |

**Why it passes (= conditions to keep when porting).** The vLLM hybrid `mamba_cache_mode=align` scheduler cuts prefill chunk ends
to the state block grid (discarding the budget remainder), and hit positions are also on the grid or multiples of `prefix-match-unit` (512), so
**every split point is a multiple of the KDA chunk (64)**. Therefore it is computed with the same splits regardless of load and history. Our values:
`--block-size 2304` (scheduler block 4608), `--max-num-batched-tokens 9236`, `--long-prefill-token-threshold 4608`,
`--mixed-prefill-token-cap 4608`, `--prefix-match-unit 512`.
⚠️ If you change these to values that are not grid multiples (especially mixed-prefill cap < block → allows splits inside a block),
or change prefix-match-unit to a value that is not a multiple of 64, you **must rerun** this experiment.
When moving to a different vLLM version, also check that its scheduler has the same alignment rules — `--mixed-prefill-token-cap` · `--prefix-match-unit` are
options on our fork (image) side, and the alignment logic is `_mamba_block_aligned_split` in `v1/core/sched/scheduler.py`.

## 4. Batch invariance (prerequisite) — `BATCH_INVARIANCE.md`

This must pass first for the concurrent tests to mean anything. Our kernel solution is GB10-specific and does not carry over as is
(only the principle carries over — fixed settings per decode/prefill row role, everything is decode during graph capture).
- e7: A alone repeated / A 3 times during another request's cold prefill / ABCD concurrent 2 times → all match
- e9: ABCD (149K) concurrent → 8/8 match with sequential
- Speed gate (A/B in the same session): single decode identical, prefill stays at 1,300+ (ours −1.5%)

## 5. What to add when moving to a different configuration (recommended)

- **If there is a device mirror (e.g. a prefill sidecar GPU)**: right after an NVMe restore, check that the mirror is refilled and that the mirror path's output
  is the same as before the restore — add one step "long prefill continuation right after restore" to the E1 · S5 procedures.
- **Confirm no rank asymmetry**: both ranks' `count.py` results must be the same. Because TP · block · group layout go into the fingerprint,
  if only one side has different settings the whole store is treated as empty.
- **Disk path**: a filesystem that supports O_DIRECT (ext4/xfs). tmpfs and some FUSE filesystems do not work.
- Slot size and slot count differ by model and KV layout — change `SB` in `corrupt*.py` · `count.py` (ours 31,027,200B) and
  the slot count (6,445) to the values from that environment's logs.
