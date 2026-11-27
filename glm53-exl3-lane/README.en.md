[한국어](README.md) | **English**

# GLM-5.3-Flash EXL3 lane (2×DGX Spark GB10) — operations document

Installation and upgrades: repository root [`README.md`](../README.en.md). Patch rationale and pitfalls: [`HANDOFF.md`](HANDOFF.en.md).
This directory must have **the same path and the same contents on both nodes (head/worker)** (`sync-lane-67.sh` keeps them aligned). Last updated 2026-11-27.

> **11-27 merged to main, operational default**: NVMe KV offload + temperature-0 batch invariance + priority scheduling + `/ab` zero trust (per-request only).
> Field tests (E1~E6, stage 2 A/B/C) passed. The three documents in the table below are the source of truth for operational settings and results.

### Document map — where to look for what

| Looking for | Document | Status |
|---|---|---|
| Start/stop/status check, current settings, measured t/s | **this document** | current |
| Images (vision) — settings, safe range, measurement method, **attempts that already failed** | **`VISION.md` §0 — must read** | current |
| Why it was patched this way, pitfalls not to step on again | `HANDOFF.md` | current |
| NVMe KV offload design, operational settings, eviction, integrity, measurements | `KV_OFFLOAD.md` | current |
| Offload verification procedure (E1~E6, S1~S5, stage 2) — follow as-is when porting to another machine | `KV_OFFLOAD_TESTPLAN.md` | current |
| Temperature-0 concurrent-batch nondeterminism fix (batch invariance) — cause, fix, pitfalls, diagnostic switches. ⚠️ Concurrent prefill of short identical prompts is still unresolved (§6-b) | `BATCH_INVARIANCE.md` | current |
| `/ab` uncensoring switch — usage and mechanism | `AB-SWITCH.md` | current |
| UTF-8 byte guard design and verification | `UTF8-GUARD.md` | current |
| Why we serve the original instead of abliterated | `ORIGINAL-VS-ABLITERATED.md` | background (Sep–Oct) |
| Chronological work history | `HISTORY.md` | history |
| Installation, upgrade, offline rebuild | root [`README.md`](../README.en.md), `../glm53-materials/HANDOFF.md` | current / September snapshot |

⚠️ **Before measuring or fixing anything image-related, read `VISION.md` §0 first.** The attempts that were already tried and
failed (early release of `.data`, lowering the limit to 4, malloc knobs, semaphores) and the conditions that contaminate measurements are listed there.

## 1. Operations (run on the head)
```
./vllm-glm53-exl3.sh RESTARTALL   # the only start command (clean up both nodes→memwatch→worker→head→READY)
./vllm-glm53-exl3.sh STATUS       # container/health/model and /ab state/available memory/memwatch
./vllm-glm53-exl3.sh LOGS | LOGSW # follow head/worker logs
./vllm-glm53-exl3.sh STOPALL      # stop both nodes (+release memwatch)
```
- ⚠️ With 2 nodes, always RESTARTALL. If only the worker is alive, it becomes a zombie waiting at rendezvous (601 seconds wasted). Use `START` alone only when the worker is already waiting.
- The worker holds a file copy, not git. After committing or switching branches on the head, you must push with `./sync-lane-67.sh` or RESTARTALL will not come up (checked automatically before start).
- Experiments go on a branch (`git switch -c exp/...`); to revert: `git switch main` → `./sync-lane-67.sh` → RESTARTALL. Do not rmi/retag the image tag v2.3-tier1.
- After a reboot, RESTARTALL manually too (there is no systemd unit). About 210~255 seconds until READY; the first request is slow because of JIT warm-up.
- No heavy work on the head while serving (unified memory). memwatch preemptively stops the containers if available memory drops below 3GB.
- API: `http://<head LAN IP>:8000/v1`, model name `glm-5.3-flash`, key is `VLLM_API_KEY` in the env. Uncensoring is only per request via `"vllm_xargs":{"ab_scale":1}` (`POST /ab` returns 410 — there is no global switch; see `AB-SWITCH.md`).

## 2. Files
| File | Role |
|---|---|
| `vllm-glm53-exl3.sh` | Control wrapper (controls both nodes, memwatch built in). Reads the worker ssh target and key from the env |
| `launch.sh` (+`launch.sh.diff`) | Copy of the Entrpi kit launcher + our changes: automatic lane path, vision gate `VISION` (1=accept images — live default, `VISION.md` §0; if 0, `--language-model-only`), API key, overlay bind-mounts, env knob pass-through, `/ab` adapter and plugin, `NSYS=1` profiling |
| `glm53-serve.env` | Settings. ① site values (edited at install) ② model and adapter paths ③ verified operational settings |
| `sync-lane-67.sh` | rsync --delete of the lane to the worker + per-file md5 comparison (`--check` is run automatically by RESTARTALL). Refuses if the old `~/glm53-hotfix-b12x` is still present |
| `chat_template.jinja` (+.orig/.diff) | Copy of the model template + backport of the zai-org 09-04 fix. Injected with `--chat-template` |
| `draft_vocab_32k.json` | The 32,768 token ids of the MTP-draft-only vocabulary |
| `overlay/exl3.py` | Dense layers FP8 (Marlin W8A16), lm_head FP8, prefill BF16 dual path, 32K draft head, SwiGLU clamp |
| `overlay/glm5next/{model,kda,mtp,attention}.py` | Projection quantization pass-through, MTP eh_proj FP8 and draft-head proxy, indexer fp32 transposed layout, `/ab` hook (KDA) |
| `overlay/glm5next/ops/kpool_compress.py` | kpool tail-cache stride fix (backport of vllm#57477) |
| `overlay/v2runner/` | '!' collapse fix (backport of vllm#51865): runner + 2 speculators |
| `overlay/mla.py`, `ab_switch.py`, `ab_plugin/` | `/ab` runtime uncensoring switch (MLA hook, hook body, HTTP route and worker propagation) |
| `overlay/sparse_utils.py`, `sparse_attn_indexer_kpool.py`, `b12x_mla_sparse.py`, `scatter_states.py` | temperature 0 determinism (§4.9), 09-28 audit fixes, null-block handling |
| `overlay/vllm_utf8_guard.py`, `gpu_sampler.py` | UTF-8 guard (`UTF8-GUARD.md`) |
| `overlay/b12x/` (+`b12x.orig`, `b12x.diff`) | The entire b12x package. Our diff = env hook for the m8 path (disabled by default) |
| `tools/fetch_oproj.py` | Fetch only the 46 o_proj tensors (3.89GB) via HTTP Range (original/abliterated) |
| `tools/build_ab_lora.py` | Build the `/ab` adapter (rank-24 approximation of original − abliterated, 29MB) |
| `tools/build_original_from_abliterated.py` | Build the original from an abliterated install (hard links) |
| `tools/prefill_tail_nan_check.py` | '!' collapse regression check (lengths with remainder 5 mod 512) |
| `tools/kpool_cache_probe.py` | kpool cache contamination check |
| `tools/kmeas.py`, `prefill_bench.py` | Decode t/s, acceptance, step / prefill t/s |
| `tools/probe_long.py`, `tfcap.py`, `lpcap.py`, `probe8.py`, `refusal_probe.py` | Hallucination probes, precision gate, refusal-rate measurement |
| `tools/agent_sim.py` | Traffic replayer that reproduces cm2 request shapes exactly (used to reproduce '!') |
| `tools/draft_vocab_*.py`, `router_gemm_bench.py`, `probe_head.rs` | Draft vocabulary regeneration, router GEMM bench (with the server stopped), probe helper |
| `tools/_lane.py` | Shared tool helper: lane path, URL, API key (read from the env) |

## 3. Final configuration (glm53-serve.env)
- Image `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`, model **original** `brandonmusic/GLM-5.3-Flash-tr3-4bpw`,
  `/ab` adapter `~/Models/glm53-ab-lora` (off by default), TP2, max-model-len 524288, KV 10e9/node (fp8_ds_mla; overflowing prefixes go to the NVMe offload), max-num-seqs 4.
- Speculative decoding MTP-4 (`SPEC=none MTP=4`). DFlash2 is available as an option but unused because it lost to MTP-4 (§5).
- Precision: MoE routed experts W4A16 (EXL3 trellis); dense, shared expert, lm_head, MTP eh_proj **W8A16 Marlin** (weights FP8, activations BF16); KV FP8.
  **Activation quantization (W8A8/W4A8) and INT4 are forbidden** (they cause hallucination).
- Prefill: dense calls with M>1024 dequantize FP8 to BF16 and use cuBLAS (`VLLM_EXL3_PREFILL_BF16=1`); decode uses Marlin.
- UTF-8 byte guard (`VLLM_UTF8_GUARD=1`), MoE router GEMM deduplication, draft-only 32K vocabulary (`VLLM_GLM53_DRAFT_VOCAB`),
  routed-expert SwiGLU clamp (`VLLM_EXL3_MOE_SWIGLU_CLAMP=1`), temperature 0 determinism (`B12X_DENSE_SPLITK_TURBO=0` + 2 overlays),
  '!' collapse fix (`overlay/v2runner/`), kpool cache fix (`overlay/glm5next/ops/`).
- Prefix cache: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=18432` (thins out mamba snapshots; rationale in the env comment).
- **Offload, batch invariance, priority (merged to main 11-27)**: KV 10e9/node + `KV_OFFLOAD=1` (NVMe 200GB/node, cache survives restarts), `VLLM_PREFIX_HASH_SEED`=`KV_OFFLOAD_HASH_SEED` (currently `glm53-kv-offload-bi-1`; change it whenever kernels or numerics change), `GLM53_BATCH_INVARIANT=1`, `SCHED_POLICY=priority` (codemason subagents = priority 1), `MTP_PER_BATCH=[[1,1,4],[2,2,3],[3,4,2]]`. The canonical block is the 'Production settings' section of [`KV_OFFLOAD.en.md`](KV_OFFLOAD.en.md).

## 4. Measurement ledger

**Latest (2026-10-05, current main, MTP-4, `/ab` off)** — kmeas (temp 0, thinking off, 300 tokens×3): hash-map **34.9** / Korean prose **38.9** / code **41.8** t/s,
acceptance 1.85 / 2.08 / 2.39, step 79~81ms. (What follows below is the record as of that time.)
**Latest (09-27, exp/step70: router dedup + 32K draft vocabulary + SwiGLU clamp, MTP-4)** — kmeas (temp 0, thinking off, 300 tokens×3)
| Item | Value |
|---|---|
| Decode English / Korean prose / code | 34.5 / 39.7 / 44.3 t/s (acceptance 1.70 / 2.05 / 2.54) |
| Step | 77~79ms (09-09: 82~84.5). nsys breakdown (09-27, before 32K): MoE verify 29.0 · Dense Marlin 20.0 · lm_head×5 8.1 · NCCL 4.7 · BF16 GEMV 3.7 · small 6.2 · idle 4.0 |
| Prefill | 12k 1,348~1,362 / 24k 1,327 / 75k cold 1,328 t/s |

**09-09 baseline**
| Item | Value |
|---|---|
| Decode Korean prose / code / English prose | 34.5 / 44.1 / 34.8 t/s (MTP-4, acceptance 1.82 / 2.66 / 1.98) |
| Step | 82.0~84.5ms = MoE 33.5 + Marlin 29.1 + NCCL 5.5 + small kernels ~12 + idle ~4 |
| Prefill | warm 12k 1,341 / 24k 1,311 t/s, first request cold 851 |
| Loading | READY 225 s |
| Comparison | 09-03 (DOCA 3.4.0): decode 36/43/35, prefill 1,333/1,312 — same level, no regression |
| Comparison | All BF16 (09-02): decode 25.6/31.1, prefill 1,395/1,078. FP8 alone: 36.2/44.1, 1,225/982 |
- Decode is byte-bound: the MoE kernel reaches 73% of the ceiling (244GB/s) at M=1; verify (M=5) reads 30+ experts and sits near the ceiling. Marlin overall 83%.
- RDMA path confirmed (09-09): mlx5_0 port_xmit/rcv +8.7GiB during the decode bench window → not an NCCL TCP fallback. `NCCL_IB_GID_INDEX=3` still resolves to RoCE v2/IPv4 (10.0.1.1·.2) consistently on both nodes even after replacing rdma-core.
- Remaining headroom from kernels ≈ -12ms (prose 42 / code 50). Beyond that, only drafter acceptance (Korean training). Details = memory `project_glm53_decode_kernel_truth`.

## 4.9 ✅ Resolved — temperature 0 nondeterminism (2026-09-29~30)

**Symptom**: with the operational settings (MTP-4), the same prompt at temperature 0 gave a different answer every time (6 runs, 6 variants).
Speculative decoding should be lossless because the target verifies, but it wasn't. The first divergence is a
~1e-4 wobble in logprob values, which accumulates in the KV and flips a near-tie argmax a few hundred tokens later.

**Three causes — all the same family: "an atomic decides the order (and tie selection)."** All three are defects in **upstream code** (b12x / vLLM; not introduced by our patches — anyone serving GLM-5.3 with the same image hits them). Whether upstream's latest main has fixed them is unverified. The ROCm lane (NONAME) also confirmed the same defect ③ with a unit test (①② do not exist in its kernels). Floating-point addition
is not associative, so if the summation order changes from run to run, the values wobble.

1. **dense GEMM split-K** (`99298bd`): partial sums were accumulated with bf16 atomicAdd (`red.global.add.bf16x2`,
   `_lib/dense_gemm.py` `scatter_add_bf16x2`). → Use the index slots that already existed in the kernel +
   fixed-order summation (`B12X_DENSE_SPLITK_TURBO=0`, the default). This alone makes MTP=1 deterministic.
2. **sparse index compaction** (`overlay/sparse_utils.py`): when converting logical tokens → physical slots and gathering the valid
   ones to the front (`COMPACT_TO_FRONT`), 64-column tiles claim positions with `tl.atomic_add`.
   If the tail tile wins first you get `[T40, T0..T39]`, if it is late `[T0..T40]` — same set, different order.
   Upstream (#50365) uses an atomic-free single tile only when the row width is a power of 2 (2048), but
   GLM-5.3 kpool rows are 2048 + tail pool (64-aligned) = **2112** wide, so they fell into the atomic path.
   → Pad the width to 4096 + column mask, one row = one program. Order = column order.
3. **indexer top-k** (`overlay/sparse_attn_indexer_kpool.py` `_canonicalize_topk_`): when a row's pool exceeds
   k (512 = 2,048 tokens), persistent_topk (decode) and top_k_per_row_prefill (prefill)
   distribute output positions via shared-memory atomicAdd, so the **order** changes, and when exactly equal values
   straddle the boundary it is first-come-first-served, so even the **set** changes. Measured: in all 3 runs, logprobs were bit-identical
   up to context 2,051, with the first divergence at 2,052. Unit test: over 20 runs of the original, the order wobbled in 16/20 rows and the set in 6/20 rows (tie rows).
   → Use the fast kernel only to get the k-th value t (deterministic regardless of ties), then a Triton kernel scans the row in ascending
   index order and re-selects all `> t` plus `== t` starting from the smallest index for the remaining slots.
   Same rule as torch.topk (threshold + stable gather), no atomics. Rows with pool ≤ 512 were
   already in ascending order, so their output does not change (unit test 4/4 bit-identical).

**How it was found** (if it recurs, follow the same order): ① print per-layer checksums without GPU sync to pinpoint the first diverging layer
(always a sparse layer) ② run only the decode kernel twice → 0/40 differences (kernel innocent); run the whole attention path twice →
6/24 differences (something upstream of the kernel is the culprit) ③ hash-compare kernel inputs across requests → same KV record set, only the order differs
④ back-compute physical slots through the block table → the tail token swaps between front and back (cause 2) ⑤ for the divergence that remained only
on long outputs, locate it at token granularity with logprobs → exactly at the 512 pool boundary (cause 3).

**Verification** (operational settings, CUDA graph + MTP-4, no probes):

| Condition | Result |
|---|---|
| effort low, 3 kinds × 5 runs × (full prefill / cache hit), ≤1,000 tokens | **30/30 identical**, also identical between full prefill and cache hit |
| Thinking on, code 2,201 tokens / prose 3,500 tokens (context exceeds 2,048) | 4/4 · 4/4 identical (before the cause 3 fix, all 4/4 differed) |
| 12,393-token prompt + 600 tokens | 3/3 identical |
| MTP off (MTP=0): short battery + thinking-on code 2,061 / prose 3,500 tokens | 30/30, 3/3, 3/3 identical |
| Control: same conditions with only the cause 2 overlay removed | prose 5/5, code 5/5 all differed |

**Cost**: re-selection kernel 9 / 20 / 127 µs per call (pool 2K / 16K / 131K, 5 rows). Decode step
prose 78.5 / code 79.2 / hash-map 80.3ms — same range as the 09-29 baseline 77.5 / 79.3 / 80.8.
Prefill 12k 1,352 · 25k 1,332 t/s (09-03 record 12k 1,333). kmeas t/s is prose 39.4 / code 42.5 /
hash-map 35.2, lower than the 09-29 record (40.6/43.5/36.4), but since the step is the same, the entire difference is **acceptance rate**
(2.147 → 2.082 etc.). The old values were samples from paths that diverged on every run; now the path is pinned
to one, and that path's acceptance rate is what comes out (re-measuring gives the same value down to the decimals).

**Confirmed not to be the cause** (do not repeat): TC-decode FC2 float atomic, `route_pack.py`
row-position atomic, CUDA graph, indexer logits (bit-identical in the valid range), the sparse MLA kernel itself
(including the split-K merge). **Only sorting** the top-k is not the answer either — with ties, the set changes.

## 4.10 abliteration runtime switch (`/ab`, 2026-10-01)

Serve the original while turning uncensoring on and off per request. Since `o_proj` is BF16 even in the 4bpw checkpoint,
the difference between the two checkpoints is extracted as a rank-24 adapter (29MB) and added to the output.
No restart, no re-prefill, no graph re-capture. Details, measurements, and porting notes are in `AB-SWITCH.md`.

Start with plain `./vllm-glm53-exl3.sh RESTARTALL` (original + adapter is the default); control is
per-request `vllm_xargs.ab_scale` (codemason `/ab off|on|auto` attaches it to every request; the global `POST /ab` was removed 11-27). Measured: refusals 5/16 → 0/16,
hallucination 4/60 with the switch off (serving abliterated wholesale: 18~22/60), speed loss 1~2ms per step,
temperature 0 determinism kept in both states (12/12). `STATUS` shows the switch state.

## 5. Experiment records (records under the conditions of the time — in some cases the target back then was abliterated)
### 09-27 — draft-only 32K vocabulary lm_head (adopted, merged to main 09-29)
- Only the MTP draft argmax uses a 32,768-token subset FP8 W8A16 (g128) head (`overlay/exl3.py` _SubsetDraftHead, `draft_vocab_32k.json`, env `VLLM_GLM53_DRAFT_VOCAB`). Verify uses the full vocabulary → output is lossless.
- kmeas step 82.9/80.6/81.9 → 79.2/76.4~77.3/78.5~79.8ms, t/s 32.2/38.0/43.6 → 33.8~34.0/39.2~39.6/44.7~45.2, no change in acceptance. Prefill 12k 1,348/1,362 · 24k 1,327.
- Set: mandatory 4,444 (special, single-byte, UTF-8 fragments, Hangul, single-character Hanja in KS X 1001) + corpus frequency + BPE merge rank. Excluded: Chinese-only Hanzi, Japanese shinjitai, kana, Cyrillic/Arabic/Thai/Devanagari. Regenerate with `tools/draft_vocab_corpus_gen.py` → `tools/draft_vocab_build.py`.
- ⚠️ The V2 speculator picks drafts via `compute_logits`+argmax. Replacing the draft head only takes effect if done in the MTP `compute_logits` (the old INT4 draft head was hooked only into get_top_tokens).
- **Fix for the missing routed-expert SwiGLU clamp (upstream Entrpi bug)** — `VLLM_EXL3_MOE_SWIGLU_CLAMP=1` (code default 0). The model's `swiglu_limit=10` is present in vLLM's `moe_config`, but the EXL3 method calls the kernels directly and did not pass it on (missing since the image's original exl3.py; the MiaAI kit applies it). All 4 paths: b12x fused Trellis `Caps.swiglu_limit` (the one actually used) / mixed Trellis `compile_mixed_trellis(swiglu_limit)` / exllamav3 `act_limit` / torch fallback. Measured with clamp ON: the 괭 (gwaeng) test and the scientists test passed, kmeas 34.5/39.7/44.3 · 79.1/77.1/79.1ms (no change). Upstream comparison: routing scale 2.5 is applied once by the router (runner 1.0); renormalize, sigmoid, and bias are in the router; the 3 dense layers and the shared expert were already clamped by vLLM `SiluAndMulWithClamp` — no other omissions. ⚠️ The causal link to long-context gibberish + repeated '!' (09-02 codemason 21.5K, NaN traces) is **unconfirmed**: intermediate values on the b12x path actually in use are BF16/FP32, so it is not an fp16 overflow path.
- **MTP k re-measurement (after applying the 32K head; kmeas t/s hash-map/prose/code · step):** MTP2 **35.8**/39.5/42.2 · 61.5~63.1ms / MTP4 33.9/**39.8**/**44.3** · 77.7~79.4ms / MTP5 30.4/35.2/43.3 · 84.5~88.0ms / MTP6 31.3/35.4/39.1 · 93.1~96.3ms / MTP7 not measured (stopped). With the draft cheaper, MTP2 leads on English by +5.6%, but prose is tied and code is -4.7% → **keep MTP4 (user decision)**.
- Rejected (do not reopen): router ll_bf16 dotprod (`VLLM_SM12X_LL_ROUTER`; the kernel is faster but the step is +2~3ms) / verify top-p Triton (`VLLM_TOPKP_TRITON_MIN_ROWS`, noise) / disabling CPU C-states (no effect). The code is kept behind env gates; the default is upstream.

Adopted: dense FP8 Marlin W8A16 / lm_head FP8 / eh_proj FP8 / fp32 indexer layout / prefill BF16 dual path.
Discarded (do not reopen; rationale in memory): b12x block-FP8 W8A8 (hallucination) / INT4 draft head (acceptance -4%, precision rule) / b12x split-K flag (no effect) / b12x tile 64x128+CTA3 (decode -3% but prefill -10%) / NCCL_PROTO=LL (no effect) / DFlash2 (prose 29 vs 36) / MTP k sweep (completed 09-02, MTP=4).

### 09-11 upstream re-survey + drafter re-check
- Upstream status: lovesenko weights (`c8f58e6a`) and the entrpi image (`v2.3-tier1`, digest `44b2dba…`) both unchanged = we are at head. The parent quantization brandonmusic tr3-4bpw has 09-04 and 09-10 commits, but only LICENSE/card (weights unchanged).
- **Adopted**: 3 backports from the zai-org 09-04 template fix — the bug where `content=None` in visible_text rendered the literal "None", `'<tool_call>' + tc.name` → `~`, and a short-circuit guard in the can_sort loop. Compared old vs new renders on 10 message shapes: only the `content:null` case differs, everything else is byte-identical. However, the vLLM OpenAI layer normalizes `content:null` to `""`, so **there was no actual harm on the current API path** (confirmed with `/tokenize`). A preventive patch for consistency with upstream.
- Replacing the template wholesale with upstream's is **forbidden**: the latest zai one (a) emits media tokens (at the time we ran `--language-model-only`. ⛔ Since 2026-10-30 we accept images with `VISION=1` — `VISION.md`), and (b) changes the `clear_thinking` default to false, preserving all past thinking. Both run counter to our intent. Backports are always minimal hooks.
- **Discarded (do not reopen)** — 2 new DFlash2 drafters, both lost to MTP-4 (same day, same harness):

| Drafter | English | Korean prose | Korean code | Step |
|---|---:|---:|---:|---:|
| MTP-4 (current) | **36.1** | **37.4** | **43.6** | 82ms |
| cfontes/Coder40-95 TR3-v3 (retrained, K=7) | 31.1 | 27.3 | 38.2 | 93~97ms |
| incoai 08-31 new checkpoint (K=7) | 30.5 | 29.9 | 40.6 | 93~96ms |

- The cfontes champion setting (K=2) cannot be used on this image. `qwen3_dflash2.py:143` checks
  `1 + num_speculative_tokens` against the drafter config's `block_size` (8) and raises `ValueError`.
  ⚠️ **This is not an arbitrary restriction of the image but a training-condition guard.** Source comment: "The convolutions'
  block modulus is baked in at training time; a mismatch drafts silently garbage-quality
  tokens." **Do not remove it** — removing it means silently using broken drafts instead of getting an error.
  (There are forks where K is unrestricted. Those may be versions that do not have this guard yet, so do not quote
  their K=2 numbers as-is.)
- The **"+68%"** on the cfontes card is **relative to the vendor DFlash drafter** (18.9→31.7), not relative to MTP.
  The champion's 31.7 t/s (4-node TP4) is itself lower than our 2-node MTP-4 prose 37.4.
- **AMD lane cross-check (09-29)**: re-measured on the original (o_proj restored) + the K-unrestricted fork — prose accept len
  1.78 (close to the card's 1.83), so **acceptance is normal**; cycle 119ms vs MTP2 82ms, the drafter adds
  37ms and loses. Both sides agree on the conclusion that **the culprit is drafter cost, not acceptance**.
  There is no record yet of the DFlash2 family beating MTP on this model.
- The drafter itself is not bad (the new incoai one has code acceptance 2.90, higher than MTP's 2.56). It loses on step time: the BF16 drafter raises the step from 82 → 96ms, eating the acceptance gain.
- ⚠️ **Conditions of the table above (09-11)**: the target was abliterated, and the drafter was plain BF16 (dense FP8
  online quantization was introduced 09-27). Both are different now. Also, **per-domain acceptance was not recorded separately
  from t/s** (only code 2.90), so it is impossible to tell whether prose is low because of acceptance or step time —
  if re-measured, always record per-domain accept/step separately.
- Recalculation (09-29): even trimming the drafter to FP8 takes it from 2.34GB → 1.2GB, about 4.4ms at the 244GB/s ceiling.
  Step 96 → 91~92ms, still short of MTP-4's 82ms. 5ms cannot close prose 29.9 vs 37.4.
  → **Decision: keep MTP-4.** To overturn it, the step time has to come down, not the acceptance rate go up.
- Note: `VLLM_EXL3_DENSE_FP8=1` converts non-EXL3 dense linear layers to FP8 Marlin at load time, and there is no code
  that excludes the drafter. If turned on together with `SPEC=dflash`, the drafter would apparently also be converted to FP8
  (unverified). The MXFP8 version (`~/Models/glm53-dflash2-mxfp8`) is an off-the-shelf HF build and has **never been measured**.

## 6. Change procedure
1. Edit lane files (never modify the image originals directly; keep `.orig`/`.diff` for overlays) → 2. **git commit** → 3. `./sync-lane-67.sh`
→ 4. RESTARTALL (compares both nodes before start) → 5. gate with `tools/prefill_tail_nan_check.py`, `tools/kmeas.py`, `tools/prefill_bench.py`,
and `tools/probe_long.py` if precision-related → 6. update this document and HISTORY.
- If you use a new env knob, it must be added to the pass-through list in launch.sh (`for _v in ...`) to reach the container. Environment variables do not pass through ssh.
- Profiling: `NSYS=1 ./launch.sh 0` (head), then `docker exec vllm_glm53 /opt/nsys/target-linux-sbsa-armv8/nsys start -o /cache/prof/X.nsys-rep` / `stop`.
  Read reports with `nsys stats` from Nsight Systems 2025.3.2 (older host nsys cannot read them).
- Standalone GPU or conversion work only after STOPALL (unified memory).

## 7. Recovery
- Containers are recreated on every start, so if one is deleted, just RESTARTALL.
- If the image is deleted: `docker pull ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` (both nodes; check image id `sha256:606b5f3c…`).
- Required assets (same paths on both nodes): this directory, `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw`, `~/Models/glm53-ab-lora`,
  `~/glm53-vllm-cache` (JIT cache; regenerated automatically if missing). How to rebuild the model and adapter: root README §5.5.
