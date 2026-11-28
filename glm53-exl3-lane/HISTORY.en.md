[한국어](HISTORY.md) | **English**

# GLM-5.3-Flash EXL3 lane — work history

Chronological record. Numbers are measurements at the time; hashes are `AmesianX/dspark-serve` commits. Details are in README §5 (experiment records), HANDOFF, and memory notes.

## 11-28 — NVMe offload: re-store and restart re-index after a corrupt chunk · 512K one-direction determinism
- Two defects the .109 (NONAME) lane found while running E6 on rank1 had the same structure in our code (cc99e33).
  ① After a load failure the store cursor stayed at the end of the hit, so recomputed chunks were never stored again → move it back to the
  chunk where the load started (not 0, so default-namespace KV of an /ab request never gets a tag).
  ② `invalidate` dropped the chunk only from the manager and left the rank0 header, so a restart indexed the bad chunk again → zero the
  header, then return the slot.
  Neither produced wrong output (crc caught it every time, then recompute). Our 11-26 E6 only checked detection and recompute, which is
  why we missed it → E6b (re-store and restart) added to the test plan.
  Verified: unit tests 9/9 (the new test fails with header clearing disabled); live E6b, 19,598 tokens — recompute matches → after restart
  NVMe 13,824 restored in 6.7s, match, no load failure.
- 512K one-direction determinism: one conversation grown 32K→65K→131K→264K→523K, per stage 1 extension + 2 cold, all 48 tokens and
  logprobs bit-identical (523K: extension restored 258,048 from NVMe, cold TTFT 439–441s, decode 57–58.5 t/s). Test tools pointing at
  a deleted worktree were fixed too (01cc627).

## 11-27 (night) — long-context decode indexer top-k bug fixed (correctness)
- Above 18K tokens two temperature-0 cold runs diverged at token 4–9 → traced to the decode top-k picking a wrong pool set (up to ~80 of
  512) for MTP verify rows, and the canonicalisation kernel trusting its minimum and overwriting the next row. Fix: verify the threshold,
  exact K-th bisection only for failing rows, bounded writes, −inf past each row's length. 18K/38K/78K cold ×3 bit-identical, all 840
  selections correct, kmeas unchanged. Details in BATCH_INVARIANCE.en.md §6-c.
- ⚠️ At first this failure was hidden by shrinking the test to 18K (the user: "ignoring above 18K was a fatal mistake"). The final
  verification was rerun at 44.7K.

## 11-27 (evening) — `/ab` cache namespaces · per-session cache isolation + shared public prefix · cold-prefill readout
- **`/ab` leak closed**: an ab=0 request reusing ab=1 KV diverged at the first token (measured) → KV identity = (tokens, scale). KV computed under ab
  gets tagged keys; original requests read untagged keys only; ab requests read own → original (GPU `block_pool` overlay + NVMe kvconn). The first
  attempt, "ab requests read only", left `/ab on` with no cache and was dropped.
- **Session isolation**: the gateway forces `cache_salt` (server-only secret); the lane shares up to the public system block unsalted. cm2 sends the
  system prompt as public/private messages. Traps caught live: `vllm_xargs` turns JSON true into 1; cm2's query joined the sections into one string so
  nothing was split.
- **cm2 prefill readout**: new input (search results etc.) used to be lumped into "to prefill"; now only cold (re-computing already-sent input) shows, as `⟲ re-read`.
- Final light verification F1~F6 all pass (KV_OFFLOAD_TESTPLAN.en.md §6). Found: a cold-reproducibility gap above 18K tokens (BATCH_INVARIANCE.en.md §6-c).
- Remaining: double recompute on `/ab auto`'s first request of a turn (fixed by applying ab only from the last user message), and next-turn recompute
  from clear_thinking (whether to use Preserved Thinking is the user's call).

## 11-26 ~ 11-27 — NVMe KV offload · temperature-0 batch invariance · priority (merged into main 11-27, on by default)
- **NVMe offload**: our own `NvmeOffloadingSpec` (O_DIRECT · slot header crc/fingerprint · SegmentLRU · failure → recompute). KV 10e9 + NVMe 200GB/node.
  Stress E1~E6 pass — 7.0s after an 82K×18 flood (cold 59.7), 10.1s after restart with 446K (cold 357.8), corruption detection · recompute. Details in `KV_OFFLOAD.md`.
- **Temperature-0 concurrent batch nondeterminism** (a bug that was also in main) fixed: fixed kernel settings per row role in 7 places, `GLM53_BATCH_INVARIANT=1`.
  Decode t/s identical, 82K prefill −1.5% (1,340~1,348). Turning off cuBLASLt split-K was rejected because it made bf16 decode 2× slower. Details in `BATCH_INVARIANCE.md`.
- **priority**: server `--scheduling-policy priority`, codemason subagents `priority=1` (main in both codemason repos).
- 11-27 light re-verification S1~S5 and **stage 2 (cache provenance)** A · B · C pass — split points are on the 4608 grid and in 512 units, so load, history, and restart don't matter. No fix needed.
  Verification plan for porting: `KV_OFFLOAD_TESTPLAN.md` (E1~E6 scripts preserved in `tools/kvtest/`). Merged into main on 11-27 (experiment branch deleted); serving moved to the main tree.

## 10-30 ~ 11-01 — image (vision) support landed in main, limit 5, early `.data` release failed
- **FF-merged** `exp/glm53-vision` **into main** (`15047c4`, 27 commits / 3,475 lines). The branch was deleted both locally and on the remote.
  The rollback point is `c45c10d`. Live came up with the vision template and image recognition was confirmed (shapes, colors, 7-digit numbers, tables all correct).
- **Measurements by image count** (clean n=2~4, 1792²): 1 image drop **0.00GB** · window 0.0s / 2 images 0.44GB · 26.2s / 5 images 0.78GB · 37.3s /
  8 images 1.00~1.26GB · window 31.2s (TTFT 30.4 = 97% of the window). Headroom = head baseline 5.3~6.0 − memwatch bar 3.0.
- **`VISION_MAX_IMAGES` 8 → 5** (`a334496`). Three concurrent 8-image requests stop at 3.06GB, but 5 images measured 2.06GB,
  **3/3 success · +0.69 above the bar**. 6 images get a 400 at 0.0s. No loss for the user's real use (usually 1 image, at most 2, 4~5 in serious cases).
- ⛔ **Early release of `.data` (pixel_values) failed** (preserved as tag `glm53-mm-data-release-failed`, branch deleted. Not in main).
  Stage 1 (after KV store) had 0 effect, stage 2 (right after encoding + the worker copy) 0.06GB. Reason: of the 1.03GB drop, `AnonPages` is
  only +0.29GB and 0.74GB (71%) is outside the process's anon memory (presumably CUDA staging, not identified). Process RSS
  changes 0 during a request. The ceiling of an approach that drops references in Python was 0.29GB. **We should have done the arithmetic before the code.**
- Two measurement contaminations self-corrected: the "38.9 s" window was the end of the window being cut off by the 6-second recovery sampling (actual 31.2 s), and the first measurement right after
  startup was misread as a 3.4GB drop because of the ViT's 3.3GiB being held. Protocol in `VISION.md` §0 · §15.
- Docs: new `VISION.md` §0 (must read before image work) · §15, a document map in the lane `README.md`, corrected stale records
  (the root README's "vision tower excluded", the deleted knob `VISION_PROFILE`).

## 10-04 ~ 10-05 — portability cleanup · original-model naming · README · install rehearsal
- Separated paths and site values: scripts compute paths from the lane location and `$HOME`; site values (rail IP · worker ssh · API key) are in one place, the env ① block. Tools also read the key from env (`tools/_lane.py`).
- Named the served model for what it actually is: `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw` (shard sha256 and accessory files confirmed to match the original repository). `/ab` mode is `original|abliterated`.
- Tools: `fetch_oproj.py` (Range-downloads only the 3.89GB o_proj, streaming · peak memory 0.3GB), `build_original_from_abliterated.py`, `build_ab_lora.py` (accepts o_proj files as input). Rebuilding the adapter from just the two o_proj files is byte-identical to the production one.
- Deleted 8 unmounted overlays and old env backups (kept only in git history). Removed the HANDOFF appendix (stale diff copies).
- New repository root README: structure, differences, `/ab`, the '!' bug, fresh install (§5), 09-05 kit upgrade (§6).
- **Ran 2 install rehearsals**: (a) cloned to a new path and followed README §5 exactly → startup and verification passed (prose 38.9 / code 41.8 t/s). (b) actually installed the 09-05 kit, brought it up as abliterated, then followed §6 exactly → original and adapter byte-identical, '!' check PASS, `/ab` working in both directions. Fixed during the rehearsals: image check (images from docker load have no digest → image id), commit failing when git user info is missing, notice about the adapter build time (about 10 minutes), hard-coded site IP wording in the sync output, telling where to get them in the missing model/adapter errors.

## 10-02 — '!' collapse root cause found and fixed (fix/glm53-prefill-tail-decode-graph)
- Symptom: only '!' (token 0) repeated in the reasoning channel, `finish=repetition`. Reproduced 100% on the first turn with the shape of a real cm2 request (system prompt + 18 tools + Korean task).
- Cause: when **prompt length ≡ 5 (mod 512)**, the last prefill piece is exactly 5 tokens (= 1 + MTP 4), and the V2 runner, looking only at the batch shape, replays the spec-verify FULL graph → the KDA recurrent state is not built, logits are NaN, and all subsequent decoding is NaN. Unrelated to prefix caching, FlashInfer, or context length.
- Fix: backport of upstream vllm#51865 (08-12; missed because the fork point is 08-10) — not uniform decode if any request is in prefill. `overlay/v2runner/` (runner + 2 speculator files).
- Verification: `tools/prefill_tail_nan_check.py` 7/7 NaN (FAIL) before the fix → PASS after. The reproducing prompt gets a normal response.
- Also included: backport of upstream vllm#57477 (`overlay/glm5next/ops/kpool_compress.py`) — a bug where the prefill tail-seed kernel ignored the padded indexer stride (measured 76032) and wrote with the dense one (1024), **overwriting other requests' indexer cache blocks**. A/B (`tools/kpool_cache_probe.py`): with the buggy version the first-token logprob of the same cached prefix changes after another request is processed (−0.222→−0.122); the fixed version is unchanged over 3 rounds. Not the cause of '!'.
- Tool: `tools/agent_sim.py` (a traffic reproducer with exactly the shape of cm2 requests — uses request bodies dumped by CODEMASON_DEBUG as templates).

## 09-02 ~ 09-03 — lane established
- Started 2-node TP2 serving with the Entrpi kit v2.3-tier1 image (lovesenko tr3-4bpw EXL3 abliterated). The NVFP4 lane was retired (attic/).
- MTP k sweep → MTP-4 settled.
- Online FP8 for dense layers (Marlin W8A16): step 117 → 86~89ms (decode +40%). b12x block FP8 (W8A8) is forbidden because it causes confabulation.
- Prefill BF16 dual path (M>1024): prefill 12k recovered from 1,225 → 1,333 t/s.
- 09-02 codemason session `d5774cad` (21.5K ctx): 1 case of gibberish from the first token + repeated '!' (traces of NaN logits) — found by the 09-27 log investigation.

## 09-09 — DOCA 3.5.0 re-measurement
- Decode 34.5/44.1/34.8 t/s (prose/code/English), step 82~84.5ms, prefill 1,341/1,311 — no regression. RDMA path confirmed.

## 09-11 — upstream re-survey
- No upstream updates (we are the head). Backported 3 zai-org 09-04 template fixes. Two new DFlash2 drafters lost to MTP-4.

## 09-12 ~ 09-14 — version control · UTF-8 guard
- `1c35d89` started tracking the lane in git. `320da97` cleaned up preservation comments.
- `544a6ad` UTF-8 byte guard: blocks in the sampler the U+FFFD caused by script switches in the middle of a character (7/8 → 0/8, no speed loss).

## 09-27 — step reduction · correctness fixes (main + exp/step70)
**main**
- `948cd83` removed duplicate MoE router GEMM: the model and MoERunner computed the gate twice (upstream bug, found first by the NONAME session). Decode 36.83 → 38.44 t/s (+4.4%).
- `d46b184` · `2747dce` · `06178e5` committed as-is the changes left by other sessions (key rotation, DSV4 512K, attic cleanup). Caches · materials (172G) · the upstream kit clone are ignored.
- `48bca07` mount the b12x overlay directly from launch.sh instead of a home-symlink hook (runtime identical, md5 c89d8c0c).
- `a0dd9ff` · `f0ee751` `sync-lane-67.sh` + .66/.67 comparison before RESTARTALL startup (on mismatch, abort before touching the server). Round-trip rollback across branches confirmed by measurement.

**exp/step70**
- nsys re-breakdown (step 76.9ms): MoE verify 29.0 (bandwidth wall) · Dense 20.0 · lm_head×5 8.1 · NCCL 4.7 · BF16 GEMV 3.7 · small 6.2 · idle 4.0.
- `a79a091` · `7f8a865` · `257eacb` router ll_bf16 dotprod (kernel 44→23us, step +2~3ms → rejected), verify top-p Triton (noise → rejected). On GB10, cuteDSL 8-CTA cluster split-K = launch failure. Disabling CPU C-states had no effect (reverted).
- `278ae82` · `93ba399` · `98830d3` draft-only 32K-vocab lm_head: step 82.9/80.6/81.9 → 79.2/76.4/78.5ms, no change in acceptance rate. Pitfall: the V2 draft goes through the `compute_logits` path.
- `66b41d1` · `a217a24` · `e8de017` · `1d580f2` audit fixes: subset head goes on the main model's lm_head (MTP shared_head.head is replaced at load), actual rows for padding, rejected overlays not mounted, proxy attribute delegation, regeneration tools (`draft_vocab_*`).
- MTP k re-measured (after 32K): MTP2 35.8/39.5/42.2 · MTP4 33.9/39.8/44.3 · MTP5 · 6 worse → keep MTP-4 (user decision).
- Prefill 12k 1,348~1,362 / 24k 1,327 — target (1,300+) maintained.
- `5200a48` · `6f26fda` routed-expert SwiGLU clamp: upstream exl3.py does not pass `swiglu_limit=10` to any MoE path (the MiaAI kit applies it). Fixed 4 paths, env-gated (code default off, lane env on). No speed change. The b12x path actually in use has BF16/FP32 intermediates, so it is not an overflow path → causal link to the gibberish symptom **unconfirmed**.
- Upstream comparison: routing scale 2.5 applied once in the router, renorm · sigmoid · bias in the router, dense · shared clamp correct — no other omissions.
- OpenWebUI: replaced the old key left in the DB (sk-dsv4) with the new key, `glm-5.3-flash` model defaults temperature 1.0 / top_p 0.95 (official recommendation), vision off. Confirmed a template/parser mismatch where reasoning text leaks into content with `enable_thinking:false` (does not affect codemason or OpenWebUI, only measurement tools) — not fixed.
- Passed to NONAME (AMD lane) how to port the 32K draft vocab, router duplication fix, and SwiGLU clamp (32K port done, step -2.1ms).
- NONAME SwiGLU clamp applied and verified (same env rule): head and worker clamp ON, U+FFFD 0 over 8 runs of the 괭 (gwaeng) · scientists tests, step 70.0ms unchanged. temp 0 acceptance rate 1.371→1.320 = generated text changed → evidence that the outlier clamp actually triggers in real use. Their mid_kernel stores the raw gate/up GEMM output in FP16 (Hadamard space) → a separate overflow path the clamp cannot prevent (our b12x path is BF16/FP32, so not applicable).
- Status: 09-29 merged into main (fast-forward, 484fc09). The exp/step70 branch was deleted — any `exp/step70` mentions remaining
  elsewhere in this document are records referring to the working branch at the time; the content is all in main.

## 09-28~29 — quality audit and confabulation root cause

- Fixed 7 upstream audit items (thinking budget bypassing the UTF-8 guard, DCP CKV-gather's
  656B writer changed to GLM 528B, kpool chunk alignment guard, blocking scatter_states' null block writes,
  persistent_topk width per pool, tokenizer hash in the UTF-8 guard cache key, V1 copy argument order).
- **Confabulation cause = abliteration.** Reverting just 46 o_proj back to the original took the fake-target traps
  from 22 → 4 out of 60. No speed loss (prose 38.3 → 40.6 t/s). Dense precision · EXL3 4bpw · KV fp8 · nondeterminism were
  all cleared. Restoring only layer 44 had no effect (22 → 25). Evidence and reproduction procedure in `OPROJ-RESTORE.md`.
- Switched the checkpoint to `glm53-tr3-4bpw-oproj-restored` (`glm53-serve.env`). Since refusal behavior
  comes back with it, pentest use is specified per session via codemason's operator directive (`~/.codemason/directive.md`)
  — measured: agent mode executes tools even without the directive.

## 09-29~30 — temperature 0 nondeterminism fixed

- Three causes, all atomics deciding summation order: dense split-K bf16 atomicAdd (`99298bd`),
  tile slot claiming in sparse index compression (`overlay/sparse_utils.py`; GLM's row width 2112 falls outside upstream's
  power-of-2-only atomic-free path), the indexer top-k's atomic ordering · first-come tie-breaking (when pool 512 =
  over 2,048 tokens, `_canonicalize_topk_`). Details in README §4.9.
- Verification (MTP-4): short battery 30/30, thinking-on outputs over 2,048 tokens 8/8, 12k prompt 3/3 identical.
  MTP=0 also 30/30 · 6/6. Control
  (only the compression fix removed) 10/10 all different. Step · prefill (12k 1,352 / 25k 1,332) unchanged.
- The tracing probes were not committed, and the debug knob that slipped into `99298bd` (`B12X_TC_DECODE_MAX_M`)
  was reverted to the 484fc09 original in `1a5942e`.
- All three causes are upstream (b12x/vLLM) defects. The ROCm lane also confirmed ③ identical (①② not applicable). 09-30 both nodes rebooted (cause not investigated), then restarted and determinism reconfirmed, `cc300f8` pushed.
