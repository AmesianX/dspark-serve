[한국어](HANDOFF.md) | **English**

# HANDOFF — for developers: patch rationale and traps (GLM-5.3-Flash EXL3, 2×DGX Spark)

Install/upgrade: repository root [`README.en.md`](../README.en.md); operating commands and measurements: [`README.en.md`](README.en.md).
This document records **what was changed, why, and how**, and **traps not to step on again**. The exact change for each overlay
is in `overlay/<file>.diff` (against the image original `.orig`). Last updated 2026-11-27.

## 0. Current state
A single original model, `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (EXL3 4bpw), served 2-node TP2 on the Entrpi v2.3-tier1 vLLM image.
Uncensoring is not in the weights but the runtime `/ab` adapter (`AB-SWITCH.md`). Our patches: dense-layer FP8 Marlin W8A16 +
prefill BF16 dual path + 32K draft vocabulary + router GEMM dedup + SwiGLU clamp + temperature 0 determinism +
UTF-8 guard + **'!' collapse fix (vllm#51865 backport)** + **kpool cache corruption fix (vllm#57477 backport)**
+ **image input (`VISION=1`, 5 images per request cap — `VISION.md` §0)**.

⚠️ For anything image-related, read `VISION.md` §0 first. Attempts that already failed and the conditions that contaminate measurements are there.

## 1. Provenance
| Component | Source | Notes |
|---|---|---|
| Serving kit | https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark commit 63f254f (v2.3-tier1) | The kit's `install.sh` is not used. The launcher `scripts/launch-glm53-vllm-tp2.sh` is copied and modified (`launch.sh.diff`) |
| Docker image | `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` (image id `sha256:606b5f3c…`, registry digest `sha256:44b2dbaf…`) | vLLM fork `Entrpi/vllm-glm-5.3-flash-spark` @ f223ff9f2 (upstream base around 2026-08-10) + b12x (sparkinfer-glmrt glm-next-backport 2f53ce3) + exllamav3 c5d9c657 |
| Serving model | HF `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ a5fee929 | Only routed experts are EXL3 4bpw, the rest BF16. `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw` |
| abliterated | HF `lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated` @ c8f58e6a | Differs from the original only in 45 o_proj. Used only as material for the adapter |
| `/ab` adapter | Generated with `tools/build_ab_lora.py` (29MB) | `~/Models/glm53-ab-lora` |
| Drafter (unused) | HF `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` rev 62f758c0 | Only when `SPEC=dflash`. Lost to MTP-4, so not used |

When moving overlays to a new image: extract the original from the new image (`cid=$(docker create <IMAGE>); docker cp $cid:<path> X.orig`),
re-apply the existing `.diff`, and resolve conflicts by hand. The original path is exactly the mount destination in launch.sh.

## 2. Rebuild
Root README §5 (fresh install) / §6 (upgrade from the 09-05 kit). No image rebuild is needed (overlay approach).

## 3. Patch rationale (what, why, how)
### 3.1 launch.sh (vs the kit launcher, `launch.sh.diff`)
- Computes the lane location (`LANE_DIR`) from the script's own location and uses the config files in the lane directory (no hardcoded paths). `--language-model-only` (default VISION=0. Sets every modality limit to 0, which skips both loading the vision tower weights and the startup dummy MM profile — the old comment "vision tower warmup OOM" is correct. ⛔ The 10-30 rewrite saying "it is only an input-rejection gate; the tower is loaded anyway" was false. VISION.md §1·§2). API key. Overlay bind-mounts (:ro; the list is the `-v "$LANE_DIR/...` lines in launch.sh). Env knob passthrough (`B12X_*`, `VLLM_EXL3_*`, NCCL_PROTO/ALGO, etc.). With `NSYS=1`, wraps in nsys launch. If pointed at a model that is already abliterated, the `/ab` adapter is automatically turned off (prevents applying the edit twice).
- In-container paths: vllm = `/usr/local/lib/python3.12/dist-packages/vllm`, b12x = `.../dist-packages/b12x`. The whole b12x package is mounted directly from `overlay/b12x` (since 09-27). The kit's `$HOME/glm53-hotfix-b12x` hook code is left as in the original, so **if that symlink still exists, a double mount lands on the same spot and startup fails** (must be deleted when coming up from the 09-05 kit).
### 3.2 overlay/exl3.py — dense-layer FP8 Marlin (core, `overlay/exl3.py.diff`)
- Background: in the EXL3 checkpoint only the routed experts are 4bpw; the attention/KDA/MLA projections, shared expert, lm_head and dense MLP are BF16. In nsys measurements, 56% of a decode step is these BF16 GEMMs (17.6GiB read per step).
- Method: in `Exl3Config.get_quant_method`, send non-EXL3 `LinearBase`/`ParallelLMHead` to the vLLM fork's `Fp8PerBlockOnlineLinearMethod` (128x128 block FP8 e4m3 quantization at load), but force the kernel to `MarlinFP8ScaledMMLinearKernel` (`force_kernel`). Marlin = **FP8 weights only, BF16 activations (W8A16)**. On sm_121, Marlin requires `VLLM_TEST_FORCE_FP8_MARLIN=1`. Note that lm_head is a `ParallelLMHead`, so its attr names are embedding_dim/num_embeddings.
- Forbidden: the fork's default `B12xFp8BlockScaledMMKernel` (MXFP8 activation quantization, W8A8) causes confabulation (probe fake bugs 4~6/8) → do not use.
- Result: step 117→87ms, prose 25.3→36.
### 3.3 overlay/glm5next/model.py, kda.py
- Where the MLA/KDA projections forced `quant_config=None`, pass quant_config through when `VLLM_EXL3_DENSE_FP8=1`. (kda.py calls the parent __init__ via the `saved_quant_config` trick)
### 3.4 Prefill BF16 dual path (inside exl3.py)
- Problem: at large M (prefill M=8192) Marlin is 2.7× slower than cuBLAS BF16 (in_proj 31 vs 11ms) → prefill -12%.
- Method: in `_Fp8BlockOnlineMarlin.apply`, if `M > VLLM_EXL3_PREFILL_BF16_MIN_M(1024)`, **invert** the Marlin-packed FP8 back into BF16 and call `F.linear`. The inversion is the exact inverse of vLLM `marlin_utils_test.marlin_weights`: 16x16 tile reordering → 1024-unit `get_weight_perm(8)` permutation → 4-byte packing. Scales: inverse of the 64-unit `get_scale_perms` permutation + removal of `fp8_fused_exponent_bias` (bf16=2^120). One-pass Triton kernel (1024-byte chunk = 16×64 output tile). Largest layer 2.25ms, 0 resident memory. Standalone verification bit-exact on 5 shapes. lm_head excluded.
### 3.5 overlay/glm5next/mtp.py
- `eh_proj` nn.Linear (BF16 cuBLAS gemv 291µs×4/step) → `ReplicatedLinear(quant_config)`, the same Marlin FP8 path. The draft INT4 head proxy code is **disabled** (user rule: no precision cuts below FP8; no measured gain either).
### 3.5.1 Draft-only reduced-vocabulary lm_head (09-27, exl3.py + mtp.py)
- exl3.py `_build_subset_draft_head`: from the BF16 lm_head before FP8 packing, take only the subset rows (draft_vocab_32k.json) of this rank's shard and build an FP8 W8A16 (g128) Marlin head. **Only on the main model's lm_head** (prefix `lm_head`). V2 `load_eagle_model` (spec_decode/eagle/utils.py) swaps MTP's `shared_head.head` for the main model's lm_head, so the draft also uses this module. MTP's own head (`…layers.<N>.head`) is created without weights and then discarded — attaching there means the head is never used (actually stepped on 09-27, found via a warning log). The subset head is attached only as an attribute, so the verify path (quant_method) is unchanged. Padding rows are copies of real rows (0 rows → FP8 scale 0 → NaN).
- `_SubsetDraftHead.apply`: writes the compressed logits into a -inf tensor of this rank's shard width, so the existing vocab-parallel gather/argmax is used as is. The layer argument of apply is the proxy, so the output width is held by the head object.
- mtp.py `compute_logits` → `_draft_head()` proxy. **The V2 speculator samples drafts with compute_logits+argmax (not get_top_tokens)** — if not hooked here, the head is built but never used. If the env is on but there is no subset head, one warning.
- verify uses the main model's full-vocabulary head → output lossless. Probabilistic drafts are also sampled from the subset distribution, and the same logits go in as q, so lossless.

### 3.5.2 Routed-expert SwiGLU clamp (09-27, exl3.py + b12x mixed_trellis.py)
- `Exl3MoEMethod._swiglu_clamp_limit()`: if env `VLLM_EXL3_MOE_SWIGLU_CLAMP=1`, `self.moe.swiglu_limit` (GLM-5.3 = 10). Upstream (the image original) did not pass this value to any path.
- Destinations: b12x `Caps(swiglu_limit=…)` (actually used; kernel `_clamp_swiglu_inputs`: gate≤L, up∈[-L,L], FP32) / `compile_mixed_trellis(swiglu_limit=…)` → `W4A16FusedMoeKernel` / exllamav3 `exl3_moe(_fused)`'s only float argument `act_limit` (0.0=off) / clamp in the torch fallback path.
- Cross-check found no other omissions: routing scale, normalization, scores and bias are in the router; dense and shared are clamped by the vLLM MLP.

### 3.6 overlay/glm5next/attention.py
- Indexer fp32 `weights_proj`: cuBLAS picks a 1-block kernel (90~130µs) for [M,4096]x[4096,64] fp32 NN at M=5 → `(W_nk @ x.T).T` gives 19µs. `.contiguous()` is a no-op at M=1 (stride(0)=1), so DeepGEMM rejects it → `clone(memory_format=contiguous_format)` is mandatory.
### 3.7 overlay/b12x (`overlay/b12x.diff`)
- One env hook for CTAs per SM on the `moe/_shared/kernels/w4a16/kernel.py` m8 (TC-decode) path. Together with a tile override it gave decode -3% but prefill -10%, so **env left empty (disabled)**. The mount is kept (harmless).

### 3.8 temperature 0 determinism (09-29~30, README §4.9)
- All three causes are **upstream code** (not created by our patches), each an atomic deciding the summation order:
  ① b12x dense split-K bf16 atomicAdd → env `B12X_DENSE_SPLITK_TURBO=0`
  ② vLLM `sparse_utils.py` index compaction (COMPACT_TO_FRONT) tile slot claiming — GLM's row width 2112 falls outside upstream's power-of-two-only atomic-free path → pad to 4096 + column mask, row = one program
  ③ vLLM top-k (`persistent_topk`, `top_k_per_row_prefill`) — above pool 512 (=2,048 tokens), atomic ordering + first-come at boundary ties → `_canonicalize_topk_` (with the k-th value t: all >t + ==t from lowest index, the torch.topk rule)
- Flow: indexer scores → top-k (③) → logical→physical slot compaction (②) → Sparse MLA kernel (innocent). The kernel adds in the order of the list it receives.

## 4. Measurement ledger and experiment records
README.md §4~§5 and memory `project_glm53_decode_kernel_truth`, `project_glm53_dense_fp8_overlay`. Key points: decode is byte-bound (MoE kernel at 73% of ceiling at M=1, verify near ceiling, Marlin 83%). Kernel upper bound ≈ -12ms (prose 42 / code 50). The only route to +10 t/s is the drafter's Korean acceptance rate (MTP head self-distillation, FastMTP recipe). Abandoned: b12x W8A8, INT4 head, split-K flag, tile override, NCCL LL, DFlash2 (prose), MTP k sweep, router ll_bf16 dotprod (step +2~3ms), verify top-p Triton (noise), disabling CPU C-states (no effect).
Additionally adopted 09-27: router dedup, 32K draft vocabulary (step -3ms), SwiGLU clamp (correctness). Remaining levers: 32K head TP rank balance (~0.8ms), draft graph start gap (~1.5ms, cause unknown), drafter acceptance rate (fine-tuning).

## 5. Traps (do not step on these again)
- **temp 0 determinism regression check**: when changing the image or overlays, do not look only at short outputs — ③ only shows up past 2,048 tokens of context. Confirm identical hashes over 4 runs of a thinking-on code request (2,200-token output).
- **Only sorting** top-k is not the answer (with boundary ties the set changes). Do not commit debugging probes — when done, prove with `git diff <production base commit>` that only the production version + fixes remain.
- The draft head replacement must be done in MTP `compute_logits` (V2). Hooking only get_top_tokens is silently ineffective (09-27 first measurement: step change 0). Attach the subset head to the main model's lm_head (MTP shared_head.head is replaced by the main model's lm_head at load).
- When replacing the image: mtp.py `_DraftHeadProxy` overrides only tp_size/shard_indices/quant_method and delegates the rest to the original lm_head. If the attributes the upstream logits_processor reads from the head change, delegation absorbs most of it, but if the quant_method.apply signature changes, adjust `_SubsetDraftHead.apply`. Confirm in the startup log 1 line of 'subset draft lm_head on …lm_head' + 0 warning lines.
- Do not estimate savings from nsys kernel totals: with multi-stream and PDL they do not match the real step (router kernel 44→23us yet step +2~3ms). Measure the step only.
- 2-node: RESTARTALL only. Worker alone Up = zombie. Using a self-made restart script for experiments means memwatch does not start → use only the wrapper.
- Environment variables do not pass over ssh. Put knobs in the env file, register them in the launch.sh passthrough list, and rsync+md5 to both nodes.
- `pkill -f <script name>` kills your own shell (exit 144). Kill via a pid file / `$!`.
- Editing a mounted overlay file while serving is seen by the container immediately (from the next import). Edit between restarts.
- `.contiguous()` no-op trap (M=1); DeepGEMM requires stride(-1)==1·stride(0)==H.
- A Marlin-packed lm_head conflicts with DFlash `quantize_draft_head` → `VLLM_DFLASH_FP8_DRAFT_HEAD=0`.
- Standalone GPU tests only after STOPALL (unified-memory OOM = the whole box). Mounting the JIT cache (~/glm53-vllm-cache/jit/triton) into a test container is harmless, and there are no cache key collisions.
- nsys: capture with the container's 2025.3.2; the host's default nsys (2024.6) cannot read the report → `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys stats`.
- Measure cuBLAS skinny GEMMs under "cold" conditions (L2-resident benchmarks are 3× optimistic).
- The kit's auto-update check (launch.sh kit_update_check) is harmless but can be turned off with `GLM53_NO_UPDATE_CHECK=1`.
### 3.9 '!' collapse — prefill tail replayed as a decode graph (10-02, `overlay/v2runner/`)
- Symptom: token 0 ('!') repeated in the reasoning channel. 100% reproducible when **prompt length ≡ 5 (mod 512)** (independent of content and cache).
- Cause: when the prefill chunk is cut at a 512 boundary and the last piece is exactly 1+MTP (=5) tokens, the V2 runner looks only at the batch shape
  (`get_uniform_token_count`) and replays the spec-verify FULL cudagraph. That graph does not build the prompt's KDA recurrent state, so
  logits are NaN → the sampler picks 0 and all subsequent decode is NaN too.
- Fix: backport of upstream vllm#51865 (08-12, after the fork base so it was missing) — if any request is still prefilling, it is not uniform decode.
  Runner + 2 speculators (`model_runner.py`, `spec_ar/`, `spec_mmtp/`). Regression check `tools/prefill_tail_nan_check.py`.
- Debugging lesson: if per-layer NaN probes are silent but only the output layer's input is NaN, that step is a **graph replay** (Python not executed).
  If you request logprobs, the server returns HTTP 400 "Out of range float values are not JSON compliant: nan" on NaN — the cheapest detector.

### 3.10 kpool tail cache stride (10-02, `overlay/glm5next/ops/kpool_compress.py`)
- Backport of upstream vllm#57477. The prefill tail-seed kernel wrote the tail cache with the dense stride (1024), so instead of the slot at the real padded
  stride (76032) it overwrote **another request's indexer cache block**. Check `tools/kpool_cache_probe.py` (the first-token logprob of the same
  cache prefix must not change after another request is processed).

### 3.11 `/ab` runtime switch — `AB-SWITCH.md`. ### 3.12 UTF-8 guard — `UTF8-GUARD.md`. ### 3.13 Image input — `VISION.md`.

### 3.14 NVMe KV offload (11-26~27, branch `exp/glm53-kv-offload` → merged into main 11-27) — `KV_OFFLOAD.md`, verification `KV_OFFLOAD_TESTPLAN.md`
- Upstream `OffloadingConnector` + our own `NvmeOffloadingSpec` (`overlay/kvoffload/glm53_nvme.py`). No CPU tier (meaningless with unified memory), GPU→pinned bounce→O_DIRECT, fixed-size slots, per-rank preallocated files, verified per rank via slot headers (crc, fingerprint).
- Upstream connector changes (`overlay/kvconn/*.diff`): failure propagation (recompute instead of assert), exclude the kpool tail (scratch) group, handle hybrid invalid blocks.
- Eviction = `SegmentLRUPolicy` (upstream LRU missed KDA group boundaries and the tail became newer than the head, so restoring a reused conversation hit 0).
- Traps: ① if the seed is not fixed, NONE_HASH comes from urandom at every boot → 0 hits after restart (`overlay/v1core/kv_cache_utils.py`) ② same key keeps serving old contents → rotate `KV_OFFLOAD_HASH_SEED` when changing kernels/numerics ③ the slot layout is part of the fingerprint, so both ranks need the same values ④ prefill split points are on the 4608 grid and 512 units (= multiples of KDA's 64), so cache provenance does not matter (stage 2) — if `--mixed-prefill-token-cap` or `--prefix-match-unit` is changed to an off-grid value, re-verify.
- When exposed to multiple users: the cache has no direct leak/corruption path, but TTFT timing can reveal 'who sent this prefix' (persists across restarts). The gateway must enforce a random per-session `cache_salt` (`KV_OFFLOAD.md` 'Isolation').

### 3.15 Temp-0 concurrent-batch nondeterminism (11-26~27, branch) — `BATCH_INVARIANCE.md`
- Kernel configuration changes with row count M, presence of prefill, and (cuBLAS) view position → 1-bit difference → amplified by the KDA recurrence. Fix = fixed configuration per row role (decode/prefill) at 7 sites; the role is passed by the runner as a host-side value; during graph capture everything is decode. `GLM53_BATCH_INVARIANT=1`. Speed: decode same, prefill −1.5%. ⛔ No cuBLAS for small GEMMs (view-position dependent).
