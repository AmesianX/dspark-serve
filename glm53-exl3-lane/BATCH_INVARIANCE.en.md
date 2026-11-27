[한국어](BATCH_INVARIANCE.md) | **English**

# Temperature-0 determinism — the bug where output changes under concurrent batching (batch invariance)

Branch `exp/glm53-kv-offload` (merged into main 11-27). **The bug was already present on main too.** Started 2026-11-26 → **resolved 2026-11-27**.

## ✅ Summary of results (2026-11-27, production settings, CUDA graphs on)
- One production switch: env `GLM53_BATCH_INVARIANT=1` → launch.sh expands it into `GLM53_BI_{DENSE,MHC,MLA,MOE,ROUTER,TOPK,INDEXER}=1` and passes it to both nodes.
- Determinism (output text fully identical): co-prefill e13 3/3 (⛔ correction 11-27: the prefix had been warmed beforehand, so only the tail prefill was shared — cold co-prefill was verified by the stage-2 experiment) · ABCD concurrent e9 8/8 · e7 alone-repeat / cold prefill during decode 3/3 / ABCD concurrent 8/8 — all pass.
- Speed (A/B within the same session): decode 10 prompts ×2 average t/s **36.6 → 36.7 (same)**, step +0.7ms (mostly expert-mix differences because the output sentences differ — varies 0 to +3ms per prompt),
  82K prefill 1,356~1,373 → **1,340~1,348 t/s (−1.5%)**.
- If you change kernels/numerics again, rotate `KV_OFFLOAD_HASH_SEED` (currently `glm53-kv-offload-bi-1`). NVMe keeps serving old contents for the same key.

### Final solution table
| Site | Cause | Fix |
|---|---|---|
| dense FP8 Marlin | Only M≤16 calls are row-invariant; M>1024 goes to cuBLAS | Decode rows: Marlin in ≤16-row units (single stream = the existing single call); prefill rows: persistent matmul (dequant BF16). lm_head and >16 rows of unknown role: split into 16-row chunks |
| mHC (vLLM TileLang) | split/path changes at M<8 / 8~16 / >16 | Decode rows = fixed small M<8 config; prefill rows = large path fixed at split 1 (`overlay/mhc_tilelang.py`) |
| MLA sparse attention | If prefill is mixed in, the whole step runs extend | In mixed steps, decode rows = decode kernel, prefill rows = separate extend call |
| MoE trellis | m≤32 trellis / m>32 prefill plan | Decode rows: trellis in 32-row units; prefill rows: always the prefill plan |
| MoE router GateLinear | GB10 has no dedicated kernel → everything is cuBLAS, dependent on M and **view position** | Decode rows: fixed-config Triton (`exl3.bi_linear`); prefill rows: persistent |
| grouped_topk | Implementation switches at M>1024 | Call in 1024-row units (+clone) |
| Indexer wk·kpool gate (bf16), head weights (fp32) | cuBLAS dependent on M and view position → **stored in the K cache**, so it blows up later | bf16: decode bi_linear / prefill persistent. fp32: decode `bi_rowdot_fp32` (10µs, originally 37µs) / prefill tf32x3 |
| Role determination | — | Computed from the model runner's host-side `is_prefilling_np` (0 syncs). **During CUDA graph capture, all rows = decode** |

### Rejected alternatives
- Disabling cuBLASLt split-K (the Hopper/Blackwell approach of vLLM's batch-invariant mode, `CUBLASLT_WORKSPACE_SIZE=1`): bf16 small-GEMM decode 2× slower (12→25µs), router view-position dependence remains (`tools/bi/cublaslt_bi.py`).
- cuBLAS fixed-shape calls (padding to 16/1024 rows): inside the server, still differs by view position (pointer alignment) (bi29).
- A "one prefill per step" scheduling restriction: became unnecessary.

## 0. Symptom and principles

- Repeating the same request **alone** at temperature 0 is bit-identical every time. Running it **concurrently with other requests** changes the output.
- Principle (user): temperature 0 must always give the same output. Otherwise it is a "random output machine". Unacceptable for a service.
- Constraint (user): **single-stream decode and prefill speed must not be cut.** Keep prefill at 1,300+ t/s (baseline 1,377~1,381).
  A loss like 1,300 → 800 just to get determinism is "insane".

## 1. Root cause in one line

Several kernels choose their split scheme/kernel based on **"the total row count (M) of this step" or "whether prefill is mixed into this step"**.
M is determined by whichever other users happen to be running at that moment → the fp32 summation order for the same token changes → last-bit difference
→ **the KDA recurrent state keeps carrying that 1 bit and amplifies it** → a few dozen tokens later the greedy choice changes.
Not a quality problem (both paths are mathematically the same computation). Purely a determinism problem.

Fix principle: choose the kernel **not by row count but by the row's own role (decode row / prefill row)**, with a **fixed configuration** per role.
Same direction as the industry's batch-invariant inference. No new kernels are written; the existing fast kernels are called at fixed sizes.

## 2. List of culprits (in order of confirmation)

| # | Site | What changes with M / batch composition | Evidence | Fixing switch |
|---|---|---|---|---|
| 1 | dense GEMM: FP8 Marlin W8A16 (`overlay/exl3.py` `_Fp8BlockOnlineMarlin.apply`) | Marlin's internal blocking / K split depends on M. **For M≤16 calls the per-row result equals M=1**; 17+ differs. M>1024 is BF16 cuBLAS (a separate kernel) | `tools/bi/marlin_bi*.py` | `GLM53_BI_DENSE_MARLIN16` (diagnostic) |
| 2 | mHC: **vLLM TileLang** `kernels/mhc/tilelang.py` | M<8: split 8·tile 2 / 8~16: split 4·tile 3 / >16: deep_gemm TF32+`compute_num_split(M)` | Layer 0 attn_out identical, mlp_in differs | `GLM53_BI_MHC` |
| 3 | MLA sparse attention `b12x_mla_sparse.py` | **If prefill is mixed in, the whole step goes to the extend kernel** (decode-only uses the split-K decode kernel) | Layer 3 attn_out 7.8e-3, only in mixed steps | `GLM53_BI_MLA` (decode rows = decode kernel, prefill rows = extend) |
| 4 | MoE trellis (`exl3.py _apply_rank_sliced`) | m≤32 trellis plan / m>32 prefill plan | (confirmed after fixing the router) | `GLM53_BI_MOE` (decode rows = trellis in 32-row units, prefill rows = prefill plan) |
| 5 | **MoE router gate** GateLinear | **GB10 (sm_121) is outside the dedicated router kernel's condition (Hopper/SM100) → every M is cuBLAS F.linear**. cuBLAS is identical among M=2~16, differs at M=1 and 17+ | MoE internal probe 3a_router | `GLM53_BI_ROUTER` (decode rows cuBLAS 2~16 rows, prefill rows persistent) |
| 6 | `ops.grouped_topk` (routing top-k) | **Implementation switches at M>1024**. ≤1024 is per-row identical to M=1. ⚠️ It may reuse the output buffer, so clone is mandatory when calling in chunks | `tools/bi/topk_bi*.py` | `GLM53_BI_TOPK` (1024-row units) |
| 7 | **Indexer bf16 cuBLAS** (`attention.py` `wk_weights_proj`, `index_kpool_compress_gate`) | Results change from M=4. The result is **stored in the indexer K cache**, so the token choice changes in decode steps after the mixed step has ended (a late-appearing divergence) | Layer 3 attn_out at the M=5 step after mixing | `GLM53_BI_INDEXER` (`exl3.bi_fixed_rows`: fixed shapes of 16 rows for decode, 1024 rows for prefill) |
| - | Indexer fp32 head-weight GEMM | Innocent (identical for all M) | `idx_bi.py` | Not needed |
| - | KDA decode recurrence `fused_recurrent_kda`, FlashKDA prefill | Innocent. Splitting FlashKDA per sequence gives identical results (→ `GLM53_BI_KDA` has no effect) | bi20 | (not needed) |
| ? | lm_head, MTP draft | Never surfaced as a problem in end-to-end tests | | |

**Trap: b12x also has an mHC implementation (`overlay/b12x/norm/mhc/`) and `B12X_MHC_*` environment variables, but this model uses vLLM TileLang mHC.**

### 2b. When the difference is not the kernel but the "cache provenance" (found 2026-11-27)

Tracing why only B, C, D were wrong in the ABCD concurrent test: the **KDA initial state** read at the same position (147,968) was **a different version on each run**
(probe `1b_c3_init_state` checksum differs by 9e-4). 4×149K does not all fit in the GPU cache, so part of it is **restored from NVMe**, and
NVMe holds **state computed under an older configuration (pre-fix kernels, different prefill boundaries)**; since an existing key is not saved again,
the old version keeps being used. → **If you change the numeric path (kernels), you must also change the NVMe key space (`KV_OFFLOAD_HASH_SEED`).**
More fundamentally, even for the same prefix, the state bits differ depending on "which chunk boundaries it was computed with" (§7-4).

## 3. Diagnostic switches (all off by default; only via the env file + sync-lane-67.sh — command-line env does not reach the worker)

| Switch | Effect |
|---|---|
| `GLM53_BI_PROBE=/cache/bi` + `GLM53_BI_PROBE_MINPOS=<pos>` | Dumps per-layer/sub-site activations for rank 0 rows with pos≥MINPOS. Key = (pos, input embedding). Records `m_total` |
| `GLM53_BI_DEBUG=1` | `[BI-DBG]` log: role (d,m), per-request qlens·seq_lens·prefilling for prefill steps |
| `GLM53_BI_DENSE_PREFILL` / `GLM53_BI_DENSE_MARLIN16` | dense: persistent for M>1024 / Marlin 16-row for all rows (diagnostic, prefill very slow) |
| `GLM53_BI_MHC` `GLM53_BI_MLA` `GLM53_BI_MOE` `GLM53_BI_ROUTER` `GLM53_BI_TOPK` `GLM53_BI_INDEXER` `GLM53_BI_KDA` | See the §2 table |

Probe sites: layer `1_attn_in` `2_attn_out` `3_mlp_in` `4_mlp_out`, KDA `1b_a_in_proj` `1b_b_f_b_proj` `1b_c2_conv` `1b_c3_init_state`
`1b_c_g_b_proj` `1b_d_recurrent` `1b_e_o_norm` `1b_f_o_proj`, MoE `3a_router` `3b_topk_w` `3c_topk_id` `3d_routed` `3e_shared`.

Role propagation: the MLA metadata builder uses `is_prefilling` to compute the number of "rows of requests that have finished their prompt (the leading block)" (`bi_decode_tokens`)
→ model forward calls `exl3.bi_set_role(d, m)` → MoE/router/indexer read it. The criterion is is_prefilling, not query length.

## 4. Reproduction procedure (tools: `tools/bi/`)

```bash
cd tools/bi && export K=<VLLM_API_KEY>
# Restart with the probe on (EAGER=1 required — put it in the env file temporarily)
GLM53_BI_PROBE=/cache/bi GLM53_BI_PROBE_MINPOS=148962 ../../vllm-glm53-exl3.sh RESTARTALL
docker exec vllm_glm53 rm -rf /cache/bi /cache/bi_alone /cache/bi_batch
python3 bi_run.py warm                       # Put X (148,962 tokens) into the GPU prefix cache (outside the probe range)
docker exec vllm_glm53 rm -rf /cache/bi
python3 bi_run.py alone 60                   # X alone, 60 tokens
docker exec vllm_glm53 bash -c 'mkdir -p /cache/bi_alone && mv /cache/bi/* /cache/bi_alone/'
python3 bi_run.py mix                        # 2 cold prompts' prefill arrive while X is decoding (or batch: X joins while 3 others decode)
docker exec vllm_glm53 bash -c 'mkdir -p /cache/bi_batch && mv /cache/bi/* /cache/bi_batch/'
docker exec -i vllm_glm53 python3 - /cache/bi_alone /cache/bi_batch < bi_compare.py     # Summary
docker exec -i vllm_glm53 python3 - /cache/bi_alone /cache/bi_batch < bi_detail_all.py  # First divergence per row + m_total
```

End-to-end (output text) test: `python3 e7_det.py 200` — alone-repeat / cold prefill during decode ×3 / ABCD concurrent ×2.
Kernel-only test (server down, one-GPU-process rule): `docker run --rm --gpus all ... python3 /w/marlin_bi2.py`.
`p60k.txt` = the prompt text (vLLM source, ~82K tokens). Determinism tests require **the same prompt**, so it is kept in the repo.

## 5. Measurement log

- Marlin (N=12800, K=4096) GEMM ms: M=5/12/16 → 0.24 (16-row split also 0.24, same) / M=20·32 → 0.24 vs split 0.47~0.49 /
  M=994 → Marlin 3.79, 16-split 15.1, persistent 2.63, cuBLAS 1.51. persistent is row-invariant for all of M=1~2048.
- 82K cold prefill: baseline (cuBLAS) 1,381/1,377 t/s → dense persistent 1,339/1,362/1,364 (−1.5%). `logs/prefill_ab_dense_bi.out`
- Probe progression (`logs/bi*.out`):
  - bi3/bi4: tail prefill rows diverge at the layer-0 GEMM (same with PREFILL_BF16 off → Marlin itself)
  - bi6: that forward pass's m_total 994 alone / 1003 concurrent → both Marlin, only M differs = Marlin M-dependence confirmed
  - bi7: switch did not reach the worker (trap below) → on the rank 0 side, identical up through d_recurrent
  - bi8: dense fixed on both nodes → the remaining divergence is layer 0 mlp_in (= mHC)
  - bi9: b12x mHC knob (no effect)
  - **bi10: dense+mHC fixed → 97/97 rows bit-identical through the final norm** (scenario: other requests decoding during X's tail prefill)
  - e7 (dense+mHC): alone-repeat ✅, cold prefill during decode ✗×3, ABCD concurrent ✗×2
  - bi11 (mix): 30 rows identical before the cold arrival; from the mixed step, layer 3 attn_out diverges = MLA kernel switch
  - bi12: MLA separated → layer 3 attn_out identical, moves to layer 3 mlp_out (MoE)
  - bi13: separating MoE alone is not invariant (role propagation confirmed correct via debug) → added MoE internal probes
  - bi15: first divergence = `3a_router` (router logits). The ll_bf16 path assumption was wrong (GB10 is cuBLAS only)
  - bi16: router fixed → `3b_topk_w` (3e-7) = grouped_topk M>1024 switch
  - bi17: top-k fixed → all rows of the mixed step identical. After mixing ends, layer 3 attn_out at the M=5 decode = indexer cache key
  - **bi18: indexer fixed → mix scenario 93/93 identical through the final norm**
  - e7 (7 sites fixed): alone-repeat ✅, **cold prefill during decode 3/3 ✅**, ABCD concurrent only A ✅
  - e8 (A+B pair): 3/3 ✅ / e9 (ABCD + qlens log): only the request that prefilled its tail alone is correct
  - bi19 (co: A co-prefilling with B·D): layer 0 d_recurrent / bi20: identical even with FlashKDA called per sequence (innocent)
  - bi21: `1b_c3_init_state` checksum differs → the KDA initial state read from cache is itself a different version (§2b)
- Router microbenchmark (4096→288): cuBLAS M=5 12.4µs / persistent 101µs (unsuitable for decode) / M=9236 cuBLAS 454 vs persistent 667µs
- Indexer microbenchmark: wk_proj·gate bf16 M=5·16 both ~12µs, M=1024 ~40µs, M=9236 ~400µs (10 fixed-1024 calls ≈ a single call)

## 6. Collected traps (do not step on these again)

1. **Command-line env does not reach the worker (.67).** `VAR=1 ./vllm-glm53-exl3.sh RESTARTALL` affects the head only. With TP2, the worker-side
   difference gets mixed into the o_proj sum and leads to wrong conclusions. Switches needed on both nodes go in the **env file + sync-lane-67.sh**. Check:
   `ssh 10.0.1.2 docker exec vllm_glm53 printenv <VAR>`.
2. The probe key is (pos, embedding). If a draft is rejected and the same (pos,token) is computed twice, you get `occ=2` — comparison invalid, ignore.
3. The vision lane has `input_ids=None` (inputs_embeds) → identify rows by embedding.
4. The EXL3 scope in the model config is `glm53_routed_experts_only` — only routed experts are trellis; **the rest of dense is the FP8 Marlin overlay**.
5. Single-GPU rule: kernel-only tests only after server STOPALL.
6. "Identical when repeated alone" is not evidence of determinism (same M always gives the same result). Always compare with a changed batch composition.
7. If there are uncommitted changes, `RESTARTALL` prints "⛔ 기동 중단: 레인 불일치" ("startup aborted: lane mismatch") and **skips the restart, so the old server keeps running**.
   Copying log files also needs a commit. Make a habit of checking switches with `printenv` after restart.
8. The debug log budget (`_bi_dbg` 120 lines) is used up first by the startup warmup dummy run (m=9236) and pure prefill steps → filtering is mandatory.
9. vLLM `GateLinear`'s dedicated ll_bf16/dsv3/fp32 paths are Hopper/SM100 only. GB10 (sm_121) is all cuBLAS.
10. Key collisions: probing several prompts with the same body and same length concurrently makes the (pos, embedding) keys overlap. Using prompts whose header lengths differ so positions are offset
    (E5-A/B/D: 148962/148955/148957) keeps them distinguishable.
11. The NVMe offload cache keeps using old contents when the key is the same. Isolate experiments that change the numeric path by changing `KV_OFFLOAD_HASH_SEED`.

## 6-b. ⚠️ Gap: concurrent prefill of short identical prompts (found 2026-11-27, unresolved)

Found during /ab isolation verification. All 7 `GLM53_BI_*` switches were on on both nodes.

| Condition | Result |
|---|---|
| 3 copies of the same ~30-token prompt sent concurrently | 2/3 differ **from the first token** (max\|Δlp\| 1.364, text also differs) |
| 2 copies of the same prompt concurrently | 2/2 differ |
| Another request cuts in while 2 neighbor requests are decoding (0.4s·2.0s intervals) | All 4 runs: neighbors' 160-token logprobs bit-identical |

This does not contradict stage 2 (17K prompt, 4608-aligned chunks) passing co-prefill — the case of several short prompts
being prefilled together in one step was never tested. Reproduce: `tools/bi/ab_iso_probe.py diff`, `plain3`.
Cause not investigated (candidates: KDA chunk boundaries when there are several short varlen prefill rows, or the MLA/indexer prefill path).

## 6-c. ✅ Fixed: the indexer top-k picking the wrong context in long-context decode (found and fixed 2026-11-27)

**Symptom.** Above 18K tokens, two temperature-0 cold runs (no concurrent requests, cache irrelevant) diverged at token 4–9.
22.7K identical; 38K, 40K, 41K diverged. Found during the 11-27 final verification — and at first hidden by shrinking the test to 18K
(a mistake; see the lesson below).

**Trace (turn things off → position → row → kernel).**
1. Batch invariance OFF / NVMe offload OFF (connector and salt code included) / `/ab` hooks OFF / the new cache overlays (block_pool,
   kv_cache_manager) removed — all four still reproduced it. Unrelated to that day's changes.
2. `prompt_logprobs` over all 38,264 positions, two runs → **identical**. Prefill is deterministic; the split happens in **decode**.
3. BI probe (`GLM53_BI_PROBE`, EAGER): the two runs had the same decode steps and even the same MTP draft tokens, yet **rows 1–2 of the
   first decode step (a 5-row MTP verify) differed in layer 7's attention output** (inputs bit-identical).
4. Decode indexer input/output dump (diagnostic only, removed afterwards): logits within the valid length identical, but `pool_topk`
   differed in rows 1–2 and **overlapped the true top-512 in only 432–451/512**. Row 2 was out of order and missing the top-scoring pool (#0, +93.3).

**Cause (two layers).**
- The fast decode top-k (`persistent_topk`) returns a non-top-512 set for some MTP verify rows (a −130.1 pool against a true cut of
  −129.8). Columns past the valid length hold stale buffer values (+1…+30) because the logits are produced with `clean_logits=False`.
- The canonicalisation kernel (`_canonical_topk_kernel`) **trusted that set's minimum as the threshold.** A too-low threshold makes
  "above threshold" exceed K, and **the writes had no bound, so they overwrote the next row's K slots** — row 2's broken order and
  missing top pool are that overwrite.
- Result: in long contexts some rows decoded with **the wrong context (up to ~80 of 512 pools)**, and since the stale bytes differ run
  to run, output was nondeterministic. It is an answer-quality bug before it is a determinism bug.

**Fix (`overlay/sparse_attn_indexer_kpool.py`).**
- The canonicalisation kernel **verifies** the threshold: it is the exact K-th value iff `n_above < K ≤ n_above + n_at`. Only rows that
  fail it find the exact K-th value with a **32-step bisection** over an order-preserving int32 key of the float — unique, hence correct
  and identical run to run. Rows that pass cost nothing extra.
- A `pos < K` bound on the writes — no more spilling into the next row.
- Logits past each row's length are set to −inf before top-k (`_mask_logits_past_row_len_`), on both decode top-k paths.

**Verification.**
- After the fix: all 840 dumped rows match the true top-512; the two runs are identical in 84/84 step-layers.
- Production settings (CUDA graphs on): 18K, 38K, 78K cold ×3 bit-identical. 44.7K F1 (hit), F2 (concurrent) and F5 (NVMe restore
  after restart, 7.45s) bit-identical (KV_OFFLOAD_TESTPLAN.en.md §6).
- Speed: kmeas 34.9 / 39.0 / 44.8 t/s (unchanged). 38K-context decode 34.0 t/s (a run alone; not an A/B against the old code).

**Lesson.** Test determinism on **logprobs**, not just output text, and **beyond 18K tokens**. This bug sat between the BI work's
text-only comparison and verification that never went past 18K. Dropping the failing length from a test hides the defect as a "pass".

## 7. Remaining work (as of 2026-11-27, items 1~5 all done)

1. (In progress) ABCD end-to-end test with a new hash seed → decide whether kernel-level batch invariance is complete.
2. Final implementation (diagnostic switches → production path): role-based branching, in a form with no speed loss
   - dense: decode rows → Marlin in 16-row units (up to 5 concurrent users is ≤16 rows, so same kernel and speed as now), prefill rows → persistent matmul (−1.5%)
   - mHC: decode rows → fixed M<8 config, prefill rows → decide after measuring cost
   - MLA·MoE·router·top-k·indexer: clean up the current diagnostic implementation into the production path
   - Confirm it works under CUDA graphs (decode-only steps)
3. Verify under production settings: kmeas single decode (≈34.2/38.3/41.2), prefill 1,300+, all e7·e9 items identical.
4. ✅ Stage 2 (cache provenance) passed 2026-11-27 — no fix needed. Under load with 8 cold prefills, cache continuation, fully cold, and partial hits at a branch point — all match.
   Reason: the align-mode scheduler aligns chunk ends to the state grid (4608) and hits to multiples of prefix-match-unit (512), so every split point is a multiple of the KDA chunk (64).
   Procedure and conditions: `KV_OFFLOAD_TESTPLAN.md` §3, script `tools/bi/s2_prov.py`.
5. When finished, remove every `[BI 진단 임시 — 지울 것]` ("BI diagnostic, temporary — delete") line from the env file (including EAGER and the hash seed), and restart with production settings.
