[한국어](VISION.md) | **English**

# Vision (image) input — how the encoder budget grows, and how it is capped

Measurement environment: .66/.67 2-node TP2, `IMAGE=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`,
`MNBT=9236`, `MAX_LEN=524288`, `KV_CACHE_MEMORY=12e9`, `GMU=0.85`.
Model `~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw`.

## 0. Read before any image work — do not repeat the same dead ends (as of 2026-11-01)

If you are going to measure or change anything image-related, read **this section first**. Everything below is either measured or verified in code.

### Live now (main)

`VISION=1`, **`VISION_MAX_IMAGES=5`**, `VISION_MAX_TOKENS=4096` (one 1792² image), `VISION_PIXEL_DTYPE=bf16`,
`VISION_LAZY_DECODE=1`, `VISION_MAX_IMAGE_PIXELS=8294400` (4K UHD). memwatch bar 3.0GB,
head baseline 5.3~6.0GB → **usable headroom 2.3~3.0GB**.

### Safe range (measured per image count, §15)

| Images | Drop | Window | 3 concurrent | Verdict |
|---|---|---|---|---|
| 1 | **0.00GB** | 0.0s | ~0.0GB | Nothing to worry about |
| 2 | 0.44GB | 26.2s | ~1.3GB | Safe |
| 5 | 0.78GB | 37.3s | **2.06GB measured, 3/3 succeeded** | The limit. Safe |
| 8 | 1.00~1.26GB | 31.3s | 3.06GB | **memwatch stop** (this is why the limit was lowered to 5) |

### Already tried and failed — do not do these again

| Attempt | Result | Where |
|---|---|---|
| Early release of `.data` (pixel_values), after KV store | **Zero effect** (drop 1.02→1.04GB) | §15, tag `glm53-mm-data-release-failed` |
| Early release of `.data`, right after encoding + worker copy | **0.06GB** (1.02→0.96GB). Not merged into main | Same branch |
| Lowering `VISION_MAX_IMAGES` to 4 | Breaks the 5-image requests the user actually sends with a 400. Wrong proposal | §15 |
| Pinning `MALLOC_MMAP_THRESHOLD_` | No effect after one-image-at-a-time processing → knob deleted | §10, commit `168416f` |
| Limiting concurrency with a semaphore/queue | The window is 31 s (97% of it prefill), so worst-case wait cost is 62 s. A limit of 5 is 4× cheaper | §15 |

**71% of the drop cannot be touched.** Of the 1.03GB drop for one 8-image request, `AnonPages` accounts for only **+0.29GB**;
the remaining **0.74GB is outside process anon** (presumed CUDA staging/pinned, unidentified). Process RSS does
**not rise at all** during the request (APIServer 2.60 / EngineCore 1.87, fixed). This means any approach that drops
references on the Python side has a ceiling of 0.29GB. Go in without knowing this and you will chase ghosts again.

### If you measure — contamination conditions (these are why 39 s was corrected to 31 s)

1. **60 s cooldown between runs.** 30 s is not enough — residue from the previous run raises the baseline and stretches the window to 42.7 s.
2. **Recovery sampling of 18 s or more.** With 6 s, the end of the window was clipped at the measurement end time and misread as 38.9 s.
3. **Discard the first measurement right after startup.** On the first image, ~3.3GiB of ViT workspace gets pinned and the drop reads as 3.4GB.
4. **No other commands (`docker exec`, `ps`, etc.) during measurement.** They are themselves a contamination source. Obtain PIDs beforehand.
5. A **new image** each run (keep prefix and mm cache misses). The same image skips encoding.
6. Acceptance criterion: only runs with baseline jitter ≤0.10GB + recovery to at least baseline −0.25GB.
7. **No offline preprocessing measurements on the head while serving** (memwatch takes the live server down).

### Easy to misread

- Do not look at `--chat-template /lane/chat_template.jinja` in `docker inspect` and conclude "it's the text template".
  **The vision template is mounted under that name** (`launch.sh:624`). Check with md5.
- "3 concurrent stop" in §13 means **3 concurrent 8-image requests**. It is a different axis from the image count within one request.
- Given 8 images and asked "how many?", the model answers 6~7 (even at temp 0). It is a counting weakness; listing the contents is accurate.
- The `/ab` default is **0 (OFF)**. There is no value called `auto`. Do not confuse it with `mode` showing as `original|abliterated`.

### Still unknown (start here if you want to touch it)

1. **What the 0.74GB outside anon is** — per request. Solving this would raise the concurrency limit fundamentally.
2. **17~18 s release delay after request end for 2 and 5 images** — not seen with 8 images. Unrelated to `.data`.
3. **Parallel subagents** — the only path to 3 concurrent requests that operating habits cannot prevent.

## 1. The model is multimodal. The vision tower is loaded only when VISION=1

- `config.json`: `Glm5NextForConditionalGeneration`, `vision_config` = ViT depth 24,
  hidden 1024, patch 14, image 448, merge 2, `out_hidden_size` 4096,
  `projection_intermediate_size` 10240.
- Weights: `model.visual.*` **347 tensors / 1.127 GB BF16** (tallied from `model.safetensors.index.json`).
  `mm_encoder_tp_mode` defaults to `"weights"` → split across TP ranks (~563 MB per node).
- `overlay/glm5next/model.py:1118` builds the tower inside `self._mark_tower_model(vllm_config, {"image","video"})`.
  This context (`model_executor/models/interfaces.py`) **replaces the tower with `StageMissingLayer` and does not load
  its weights when the limits for those modalities are all 0** — docstring:
  "Tower model components are automatically skipped when `--limit-mm-per-prompt` is set to zero
  for all of their modalities."

> ⛔ **Correction (2026-10-30)**: The original content of this section ("the tower is always built; the memory saved by
> `--language-model-only` is 0") was **false**. I looked only at the `self.visual = ...` assignment and did not read the enclosing context.
> The startup log line `Using FLASH_ATTN for MMEncoderAttention` is just a module-construction log, not evidence that weights were loaded.

## 2. `--language-model-only` does prevent OOM (the original comment was right)

`vllm/config/multimodal.py`: "If True, disables all multimodal inputs by setting all modality
limits to 0." When all limits are 0, two things happen together:

1. Per §1 above, the vision tower weights are not loaded.
2. The startup dummy MM profile is skipped — in `multimodal/encoder_budget.py`, `tower_modalities`
   (those with limit > 0) is empty, so `mm_max_toks_per_item` is empty, and `get_dummy_encoder_profile_inputs`
   returns an empty list (assuming `enable_mm_embeds=False`).

`--skip-mm-profiling` is a separate flag that does only #2. Having both attached at VISION=0 was redundant,
not wrong. The old comment "vision tower warmup causes unified-memory OOM → `--language-model-only`" is correct.

> ⛔ **Correction (2026-10-30)**: The original title and content of this section ("`--language-model-only` is not an OOM guard,
> it is purely an input-rejection gate; what prevented OOM was `--skip-mm-profiling`") were **false**.

Side check: enabling multimodal does not break the attention backend. `b12x_mla_sparse.py` does not declare
`supports_mm_prefix()`, so it gets the base default `False`, but `glm5_next` is not in the
`MM_PREFIX_LM_MODELS` list and `config.json` has no `is_mm_prefix_lm` key either
(`transformers_utils/model_arch_config_convertor.py`) → `use_mm_prefix=False`.
The "Disabled mm_prefix attention mode …" line in the startup log is just an `info_once` announcing a value
that would have been False anyway. **The B12X_MLA_SPARSE / fp8_ds_mla / MTP=4 configuration is kept as is.**

## 3. Where the GPU encoder budget grows

> ⛔ **Correction (2026-10-30)**: The original title was "Where memory grows dynamically (only one place)", and it is **false**. This section
> covers only the GPU encoder budget. The head API server's **host preprocessing memory grows without bound in proportion to image count × pixels**,
> and every actual death was caused there (§8 correction table). I asserted "only one place" without measuring.

`vllm/v1/core/encoder_cache_manager.py`:

```python
encoder_compute_budget = max(scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item)
encoder_cache_size     = max(scheduler_config.encoder_cache_size,           max_tokens_per_mm_item)
```

And in `config/scheduler.py`, the defaults of both scheduler values are **both MNBT**:

```python
self.max_num_encoder_input_tokens = self.max_num_batched_tokens
self.encoder_cache_size           = self.max_num_batched_tokens
```

`max_tokens_per_mm_item` is **the maximum, over enabled modalities, of the max token count of a single item**.
So **the moment one item exceeds MNBT, the budget is lifted to that item's size.** The ViT converts that budget into
patches and runs it all in one step, so activations grow by the same factor.
Neither the KV budget nor `GMU` stops this — it must be stopped at the item size.

Measured item sizes (computed by actually calling `_pixel_budget` + `smart_resize`.
One vision token = `temporal_patch_size * (patch_size*merge_size)^2` = **1,568 px**):

| max_image_tokens | max_pixels | Aligned canvas | ViT patches | Tokens/item |
|---|---|---|---|---|
| 8000 (checkpoint) | 12,544,000 | 2492×2492 | 31,684 | **7,921** |
| 4096 | 6,422,528 | 1792×1792 | 16,384 | 4,096 |
| 2048 | 3,211,264 | 1260×1260 | 8,100 | 2,025 |
| 1024 | 1,605,632 | 896×896 | 4,096 | 1,024 |

- **Image**: 7,921 < MNBT 9,236 → budget not lifted. But the margin is only **1,315 tokens (14%)**.
- **Video**: the `_MAX_VIDEO_TOKENS = 30000` cap in `transformers_utils/processors/glm5next.py`
  applies first, making 240,000 → 30,000, and timestamp tokens
  `max_grid_t*(2+ts)+2` (`max_grid_t = 640//2 = 320`) are added on top for **≈32,242 tokens/item**.
  **3.5×** MNBT → the budget is lifted to 32,242 and the ViT runs 128,968 patches in one step
  (~2.1 GB for the MLP intermediate tensor alone). ~~This is the only real OOM path.~~ (False — host preprocessing was the real OOM path)

Two traps:
- `enable_mm_embeds` must stay at its default `False`. If `True`, modalities with limit 0 also enter
  `active_modalities` as `embed_only_modalities` and **keep participating** in the `max()` above
  (`multimodal/encoder_budget.py`). Even with video closed at 0, the budget is lifted.
- `max_pixels` and `max_image_tokens` take **different paths**. The budget calculation
  looks only at `_get_image_max_pixels()` → `max_pixels`, while the runtime preprocessing geometry
  looks only at `images_kwargs["max_image_tokens"]`. If they disagree, budget < actual and the encoder cache
  overflows (same shape as upstream vllm#34040). That is why both are passed **together** as
  `max_pixels = tokens × 1568`.

## 4. How it is capped — the `VISION` gate (`launch.sh`)

```
VISION=0   (launch.sh default)  --language-model-only + --skip-mm-profiling
VISION=1           --limit-mm-per-prompt {"image":N,"video":0}
                   --mm-processor-kwargs {"max_pixels":T*1568,"max_image_tokens":T}
                   swap in chat_template.vision.jinja, remove --language-model-only
```

Knobs (default = `launch.sh` default / **bold = live `glm53-serve.env` value**):

| Knob | Default | Live | Meaning |
|---|---|---|---|
| `VISION` | 0 | **1** | Accept images (video stays 0) |
| `VISION_MAX_TOKENS` | 4096 | **4096** | Per-image cap = canvas 1792² |
| `VISION_MAX_IMAGES` | 1 | **5** | Images per request. Excess → 400 before decode. At 8, 3 concurrent requests stop the server (§15) |
| `VISION_PIXEL_DTYPE` | bf16 | **bf16** | Preprocessed patch dtype (fp32 = upstream as is) |
| `VISION_LAZY_DECODE` | 0 | **1** | Read only the header and defer decode — the peak becomes independent of image count (§12) |
| `VISION_MAX_IMAGE_PIXELS` | (none) | **8294400** | Per-original-image cap 4K UHD. Over → 400 (§14) |

⛔ `VISION_PROFILE` and `VISION_MALLOC_MMAP_THRESHOLD_` are **deleted knobs** (2026-10-30, `168416f`).
If you see them in docs or notes, those are stale records.

Pre-launch guards — violate them and it **will not start**:
- Refuse if `MNBT` is empty (the cap cannot be validated).
- Refuse if `VISION_MAX_TOKENS >= MNBT` (exactly the condition that lifts the budget).
- Warn if `VISION_MAX_TOKENS > MNBT*0.8`.

`VISION_PROFILE=1` does **not attach** `--skip-mm-profiling`. With the cap in place,
the dummy profile acts as an OOM canary — if the peak doesn't fit, it dies at startup, not under traffic.
`get_dummy_encoder_profile_inputs` uses `max_items_per_batch` items, and that value is
bound by `encoder_budget // max_toks_per_item`.

## 5. GPU encoder peak with the cap in place

> ⛔ **Correction (2026-10-30)**: "Fixed value, does not grow further" in the original title applies only to the GPU encoder. Host preprocessing does grow.

Because the budget is fixed at MNBT, the per-step ViT input ceiling is `9,236 × merge²` = **36,944 patches**.
`max_num_encoder_input_tokens` / `encoder_cache_size` are `Field(init=False)`, derived from MNBT,
with no CLI knob → **MNBT determines the per-step peak.**

| Item | Size |
|---|---|
| Encoder cache | 9,236 × 4096 × 2 B = **75.7 MB** |
| ViT hidden | 36,944 × 1024 × 2 B = 75.7 MB |
| qkv | 227 MB |
| MLP gate+up (intermediate 4096) | 605 MB |
| merger input (9,236 × 10240 after 4:1 merge) | 189 MB |

Dimensions split by the TP=2 `"weights"` partition are halved per rank. Everything is eager
(`compile_mm_encoder: False`, `cudagraph_mm_encoder: False`,
`encoder_cudagraph_max_vision_items_per_batch: 0`) → no separate graph pool is allocated.

**Important**: lowering `VISION_MAX_TOKENS` does **not reduce** the per-step peak (the budget has a floor of MNBT,
so smaller items just get batched in greater numbers). The purpose of lowering it is not peak reduction but
**securing margin so that an item can never exceed MNBT**.

## 6. Unresolved / needs checking

- `vision_config.rms_norm_eps` is serialized in the checkpoint as **1e-05**. However, the comment in
  `overlay/glm5next/model.py` states that the vision tower's own value is **1e-6**, and that it deliberately reads
  from the vision sub-config to avoid the top-level 1e-5 being mirrored into `text_config`.
  So that defense is neutralized with the current checkpoint. Needs comparison against the upstream zai config.
  **This is an image-quality issue, so it must be checked before judging vision fit for real use.**
- The MTP drafter does not receive the target's multimodal embeddings on image turns
  (`speculator.py:232` — "using text-only draft inputs instead"). It works, but the acceptance rate of turns
  containing images must be looked at separately.

## 7. Measurements (2026-10-30, VISION=1 / VISION_MAX_TOKENS=4096 / MNBT=9236)

Startup 180 s, both nodes Up. Key confirmation:

```
[encoder_runner.py:82] Encoder cache will be initialized with a budget of 9236 tokens,
                       and profiled with 2 image items of the maximum feature size.
```

**Budget = 9,236 = MNBT unchanged. Not lifted.** `2 image items` matches `9236 // 4096 = 2`.

| | Step | Acceptance | t/s |
|---|---|---|---|
| Text | **77.8 / 78.9 ms** | 40.2 / 41.7 % | 33.05 / 34.27 |
| Image (4,053 tok) | 81.1 / 82.0 ms | 51.6 / 80.9 % | 37.78 / 51.66 |

The text step stays at the 79~81 ms baseline (no regression). The higher t/s on image turns is because
answers like listing shapes are short and formulaic, which raises acceptance; the step itself is +3 ms.
However, the image runs are short (17~31 steps), so these are not sustained-load values.

TTFT: text (32 tok) 0.380 s / image 1,024 tok 0.868·0.874 s / image 4,053 tok
0.993·1.000 s. 1,024 → 4,053 (4×) adds only +0.13 s, so fixed overhead (decode, preprocessing)
dominates over ViT per-patch compute.

Cap behavior:
- 4000×3000 = 12.0 M px (1.87× max_pixels 6,422,528) → `prompt=4101`
  = vision tokens **4,053** + 48. It did not go to 7,921; it was cut below the cap. Identical across all 4 repeats.
- 896×896 → `prompt=1072` = vision tokens 1,024 + 48 (= 64×64 patches ÷ 4, matches prediction).
- 2 images → `400 At most 1 image(s) may be provided in one prompt.`
- Video → `400 At most 0 video(s) may be provided in one prompt.`
  (not a crash — rejected at validation; the engine never even attempts it)

Memory: over 4 consecutive large images, HEAD MemAvailable 6.0 → 5.9 → 5.8 → 5.8 → 5.9 → 6.0 GB,
worker flat at 9.6 GB. GPU 7W → 17~18W.
> ⛔ **Correction (2026-10-30)**: "Does not grow" was observed only for **repeated single-image requests**. It grows as the image count increases —
> generalizing this single-image result to all image counts was the starting point of the 128-image setting.

⚠️ One caveat — because `kv_cache_memory_bytes` is set explicitly, the engine skips memory profiling
entirely (`gpu_worker.py:597`: "skipped memory profiling"). The MM dummy forward did
run (= evidence the peak fits) and did not die, but that peak is not converted
into KV size. In other words, protection comes from the **item size cap**, not from profiling.
KV is pinned and unrelated to vision: 11.18 GiB / 1,356,053 tokens / concurrency 2.59x.

- Remaining measurements: multi-turn with images mixed into long contexts, and image-turn steps under sustained load.
- Because `chat_template.vision.jinja` fixed the `visible_text` macro, it emits tokens even when an image item
  appears inside a tool-response. If vLLM does not register that image as an mm item,
  the placeholder scan errors (not a crash; the request fails).

## 8. Count vs per-image size — different axes

The image value of `--limit-mm-per-prompt` counts **the total number of image blocks in the request body**.
It does not distinguish whether they arrived side by side in one turn or accumulated in multi-turn history. The chat API is
stateless and resends the entire history every turn, so even sending 1 image per turn, the request at turn N
contains N images. Exceeding the limit gives `400 At most N image(s) may be provided in one prompt`, and
**the server does not drop old ones** — trimming is the client's job.

| Knob | Safe to raise? | Why |
|---|---|---|
| `VISION_MAX_IMAGES` (count) | ⛔ Risky | The GPU encoder budget `max(MNBT, max_tokens_per_item)` is independent of count (that part is true). But **the head's host preprocessing memory scales with count**. |
| `VISION_MAX_TOKENS` (per image) | ⛔ Risky | Lifts that `max()` directly → ViT activations grow by the same factor. Host preprocessing also scales with pixels. |

⛔ **Correction (2026-10-30)**: What was here before — "count is memory-safe, the only cost is context length,
structural cap 128 images" — was **false**. On that basis, 128 images @4096tok was live from 07:34~14:29 (`661be22`).
I looked only at the GPU side and not the host side.

Measured (warm, alternating 6/8/6/8, warmup discarded, 0.05s probe, head MemAvailable minimum):

| Config | Min GiB | Margin to memwatch bar 3GB | t/s |
|---|---|---|---|
| 8 images @1792² | 3.63 / 3.70 | +0.63 / +0.70 | 1079 |
| 6 images @1792² | 4.02 / 4.15 | +1.02 / +1.15 | 1080 |

The warm baseline is ~5 GiB, so even 8 images cannot keep the operational headroom requirement (1.0~1.5 GiB above the bar).
Per-image cost is not yet settled because the decrease over two 6-image runs scattered at 1.23 / 0.61 — judge only by measured minimums.

⚠️ **Do not confuse (the point where I actually got it wrong on 2026-10-30)**: 100 images × 4,096 = 409,600 tokens is
**the length of one conversation**. It must not be compared with S≤270,000 in `project_glm53_prefix_cache_retention_limits` —
that experiment fixed the parent conversation at **441,165 tokens**, then prefilled a *separate* conversation S
to see whether the parent cache survived; 270K is **"the boundary at which other traffic evicts my cache"**.
The parent itself, at 441K, is longer than 270K. The two numbers are on different axes, so measuring one conversation's length
against 270K is wrong. (That memo's KV figures, 8GB/847,533 tokens, also differ from the current live 12GB/1,356,053,
so the boundary value itself should not be quoted as fixed.)

## 9. Multi-turn / prefix cache measurements (2026-10-30)

Once image tokens have passed through the ViT, they are **ordinary tokens**. They go into KV, are prefix-cached
exactly like text, and are reused on the next turn. The encoder cache (9,236 tokens) is a **temporary buffer** that holds
ViT output embeddings only until they are moved into KV, and is unrelated to conversation retention — it only determines
"how many images per step".
> ⛔ **Correction (2026-10-30)**: "It's a temporary buffer, so even 128 images won't overflow" refers to the GPU encoder cache. Before that, the head
> **preprocesses all N images at once**, so image count translates directly into host memory.

Conversation with 4 images (1,024 tok each), `prompt_tokens_cached_total` deltas:

| | prompt | Cache hit | New prefill | wall |
|---|---|---|---|---|
| Turn 1 (cold) | 4,132 | 0 | 4,132 | 4.62 s |
| Turn 2 | 4,195 | 3,584 | 611 | 1.69 s |
| Turn 3 | 4,231 | 3,584 | 647 | 1.72 s |
| Turn 4 | 4,269 | 3,584 | 685 | 1.44 s |

From the second turn on, images are not re-encoded. The hit is 3,584 rather than 4,096 because of
our fix that deliberately does not store the trailing 512 (3.8s→1.0s per turn).

**Rejected hypothesis**: "the trailing 512 falls on the last image, so don't put images at the end of a message".
Appending text after the image to move the tail onto text does increase the hit range 3,584 → 4,608, but
steady-state new prefill (590~625 vs 611~685) and wall (1.44 s for both) are the same. No ordering rule is needed.

## 10. Preprocessing `pixel_values` in bf16 (2026-10-30, `3f02b9f`)

Knob `VISION_PIXEL_DTYPE=bf16|fp32` (default bf16), overlay `overlay/processors/glm5next.py`.

**Bit-identical**: there is no arithmetic between `rescale_and_normalize` and tower consumption (`.type(self.visual.dtype)`),
and the tower dtype is bf16, so moving the rounding earlier still yields a single rounding. Measured — comparing, for the same input,
the fp32 path's value as downcast in the tower against the bf16 path as int16 bits: identical, NaN 0.
fp16 **does not qualify** (fp16 → bf16 double rounding).

**Cast location is decisive**: casting after `torch.cat` leaves a full fp32 copy and the bf16 alive at the same time,
which actually raises the peak. It must be cast inside the group loop right after `reshape` (before cat).

| 896² offline peak (maxrss, MiB/image) | N=16 | N=32 |
|---|---|---|
| fp32 (upstream) | 41.8 | 42.3 |
| bf16, after cat (`bec1b73`, discarded) | 48.4 | 50.7 |
| bf16, before cat (`3f02b9f`) | 20.1 | 21.8 |

896² live ladder: fp32 hit a memwatch stop at 28 images; bf16 reached a minimum of 3.03 GiB at 32 images.
⚠️ The bf16 ladder ran **while the head memwatch was dead** (minimums reconstructed from probes).
⚠️ Running this offline measurement **on the serving head** caused memwatch to take down the live server twice
(10-30 11:38, 14:27). Run offline measurements only after taking the server down.
For 1792² operational figures, see the §8 correction table.

## 11. Memory pinned on the head after the first image (2026-10-30, per-process samples every 0.2s from right after startup)

Order: startup → 1 image → 8 images ×3 @1792². GPU = nvidia-smi usage of the Worker process, anon = RssAnon.

| Head (GiB) | Right after startup | After 3× 8 images | Pinned amount |
|---|---|---|---|
| Worker GPU usage | 99.25 | 101.15 | +1.90 (worker node also +1.90~1.96) |
| API server anon | 1.54 | 2.49~2.73 | +1.0~1.2 |
| EngineCore anon | 1.09 | 1.38 | +0.29 |
| MemAvailable | 7.71 | 4.24~4.39 | −3.3 |

- GPU +1.9: the startup `profile_run` runs the ViT dummy and then returns the memory via `_release_unoccupied_accelerator_memory()`
  (`v1/worker/gpu_worker.py:576-578`); on the first real request the caching allocator grabs it again and does not let go.
  It is a fixed cost that does not grow on repetition. The ViT uses the same amount again during a request, so releasing it would not raise the minimum.
- During a request, an additional ~1.0 GiB is held for the whole request (~30s). It does not show up in any process's anon (unidentified).
- `MALLOC_MMAP_THRESHOLD_=4194304` A/B (same order, 2 restarts):

| | Default | Pinned at 4 MiB |
|---|---|---|
| Minimum during 8-image request | 3.26 / 3.27 / 3.30 | 4.02 / 3.82 / 3.79 |
| Amount held by API server | 2.49~2.73 | 2.17~2.41 |
| Decode t/s (400 tok ×3) | 33.8 / 34.3 / 35.1 | 32.6 / 33.3 / 34.0 |

  Of the API server's share, ~0.3 was memory glibc merely had not returned; the remaining ~0.8 is actually referenced.
  Re-judged after §12 (one image at a time + lazy decode) (run D): on +1.34~1.54 / off +1.43~1.50 — **no effect, knob deleted.**
  The 4 MiB experimental state is in commit `78abf73` (in this branch's history).

After the preprocessing loop fix (`4e292b3`: bf16 right after rescale, results written directly into preallocated storage — bit-identical, offline peak
8 images@1792² 0.741 -> 0.411 GiB), re-measured in the same order:

| 8 images@1792² | Default | malloc 4 MiB | malloc + loop fix |
|---|---|---|---|
| Minimum during request | 3.26 / 3.27 / 3.30 | 4.02 / 3.82 / 3.79 | 4.17 / 4.13 / 4.06 |
| API server anon (after request) | 2.49~2.73 | 2.17~2.41 | 1.75~2.04 |
| prefill tok/s | 1069~1078 | 1075~1083 | 1065~1081 |
| Decode t/s | 33.8 / 34.3 / 35.1 | 32.6 / 33.3 / 34.0 | (30.8) / 34.6 / 35.4 |

Each is a single run, and restart-to-restart variance is ~0.35 GiB — the improvement is unconfirmed until repeated.

⚠️ The head floor of 5.90 GiB for a 400K cold prefill (10-10) was measured with vision off. Subtracting the amount pinned
after the first image puts it at the memwatch bar — **this is arithmetic, not measured.** A long prefill after an image request must be measured separately.

## 12. One-image-at-a-time preprocessing + lazy decode (2026-10-30)

Upstream handles a request's N images all at once: (1) the vLLM parser decodes all N originals and stores them
in the request object, (2) the HF preprocessor converts all N into original-size tensors, (3) only then resizes. So the peak was
"original size × N". CPU preprocessing has grouping disabled, so computation is one image at a time anyway and batching gains nothing.

- `overlay/processors/glm5next.py` `_preprocess_image_like_inputs`: convert to tensor one image at a time → resize/patchify
  → free the original; results are written directly into preallocated slots (gated by `VISION_PIXEL_DTYPE=bf16`).
- `overlay/multimodal_media/image.py` `load_bytes`: with `VISION_LAZY_DECODE=1`, reads only the header (the pixel cap
  check is unchanged) and defers decode. The preprocessing loop does the same EXIF normalization, decode and mode conversion as upstream
  one image at a time via `finish_lazy_image()` and closes it immediately. The hash is unchanged: original compressed bytes + io_config.

Worker offline, server path (base64 → ImageMediaIO → preprocessing), maxrss after warmup with 64x64 (includes the verification hash copy:
8 images ~0.31, 1 image ~0.04 GiB). Do not warm up with the same size — the one-image peak gets buried in the baseline and drops out.

| Original | Decode all, then process | Lazy decode + one at a time |
|---|---|---|
| 1792² x8 | 0.92 | 0.74 |
| 12MP x8 | 1.18 | 0.70 |
| 48MP x8 | OOM(>3.5) | 0.83 |
| 48MP x1 | 0.25 | 0.20 |
| 169MP x1 | 0.64 | 0.64 |

The peak is independent of image count and proportional to **the original size of the largest single image** (one image's worth: 48MP 0.2, 169MP ~0.6).
That single image's cap is set by `VLLM_MAX_IMAGE_PIXELS` (default 178,956,970).

Bit-identity confirmed: the 4 cases above, 1792² and mixed sizes (vs upstream fp32), RGBA PNG, EXIF-rotated JPEG, palette+transparency, grayscale
(pixel_values, grid, and hash inputs io_config/bytes all identical).

## 13. Safety tests (2026-10-30, `5145dc3`, malloc 4 MiB on, head MemAvailable minimum, 0.05s probe)

| Test | Result | Head minimum | Above bar (3.0) |
|---|---|---|---|
| 8 images @1792² alone, 3× from right after startup × 2 restarts | OK, prompt 32,830 | 4.34~4.54 | +1.34~+1.54 |
| Original 48MP JPEG ×8 in one request (63MB), warm | OK, 8/8 answered | 3.77 | +0.77 |
| Original 169MP PNG ×1 (0.7MB), warm | OK | 3.40 | +0.40 |
| 8 images @1792² ×2 concurrent | OK, ~540 tok/s each | 3.21~3.24 | +0.21~0.24 |
| 8 images @1792² ×3 concurrent | **memwatch stop** (3 requests cut), box OK | 2.79 | −0.21 |
| 400K text prefill after image processing | OK, 399,402 tok 316s (1,264 tok/s) | 5.03 | +2.03 |

- Max decrease within 0.4 s is 0.84~1.25 GiB — even when it stops the moment guard 3.0 is crossed, the box halts at ~1.75 or above (confirmed with 3 concurrent).
- Memory held only during a request: per 8-image request, ~0.5 GiB outside process anon (MemFree decreases; AnonPages, Shmem, slab
  unchanged — looks driver-side but unidentified) + EngineCore/Worker hold `pixel_values` until the end of prefill.
  When concurrent requests overlap, this stacks. Do not block it with queuing (waiting) — reduce per-request holdings instead.
- A single large original uses more on the live API server than offline (169MP: live +1.2~1.6, offline 0.64). Unidentified.
- Asked to count 8 images, it answers 6~7 (7 even at temp 0). Per prompt tokens, all 8 images are present — a counting weakness of the model.
  Shape/color tests (4 images 4/4, 3 positions in 1 image) give the same answers as at the first vision startup.
- Decode 30.0~35.4 t/s (same-day range) — within thermal drift range; malloc knob effect undetermined.

## 14. Per-original-image pixel cap = 8,294,400 (4K UHD, 2026-10-30)

`VISION_MAX_IMAGE_PIXELS` → container `VLLM_MAX_IMAGE_PIXELS`. The check happens in `load_bytes` on the header size, before
decode (same with lazy decode). The server accepts up to 4K only; larger originals (5K/6K captures, original phone photos, etc.) are downscaled
by the client (codemason) before sending. Over → 400. Boundary check: 3840x2160 passes / 3841x2160 400. vLLM default is 178,956,970 (one 169MP PNG used
1.2~1.6 GiB on the live head, §13).

## 15. Image-count ladder and occupancy window — and why early release of `.data` doesn't work (2026-10-31~11-01)

Measurement protocol: 60 s cooldown between runs (30 s is not enough — there was a run where residue from the previous measurement
raised the baseline and stretched the window to 42.7 s), 3 s baseline re-measurement, 18 s recovery sampling after the request, a new image each run (cache miss),
only runs with baseline jitter ≤0.10GB + recovery to at least baseline −0.25GB accepted. No other commands during measurement.

| Images @1792² | Drop | Occupancy window | TTFT | Done |
|---|---|---|---|---|
| 1 | **0.00GB** | 0.0s | 4.5s | 5.5s |
| 2 | 0.44GB | 26.2s | 7.5s | 8.5s |
| 5 | 0.78GB | 37.3s | 18.9s | 19.9s |
| 8 | 1.00~1.26GB | 30.7~31.9s | 30.3~30.7s | 31.4~32.2s |

- 1/2/5 images clean n=2, 8 images clean n=4. Highly reproducible (1 image 0.00/0.00, 2 images 0.46/0.43, 5 images 0.77/0.79).
- Window ≈ request lifetime, released immediately when done. For 8 images, 97% of the window is prefill (30.4 of 31.2).
- ⛔ The earlier record "window 38.9 s / held 8.9 s more after response start" was measurement contamination (recovery sampling
  ran only 6 s, so the end of the window was clipped at the measurement end time). Retracted.
- The 2- and 5-image windows being 17~18 s longer than completion is a release delay and is unrelated to `.data`. Unidentified.

### Composition of the drop — 71% cannot be touched

`/proc/meminfo` breakdown during one 8-image request (375 samples at 0.3 s intervals):

```
MemAvailable  -1.03GB      AnonPages  +0.29GB   ← pixel_values share
MemFree       -0.80GB      Shmem/Cached/SUnreclaim  0.00
process RSS   change 0 (APIServer 2.60 / EngineCore 1.87 fixed)
```

**0.74GB of the drop is outside process anon** — what §13 described as "looks driver-side but unidentified";
it appears to be CUDA staging/pinned. No matter how `.data` is released, this part does not shrink.

### `.data` early release A/B conclusion (preserved as tag `glm53-mm-data-release-failed`, not merged into main)

There is no code in upstream `v1/` or `worker/` that clears `MultiModalFeatureSpec.data` (it survives until request end).
Two copies: EngineCore `Request.mm_features`, worker `EncoderCache.mm_features` (popped only in `remove_request()`).
`free_encoder_input()` drops only the reference to the GPU embedding. The GPU embedding itself is released sequentially by `_free_encoder_inputs()`
starting from consumed images, so the chunked prefill path is fine.

| | Drop | Window | TTFT |
|---|---|---|---|
| A original | 1.02GB | 31.3s | 30.45s |
| B stage 1 (release after KV store) | 1.04GB | 31.3s | 30.47s |
| C stage 2 (right after encoding + worker copy) | 0.96GB | 31.6s | 30.42s |

- Why B is 0: with 8 images, vision tokens 32,768 are 99.8% of the 32,830-token prompt, so "the point when everything is in KV"
  coincides with the end of prefill. There is no window to reclaim.
- Why C is only 0.06GB: the reclaim ceiling is AnonPages 0.29GB, and the expectation of 2 copies (616MB) was wrong —
  effectively only one copy exists at a time.
- No quality or speed regression (TTFT within 0.03 s, all 8-image recognition correct, no server stop). It does no harm, but
  0.06GB is not worth maintaining 2 vLLM core overlays, so it was not merged into main.
- The preempt trap (resuming requires re-encoding, but `prepare_mm_inputs()` skips when `data is None`
  → silent wrong answers) and its workaround (exclude released requests from preempt victims) are implemented on the branch.
  If this ever needs revisiting, start from that tag (`git show glm53-mm-data-release-failed`).
  The branch was deleted — it clutters the list and invites confusion, so failed experiments are kept only as tags.

### What actually works

`VISION_MAX_IMAGES` was lowered to **5** (2026-11-01). Per request 1.02 → 0.78GB, 3 concurrent 3.06 → 2.34GB,
which fits within the headroom (baseline 5.3~6.0 − bar 3.0 = 2.3~3.0GB). It fully accommodates the user's real-use ceiling (usually 1 image, at most 2,
4~5 in serious cases), so there is no loss. Excess requests get a 400 before decode.
The one remaining risk is **parallel subagents each reading images** — operating habits cannot prevent it.

#### Limit-5 verification (2026-11-01, main@a334496, warm)

| Test | Baseline | Minimum during request | vs bar (3.0) | Result |
|---|---|---|---|---|
| 5 images × 2 concurrent | 5.75 | 4.34 | **+1.34** | 2/2 succeeded, wall clock 38.9s |
| 5 images × 3 concurrent | 5.76 | 3.69 | **+0.69** | **3/3 succeeded**, wall clock 58.5s, server survived |
| 5 images alone | — | — | — | 200, prompt 20,516 tok, 19.5s |
| 6 images (over limit) | — | — | — | **400 at 0.0 s** (blocked before decode) |

The 3-concurrent drop of 2.06GB matches the arithmetic expectation (0.78x3=2.34GB). The same condition with 8 images is 3.06GB, so
memwatch takes the server down (§13). With 3 concurrent, each request's wall clock stretches to 32~58 s — not because they wait on each other
but because they share the prefill token budget.
