[한국어](ORIGINAL-VS-ABLITERATED.md) | **English**

# Original vs abliterated — why we serve the original

The served model is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (the original). In early September we served lovesenko's abliterated version
wholesale. On September 29 we switched to the original, and since October 1 uncensoring is done with a runtime switch (`AB-SWITCH.md`).
This directory used to be called the "o_proj restored copy (restored)", but that name only came from how it was built — by reverting o_proj in the abliterated
version to the original — and **the result is identical to the original repository file for file** (see "Evidence that it is identical to the original" below).

## Why we switched

`lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated` hallucinated badly, describing nonexistent targets
as if they were real. Investigation on 2026-09-28~29 found that **the cause is abliteration**.

Fake-target trap, 20 items × 3 runs = 60, server defaults (effort high):

| Condition | Hallucinations |
|---|---|
| abliterated, dense FP8 (production) | 26/60 |
| abliterated, dense BF16 | 22/60 |
| **Original (all o_proj restored)** | **4/60** |
| Only layer 44 restored | 25/60 |
| Official Z.AI fp8 (OpenRouter) | 8/60 |

Ruled out (innocent): dense precision (FP8/BF16, including AMD INT8), routed experts EXL3 4bpw,
KV fp8_ds_mla, serving nondeterminism. All of these remain enabled on the original as well.

**Restoring only layer 44 has no effect** (22 → 25). The edit is concentrated in layer 44 (84% of rows changed,
other layers 0.2~4%), but the hallucination is the cumulative result of the edits across 45 layers. There is no option of trading
refusal behavior against hallucination via partial restoration.

No speed penalty. If anything, draft acceptance went up and prose got faster.

| Item | abliterated | Original |
|---|---|---|
| Korean prose | 38.3 t/s (accept 2.061) | 40.6 t/s (2.147) |
| Korean code | 43.5 t/s (2.443) | 43.5 t/s (2.448) |
| hash-map | 35.0 t/s (1.857) | 36.4 t/s (1.941) |

**Trade-off: refusal behavior comes back with it.** The original purpose of abliteration is lost.

**Remaining limitation**: the original is not perfect either. In sparse corners of the vocabulary it still makes things up
(e.g. the 괭테스트 (gwaeng test) answer "괭이밥(돌멩이를 부르는 말)" ("gwaeng-i-bap (a word for a pebble)")). Official Z.AI is also 8/60.

## Evidence

Model card: "The abliteration edits only the BF16 `o_proj` tensors; all
TR3-quantized expert tensors are byte-identical to the base quantization."
Base = `brandonmusic/GLM-5.3-Flash-tr3-4bpw`.

Verified directly:
- The 150,226 tensors in `weight_map` are **exactly identical** to the base
- **45 of the 46** o_proj tensors differ, relative change `‖Δ‖_F/‖W‖_F` averaging 0.10
- **Layer 45 (MTP) is byte-identical** — the MTP head was not edited. The card's "45 decoder
  o_proj" is the accurate wording; the 46th is untouched original
- Median row-norm change 0.003~0.004 → the card's `preserving row magnitudes` is true
  (except layer 44, with median 0.046 and 84% of rows changed by more than 1%)

## Evidence that it is identical to the original (2026-10-04)

- Compared against the LFS sha256 published by the original repository (`brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `a5fee929`): the 2 shards whose o_proj
  was overwritten and 1 shard brought in via hard link **match**. 9 files — config, tokenizer, chat template, index, quantization settings — **match**.
  The only differences are lovesenko's model card documents (README/LICENSE/PROVENANCE/NOTICES) — irrelevant to serving.
- Extracting o_proj from the checkpoint that was being served gives the same md5 as the file downloaded from the original (`5d1f22fd…`).

## How to build it

- **Fresh install**: just download the original (`hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw`). Root README §5.5.
- **If you already have abliterated**: fetch only the original 46 o_proj tensors (3.89GB) via HTTP Range and overwrite. Root README §6.

   ```
   python3 tools/fetch_oproj.py --source original --out ~/Models/glm53-oproj/o_proj_original.safetensors   # md5 5d1f22fd…
   python3 tools/build_original_from_abliterated.py --src <abliterated snapshot> \
       --base ~/Models/glm53-oproj/o_proj_original.safetensors --dst ~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw
   ```

   Only the 44 affected shards are copied, then the original bytes are written over them at the same offsets. The other 76 shards and config/tokenizer
   are **hard links**. Actual extra disk usage is about 72GB. Weights must be local to each node, so build it on each of the two nodes.

## Pitfalls

- **No symlinks.** If they point to absolute host paths, they break inside the container
  (`Invalid repository ID or local directory specified`). Use hard links.
  Directories cannot be hard-linked, but they are not needed for loading, so skip them.
- **Fixing only the index does not work.** `instanttensor_weights_iterator` reads only the tensors that match a priority
  prefix based on the index; the rest are read by
  `safetensors_weights_iterator` as **all tensors per file**. Even if you fix the index,
  the original shard's o_proj is read again, and the result depends on load order.
- Because they are hard links, the original survives even if the abliterated snapshot is deleted. However, **modifying the original in place**
  changes both.

## Caution on judgment method

Judging hallucination by keywords causes misclassification. What actually happened:
it caught `"없습니다"` ("there is none") but missed `"없고"`/`"없으며"` (other inflections of "there is no"), and `"아닙니다"` ("it is not") appeared in a different context
("Вan? 아닙니다 — 정확히는 ..." ("Вan? No — more precisely ...")) so a hallucination was counted as correct. `"표준"` ("standard") is also risky.
**Save the raw text and check it by eye.** The numbers in the table above are also keyword judgments; the original's 4 and
abliterated's 22 were confirmed by reading the raw text (the original's real hallucinations are 1~2).
