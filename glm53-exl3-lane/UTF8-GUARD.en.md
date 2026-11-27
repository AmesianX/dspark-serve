[한국어](UTF8-GUARD.md) | **English**

# UTF-8 byte guard — design, patch and verification

Written 2026-09-14 · lane `glm53-exl3-lane` · vLLM `0.26.1rc1.dev1006+gf223ff9f2.d20260902`
Image `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1`

---

## 1. Symptom

In GLM-5.3-Flash responses, **one character of a Korean name comes out broken as U+FFFD (`�`)**. The reproducing prompt is
a list request like "10 scientists who contributed to quantum mechanics", and the broken spot is almost always the same:

```
4. **�erner 하이젠베르크(Werner Heisenberg) - 행렬역학 창시, 불확정성 원리
     ^^^ where "베르너" (Werner) should be
```

(Model output, verbatim; gloss: "4. **�erner Heisenberg (Werner Heisenberg) - founded matrix mechanics, uncertainty principle".)

This went on for a long time. In interactive use, "the name before Heisenberg is always broken" had been observed for months.

## 2. Root cause (ruling things out one step at a time)

### 2.1 The client is innocent
Same symptom in both codemason clients (Python and Rust). Replaying a recorded clean SSE stream to the client split
**one byte at a time** → 0 FFFDs. The stream decoder uses a `codecs` incremental decoder that holds incomplete
bytes, so it is safe against boundary splits.

### 2.2 The bytes the server sent are already broken
Capturing the raw SSE with a transparent proxy (`scratchpad/proxy.py` style):

```
data: {... "delta":{"content":" **\xef\xbf\xbderner"} ...}
```

`EF BF BD` = U+FFFD is **inside the server's response bytes**. The client did not create it.

### 2.3 Not a detokenizer bug either
`vllm/v1/engine/detokenizer.py` uses `tokenizers.decoders.DecodeStream`. It holds incomplete
bytes and emits U+FFFD **when completion has become impossible**. In other words, it behaved honestly.
(The known vLLM #27300 is limited to logprobs output and was already fixed by PR #34875, which is in our build.
This is not an old-version problem.)

### 2.4 The real cause — the model switches script in the middle of a character
Result of probing that position with `logprobs` (top 8):

```
token 1  chosen ''  bytes[]  logprob -0.358      ← standalone 0xEB byte token (id 167)
   candidates:  '루'[235,163,168] -1.23 · '르'[235,165,180] -5.42
                '버'[235,178,132] -7.61 · ' Werner'[32,87,...] -7.98
token 2  chosen 'erner' -0.578                   ← switches to Latin
   candidates:  'erner' -0.578  vs  '베'[235,178,160] -0.828     (≈ 5:4)
```

The model, intending to write "베르너" (Werner), **sampled 0xEB, the lead byte of a Hangul syllable (correct)**. At the next
step, `'베'`, which completes the Hangul character, and the Latin `'erner'` were nearly 50/50, and `'erner'` was sampled.
At that moment the preceding 0xEB **can never become a character**.

**Where the defect is**: not the first choice, but *that a Latin token remained a candidate while in the state of waiting
for continuation bytes*. No layer prevented the byte stream from leaving the UTF-8 automaton.

### 2.5 Temperature is a frequency knob, not the cause
Via codemason, two profiles **alternated** for 8 pairs (removing time-of-day bias), number of responses containing U+FFFD:

| Arm | Temperature | top_k | Broken |
|---|---|---|---|
| A | 1.0 (vendor) | none | **7/8** |
| B | 0.6 | 40 | **1/8** |
| D | 0.6 | none | **2/6** |

Sign test p≈0.016. Temperature is the main axis, top_k contributes somewhat. **But 0.6 still breaks** — it only lowers the
symptom frequency; the defect remains. So temperature was returned to the vendor-recommended 1.0 and the defect is fixed directly.

---

## 3. Design

### 3.1 Principle
For each request, track the **UTF-8 decoder state** (Björn Höhrmann DFA) over the actually generated tokens.
While a character is incomplete, **mask to -inf every token that cannot complete that character**.
At a character boundary (ACCEPT), mask nothing.

This is a **format (well-formedness) constraint**, not a content constraint. At a boundary the whole vocabulary is open;
in the middle, only pieces that complete that character are open. No reachable text is forbidden.

### 3.2 Tables
- States: `0(ACCEPT), 24, 36, 48, 60, 72, 84, 96` + `12(REJECT)` — multiple-of-12 encoding
- `bad[state_row, token]` bool — must this token be blocked in this state
- `next_state[state_row, token]` uint8 — the state after this token
- Size: 8 × 154,856 × (1+1) ≈ **2.4MB**, built once at startup then cached in `/root/.cache/vllm/utf8_guard/`
- Token→bytes: `xgrammar.TokenizerInfo.from_huggingface(...).decoded_vocab`
  (on failure, falls back to the GPT-2 byte-level alphabet)

### 3.3 Cost
Rows in a boundary state are not masking targets, so **zero tensor copies in the normal case**. Only when some row is in an
incomplete state: `bad[rows]` gather + `masked_fill_`. On the real vocabulary, 256 tokens are blocked at a boundary
(those that start with a continuation byte only), and at most 211 tokens per state are allowed in the middle of a character.

---

## 4. The MTP (speculative decoding) problem — the core of this patch

Stock vLLM blocks custom logits processors in two places.

### 4.1 Outright rejection
`vllm/v1/sample/logits_processor/__init__.py:build_logitsprocs`

```python
if vllm_config.speculative_config:
    if custom_logitsprocs:
        raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
```

→ If spec decode is on, custom processors are **rejected at startup**.

### 4.2 Why it rejects (the hole)
`vllm/v1/sample/rejection_sampler.py:apply_logits_processors`

```python
for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
    if isinstance(processor, MinTokensLogitsProcessor):
        logits = processor.apply_with_spec_decode(logits, metadata.num_draft_tokens)
```

Only **one** processor, `MinTokens`, is applied to the draft-verification logits. The rest are silently skipped. With MTP,
most tokens are decided on this path, so without the rejection the guard would have been effectively powerless.

### 4.3 Our fix — refine the guard without breaking it
1. **Plug the hole**: for processors that expose `apply_with_spec_decode_tokens(logits, num_draft_tokens, spec_token_ids)`,
   call it per draft position. The row layout is request-major, and row j of request r is sampled *with that request's
   drafts 0..j-1 already in context*, so the entry state is obtained by advancing from the committed state along the draft prefix.
2. **Turn the rejection into a capability check**: only processors that have that method pass; processors without it are
   **rejected with exactly the same error as before**. The intent of the upstream guard (do not accept a powerless processor)
   is preserved.

---

## 5. Files (final layout)

| File | Role |
|---|---|
| `overlay/vllm_utf8_guard.py` | Guard body (new). Mounted as container `site-packages/vllm_utf8_guard.py` |
| `overlay/gpu_sampler.py` | One guard-call line in the **V2 runner sampler** `vllm/v1/worker/gpu/sample/sampler.py` (+`.orig`, `.diff`) |
| `launch.sh` | 2 mount lines + `-e VLLM_UTF8_GUARD` / `-e VLLM_UTF8_GUARD_TOKENIZER=$MODEL_PATH` |

**Turning it on/off**: `UTF8_GUARD=0 ./vllm-glm53-exl3.sh RESTARTALL` (same effect as `VLLM_UTF8_GUARD=0` inside the container; no need to revert the overlay)

### 5.1 Why this spot — three abandoned paths
1. **Registering a custom LogitsProcessor** → vLLM demotes the model runner from V2 to V1
   (`Model Runner V2 does not yet support custom logits processors`). This model cannot come up on the V1 path:
   the drafter dies on `Glm5NextConfig.image_token_index`, and if you fill that with an alias, it next dies on
   `assert len(mla_pages) == 1` in `_get_kv_cache_groups_glm5_next`.
   **An upstream defect, on a different path unrelated to our problem.** → abandoned.
2. **Patching `v1/sample/sampler.py`, `v1/sample/rejection_sampler.py`** → this lane uses the V2 runner, so those files
   are **never called** (caught because the guard init log never printed at all). → abandoned.
   (The leftover overlays from that attempt — `logitsproc_init.py`, `glm5_next_config.py`, V1 `sampler.py`, `rejection_sampler.py`,
   `gpu_input_batch.py` — were removed from the repository on 2026-10-04. They exist only in git history.)
3. **Adopted**: a single spot, `apply_sampling_params()` in the V2 sampler. Both normal decode and **MTP draft verification**
   (`gpu/spec_decode/rejection_sampler.py` puts the draft tokens in the `input_ids` slot and calls the same function)
   **go through this function**. So one line covers both paths.

### 5.2 Where the state comes from
The V2 sampler already holds each row's **previous token** on the GPU as `input_ids`. If you precompute a table of
"the state reached when starting from a boundary" for a single token, each step finishes with **1 gather + 1 masked_fill**.
There is no host synchronization (D2H copy), so the stream does not stall.
- Per-request state tracking and handling of batch reordering all became unnecessary.
- When one character is split into **3 or more tokens**, the row after a middle piece cannot read the state from one token,
  so it is sent as an "allowed row" (not masked). The observed failure is the 2-token form; this hole remains.
- No dependency on engine config: vocabulary size comes from the **logits width**, the tokenizer path from
  `VLLM_UTF8_GUARD_TOKENIZER` (at sampling time `get_current_vllm_config()` is invalid).

## 6. Verification

### 6.1 Before deployment (server untouched)
- **DFA correctness**: **every substring** of real text (Korean, Latin, Japanese, emoji, symbols) checked against Python's
  `bytes.decode('utf-8')` verdict → all match. 60,000 random byte strings → all match.
  Overlong encodings (`C0 AF`, `E0 80 80`), surrogates (`ED A0 80`), out of range (`F4 90 80 80`), isolated
  continuation byte (`80`), 5-byte sequences → all rejected.
- **Waiting states**: `EB` → state 36, `EB B2` → state 24 (neither ACCEPT nor REJECT)
- **Bug case**: in state 36, `'e'` → REJECT; a following `B2` → proceeds normally
- **Real vocabulary (154,856)**: a mixed Korean/English/emoji sentence finishes without a single token blocked and returns to a
  boundary at the end. In state 36 after token 167 (`0xEB`), token 41138 (`'erner'`) **confirmed blocked**

### 6.2 Measured after deployment (2026-09-14 02:4x, 2-node live)

**Breakage rate** — arm A (vendor temp 1.0, via codemason, same prompt), 8 runs:

| | Broken |
|---|---|
| Before patch | **7/8** (`**\ufffderner 하이젠베르크`) |
| After patch | **0/8** (`**베르너 하이젠베르크`) |

All 8 runs produced the name intact. By sign test, the probability of this being chance is under 1%.

**Speed** — average of 3 runs of the same query:

| | t/s |
|---|---|
| Before patch (measured same day) | 37.9 |
| After patch | **36.23** (35.58 / 36.62 / 36.48) |

Within error. As designed, the per-step cost is one gather, so it does not show up in measurements.

**Startup**: API READY 210 s, KV 847,533 tokens — same as usual.

## 7. Risks and maintenance

- **When replacing the image**: the two files (`vllm_utf8_guard.py`, `gpu_sampler.py`) are fitted to this build's source. If the image changes,
  diff `.orig` against the new image's original and re-port. Just dropping them on top silently turns them into old-version code.
- **The sampler is a hot path.** Keep changes minimal (1 gather + 1 masked_fill), and if problems appear,
  disable immediately with `UTF8_GUARD=0`.
- **V1 runner demotion**: as warned above, a custom LP currently cannot coexist with the V2 runner. If the t/s loss
  is large, either (a) turn the guard off or (b) build a separate path for the V2 runner.
- **Upstream contribution candidate**: the two patches in 4.3 are model- and lane-independent. "Allow custom LPs under spec decode
  based on capability" is worth submitting to vLLM itself.

## 8. Reproduction tools

- Proxy capture: insert between server and client to save the raw SSE bytes
- `logprobs` probe: with `logprobs:true, top_logprobs:8`, inspect the bytes of the chosen/candidate tokens of the broken delta
- Harness: clone the config directory via `CODEMASON_CONFIG_DIR`, run N times per profile, then
  count responses containing U+FFFD (real config.json untouched)

---

## 9. Status: applied

Live on both nodes with the guard ON. To revert: `UTF8_GUARD=0 ./vllm-glm53-exl3.sh RESTARTALL`.

**To work on next**
- Covering characters split into 3 or more tokens (§5.2) requires holding per-request tail state.
  It can be added on top of the current structure, but since it is not an observed failure, it is deferred.
- Porting to the DeepSeek lane: the guard body is model-independent (tables are built from the tokenizer), so it can be reused.
  But ① `gpu_sampler.py` must be **re-extracted from that image's original** and the diff re-applied
  ② if the tokenizer uses `<0xEB>`-style byte-fallback notation, the vocabulary extractor must be extended
  (currently based on the GPT-2 byte-level alphabet; undecoded tokens fall through as "no bytes" and the guard **can silently
  become powerless** — when porting, always check the startup log's undecoded-token warning).
- Upstream contribution candidate: "allow custom LPs under spec decode based on capability" (§4.3) is model-independent.
