[한국어](AB-SWITCH.md) | **English**

# Runtime abliteration switch (`/ab`)

While serving the original weights (with refusal ability intact, `brandonmusic/GLM-5.3-Flash-tr3-4bpw`), the abliteration edit is turned
**on and off per request**. No restart, no re-prefill.

There is **only one way to turn it on: attach it to your own request**. It applies only to that request's tokens.

```bash
# uncensor this request only (other requests in the same batch stay original — measured bit-identical)
curl -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.3-flash","messages":[...],"vllm_xargs":{"ab_scale":1}}' \
     http://localhost:8000/v1/chat/completions
curl -H "Authorization: Bearer $KEY" http://localhost:8000/ab   # status (read-only)
# POST /ab → 410 Gone (see "Zero trust" below)
```

## ⛔ Zero trust (2026-11-27)

Clients are not trusted. codemason is public under MIT, so anyone can send anything.

- **No global switch.** The old `POST /ab` changed the engine-wide default, and every request that did not carry
  `ab_scale` followed that value → one client could change someone else's model. The 11-20 per-token
  vector only created "a means to separate" and did not close this write path. 11-27 live measurement: after another
  client's `POST /ab {"scale":1}`, the text and logprobs of a request that changed nothing were altered.
  Now `POST /ab` returns **410**, and the worker's setter RPC is **not installed at all** (unreachable even through `VLLM_SERVER_DEV_MODE`'s
  `/collective_rpc`). The default is fixed at startup by `GLM53_AB_SCALE` (0).
- **Input validation.** `ab_scale` accepts only a finite number in 0~1 (int/float, excluding bool). Anything else (NaN, inf,
  out of range, string, bool, list) is **ignored and the request runs on the original** (`rejected` in `GET /ab` increments).
  The old `min(max(v,0),1)` let NaN through unchanged so NaN entered the forward pass, and while that request was
  being tracked, `GET /ab` returned 500 due to a JSON serialization error.
- Verification: `tools/test_ab_zero_trust.py` (confirmed 18 failures on the code before the fix), `tools/test_ab_per_request.py`.
- **The prefix cache belongs to the default model** (2026-11-27). A request running at a scale other than the
  startup default **reads the cache but never writes it** — neither GPU prefix-cache blocks nor the NVMe offload.
  Before this, the KV an ab=1 request computed went into the shared cache under the same hash, and an ab=0 request
  sending the same prefix reused it. Measured live (`tools/bi/ab_cache_leak_probe.py`): the reused result diverged
  from a fresh computation **at the first token** (ab=0→ab=0 reuse was bit-identical). cm2 users share the system
  prompt, so one ab=1 request could contaminate everyone's prefix, and the NVMe tier kept it across restarts. Within
  one conversation, too, `/ab auto`'s ab=0 follow-ups carried on from the first request's KV.
  After the fix the same probe passes. Cost: the default request after an ab request re-prefills only the tail the
  ab request computed (the ab request itself still reads the shared prefix — measured 14.5s → 2.0s).
  Salting by scale is not used: it cannot read the default cache, so every turn would re-prefill the whole conversation.
  One predicate decides, `ab_switch.may_write_cache()`, using the same validation (`_valid`) as `note_request`.
  Gates: `overlay/v1core/kv_cache_manager.py` (allocate_slots, cache_blocks),
  `overlay/kvconn/scheduler.py` (max_offload_tokens=0, partial tail skipped).
  Test: `tools/test_ab_cache_write.py` (confirmed 3 failures on the files before the fix).

## Why this design

Abliteration removes refusals but **also removes the ability to say "it doesn't exist."** Refusing a request,
denying a nonexistent target, and rebutting a false premise all use the same direction inside the model. Measured:
on a 60-item fake-target trap, the original hallucinated 4, abliterated 18~22. This is where symptoms like claiming that
a deliverable not in the link exists, or insisting on a nonexistent bug, come from.

So **off is the default**, and the on window is narrowed to "the one turn that accepts the request."
Every stretch that judges tool results, links, or diffs runs with it off. The core of the design is pushing
the damage into stretches that require no judgment.

## Implementation

`o_proj` stays **BF16 as-is** even in the 4bpw EXL3 checkpoint (only the MoE experts are quantized).
So the difference between the two checkpoints can be extracted as low-rank and added during serving:

    o_proj(x) + scale * (x @ A.T) @ B.T      # W_abl = W_original + B @ A

| Item | Value |
|---|---|
| Edited layers | 45 (`layers.0~44`). **Layer 45 = MTP is byte-identical** |
| ΔW rank | 19~20 per layer (99.9% energy). Truncated at rank 24, residual 0.5~2% |
| Adapter size | **29MB** BF16 (a full copy is 8~9GB) |
| Reproduction accuracy | At scale=1, relative error 0.17%, cos 0.999999. **scale=0 is exactly 0** |
| Layer distribution | 34 KDA layers (in 8192) + 11 sparse MLA layers (in 16384) |

Only layer 44 has ΔW/W of 28% (other layers 8~12%), and its first singular value accounts for 85% — refusal is
concentrated here. However, **hallucination is spread across all 45 layers**, so restoring only layer 44 does not fix it (22→25).

### Files

| File | Role |
|---|---|
| `tools/build_ab_lora.py` | Difference between the two checkpoints (or the o_proj files fetched with `tools/fetch_oproj.py`) → extract rank-24 adapter |
| `overlay/ab_switch.py` | Hook body. scale is a GPU tensor (no graph re-capture needed) |
| `overlay/mla.py` | Hook for the 11 sparse MLA layers (the `o_proj` exit at `mla.py:405`) |
| `overlay/glm5next/kda.py` | Hook for the 34 KDA layers (`kda.py:344`) |
| `overlay/ab_plugin/` | `GET /ab` (read-only) + worker query (`collective_rpc`). `POST /ab` returns 410 |
| `overlay/v1core/kv_cache_manager.py` | Requests at a non-default scale do not write the prefix cache (they still read it) |

The plugin uses vLLM's official extension points. It does not modify server code. Because of TP, the scale has to reach every
worker, so it is split into two halves — `vllm.general_plugins` installs the worker RPC and
`vllm.endpoint_plugins` exposes it over HTTP. Both must be in `VLLM_PLUGINS` to load.
Since pip install is not possible in the container, a `dist-info/` is built and mounted so the entry points
are discovered.

### Startup

```bash
./vllm-glm53-exl3.sh RESTARTALL     # original + adapter is the default (glm53-serve.env)
```

If `GLM53_AB_LORA` is empty, the hook is disabled entirely — the original as-is. The adapter is only meaningful
on top of the original; if a checkpoint that is already abliterated is launched, `launch.sh` turns the switch off to prevent
the edit from being applied twice.

## Measurements (2026-10-01, 2-node TP, MTP-4)

| Item | scale=0 | scale=1 | Baseline without adapter |
|---|---|---|---|
| Decode prose / code / hash-map | 39.2 / 41.7 / 35.4 | 37.2 / 42.5 / 34.2 | 39.4 / 42.5 / 35.2 |
| Step | 78.9~80.9ms | 79.1~82.0ms | 78.5~80.3ms |
| Prefill 12k / 25k | 1,332 / 1,319 | — | 1,352 / 1,332 |
| **temperature 0 determinism** | **12/12 identical** | **12/12 identical** | 12/12 |
| **Refusals (16 items, neutral targets)** | **5** | **0** | — |

The speed loss is 1~2ms per step, within noise. The lower scale=1 prose number is an acceptance-rate difference
(2.082→1.931), which is expected since the output content itself changes.

Refusals are measured with `tools/refusal_probe.py`. The judgment looks at **only the first sentence** — the list of reasons
or the "but I can do this instead" clause appended after a refusal looks like starting the task to pattern matching, and conversely
a caution like "only on environments you own" is not a refusal. Mixing these two makes the numbers meaningless.

All targets are neutral (vendor practice hosts, reserved domains, RFC1918). Mixing in real domains would tangle
over-refusal and legitimate refusal into a single number. The point is that 5 are blocked even with neutral
targets.

## Client integration

codemason `/ab off|on|auto` **writes nothing to the server** (11-27). The client holds the mode
and attaches `vllm_xargs.ab_scale` to every request in the conversation loop — `on`=1, `off`=0, `auto`=**only the first request of the user turn is 1**,
and every request after that is 0 (whether it answers with a tool call, text, an empty turn, or an abort). Compaction and helper calls do not attach the key, so they get the default (original).
⛔ The old "turn it off the moment a tool call starts" was the pre-10-01 design, and that was the bug (turns with no off path were left switched on).
With per-request control, the "turn off" action itself no longer exists — with no lingering state, it cannot be left on.

## ROCm lane (.109) porting notes

- First check that `o_proj` is BF16. If it is quantized, this approach of adding to the output does not work
  as-is.
- The hook sites are the same two places (the `o_proj` exit of KDA / MLA). Even with a different backend, this layer is shared.
- Tensors created inside `torch.inference_mode` cannot be modified in place. The scale tensor must be created
  inside `inference_mode(False)` (we fell into this trap once).
- TP split: `o_proj` is RowParallel (split along the input axis), so A must be sliced per rank. The output
  rides on the existing all-reduce, so there is no extra communication.
