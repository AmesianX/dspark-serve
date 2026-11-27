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
- **KV identity = (tokens, scale) — a prefix-cache namespace per scale** (2026-11-27).
  The chain hash stays tokens-only; KV **computed** at a non-default scale is registered under
  `tag_hash(chain_hash, scale)` (GPU prefix cache and NVMe alike). A default request looks up untagged keys
  only, so it can **never** be served ab KV. An ab request looks up, per block/chunk, its own namespace first
  and the default one second. The tag sits outside the chain, so "default prefix + ab tail" links up →
  both `/ab on` and `/ab auto` stay cached.
  - Why: before, ab=1 KV went into the shared cache under the same hash, and an ab=0 request reusing that prefix
    diverged **at the first token** (`tools/bi/ab_cache_leak_probe.py`). cm2 users share the system prompt, so one
    ab=1 request could contaminate everyone, and the NVMe tier kept it across restarts. The first fix ("ab requests
    read only") closed the leak but left `/ab on` with no cache at all (three ab=1 requests on one prefix: 14.2s each).
    Salting the chain by scale (LoRA-style) would stop `/ab auto`'s first request reading the default prefix — hence
    the tag outside the chain.
  - NVMe: stores always use the request's own tagged keys, and the hit range is never re-stored (default KV is not
    copied under a tag). Lookups, inside the contiguous hit range, try own → default per chunk and boundary key; a
    fallback hit is recorded and that key is used for load and touch. There is structurally no path that stores ab
    KV under an untagged key.
  - Measured (`tools/bi/ab_cache_namespace_probe.py`, all PASS, including a real RESTARTALL):

    | Case | Result |
    |---|---|
    | ab=0 reusing a prefix first prefilled by ab=1 | bit-identical to a fresh computation (no leak) |
    | `/ab on`: three ab=1 requests on one prefix | 14.8 → 2.4 → 2.5s |
    | `/ab auto` turn 2 ab=1 / following ab=0 (22.7K, fresh 18.7s) | 7.1s / 7.2s, ab=0 bit-identical to fresh |
    | After restart, NVMe: ab=1 turn 2 / ab=0 turn 2 (24.3K, fresh 19.9s) | **6.2s** (default + own chunks) / 12.9s (default chunks only, bit-identical to fresh) |
    | After restart, NVMe: `/ab on` chain (no default key exists) | 8.2s (fresh 14.8s), output bit-identical to a fresh ab=1 computation |
    | Decode t/s (kmeas) | 35.0 / 39.1 / 44.9 — unchanged |
  - One predicate: `ab_switch.cache_namespace()` (same `_valid` as `note_request`). Entry: `KVCacheManager.get_computed_blocks`
    (the single entry of every lookup; `block_pool.lookup_namespace` is set only for that call); registration:
    `block_pool.cache_full_blocks` / `cache_partial_block`.
  - Test: `tools/test_ab_cache_namespace.py` (confirmed leak / no-namespace / no-fallback failures on the original files).

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
| `overlay/v1core/block_pool.py` | Per-scale cache keys (tags); an ab request looks up own → default |
| `overlay/v1core/kv_cache_manager.py` | Sets the request's namespace for exactly one lookup |
| `overlay/kvconn/scheduler.py` | Same rule for NVMe keys (store = own tag, lookup = own → default) |

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
