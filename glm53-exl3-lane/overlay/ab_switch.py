"""Runtime on/off switch for the abliteration edit (o_proj only).

The lovesenko abliteration is an output-space projection on ``self_attn.o_proj``
in 45 of the 46 decoder layers. Those tensors stay BF16 even in the 4bpw EXL3
checkpoint -- only the MoE experts are trellis-quantized -- so the edit can be
kept out of the weights and applied at serving time instead:

    o_proj(x) + scale * (x @ A.T) @ B.T

with ``W_abliterated = W_original + B @ A`` (rank 24, residual 0.5-2%). The
served checkpoint is the original one, so ``scale = 0`` is the original model
and ``scale = 1`` is the abliterated one, switchable without reloading,
re-capturing CUDA graphs or re-prefilling -- and switchable **per request**:
each token carries its own scale, so co-batched sequences from different
clients do not share the knob (see "per-request scale" below).

Why a switch instead of picking one checkpoint: abliteration removes the
model's ability to say "no", and refusing a request and saying "that does not
exist" share that machinery -- the abliterated model fabricates 18-22/60 on
nonexistent-subject traps versus 4/60 original. Keeping it off except for the
turn that actually needs a request accepted confines the damage to a turn that
only has to say yes, while every tool result, link and diff after it is judged
by the original weights.

``scale`` lives in a GPU tensor that CUDA graphs capture by reference, so
flipping it needs no re-capture. The adapter is ~29MB BF16 for all 45 layers.

Wiring: ``glm5next/kda.py`` (34 KDA layers, in_features 8192) and the lane's
``mla.py`` overlay (11 sparse-MLA layers, in_features 16384) call
``apply(prefix, x, out)`` right after their ``o_proj``. Tensor-parallel splits
along in_features are handled by slicing A to the local shard: o_proj is a
RowParallelLinear, so each rank owns a slice of the input. The partial sums do
NOT ride o_proj's all-reduce -- the hook runs after it -- so `apply` reduces
them itself, at the rank-r intermediate. Getting that wrong landed half the
edit at TP=2 and went unnoticed for eleven days (see the note in `apply`).

Env:
    GLM53_AB_LORA   path to the adapter directory (unset = feature off)
    GLM53_AB_SCALE  default 0.0: the scale of every request that does not carry
                    its own ``ab_scale``. Fixed at startup -- nothing reachable
                    at runtime (HTTP or worker RPC) can change it.

Zero trust (2026-11-27). The only way to get the edit is to ask for it on your
own request, ``{"vllm_xargs": {"ab_scale": 1}}``, and that changes only your
request's tokens. There used to be a second way, ``POST /ab``, which wrote the
engine-wide default: every request that did not carry its own scale followed
it, so one client flipping it changed every other client's model (measured on
the live lane: a plain request's text and logprobs changed after another client
posted ``{"scale":1}``). Clients are not trusted -- the code is MIT-licensed
and anyone can send anything -- so the server no longer offers that knob;
``POST /ab`` is refused and the worker RPC that set it is gone.
"""

from __future__ import annotations

import math
import os
import re
import threading

import torch

from vllm.logger import init_logger

# Named under "vllm." on purpose. vLLM configures a handler on the "vllm"
# logger with propagate=False and leaves the root logger bare, so a logger
# named "ab_switch" has no handler anywhere on its chain and sits at the root's
# WARNING level -- every line below was silently dropped. Verified in the live
# container 2026-11-20, while chasing a switch that looked dead because it had
# never logged a thing in six weeks. The hook was fine; the logging was not.
logger = init_logger("vllm." + __name__)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")

_lock = threading.Lock()
_adapters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_scale: torch.Tensor | None = None
_loaded_from: str | None = None

# --- per-request scale ------------------------------------------------------
#
# ``scale`` above is ONE engine-global scalar, so on a shared server every
# client writes the same knob. One developer's turn ending pushed 0.0 into the
# middle of another's generation and no client-side fix could prevent it: the
# unit of the knob is the forward pass, not the request (2026-11-20).
#
# The fix is to stop sharing it. ``delta`` is computed for every token whatever
# the scale is -- scale=0 still runs both GEMMs -- so turning the multiplier
# into a per-token vector costs no extra FLOPs and lets co-batched sequences
# carry different scales. A 0 row reproduces the original weights exactly
# (0 * delta is added), so a client that asks for the original gets it even
# while another sequence in the same batch is abliterated.
#
# The prefix cache is the default model's alone (2026-11-27). A request that
# runs at a scale other than the startup default READS the shared cache but
# never WRITES to it -- neither GPU prefix-cache blocks (kv_cache_manager.py)
# nor the NVMe offload (kvconn/scheduler.py, max_offload_tokens=0). Before this,
# the KV an ab_scale=1 request computed went into the shared cache under the
# same hash, and any later request with that prefix reused it: measured live
# with tools/bi/ab_cache_leak_probe.py, an ab=0 request reusing an ab=1-primed
# prefix diverged from its fresh computation at the FIRST token, while ab=0
# reusing ab=0 was bit-identical. Every cm2 user shares the system prompt, so
# one ab=1 request landing first after a restart or eviction abliterated that
# prefix for everyone, and the NVMe tier kept it across restarts. It also
# leaked inside one conversation: /ab auto's ab=0 follow-ups reused the KV of
# what the turn's ab=1 request had prefilled.
#
# Salting the hash with the scale (the earlier idea) separates them too, but
# it cannot read the default cache either, so every scale change re-prefills
# the whole conversation (15.66s cold vs 0.89s warm on 21,294 tokens). Read-
# only costs far less: the ab request reuses the shared prefix like anyone
# else, and the next default request recomputes only what the ab request
# added on top of it.

_scale_vec: torch.Tensor | None = None
_scale_stage: torch.Tensor | None = None
_default_scale: float = float(os.getenv("GLM53_AB_SCALE", "0") or 0.0)
_req_scales: dict[str, float] = {}
_fills: int = 0
_rejected: int = 0


def enabled() -> bool:
    return bool(os.getenv("GLM53_AB_LORA"))


def _scale_tensor(device: torch.device) -> torch.Tensor:
    """The live scale, as a tensor CUDA graphs can capture by reference.

    Built under ``inference_mode(False)``: a tensor allocated inside inference
    mode is marked an inference tensor and PyTorch then refuses the in-place
    write ``set_scale`` needs, from any context. The forward pass that first
    reaches here is inside inference mode, so the exemption has to be here.
    """
    global _scale
    if _scale is None or _scale.device != device:
        init = float(os.getenv("GLM53_AB_SCALE", "0") or 0.0)
        with torch.inference_mode(False):
            _scale = torch.tensor(init, dtype=torch.float32, device=device)
    return _scale


def get_scale() -> float:
    """The scale a request that carries no ``ab_scale`` of its own will get.

    ``_default_scale``, not the scalar tensor. In per-token mode that mirror is
    what ``fill_token_scales`` lays into the step's vector, and in the fallback
    path ``set_scale`` keeps the scalar in step with it -- so this is the honest
    answer in both, while ``_scale`` reads 0.0 forever once the per-token buffer
    exists, because ``_token_scale`` then never calls ``_scale_tensor``.

    Returning the scalar is what made ``GET /ab`` answer
    ``{"scale":0.0,"mode":"original"}`` over a server holding the edit ON, with
    ``default_scale`` 1.0 sitting in the same body; codemason believed the top
    level and printed ``server: off (restored)`` for it (2026-11-21). It also
    killed the endpoint's ``consistent`` without a sound: that is derived from
    these per-worker values, so every rank answering 0.0 made a real
    disagreement -- the halves of the model on different weights -- unreportable.

    The raw scalar stays visible as ``info()["scalar"]``, where ``None`` says
    the per-token buffer owns the value.
    """
    return float(_default_scale)


def set_scale(value: float) -> float:
    """Set the engine-wide default. In-process only: startup and tests.

    Deliberately NOT reachable from outside -- no HTTP route and no worker RPC
    calls this (see the zero-trust note at the top). Exposing it is what let
    one client change every other client's model.

    Written in place so any CUDA graph that captured the tensor sees the new
    value without being re-captured.
    """
    global _default_scale
    # The per-token path reads this mirror when it builds a step's vector, so
    # the global knob has to land here too or `/ab` would stop having any
    # effect on requests that do not carry their own ab_scale.
    _default_scale = float(value)
    if _scale is None:
        # No forward has run yet, so there is no tensor to write; make the
        # value the starting point the first forward will pick up.
        os.environ["GLM53_AB_SCALE"] = str(float(value))
        return float(value)
    with torch.inference_mode(False):
        _scale.fill_(float(value))
    return float(value)


def _layer_of(prefix: str) -> int | None:
    m = _LAYER_RE.search(prefix)
    return int(m.group(1)) if m else None


def _load(layer: int, device: torch.device, dtype: torch.dtype):
    """Load one layer's (A, B) from the adapter file, sliced for this rank."""
    from safetensors import safe_open

    root = os.getenv("GLM53_AB_LORA", "")
    path = os.path.join(root, "ab_lora.safetensors")
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        ka, kb = f"layers.{layer}.A", f"layers.{layer}.B"
        if ka not in keys:
            return None  # layer was not edited (the MTP layer)
        a = f.get_tensor(ka)
        b = f.get_tensor(kb)
    return a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def reserve(max_tokens: int, device: torch.device) -> None:
    """Allocate the per-token scale buffer before any forward runs.

    This has to happen before CUDA graph capture. A graph bakes in whichever
    branch ``apply`` took, and ``_dummy_run`` builds its own scheduler output
    instead of going through ``prepare_inputs`` -- so the profiling and capture
    forwards reach ``apply`` with no fill of their own. The model runner calls
    this from ``__init__``, which cannot be reordered after them.

    Sized once at ``max_num_batched_tokens`` and never grown: every captured
    graph holds this pointer, so reallocating would orphan all of them.
    """
    global _scale_vec, _scale_stage
    if not enabled() or _scale_vec is not None:
        return
    with torch.inference_mode(False):
        _scale_vec = torch.full(
            (int(max_tokens),), _default_scale, dtype=torch.float32, device=device
        )
        _scale_stage = torch.empty(
            (int(max_tokens),), dtype=torch.float32, device="cpu", pin_memory=True
        )
    logger.info(
        "[ab] 토큰별 scale 버퍼 %d 예약 (기본 %.2f)", int(max_tokens), _default_scale
    )


def _valid(raw) -> bool:
    """An ``ab_scale`` the server honours: a real, finite number in [0, 1].

    int or float, not bool. Shared by ``note_request`` (what the forward pass
    uses) and ``may_write_cache`` (what the cache may keep), so the two can
    never disagree about which scale a request actually runs at.
    """
    return (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(raw)
        and 0.0 <= float(raw) <= 1.0
    )


def may_write_cache(sampling_params) -> bool:
    """Whether a request's KV may enter the shared prefix cache / NVMe tier.

    Only a request that runs at the startup default may write. One that runs
    at another scale reads the cache like everyone else but keeps what it
    computes to itself -- see the block above ``_scale_vec``. Judged exactly
    as ``note_request`` judges it: a missing or refused ``ab_scale`` runs at
    the default and therefore may write.
    """
    if not enabled():
        return True
    extra = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra, dict) or "ab_scale" not in extra:
        return True
    raw = extra.get("ab_scale")
    if not _valid(raw):
        return True
    return float(raw) == _default_scale


def note_request(req_id: str, sampling_params) -> None:
    """Record a request's own scale, sent as ``{"vllm_xargs": {"ab_scale": 0|1}}``.

    The value comes from a client and is not trusted. Only a real number
    (int or float, not bool), finite, in [0, 1] is accepted; anything else is
    ignored and the request runs on the default -- the original weights. It
    used to be clamped with ``min(max(v, 0), 1)``, which passes NaN through
    unchanged (both comparisons are false), so ``"ab_scale": "nan"`` stored a
    NaN that went into the forward pass and also broke ``GET /ab``'s JSON while
    the request was tracked. Out-of-range values are refused rather than
    clamped for the same reason: a client that sends 7.5 is not asking for 1.

    Absent leaves the request on the default, so a client that knows nothing
    about this keeps working unchanged.
    """
    global _rejected
    if not enabled():
        return
    extra = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra, dict) or "ab_scale" not in extra:
        _req_scales.pop(req_id, None)
        return
    raw = extra.get("ab_scale")
    if not _valid(raw):
        _rejected += 1
        _req_scales.pop(req_id, None)
        logger.warning("[ab] req %s: ab_scale=%r 거부 — 유한한 0~1 숫자만. 원본으로 돈다", req_id, raw)
        return
    _req_scales[req_id] = float(raw)


def forget_request(req_id: str) -> None:
    _req_scales.pop(req_id, None)


def fill_token_scales(req_ids, num_scheduled_tokens, num_padded: int) -> None:
    """Write this step's per-token scales into the reserved buffer.

    Called from ``prepare_inputs``, which runs before the forward of every real
    step. Tokens are laid out contiguously per request in ``req_ids`` order --
    ``query_start_loc`` is the cumsum of ``num_scheduled_tokens`` -- so the
    vector is each request's scale repeated by its token count. Speculative
    draft tokens are counted in a request's own total, so they inherit its
    scale, which is what verification against that request needs.
    """
    global _fills
    vec = _scale_vec
    if vec is None:
        return
    import numpy as np

    num_tokens = int(num_scheduled_tokens.sum())
    if num_tokens > vec.numel():
        # Cannot happen below max_num_batched_tokens; if it ever does, `apply`
        # falls back to the global scalar and this fill would only mislead.
        logger.error(
            "[ab] 토큰 %d > 버퍼 %d — 이 스텝은 전역값으로 돈다", num_tokens, vec.numel()
        )
        return
    scales = np.fromiter(
        (_req_scales.get(r, _default_scale) for r in req_ids),
        dtype=np.float32,
        count=len(req_ids),
    )
    with torch.inference_mode(False):
        if scales.size and bool((scales == scales[0]).all()):
            # One value across the batch: everyone on the global knob, or a
            # single client. Fill on device and skip the host round trip.
            vec[:num_tokens].fill_(float(scales[0]))
        else:
            assert _scale_stage is not None
            _scale_stage.numpy()[:num_tokens] = np.repeat(scales, num_scheduled_tokens)
            vec[:num_tokens].copy_(_scale_stage[:num_tokens], non_blocking=True)
        if num_padded > num_tokens:
            # Padded rows are discarded downstream; 0 keeps them from adding
            # anything in the meantime.
            vec[num_tokens:num_padded].fill_(0.0)
    _fills += 1


def info() -> dict:
    """What the switch is actually doing, for ``GET /ab``.

    Worth exposing because every ``[ab]`` log line was dropped on the floor for
    six weeks (see the logger comment at the top), so the HTTP surface is the
    only honest witness this thing has.
    """
    return {
        "scale": get_scale(),
        "default_scale": _default_scale,
        # None once `reserve` has run: the per-token vector owns the value and
        # `_scale` is dead weight kept only for the fallback path.
        "scalar": None if _scale is None else float(_scale.item()),
        "per_token": _scale_vec is not None,
        "buffer_tokens": 0 if _scale_vec is None else int(_scale_vec.numel()),
        "fills": _fills,
        "tracked_requests": len(_req_scales),
        "overrides": sorted(set(_req_scales.values())),
        # Requests whose ab_scale was refused (non-finite, out of range, not a
        # number) and ran on the default instead.
        "rejected": _rejected,
        # The default cannot be changed at runtime; say so, so a client that
        # still tries `POST /ab` can see why it got refused.
        "global_write": False,
    }


def _token_scale(device: torch.device, num_tokens: int) -> torch.Tensor:
    """The multiplier ``apply`` uses: a ``[tokens, 1]`` column, or the scalar.

    The branch is captured by CUDA graphs, so it must not differ between
    capture and replay. That is why ``reserve`` runs before any forward and the
    buffer is never reallocated: once the vector exists, this always returns
    the vector for every shape a graph can hold.
    """
    vec = _scale_vec
    if vec is None or vec.device != device or num_tokens > vec.numel():
        return _scale_tensor(device)
    return vec[:num_tokens].unsqueeze(1)


def apply(prefix: str, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Add ``scale * (x @ A.T) @ B.T`` to ``out``, an o_proj result.

    ``x`` is the o_proj input on this rank (already the local shard under TP),
    ``out`` its output. Returns ``out`` unchanged when the feature is off or
    this layer was never edited.
    """
    if not enabled():
        return out
    layer = _layer_of(prefix)
    if layer is None:
        return out

    with _lock:
        if layer not in _adapters:
            try:
                _adapters[layer] = _load(layer, out.device, out.dtype)
            except Exception:
                logger.exception("[ab] layer %d 어댑터 로드 실패 — 이 층은 끔", layer)
                _adapters[layer] = None
            global _loaded_from
            if _loaded_from is None:
                _loaded_from = os.getenv("GLM53_AB_LORA")
                logger.info("[ab] 어댑터 %s, 초기 scale %.1f", _loaded_from, get_scale())
        pair = _adapters[layer]
    if pair is None:
        return out

    a, b = pair
    # Under TP the rank owns x's last dim slice; A must match it. The adapter is
    # stored whole, so take this rank's columns the first time we see the shape.
    if a.shape[1] != x.shape[-1]:
        if a.shape[1] % x.shape[-1] != 0:
            logger.error(
                "[ab] layer %d: A in_features %d 가 입력 %d 와 안 맞음 — 이 층은 끔",
                layer, a.shape[1], x.shape[-1],
            )
            with _lock:
                _adapters[layer] = None
            return out
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        width = x.shape[-1]
        a = a[:, rank * width : (rank + 1) * width].contiguous()
        with _lock:
            _adapters[layer] = (a, b)

    # Two skinny GEMMs: [tokens, in] @ [in, r] then [tokens, r] @ [r, out].
    # scale is a device tensor so a captured CUDA graph picks up a flip without
    # being re-captured; it multiplies the rank-r intermediate, which is the
    # cheapest place to put it and keeps the off state exact (adds 0).
    # Per token, not per engine: see the block above `_scale_vec`. Falls back
    # to the global scalar when the buffer was never reserved (feature off at
    # startup, or a forward wider than max_num_batched_tokens).
    scale = _token_scale(out.device, x.shape[0])
    delta = torch.nn.functional.linear(x, a)
    # LANE FIX (2026-10-12): the rank-local term has to be all-reduced, and was
    # not. o_proj is a RowParallelLinear with reduce_results left at its default,
    # so the `out` this is called with has ALREADY been summed across ranks --
    # the partial sums do not "ride the existing all-reduce", they arrive after
    # it. Each rank was therefore adding only its own share of the edit, and the
    # ranks' hidden states diverged from this layer on. It read as working
    # because it was not nothing: at TP=2 the adapter landed about half the edit,
    # so a flat refusal softened into a conditional one instead of opening. The
    # pure abliterated checkpoint answers the same prompt outright; this path did
    # not, with an adapter verified to reconstruct the delta to 1.7%.
    #
    # Reduced at the rank-r intermediate, not at the output: r is 24, so this is
    # 24 numbers per token against hidden-size, and the identity holds either way
    # because B is replicated -- (sum_r x_r A_r^T) B^T == sum_r (x_r A_r^T) B^T.
    # TP=1 keeps the old path exactly (world size 1, no collective).
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )

    if get_tensor_model_parallel_world_size() > 1:
        delta = tensor_model_parallel_all_reduce(delta)
    delta = delta * scale.to(delta.dtype)
    return out + torch.nn.functional.linear(delta, b)
