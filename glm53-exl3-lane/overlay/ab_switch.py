"""Runtime on/off switch for the abliteration edit (o_proj only).

The lovesenko abliteration is an output-space projection on ``self_attn.o_proj``
in 45 of the 46 decoder layers. Those tensors stay BF16 even in the 4bpw EXL3
checkpoint -- only the MoE experts are trellis-quantized -- so the edit can be
kept out of the weights and applied at serving time instead:

    o_proj(x) + scale * (x @ A.T) @ B.T

with ``W_abliterated = W_original + B @ A`` (rank 24, residual 0.5-2%). The
served checkpoint is the original one, so ``scale = 0`` is the original model
and ``scale = 1`` is the abliterated one, switchable per forward pass without
reloading, re-capturing CUDA graphs or re-prefilling.

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
    GLM53_AB_SCALE  initial scale, default 0.0
"""

from __future__ import annotations

import os
import re
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")

_lock = threading.Lock()
_adapters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_scale: torch.Tensor | None = None
_loaded_from: str | None = None


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
    return 0.0 if _scale is None else float(_scale.item())


def set_scale(value: float) -> float:
    """Flip the edit on (1.0) or off (0.0). Safe to call between forwards.

    Written in place so any CUDA graph that captured the tensor sees the new
    value without being re-captured.
    """
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
    scale = _scale_tensor(out.device)
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
