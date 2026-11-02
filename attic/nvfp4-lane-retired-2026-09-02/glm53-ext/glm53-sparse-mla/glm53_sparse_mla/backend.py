# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend for NoPE sparse MLA on sm_120 / sm_121.

Fills the gap that stops GLM-5.3-Flash from running under vLLM on GB10: the
model is NoPE (`qk_rope_head_dim == 0`), and on capability 12.x every MLA
*prefill* backend rejects its dims while the only sparse *decode* backend
accepts `fp8_ds_mla` alone, whose cache kernel asserts `pe_dim == 64`.

Both problems dissolve here:

* **head size 512, not 576.** With `pe_dim == 0` there is no rope block to pack,
  so the `fp8_ds_mla` layout -- and its assert -- is never reached, and the
  "NoPE pe-pad" (a fabricated 64-wide all-zero pe, with its interleaved
  `q_b_proj` padding and its softmax-temperature footgun) becomes unnecessary.
* **no dense-MHA prefill path.** Every token, prefill included, goes through the
  top-k sparse MQA path. That is exact rather than an approximation: when
  `seq_len <= topk` the indexer selects everything, and above it the sparse path
  is what the model was trained for. Following the XPU sparse backend, this is
  expressed by reporting all tokens as decode tokens so the shared MLA layer's
  `num_mha_tokens` stays 0 and the dense branch is never entered.

Registration deliberately overrides the existing `FLASHINFER_MLA_SPARSE_SM120`
enum slot rather than using `CUSTOM`: `platforms/cuda.py` already lists that slot
in the `major == 12` MLA priority list, so selection needs no plumbing and no
`--attention-backend` flag. Gate with `VLLM_GLM53_CUDA_SPARSE_MLA=1`; unset, this
module is inert.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
# --- vLLM API compatibility -------------------------------------------------
# The deployed image is a snapshot of PR #53906 and predates two changes that
# landed in main. Feature-detect rather than pinning to either, so the same file
# works on the image today and after an image bump:
#
#   * `flat_kv_row_view` (sparse_utils) does not exist in the image. It handles
#     the case where other layers' pages sit between consecutive blocks of this
#     cache, making the flat row stride exceed block_size. The image's own XPU
#     sparse backend just does `.view(-1, head_dim)`, and the image's index
#     converter assumes the same contiguity, so the two are consistent -- we
#     assert it rather than trusting it.
#   * `BLOCK_STRIDE_ROWS` is not a parameter of the image's converter.
#
# The converter itself lives in sparse_utils in both, but the image's XPU
# backend imports it from flashmla_sparse, so try both paths.
import inspect as _inspect

try:
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )
except ImportError:  # pragma: no cover
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index,
    )

try:
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
except ImportError:
    flat_kv_row_view = None

_CONVERTER_TAKES_STRIDE = (
    "BLOCK_STRIDE_ROWS"
    in _inspect.signature(triton_convert_req_index_to_global_index).parameters
)
from vllm.v1.kv_cache_interface import AttentionSpec

import glm53_sparse_mla  # registers torch.ops.glm53_sparse_mla.sparse_fwd

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

KV_LORA_RANK = glm53_sparse_mla.KV_LORA_RANK       # 512
HEADS_PER_RANK = glm53_sparse_mla.HEADS_PER_RANK   # 32


class Glm53SparseMLABackend(AttentionBackend):
    # bf16 KV only. The kernel reads the cache directly as bf16 rows; there is no
    # fp8 path yet (see README -- an fp8 variant is the KV-capacity follow-up,
    # since bf16 rows are 1024 B against fp8_ds_mla's 656 B).
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        # Keep the overridden slot's name so logs and the backend selector agree.
        return "FLASHINFER_MLA_SPARSE_SM120"

    @staticmethod
    def get_metadata_cls() -> type["Glm53SparseMLAMetadata"]:
        return Glm53SparseMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["Glm53SparseMLAMetadataBuilder"]:
        return Glm53SparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["Glm53SparseMLAImpl"]:
        return Glm53SparseMLAImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Present in the deployed image's backends; harmless on main.
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank + qk_rope_head_dim = 512 + 0. This single line is what
        # makes the pe-pad unnecessary.
        return [KV_LORA_RANK]

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        if not (use_mla and use_sparse):
            return "glm53 sparse MLA requires sparse MLA attention"
        if head_size != KV_LORA_RANK:
            return (
                f"glm53 sparse MLA is a NoPE kernel: head_size must be "
                f"{KV_LORA_RANK} (qk_rope_head_dim == 0), got {head_size}"
            )
        if has_sink:
            return "glm53 sparse MLA does not support attention sinks"
        return None


@dataclass
class Glm53SparseMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    block_size: int = 1
    topk_tokens: int = 2048

    # `mla_attention.py::forward_impl` reads these unconditionally to split MQA
    # from dense-MHA tokens. Reporting every token as a decode token keeps
    # `num_mha_tokens` at 0, so the dense branch -- which on sm_121 has no
    # backend at all -- is never entered. This is the mechanism that replaces
    # the FlashAttn prefill-dims whitelist patch.
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0


@dataclass
class Glm53SparseMLAMetadataBuilder(
    AttentionMetadataBuilder[Glm53SparseMLAMetadata]
):
    # CUDA graphs are not wired up yet: the kernel is graph-safe (no host sync,
    # fixed launch shape per token count), but nothing here has been validated
    # under capture, so do not claim it.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Glm53SparseMLAMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        return Glm53SparseMLAMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=common_attn_metadata.num_reqs,
            num_prefills=0,
            num_decode_tokens=num_tokens,
        )


class Glm53SparseMLAImpl(MLAAttentionImpl[Glm53SparseMLAMetadata]):
    is_sparse = True
    # No dense-MHA prefill path: everything runs through forward_mqa.
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "glm53 sparse MLA does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        if num_heads != HEADS_PER_RANK:
            raise NotImplementedError(
                f"glm53 sparse MLA's warp decomposition is fixed to "
                f"{HEADS_PER_RANK} heads per rank (TP=2 for GLM-5.3-Flash); got "
                f"{num_heads}. 64 heads would need 114944 B of shared memory "
                f"against the 101376 B sm_12x ceiling."
            )
        # The indexer carries the shared buffer for normal layers; the explicit
        # buffer covers backbone skip layers, whose indexer is not constructed.
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: Glm53SparseMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "glm53 sparse MLA is bf16-KV only; pass --kv-cache-dtype bfloat16"
            )
        # With pe_dim == 0 there is no q_pe half, so q never arrives as a tuple;
        # concatenate defensively rather than asserting on a hot path.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # The kpool indexer emits topk + (kpool - 1) columns, which vLLM rounds up
        # to a multiple of 128 (2048 -> 2176). Narrow at the call site rather than
        # resizing the indexer's buffer: touching what the indexer writes risks it
        # selecting nothing, and an all--1 row makes MLA return zeros, which
        # presents as the model copying its prompt verbatim.
        cap = int(attn_metadata.topk_tokens)
        if topk_indices.shape[1] > cap:
            topk_indices = topk_indices[:, :cap].contiguous()

        head_dim = kv_c_and_k_pe_cache.shape[-1]
        conv_kwargs = {}
        if flat_kv_row_view is not None:
            kv_rows, block_stride_rows = flat_kv_row_view(
                kv_c_and_k_pe_cache, attn_metadata.block_size
            )
            if _CONVERTER_TAKES_STRIDE:
                conv_kwargs["BLOCK_STRIDE_ROWS"] = block_stride_rows
        else:
            # Older vLLM: no flat_kv_row_view, and the converter emits
            # block_table[...] * BLOCK_SIZE + offset, which is only a valid flat
            # row index when pages are contiguous. Check it instead of assuming
            # -- a violated assumption here silently reads the wrong KV rows.
            stride0 = kv_c_and_k_pe_cache.stride(0)
            assert stride0 == attn_metadata.block_size * head_dim, (
                f"paged KV block stride {stride0} != block_size * head_dim "
                f"({attn_metadata.block_size} * {head_dim}); this vLLM's index "
                f"converter cannot express that layout"
            )
            kv_rows = kv_c_and_k_pe_cache.view(-1, head_dim)

        topk_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            **conv_kwargs,
        )

        out = q.new_empty((num_actual_toks, self.num_heads, self.kv_lora_rank))
        torch.ops.glm53_sparse_mla.sparse_fwd(
            q, kv_rows, topk_global, self.softmax_scale, out
        )
        return out, None


def register() -> bool:
    """Override the sm120 sparse-MLA slot. No-op unless the env gate is set."""
    if os.environ.get("VLLM_GLM53_CUDA_SPARSE_MLA", "") not in ("1", "true", "TRUE"):
        return False
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
        "glm53_sparse_mla.backend.Glm53SparseMLABackend",
    )
    logger.info(
        "glm53_sparse_mla: overrode FLASHINFER_MLA_SPARSE_SM120 with the NoPE "
        "CUDA sparse-MLA backend (head_size=%d, bf16 KV)", KV_LORA_RANK
    )
    return True
