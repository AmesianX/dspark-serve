# SPDX-License-Identifier: Apache-2.0
# [ours] GLM-5.3 lane: per-rank NVMe KV offload tier for vLLM's OffloadingConnector.
#
# Independent implementation against vLLM's OffloadingSpec/OffloadingManager/
# OffloadingWorker API (Apache-2.0). Design notes: glm53-exl3-lane/KV_OFFLOAD.md.
#
#  * No CPU tier. The scheduler-side manager is vLLM's LRU CPUOffloadingManager
#    with "block id" reinterpreted as an NVMe slot number.
#  * Each rank owns one preallocated file <root>/<fingerprint>/r<rank>.kv and
#    writes/reads only its own shard. Nothing crosses nodes; the scheduler
#    broadcasts identical slot decisions to every rank.
#  * Slot = 4 KiB header + payload, O_DIRECT, through a per-thread pinned
#    bounce buffer. No page cache, so no dirty-page build-up on GB10's unified
#    memory.
#  * Every slot carries magic, fingerprint, key, group, length and crc32. Each
#    rank verifies on load; a mismatch fails the job (-> recompute), it never
#    feeds stale bytes to the model.
#  * The rank-0 headers are the persistent index: after a restart the manager
#    rebuilds its LRU from them, so the cache survives restarts.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import threading
import time
import zlib
from collections.abc import Collection, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    ReqContext,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = init_logger(__name__)

FORMAT_VERSION = 1
ALIGN = 4096
HEADER_BYTES = 4096
MAGIC = b"GLMKVNV1"
_KEY_MAX = 64
# magic, version, header_crc, fingerprint, key_len, group, payload_len,
# payload_crc, write_seq, wall_time, key
_HDR = struct.Struct("<8sII32sHIQIQd64s")
assert _HDR.size <= HEADER_BYTES


def _round_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------
# slot header
# --------------------------------------------------------------------------


def pack_header(
    fingerprint: str,
    key: bytes,
    group: int,
    payload_len: int,
    payload_crc: int,
    write_seq: int,
) -> bytes:
    if len(key) > _KEY_MAX:
        raise ValueError(f"offload key too long: {len(key)}")
    fp = fingerprint.encode()[:32]
    fields = [MAGIC, FORMAT_VERSION, 0, fp, len(key), group, payload_len,
              payload_crc, write_seq, time.time(), key]
    raw = _HDR.pack(*fields)
    fields[2] = zlib.crc32(raw)
    return _HDR.pack(*fields)


def unpack_header(buf: bytes | memoryview, fingerprint: str):
    """Return (key, group, payload_len, payload_crc, write_seq) or None."""
    try:
        (magic, ver, hcrc, fp, klen, group, plen, pcrc, seq, _wall,
         key) = _HDR.unpack_from(buf, 0)
    except struct.error:
        return None
    if magic != MAGIC or ver != FORMAT_VERSION:
        return None
    raw = bytearray(bytes(buf[: _HDR.size]))
    struct.pack_into("<I", raw, 12, 0)
    if zlib.crc32(bytes(raw)) != hcrc:
        return None
    if fp.rstrip(b"\0") != fingerprint.encode()[:32]:
        return None
    if klen > _KEY_MAX:
        return None
    return bytes(key[:klen]), group, plen, pcrc, seq


# --------------------------------------------------------------------------
# config fingerprint
# --------------------------------------------------------------------------


def _checkpoint_identity(model_name: str) -> list:
    ident: list = [model_name]
    try:
        if os.path.isdir(model_name):
            cfg = os.path.join(model_name, "config.json")
            if os.path.isfile(cfg):
                with open(cfg, "rb") as f:
                    ident.append(hashlib.sha256(f.read()).hexdigest())
            sizes = sorted(
                (n, os.path.getsize(os.path.join(model_name, n)))
                for n in os.listdir(model_name)
                if n.endswith(".safetensors")
            )
            ident.append(sizes)
    except OSError:
        pass
    return ident


def compute_fingerprint(config: OffloadingConfig, slot_bytes: int) -> str:
    parts = {
        "v": FORMAT_VERSION,
        "model": _checkpoint_identity(config.model.name),
        "dtype": config.model.dtype,
        "abi": config.model.kv_cache_abi,
        "tp": config.parallel.tp_size,
        "pp": config.parallel.pp_size,
        "world": config.parallel.world_size,
        "tph": config.cache.tokens_per_hash,
        "bpc": config.cache.blocks_per_chunk,
        "wkb": config.worker_kv_bytes_per_block,
        "slot": slot_bytes,
        "groups": [[g.tokens_per_block, list(g.layer_names)] for g in config.groups],
        "salt": str(config.extra_config.get("fingerprint_salt", "")),
    }
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


# --------------------------------------------------------------------------
# scheduler side
# --------------------------------------------------------------------------


class NvmeLoadStoreSpec(BlockIDsLoadStoreSpec):
    """NVMe slot numbers plus the offload key expected in each slot."""

    def __init__(self, block_ids: Sequence[int], keys: Sequence[bytes]):
        super().__init__(list(block_ids))
        self.keys: list[bytes] = [bytes(k) for k in keys]
        assert len(self.keys) == len(self.block_ids)


class SegmentLRUPolicy(CachePolicy):
    """LRU over scheduler steps, deepest chunk first within a step.

    A hybrid (MLA + KDA) prefix is only usable up to a boundary whose KDA state
    is stored. Plain per-key LRU (measured 2026-11-26, E2) evicted one
    conversation's KDA states while keeping its MLA chunks, which then held
    slots and could never be loaded. Here every key carries
      tick  - scheduler step of its last use (all groups of one request touched
              in one step share it), and
      depth - its chunk index in the conversation (the anchor/MLA group's touch
              list is ordered from chunk 0; other groups take the depth of the
              MLA chunk with the same block hash).
    Victims are taken by (tick ascending, depth descending): the least recently
    used conversation loses its tail first, MLA chunk and the KDA state at that
    chunk together, so what remains is still a loadable prefix.

    Two rules found live (E2c, 2026-11-26) keep a conversation's keys on ONE tick:
      * touching an anchor (MLA) key refreshes every stored key with the same
        block hash. Upstream's _touch only refreshes a sliding group's last
        window, which never contains the sparse KDA boundaries, so a re-used
        conversation kept its MLA chunks fresh and lost its KDA states;
      * store/load completion (mark_evictable) is not a use. A long prompt is
        stored over many steps; bumping on completion made its tail newer than
        its head, and the head (the part every restore needs) went first.
    """

    def __init__(self, cache_capacity: int, anchor_group: int = 0) -> None:
        super().__init__(cache_capacity)
        self.anchor_group = anchor_group
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.evictable: set[OffloadKey] = set()
        self.tick: dict[OffloadKey, int] = {}
        self.depth: dict[OffloadKey, int] = {}
        self.hash_depth: dict[bytes, int] = {}
        self.hash_keys: dict[bytes, set[OffloadKey]] = {}
        self.step = 0

    @staticmethod
    def _split(key: OffloadKey) -> tuple[bytes, int]:
        return bytes(key[:-4]), int.from_bytes(key[-4:], "big")

    def _depth_of(self, key: OffloadKey) -> int:
        h, _ = self._split(key)
        return self.hash_depth.get(h, self.depth.get(key, 0))

    def get(self, key):
        return self.blocks.get(key)

    def insert(self, key, block):
        self.blocks[key] = block
        self.tick[key] = self.step
        self.depth[key] = self._depth_of(key)
        self.hash_keys.setdefault(self._split(key)[0], set()).add(key)
        if block.ref_cnt == 0:
            self.evictable.add(key)

    def remove(self, key):
        del self.blocks[key]
        self.evictable.discard(key)
        self.tick.pop(key, None)
        self.depth.pop(key, None)
        h, g = self._split(key)
        keys = self.hash_keys.get(h)
        if keys is not None:
            keys.discard(key)
            if not keys:
                del self.hash_keys[h]
        if g == self.anchor_group:
            self.hash_depth.pop(h, None)

    def touch(self, keys, req_context):
        keys = list(keys)
        if not keys:
            return
        anchor = self._split(keys[0])[1] == self.anchor_group
        for i, key in enumerate(keys):
            if anchor:
                h = self._split(key)[0]
                self.hash_depth[h] = i
                for k in self.hash_keys.get(h, ()):   # this chunk in every group
                    self.tick[k] = self.step
                    self.depth[k] = i
            elif key in self.blocks:
                self.tick[key] = self.step
                self.depth[key] = self._depth_of(key)

    def mark_evictable(self, key):
        # Completion is not a use (see class docstring): keep the tick that
        # insert/touch gave the key.
        self.evictable.add(key)
        self.tick.setdefault(key, self.step)

    def mark_non_evictable(self, key):
        self.evictable.discard(key)

    def evict(self, n, protected):
        if n == 0:
            return []
        import heapq
        cands = [k for k in self.evictable if k not in protected]
        if len(cands) < n:
            return None
        victims = heapq.nsmallest(n, cands, key=lambda k: (self.tick[k], -self.depth[k]))
        out = []
        for key in victims:
            block = self.blocks[key]
            assert block.ref_cnt == 0
            out.append((key, block))
            self.remove(key)
        return out

    def clear(self):
        self.blocks.clear(); self.evictable.clear(); self.tick.clear()
        self.depth.clear(); self.hash_depth.clear(); self.hash_keys.clear()


class NvmeOffloadingManager(CPUOffloadingManager):
    """LRU slot manager; block_id == NVMe slot index on every rank."""

    def __init__(self, num_slots: int, enable_events: bool,
                 restored: Sequence[tuple[int, bytes, int]] = ()):
        super().__init__(num_blocks=num_slots, cache_policy="lru",
                         enable_events=enable_events)
        self.medium = Medium.STORAGE
        self._doomed: set[OffloadKey] = set()
        self._policy = SegmentLRUPolicy(num_slots)
        self._restore(restored)

    def _restore(self, restored: Sequence[tuple[int, bytes, int]]) -> None:
        """Rebuild the index from rank-0 headers, oldest write first."""
        if not restored:
            return
        used: set[int] = set()
        n = 0
        # Oldest write first, at negative ticks: everything restored is older
        # than anything this process touches, and keeps its relative order.
        self._policy.step = -len(restored) - 1
        for _seq, key, slot in sorted(restored):
            self._policy.step += 1
            k = OffloadKey(key)
            if slot in used or self._policy.get(k) is not None:
                continue
            block = BlockStatus(slot)
            block.ref_cnt = 0
            self._policy.insert(k, block)
            used.add(slot)
            n += 1
        self._policy.step = 0
        self._num_evictable_cache_blocks += n
        self._num_allocated_blocks = self._num_blocks
        self._free_list = [s for s in range(self._num_blocks - 1, -1, -1)
                           if s not in used]
        logger.info("glm53-nvme: restored %d offloaded chunks from disk", n)

    def _get_load_store_spec(self, keys: Iterable[OffloadKey],
                             blocks: Iterable[BlockStatus]) -> NvmeLoadStoreSpec:
        keys = list(keys)
        blocks = list(blocks)
        return NvmeLoadStoreSpec([b.block_id for b in blocks], keys)

    def invalidate(self, keys: Collection[OffloadKey]) -> None:
        """Drop chunks that failed verification on some rank."""
        dropped = []
        for key in keys:
            block = self._policy.get(key)
            if block is None:
                continue
            if block.ref_cnt == 0:
                self._policy.remove(key)
                self._free_block(block)
                self._num_evictable_cache_blocks -= 1
                dropped.append(key)
            else:
                self._doomed.add(key)
        if dropped:
            logger.warning("glm53-nvme: invalidated %d chunks", len(dropped))
            if self.events is not None:
                from vllm.v1.kv_offload.base import OffloadingEvent
                self.events.append(OffloadingEvent(keys=dropped, medium=self.medium,
                                                   removed=True))

    def complete_load(self, keys: Collection[OffloadKey],
                      req_context: ReqContext) -> None:
        super().complete_load(keys, req_context)
        if self._doomed:
            ready = [k for k in keys if k in self._doomed]
            if ready:
                self._doomed.difference_update(ready)
                self.invalidate(ready)

    def on_schedule_end(self, context) -> None:
        # One tick per scheduler step: every group a request touches in a step
        # shares the same recency (see SegmentLRUPolicy).
        self._policy.step += 1
        super().on_schedule_end(context)

    def reset_cache(self) -> None:
        super().reset_cache()
        self._doomed.clear()
        self._num_allocated_blocks = 0
        self._free_list = []


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------


class _ThreadState(threading.local):
    bounce: torch.Tensor | None = None
    view = None
    stream: torch.cuda.Stream | None = None


class NvmeOffloadingWorker(OffloadingWorker):
    def __init__(self, kv_caches: CanonicalKVCaches, path: str, num_slots: int,
                 slot_bytes: int, fingerprint: str, io_threads: int,
                 verify_crc: bool, device: torch.device):
        self.gpu = [t.tensor for t in kv_caches.tensors]
        for t in self.gpu:
            assert t.dtype == torch.int8 and t.ndim == 2
        self.refs = [[(r.tensor_idx, r.page_size_bytes) for r in refs]
                     for refs in kv_caches.group_data_refs]
        self.payload = [sum(sz for _, sz in refs) for refs in self.refs]
        self.slot_bytes = slot_bytes
        if max(self.payload) + HEADER_BYTES > slot_bytes:
            raise RuntimeError(
                f"glm53-nvme: group payload {max(self.payload)} exceeds slot "
                f"{slot_bytes - HEADER_BYTES}")
        self.num_slots = num_slots
        self.fingerprint = fingerprint
        self.verify_crc = verify_crc
        self.device = device
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
        size = num_slots * slot_bytes
        if os.fstat(self.fd).st_size < size:
            os.posix_fallocate(self.fd, 0, size)
        self._seq = int(time.time() * 1e6)
        self._seq_lock = threading.Lock()
        self._tls = _ThreadState()
        self._pool = ThreadPoolExecutor(max_workers=io_threads,
                                        thread_name_prefix="glm53-nvme")
        # job_id -> (futures, is_store, t0)
        self._jobs: dict[int, tuple[list[Future], bool, float]] = {}
        logger.info("glm53-nvme worker: %s slots=%d slot=%.1fMiB threads=%d "
                    "groups_payload=%s", path, num_slots, slot_bytes / 2**20,
                    io_threads, self.payload)

    # -- per-thread resources ------------------------------------------------
    def _bounce(self):
        tls = self._tls
        if tls.bounce is None:
            torch.cuda.set_device(self.device)
            buf = torch.empty(self.slot_bytes + ALIGN, dtype=torch.int8,
                              pin_memory=True)
            off = (-buf.data_ptr()) % ALIGN
            tls.bounce = buf
            tls.view = buf[off: off + self.slot_bytes]
            assert tls.view.data_ptr() % ALIGN == 0
            tls.stream = torch.cuda.Stream(device=self.device)
        return tls.view, tls.stream

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    @staticmethod
    def _expand(spec: GPULoadStoreSpec) -> list[tuple[int, int]]:
        out = []
        ids = spec.block_ids.tolist()
        pos = 0
        for g, n in enumerate(spec.group_sizes):
            for bid in ids[pos: pos + n]:
                out.append((int(bid), g))
            pos += n
        assert pos == len(ids)
        return out

    # -- transfers ------------------------------------------------------------
    def _store_one(self, event: torch.cuda.Event, bid: int, g: int, slot: int,
                   key: bytes) -> int:
        view, stream = self._bounce()
        plen = self.payload[g]
        stream.wait_event(event)
        off = HEADER_BYTES
        with torch.cuda.stream(stream):
            for t_idx, sz in self.refs[g]:
                view[off: off + sz].copy_(self.gpu[t_idx][bid, :sz], non_blocking=True)
                off += sz
        stream.synchronize()
        mv = view.numpy()
        pcrc = zlib.crc32(mv[HEADER_BYTES: HEADER_BYTES + plen]) if self.verify_crc else 0
        hdr = pack_header(self.fingerprint, key, g, plen, pcrc, self._next_seq())
        mv[: HEADER_BYTES] = 0
        mv[: len(hdr)] = np.frombuffer(hdr, dtype=np.int8)
        n = _round_up(HEADER_BYTES + plen)
        written = os.pwrite(self.fd, memoryview(mv)[:n], slot * self.slot_bytes)
        if written != n:
            raise OSError(f"short write {written}/{n} slot {slot}")
        return n

    def _load_one(self, bid: int, g: int, slot: int, key: bytes) -> int:
        view, stream = self._bounce()
        plen = self.payload[g]
        n = _round_up(HEADER_BYTES + plen)
        mv = view.numpy()
        got = os.preadv(self.fd, [memoryview(mv)[:n]], slot * self.slot_bytes)
        if got != n:
            raise OSError(f"short read {got}/{n} slot {slot}")
        hdr = unpack_header(memoryview(mv)[:HEADER_BYTES].cast("B"), self.fingerprint)
        if hdr is None:
            raise ValueError(f"slot {slot}: bad header")
        hkey, hgroup, hplen, hcrc, _ = hdr
        if hkey != key or hgroup != g or hplen != plen:
            raise ValueError(f"slot {slot}: key/group/len mismatch")
        if self.verify_crc and zlib.crc32(mv[HEADER_BYTES: HEADER_BYTES + plen]) != hcrc:
            raise ValueError(f"slot {slot}: payload crc mismatch")
        off = HEADER_BYTES
        with torch.cuda.stream(stream):
            for t_idx, sz in self.refs[g]:
                self.gpu[t_idx][bid, :sz].copy_(view[off: off + sz], non_blocking=True)
                off += sz
        stream.synchronize()
        return n

    def submit_store(self, job_id: int, src_spec: GPULoadStoreSpec,
                     dst_spec: LoadStoreSpec) -> bool:
        assert isinstance(dst_spec, NvmeLoadStoreSpec)
        items = self._expand(src_spec)
        assert len(items) == len(dst_spec.block_ids)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        futs = [self._pool.submit(self._store_one, event, bid, g, int(slot), key)
                for (bid, g), slot, key in zip(items, dst_spec.block_ids.tolist(),
                                               dst_spec.keys)]
        self._jobs[job_id] = (futs, True, time.perf_counter())
        return True

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec,
                    dst_spec: GPULoadStoreSpec) -> bool:
        assert isinstance(src_spec, NvmeLoadStoreSpec)
        items = self._expand(dst_spec)
        assert len(items) == len(src_spec.block_ids)
        futs = [self._pool.submit(self._load_one, bid, g, int(slot), key)
                for (bid, g), slot, key in zip(items, src_spec.block_ids.tolist(),
                                               src_spec.keys)]
        self._jobs[job_id] = (futs, False, time.perf_counter())
        return True

    def _result(self, job_id: int) -> TransferResult:
        futs, is_store, t0 = self._jobs.pop(job_id)
        ok, nbytes = True, 0
        for f in futs:
            try:
                nbytes += f.result()
            except Exception as e:  # noqa: BLE001 - reported, never swallowed
                ok = False
                logger.error("glm53-nvme %s job %d failed: %s",
                             "store" if is_store else "load", job_id, e)
        return TransferResult(job_id=job_id, success=ok, transfer_size=nbytes,
                              transfer_time=time.perf_counter() - t0)

    def get_finished(self) -> list[TransferResult]:
        done = [j for j, (futs, _, _) in self._jobs.items()
                if all(f.done() for f in futs)]
        return [self._result(j) for j in done]

    def wait(self, job_ids: set[int]) -> None:
        for j in job_ids:
            entry = self._jobs.get(j)
            if entry is not None:
                for f in entry[0]:
                    f.exception()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        try:
            os.close(self.fd)
        except OSError:
            pass


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


def scan_index(path: str, num_slots: int, slot_bytes: int,
               fingerprint: str) -> list[tuple[int, bytes, int]]:
    """Read every slot header of a rank file: [(write_seq, key, slot)]."""
    out: list[tuple[int, bytes, int]] = []
    if not os.path.isfile(path):
        return out
    fd = os.open(path, os.O_RDONLY)
    try:
        for slot in range(num_slots):
            buf = os.pread(fd, _HDR.size, slot * slot_bytes)
            hdr = unpack_header(buf, fingerprint)
            if hdr is not None:
                key, _g, _pl, _pc, seq = hdr
                out.append((seq, key, slot))
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    finally:
        os.close(fd)
    return out


class NvmeOffloadingSpec(OffloadingSpec):
    """kv_connector_extra_config:
        spec_name: NvmeOffloadingSpec
        spec_module_path: vllm.v1.kv_offload.glm53_nvme
        root_dir: directory on each node's NVMe (same path on every node)
        capacity_bytes: per-rank store size (preallocated)
        io_threads: default 8
        verify_crc: default true
        restore_index: default true (rebuild the cache from disk at boot)
        drop_foreign: default true (delete other fingerprints' stores)
    """

    @classmethod
    def build_metric_definitions(cls, extra_config):
        # The manager is a CPUOffloadingManager and reports the same gauges
        # (slot usage etc.); the metrics layer asserts every reported key is
        # declared by the spec.
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        ec = self.extra_config
        if self.blocks_per_chunk != 1:
            raise ValueError("glm53-nvme requires blocks_per_chunk == 1")
        self.root = str(ec["root_dir"])
        capacity = int(ec["capacity_bytes"])
        self.io_threads = int(ec.get("io_threads", 8))
        self.verify_crc = bool(ec.get("verify_crc", True))
        self.restore_index = bool(ec.get("restore_index", True))
        self.drop_foreign = bool(ec.get("drop_foreign", True))
        if config.worker_kv_bytes_per_block <= 0:
            self.slot_bytes = HEADER_BYTES + ALIGN
            self.num_slots = 0
        else:
            self.slot_bytes = HEADER_BYTES + _round_up(config.worker_kv_bytes_per_block)
            self.num_slots = capacity // self.slot_bytes
        self.fingerprint = compute_fingerprint(config, self.slot_bytes)
        self.store_dir = os.path.join(self.root, self.fingerprint)
        self._manager: NvmeOffloadingManager | None = None
        self._worker: NvmeOffloadingWorker | None = None

    def rank_path(self, rank: int) -> str:
        return os.path.join(self.store_dir, f"r{rank}.kv")

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            restored: list = []
            if self.restore_index and self.num_slots:
                t0 = time.perf_counter()
                restored = scan_index(self.rank_path(0), self.num_slots,
                                      self.slot_bytes, self.fingerprint)
                logger.info("glm53-nvme: scanned %d slots in %.2fs", self.num_slots,
                            time.perf_counter() - t0)
            self._manager = NvmeOffloadingManager(
                self.num_slots, self.kv_events_config.enable_kv_cache_events,
                restored)
        return self._manager

    def _drop_foreign_stores(self) -> None:
        if not self.drop_foreign or not os.path.isdir(self.root):
            return
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if name != self.fingerprint and os.path.isdir(p) and len(name) == 32:
                logger.warning("glm53-nvme: removing store of other config %s", p)
                shutil.rmtree(p, ignore_errors=True)

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            rank = self.config.parallel.rank
            # one rank per node in this lane: each node cleans its own disk
            self._drop_foreign_stores()
            os.makedirs(self.store_dir, exist_ok=True)
            self._worker = NvmeOffloadingWorker(
                kv_caches, self.rank_path(rank), self.num_slots, self.slot_bytes,
                self.fingerprint, self.io_threads, self.verify_crc,
                torch.device("cuda", torch.cuda.current_device()))
        return self._worker
