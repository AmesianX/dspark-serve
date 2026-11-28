# [ours] CPU-only tests for overlay/kvoffload/glm53_nvme.py (no CUDA).
import os, struct, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "overlay", "kvoffload"))
import glm53_nvme as m  # noqa: E402
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext  # noqa: E402

FP = "a" * 32
ctx = ReqContext(req_id="r")


def test_header_roundtrip_and_tamper():
    h = m.pack_header(FP, b"k" * 36, 2, 12345, 0xDEADBEEF, 7)
    assert m.unpack_header(h, FP) == (b"k" * 36, 2, 12345, 0xDEADBEEF, 7)
    assert m.unpack_header(h, "b" * 32) is None          # other config
    bad = bytearray(h); bad[60] ^= 1
    assert m.unpack_header(bytes(bad), FP) is None        # bit flip
    assert m.unpack_header(b"\0" * 4096, FP) is None      # empty slot


def _write_slot(fd, slot, slot_bytes, key, seq, fp=FP):
    hdr = m.pack_header(fp, key, 0, 10, 0, seq)
    os.pwrite(fd, hdr, slot * slot_bytes)


def test_scan_index_and_restore_lru_order():
    slot_bytes, n = 8192, 6
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r0.kv")
        fd = os.open(p, os.O_RDWR | os.O_CREAT); os.ftruncate(fd, n * slot_bytes)
        _write_slot(fd, 1, slot_bytes, b"A" * 36, seq=30)
        _write_slot(fd, 3, slot_bytes, b"B" * 36, seq=10)
        _write_slot(fd, 4, slot_bytes, b"C" * 36, seq=20, fp="c" * 32)  # foreign
        os.close(fd)
        idx = m.scan_index(p, n, slot_bytes, FP)
        assert sorted((k, s) for _, k, s in idx) == [(b"A" * 36, 1), (b"B" * 36, 3)]
        mgr = m.NvmeOffloadingManager(n, False, idx)
        assert mgr.lookup(OffloadKey(b"A" * 36), ctx) is LookupResult.HIT
        assert mgr.lookup(OffloadKey(b"C" * 36), ctx) is LookupResult.MISS
        # 4 free slots, none of them 1 or 3
        out = mgr.prepare_store([OffloadKey(bytes([i]) * 36) for i in range(4)], ctx)
        assert out is not None and not out.evicted_keys
        assert not {1, 3} & set(out.store_spec.block_ids.tolist())
        mgr.complete_store(out.keys_to_store, ctx)
        # next store must evict the OLDEST restored chunk first (B, seq 10)
        out2 = mgr.prepare_store([OffloadKey(b"Z" * 36)], ctx)
        assert out2.evicted_keys == [OffloadKey(b"B" * 36)]
        assert out2.store_spec.block_ids.tolist() == [3]
        assert out2.store_spec.keys == [b"Z" * 36]


def test_invalidate_frees_slot_and_defers_while_loading():
    mgr = m.NvmeOffloadingManager(2, False)
    k1, k2 = OffloadKey(b"1" * 36), OffloadKey(b"2" * 36)
    out = mgr.prepare_store([k1, k2], ctx); mgr.complete_store(out.keys_to_store, ctx)
    mgr.prepare_load([k1], ctx)                 # k1 in use
    mgr.invalidate([k1, k2])
    assert mgr.lookup(k2, ctx) is LookupResult.MISS
    assert mgr.lookup(k1, ctx) is LookupResult.HIT   # still protected
    mgr.complete_load([k1], ctx)
    assert mgr.lookup(k1, ctx) is LookupResult.MISS  # dropped after load
    out = mgr.prepare_store([OffloadKey(b"x" * 36), OffloadKey(b"y" * 36)], ctx)
    assert out is not None and not out.evicted_keys


def test_invalidate_clears_rank0_header_so_restart_does_not_reindex():
    # A chunk that failed verification must not come back after a restart: its
    # rank-0 header is zeroed before the slot is reused (immediately, or after
    # the load holding it completes). Needs O_DIRECT, so not on tmpfs.
    slot_bytes, n = 8192, 4
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__))) as d:
        p = os.path.join(d, "r0.kv")
        fd = os.open(p, os.O_RDWR | os.O_CREAT); os.ftruncate(fd, n * slot_bytes)
        for slot, key, seq in ((0, b"A" * 36, 1), (1, b"B" * 36, 2), (2, b"C" * 36, 3)):
            _write_slot(fd, slot, slot_bytes, key, seq)
        os.close(fd)
        mgr = m.NvmeOffloadingManager(n, False, m.scan_index(p, n, slot_bytes, FP),
                                      index_path=p, slot_bytes=slot_bytes)
        a, b = OffloadKey(b"A" * 36), OffloadKey(b"B" * 36)
        mgr.prepare_load([b], ctx)                  # B in use -> deferred
        mgr.invalidate([a, b])
        left = sorted(k for _, k, _ in m.scan_index(p, n, slot_bytes, FP))
        assert left == [b"B" * 36, b"C" * 36]       # A cleared now, B still protected
        mgr.complete_load([b], ctx)
        left = sorted(k for _, k, _ in m.scan_index(p, n, slot_bytes, FP))
        assert left == [b"C" * 36]                  # B cleared after its load
        idx = m.scan_index(p, n, slot_bytes, FP)
        mgr2 = m.NvmeOffloadingManager(n, False, idx)   # "restart"
        assert mgr2.lookup(a, ctx) is LookupResult.MISS
        assert mgr2.lookup(b, ctx) is LookupResult.MISS
        assert mgr2.lookup(OffloadKey(b"C" * 36), ctx) is LookupResult.HIT


def test_failed_store_releases_slot():
    mgr = m.NvmeOffloadingManager(1, False)
    k = OffloadKey(b"f" * 36)
    out = mgr.prepare_store([k], ctx)
    mgr.complete_store([k], ctx, success=False)
    assert mgr.lookup(k, ctx) is LookupResult.MISS
    assert mgr.prepare_store([OffloadKey(b"g" * 36)], ctx) is not None




def test_reported_stats_are_declared():
    # 11-26 live crash: the metrics layer asserts every reported key is declared
    mgr = m.NvmeOffloadingManager(4, False)
    out = mgr.prepare_store([OffloadKey(b"s" * 36)], ctx)
    mgr.complete_store(out.keys_to_store, ctx)
    stats = mgr.get_stats()
    reported = set(stats.data["types"])
    declared = set(m.NvmeOffloadingSpec.build_metric_definitions({}))
    assert reported, vars(stats)
    assert reported <= declared, reported - declared


def _k(conv, chunk, group):
    return OffloadKey(f"{conv}{chunk:03d}".encode().ljust(32, b"_") + group.to_bytes(4, "big"))


def test_segment_lru_evicts_tail_segment_of_oldest_conversation():
    pol = m.SegmentLRUPolicy(100)
    def add(conv, step):
        pol.step = step
        mla = [_k(conv, c, 0) for c in range(8)]
        kda = [_k(conv, c, 2) for c in (3, 7)]          # boundary states at chunks 3 and 7
        for i, key in enumerate(mla + kda):
            b = m.BlockStatus(i); b.ref_cnt = 0; pol.insert(key, b)
        pol.touch(mla, ctx); pol.touch(kda, ctx)        # same step: same tick
    add("X", 1); add("Y", 2)
    out = [k for k, _ in pol.evict(3, set())]
    # oldest conversation first, deepest chunk first, its KDA state with it
    assert set(out) == {_k("X", 7, 0), _k("X", 7, 2), _k("X", 6, 0)}, out
    out = [k for k, _ in pol.evict(5, set())]
    assert all(k[:1] == b"X" for k in out), out          # Y untouched while X remains
    assert _k("X", 3, 2) in out and _k("X", 3, 0) in out # next segment, state included
    assert pol.evict(3, {_k("X", 0, 0), _k("X", 1, 0)} | {_k("Y", c, 0) for c in range(8)}
                     | {_k("Y", 3, 2), _k("Y", 7, 2)}) is None   # protected keys never taken


def test_segment_lru_recent_use_protects_conversation():
    pol = m.SegmentLRUPolicy(100)
    for conv, step in (("X", 1), ("Y", 2)):
        pol.step = step
        keys = [_k(conv, c, 0) for c in range(4)]
        for i, key in enumerate(keys):
            b = m.BlockStatus(i); b.ref_cnt = 0; pol.insert(key, b)
        pol.touch(keys, ctx)
    pol.step = 3; pol.touch([_k("X", c, 0) for c in range(4)], ctx)   # X used again
    out = [k for k, _ in pol.evict(2, set())]
    assert out == [_k("Y", 3, 0), _k("Y", 2, 0)], out


def test_reuse_refreshes_kda_states_and_completion_is_not_a_use():
    """E2c live failure: a re-used conversation lost its KDA states (upstream
    touches only a sliding group's last window), and a prompt stored over many
    steps had its HEAD evicted first because completion bumped the tail."""
    pol = m.SegmentLRUPolicy(100)
    def store(conv, step):
        pol.step = step
        mla = [_k(conv, c, 0) for c in range(8)]
        kda = [_k(conv, c, 2) for c in (3, 7)]
        for i, key in enumerate(mla + kda):
            b = m.BlockStatus(i); b.ref_cnt = -1; pol.insert(key, b)
        pol.touch(mla, ctx)
        for key in mla + kda:                       # completes later, chunk by chunk
            pol.step += 1; pol.blocks[key].ref_cnt = 0; pol.mark_evictable(key)
    store("X", 1); store("Y", 100)
    pol.step = 200
    pol.touch([_k("X", c, 0) for c in range(8)], ctx)      # X re-used: MLA list only,
    pol.touch([], ctx)                                     # KDA window misses 3 and 7
    out = [k for k, _ in pol.evict(3, set())]
    assert all(k[:1] == b"Y" for k in out), out            # Y is now the LRU one
    assert set(out) == {_k("Y", 7, 0), _k("Y", 7, 2), _k("Y", 6, 0)}, out   # its tail, not head


if __name__ == "__main__":  # noqa
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
