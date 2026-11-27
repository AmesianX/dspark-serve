"""/ab: KV identity is (tokens, scale) -- prefix-cache namespaces.

KV computed at a non-default ab_scale is cached under a tagged key
(ab_switch.tag_hash), in the GPU prefix cache (overlay block_pool.py) and the
NVMe offload (overlay kvconn/scheduler.py):
  - a default request never finds tagged KV          (zero trust: no leak)
  - a non-default request finds its own namespace     (/ab on stays cached)
  - ... and falls back to the default namespace       (/ab auto reuses the prefix)
Measured live 2026-11-27 before this: an ab=0 request reusing an ab=1-primed
prefix diverged at the first token (tools/bi/ab_cache_leak_probe.py), and the
read-only stopgap left /ab on with no cache at all (14.2s per request).

CPU only. Loads the OVERLAY files by path (the tree, not the deployment):

    for f in tools/test_ab_cache_namespace.py overlay/ab_switch.py; do docker cp $f vllm_glm53:/tmp/; done
    docker cp overlay/v1core/block_pool.py vllm_glm53:/tmp/ovl_block_pool.py
    docker cp overlay/v1core/kv_cache_manager.py vllm_glm53:/tmp/ovl_kv_cache_manager.py
    docker cp overlay/kvconn/scheduler.py vllm_glm53:/tmp/ovl_kvconn_scheduler.py
    docker exec -w /tmp -e CUDA_VISIBLE_DEVICES= vllm_glm53 python3 /tmp/test_ab_cache_namespace.py
"""

import hashlib
import importlib.util
import os
import sys
import types

os.environ.setdefault("GLM53_AB_LORA", "/does/not/need/to/exist")
os.environ.setdefault("GLM53_AB_SCALE", "0")
sys.path.insert(0, os.getcwd())

import ab_switch as ab  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILED.append(name)


def load(path, name, package):
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


class SP:
    def __init__(self, extra_args=None):
        self.extra_args = extra_args


def H(i):
    return hashlib.sha256(f"block-{i}".encode()).digest()


def req(scale=None, hashes=(), rid="r"):
    extra = None if scale is None else {"ab_scale": scale}
    return types.SimpleNamespace(
        request_id=rid, sampling_params=SP(extra), kv_transfer_params=None,
        block_hashes=list(hashes), lora_request=None,
    )


print("1. cache_namespace / tag_hash")
for name, sp, want in [
    ("ab_scale 없음", SP(None), None), ("ab=0", SP({"ab_scale": 0}), None),
    ("ab=1", SP({"ab_scale": 1}), "1.0"), ("ab=0.5", SP({"ab_scale": 0.5}), "0.5"),
    ("거부값 nan", SP({"ab_scale": float("nan")}), None), ("거부값 '1'", SP({"ab_scale": "1"}), None),
    ("거부값 True", SP({"ab_scale": True}), None), ("extra_args 비dict", SP(["ab_scale"]), None),
]:
    got = ab.cache_namespace(sp)
    check(f"{name} → {want!r}", got == want, repr(got))
t1 = ab.tag_hash(H(0), "1.0")
check("tag 는 결정적", t1 == ab.tag_hash(H(0), "1.0"))
check("tag 길이 유지", len(t1) == len(H(0)))
check("tag ≠ 원 해시", t1 != H(0))
check("scale 마다 다른 tag", t1 != ab.tag_hash(H(0), "0.5"))

print("2. GPU 블록 풀 — 기본 요청은 tag 를 못 보고, ab 요청은 자기 것 → 기본 순")
bp_mod = load("/tmp/ovl_block_pool.py", "block_pool", "vllm.v1.core")
bp = bp_mod.BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
G = [0]


def cache(r, n):
    blocks = bp.get_new_blocks(n)
    bp.cache_full_blocks(r, blocks, 0, n, 16, 0)
    return blocks


def look(h, ns=None):
    bp.lookup_namespace = ns
    try:
        got = bp.get_cached_block(h, G)
    finally:
        bp.lookup_namespace = None
    return None if got is None else got[0].block_id


d_blocks = cache(req(None, [H(0), H(1)], "d"), 2)          # default prefix
a_blocks = cache(req(1, [H(0), H(1), H(2)], "a"), 3)  # ab computed all 3
check("기본 요청: 기본 블록 적중", look(H(0)) == d_blocks[0].block_id)
check("기본 요청: ab 만 가진 블록(H2)은 미적중 (누출 없음)", look(H(2)) is None, repr(look(H(2))))
check("기본 요청: 같은 해시라도 ab 블록이 아닌 기본 블록", look(H(1)) == d_blocks[1].block_id)
check("ab 요청: 자기 네임스페이스 우선", look(H(1), "1.0") == a_blocks[1].block_id)
check("ab 요청: 자기 것만 있는 H2 적중", look(H(2), "1.0") == a_blocks[2].block_id)
o_blocks = cache(req(None, [H(5)], "d2"), 1)
check("ab 요청: 자기 것 없으면 기본으로 대체", look(H(5), "1.0") == o_blocks[0].block_id)
check("다른 scale(0.5)은 1.0 블록을 못 봄", look(H(2), "0.5") is None)
check("조회 뒤 lookup_namespace 원복", bp.lookup_namespace is None)

print("3. kv_cache_manager — 조회 한 번 동안만 네임스페이스")
kvm = load("/tmp/ovl_kv_cache_manager.py", "kv_cache_manager", "vllm.v1.core")
seen = []


class Coord:
    def find_longest_cache_hit(self, block_hashes, max_len):
        seen.append(m.block_pool.lookup_namespace)
        return (), 0, 0


m = object.__new__(kvm.KVCacheManager)
m.enable_caching = True
m.coordinator = Coord()
m.block_pool = types.SimpleNamespace(lookup_namespace=None)
m.draft_replay_reserve = 0
m.enable_kv_cache_events = False
m.empty_kv_cache_blocks = "EMPTY"
for r, want in ((req(None), None), (req(1), "1.0"), (req(0), None)):
    r.num_tokens = 100
    r.skip_reading_prefix_cache = False
    seen.clear()
    m.get_computed_blocks(r)
    check(f"조회 중 ns={want!r}", seen == [want], repr(seen))
    check("조회 뒤 원복", m.block_pool.lookup_namespace is None)

print("4. NVMe 오프로드 — 키·대체 조회·적재 키")
sch = load("/tmp/ovl_kvconn_scheduler.py", "scheduler",
           "vllm.distributed.kv_transfer.kv_connector.v1.offloading")
gcfg = types.SimpleNamespace(is_scratch=False, hashes_per_chunk=1, group_idx=0)
cfg = types.SimpleNamespace(kv_group_configs=[gcfg], tokens_per_hash=16)
hs = [H(10), H(11), H(12)]
st_d = sch.RequestOffloadState(config=cfg, req=req(None, hs), req_context=None, offloading_context=None)
st_d.update_offload_keys()
st_a = sch.RequestOffloadState(config=cfg, req=req(1, hs), req_context=None, offloading_context=None)
st_a.update_offload_keys()
mk = sch.make_offload_key
check("기본 요청: 원 해시 키, 대체 없음",
      st_d.group_states[0].offload_keys == [mk(h, 0) for h in hs] and not st_d.group_states[0].fallback_keys)
check("ab 요청: 저장 키는 tag 키",
      st_a.group_states[0].offload_keys == [mk(ab.tag_hash(h, "1.0"), 0) for h in hs])
check("ab 요청: 대체 키는 원 해시", st_a.group_states[0].fallback_keys == [mk(h, 0) for h in hs])

stored = {mk(H(10), 0), mk(ab.tag_hash(H(11), "1.0"), 0)}   # chunk0 default, chunk1 ab
L = sch.LookupResult
mgr = types.SimpleNamespace(lookup=lambda k, ctx: L.HIT if k in stored else L.MISS)
s = object.__new__(sch.OffloadingConnectorScheduler)
s.manager = mgr
s.config = cfg
gs_a, gs_d = st_a.group_states[0], st_d.group_states[0]
res = [s._lookup_chunk(None, gs_a, i, gs_a.offload_keys[i]) for i in range(3)]
check("ab: 청크0 기본 대체 적중, 청크1 자기 적중, 청크2 미적중", res == [L.HIT, L.HIT, L.MISS], repr(res))
check("ab: 적재 키 = [기본 키, tag 키]",
      sch._chunk_keys(gs_a, 0, 2) == [mk(H(10), 0), mk(ab.tag_hash(H(11), "1.0"), 0)])
res_d = [s._lookup_chunk(None, gs_d, i, gs_d.offload_keys[i]) for i in range(3)]
check("기본: 청크1(ab 만 있음) 미적중 — 누출 없음", res_d == [L.HIT, L.MISS, L.MISS], repr(res_d))
stored.add(mk(H(11), 0))
check("기본: 기본 키 생기면 적중", s._lookup_chunk(None, gs_d, 1, gs_d.offload_keys[1]) is L.HIT)

stored.add(mk(H(11), 3))                                   # default boundary key, group 3
r_b, k_b = s._lookup_boundary(st_a, 3, 32)
check("ab: 경계 키 기본 대체", r_b is L.HIT and k_b == mk(H(11), 3), repr(r_b))
check("ab: 적재용 경계 키 = 대체된 키", s._boundary_key(st_a, 3, 32) == mk(H(11), 3))
check("ab: 저장용 경계 키 = tag", s._make_boundary_key(st_a.req, 3, 32) == mk(ab.tag_hash(H(11), "1.0"), 3))
stored.discard(mk(H(11), 3)); stored.add(mk(ab.tag_hash(H(12), "1.0"), 3))
r_b2, _ = s._lookup_boundary(st_d, 3, 48)
check("기본: ab 경계 키 미적중 — 누출 없음", r_b2 is L.MISS, repr(r_b2))

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
