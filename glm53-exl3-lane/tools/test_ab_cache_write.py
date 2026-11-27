"""/ab: the prefix cache belongs to the default model.

A request that runs at a non-default ab_scale may READ the shared prefix cache
but must never WRITE it -- neither GPU prefix-cache blocks nor the NVMe tier.
Measured on the live lane 2026-11-27 (tools/bi/ab_cache_leak_probe.py): before
this, an ab=0 request reusing a prefix first prefilled by an ab=1 request
diverged from its own fresh computation at the first token.

CPU only. Loads the OVERLAY files by path (not the copies mounted in a running
container), so it tests the tree, not the deployment:

    docker cp tools/test_ab_cache_write.py vllm_glm53:/tmp/ && \
    docker cp overlay/v1core/kv_cache_manager.py vllm_glm53:/tmp/ovl_kv_cache_manager.py && \
    docker cp overlay/kvconn/scheduler.py vllm_glm53:/tmp/ovl_kvconn_scheduler.py && \
    docker cp overlay/ab_switch.py vllm_glm53:/tmp/ab_switch.py && \
    docker exec -w /tmp vllm_glm53 python3 /tmp/test_ab_cache_write.py
"""

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


def req(ab_scale=None, absent=False):
    extra = None if absent else {"ab_scale": ab_scale}
    return types.SimpleNamespace(
        request_id=f"r{ab_scale}", sampling_params=SP(extra), kv_transfer_params=None
    )


print("1. may_write_cache — note_request 와 같은 판정")
cases = [
    ("ab_scale 없음", SP(None), True),
    ("ab=0 (기본)", SP({"ab_scale": 0}), True),
    ("ab=0.0", SP({"ab_scale": 0.0}), True),
    ("ab=1", SP({"ab_scale": 1}), False),
    ("ab=0.5", SP({"ab_scale": 0.5}), False),
    ("거부값 nan → 기본으로 돈다", SP({"ab_scale": float("nan")}), True),
    ("거부값 '1' → 기본으로 돈다", SP({"ab_scale": "1"}), True),
    ("거부값 True → 기본으로 돈다", SP({"ab_scale": True}), True),
    ("extra_args 가 dict 아님", SP(["ab_scale"]), True),
]
for name, sp, want in cases:
    got = ab.may_write_cache(sp)
    check(f"{name} → 쓰기 {'허용' if want else '금지'}", got is want, repr(got))

print("2. GPU 프리픽스 캐시 — cache_blocks / allocate_slots 게이트")
kvm = load("/tmp/ovl_kv_cache_manager.py", "kv_cache_manager", "vllm.v1.core")


class Coord:
    def __init__(self):
        self.cached = []

    def cache_blocks(self, request, n):
        self.cached.append(request.request_id)


m = object.__new__(kvm.KVCacheManager)
m.enable_caching = True
m.coordinator = Coord()
for r, want in ((req(0), True), (req(absent=True), True), (req(1), False)):
    m.coordinator.cached.clear()
    m.cache_blocks(r, 4096)
    check(f"cache_blocks {r.request_id} → {'기록' if want else '건너뜀'}",
          bool(m.coordinator.cached) is want, repr(m.coordinator.cached))
import inspect  # noqa: E402
src = inspect.getsource(kvm.KVCacheManager.allocate_slots)
check("allocate_slots 가 같은 게이트를 쓴다", "_may_write_prefix_cache(request)" in src)

print("3. NVMe 오프로드 — ab 요청은 저장 0")
sch = load("/tmp/ovl_kvconn_scheduler.py", "scheduler",
           "vllm.distributed.kv_transfer.kv_connector.v1.offloading")
cfg = types.SimpleNamespace(kv_group_configs=[])
for r, want in ((req(0), None), (req(absent=True), None), (req(1), 0)):
    st = sch.RequestOffloadState(config=cfg, req=r, req_context=None, offloading_context=None)
    check(f"{r.request_id}: max_offload_tokens={want}", st.max_offload_tokens == want,
          repr(st.max_offload_tokens))

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
