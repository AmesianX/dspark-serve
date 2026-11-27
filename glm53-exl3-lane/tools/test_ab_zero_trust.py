"""/ab zero trust: no request, and no HTTP call, can change another request's model.

The 11-20 per-token change made it POSSIBLE for co-batched requests to carry
different scales, but left the engine-wide default writable over HTTP
(`POST /ab`). Every request that does not carry its own `ab_scale` follows that
default, so one client flipping it changed everyone else's model -- measured on
the live lane 2026-11-27: a plain request's text and logprobs changed after a
different client sent `POST /ab {"scale":1}`. The client side cannot fix that;
the server must not offer the knob.

What this pins down (CPU only, nothing touches the GPU):
  1. `POST /ab` is refused and never reaches the workers.
  2. No worker RPC can set the default (so not even VLLM_SERVER_DEV_MODE's
     /collective_rpc can reach it).
  3. A request's own `ab_scale` is accepted only as a finite number in [0, 1];
     anything else (NaN, inf, out of range, strings, bools, lists) is ignored
     -- the request runs on the default, i.e. the original weights -- instead
     of being clamped or stored. NaN used to slip through `min(max(v,0),1)`.

    docker cp tools/test_ab_zero_trust.py vllm_glm53:/tmp/ && \
    docker exec vllm_glm53 python3 /tmp/test_ab_zero_trust.py
"""

import asyncio
import os
import sys

os.environ.setdefault("GLM53_AB_LORA", "/does/not/need/to/exist")
os.environ.setdefault("GLM53_AB_SCALE", "0")

import ab_switch as ab

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILED.append(name)


class SP:
    def __init__(self, extra_args=None):
        self.extra_args = extra_args


print("1. 요청별 ab_scale 검증")
ab._req_scales.clear()
good = {"int0": 0, "int1": 1, "f0": 0.0, "f1": 1.0, "half": 0.5}
bad = {
    "nan_str": "nan", "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"),
    "neg": -0.5, "big": 1.5, "huge": 7.5, "true": True, "false": False,
    "str1": "1", "list": [1], "dict": {"v": 1},
}
for k, v in {**good, **bad}.items():
    ab.note_request(k, SP({"ab_scale": v}))
for k, v in good.items():
    check(f"수락 {k}={v!r}", ab._req_scales.get(k) == float(v), repr(ab._req_scales.get(k)))
for k, v in bad.items():
    check(f"거부 {k}={v!r} → 기본값(원본)", k not in ab._req_scales, repr(ab._req_scales.get(k)))
check("거부 건수가 info 에 보인다", ab.info().get("rejected", 0) >= len(bad), repr(ab.info().get("rejected")))

print("2. 워커 RPC 에 전역 setter 없음")
import glm53_ab_plugin as plugin


class FakeWorker:
    pass


import types

fake_mod = types.ModuleType("vllm.v1.worker.gpu_worker")
fake_mod.Worker = FakeWorker
real = sys.modules.get("vllm.v1.worker.gpu_worker")
sys.modules["vllm.v1.worker.gpu_worker"] = fake_mod
try:
    plugin.Glm53AbWorkerPlugin()()
finally:
    if real is not None:
        sys.modules["vllm.v1.worker.gpu_worker"] = real
check("glm53_ab_set_scale 미등록", not hasattr(FakeWorker, "glm53_ab_set_scale"))
check("조회 RPC 는 유지", hasattr(FakeWorker, "glm53_ab_info"))

print("3. POST /ab 거부, 워커에 닿지 않음")
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeEngine:
    def __init__(self):
        self.calls = []

    async def collective_rpc(self, name, args=()):
        self.calls.append(name)
        if name == "glm53_ab_get_scale":
            return [0.0, 0.0]
        if name == "glm53_ab_info":
            return [ab.info(), ab.info()]
        return [args[0] if args else None] * 2


app = FastAPI()
app.include_router(plugin.router)
eng = FakeEngine()
app.state.engine_client = eng
client = TestClient(app)
before = ab._default_scale
for body in ({"scale": 1}, {"scale": 0}, {"on": True}, {"scale": "nan"}):
    r = client.post("/ab", json=body)
    check(f"POST /ab {body} 거부", r.status_code in (403, 405, 410), f"{r.status_code} {r.text[:80]}")
check("워커 set RPC 호출 0회", not any("set" in c for c in eng.calls), repr(eng.calls))
check("전역 기본값 불변", ab._default_scale == before)
r = client.get("/ab")
check("GET /ab 는 동작", r.status_code == 200, str(r.status_code))
check("GET 이 전역 쓰기 불가를 알린다", r.json().get("global_write") is False, repr(r.json().get("global_write")))

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
