"""Per-token abliteration scale: the batch really does carry mixed scales.

Runs on device="cpu" on purpose. The logic under test is the bookkeeping --
which token gets which multiplier -- and that is device independent, while the
box this lands on is a GB10 with unified memory and the server holding ~113GB
of it. Nothing here touches the GPU.

    docker exec vllm_glm53 python3 /lane/tools/test_ab_per_request.py
"""

import os
import sys

os.environ.setdefault("GLM53_AB_LORA", "/does/not/need/to/exist")
os.environ.setdefault("GLM53_AB_SCALE", "0")

import numpy as np
import torch

import ab_switch as ab

CPU = torch.device("cpu")
MAX = 64
FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILED.append(name)


class SP:
    """Stand-in for SamplingParams: only extra_args is read."""

    def __init__(self, extra_args=None):
        self.extra_args = extra_args


def reset():
    ab._req_scales.clear()
    ab._fills = 0
    ab.set_scale(0.0)


print("reserve")
ab.reserve(MAX, CPU)
check("버퍼 할당", ab._scale_vec is not None and ab._scale_vec.numel() == MAX)
check("기본값으로 채워짐", bool((ab._scale_vec == 0.0).all()))
first = ab._scale_vec
ab.reserve(MAX * 2, CPU)
check("재호출은 무효 (캡처된 그래프의 포인터 보존)", ab._scale_vec is first)

print("note_request")
reset()
ab.note_request("a", SP({"ab_scale": 1}))
ab.note_request("b", SP({"ab_scale": 0}))
ab.note_request("c", SP(None))
ab.note_request("d", SP({"ab_scale": "nope"}))
ab.note_request("e", SP({"ab_scale": 7.5}))
check("ab_scale=1 기록", ab._req_scales.get("a") == 1.0)
check("ab_scale=0 기록", ab._req_scales.get("b") == 0.0)
check("미지정은 비추적 → 전역값", "c" not in ab._req_scales)
check("숫자 아닌 값은 비추적", "d" not in ab._req_scales)
check("[0,1] 밖은 제한", ab._req_scales.get("e") == 1.0)
ab.forget_request("a")
check("forget_request", "a" not in ab._req_scales)

print("fill_token_scales — 섞인 배치")
reset()
ab.note_request("r1", SP({"ab_scale": 1}))
ab.note_request("r2", SP({"ab_scale": 0}))
ab.note_request("r3", SP({"ab_scale": 0.5}))
req_ids = ["r1", "r2", "r3"]
counts = np.array([3, 4, 2], dtype=np.int32)
ab.fill_token_scales(req_ids, counts, num_padded=12)
got = ab._scale_vec[:12].clone().numpy()
want = np.array([1, 1, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0], dtype=np.float32)
check("토큰별 벡터가 요청 순서대로", np.array_equal(got, want), f"{got.tolist()}")
check("fills 증가", ab._fills == 1)

print("fill_token_scales — 전역값만 (빠른 경로)")
reset()
ab.set_scale(1.0)
ab.fill_token_scales(["x", "y"], np.array([2, 3], dtype=np.int32), num_padded=5)
check("set_scale 이 전역 기본을 움직인다", ab._default_scale == 1.0)
check(
    "미지정 요청 전부 전역값",
    bool((ab._scale_vec[:5] == 1.0).all()),
    str(ab._scale_vec[:5].tolist()),
)

print("fill_token_scales — 전역 on 인데 한 요청만 off")
reset()
ab.set_scale(1.0)
ab.note_request("off", SP({"ab_scale": 0}))
ab.fill_token_scales(["off", "dflt"], np.array([2, 2], dtype=np.int32), num_padded=4)
got = ab._scale_vec[:4].clone().numpy()
check(
    "전역이 켜져 있어도 요청이 끌 수 있다",
    np.array_equal(got, np.array([0, 0, 1, 1], dtype=np.float32)),
    f"{got.tolist()}",
)

print("패딩")
reset()
ab._scale_vec.fill_(9.0)
ab.note_request("p", SP({"ab_scale": 1}))
ab.fill_token_scales(["p"], np.array([2], dtype=np.int32), num_padded=6)
got = ab._scale_vec[:6].clone().numpy()
check(
    "패딩 구간은 0",
    np.array_equal(got, np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)),
    f"{got.tolist()}",
)

print("_token_scale")
reset()
ab.note_request("v", SP({"ab_scale": 1}))
ab.fill_token_scales(["v"], np.array([4], dtype=np.int32), num_padded=4)
col = ab._token_scale(CPU, 4)
check("모양이 [tokens, 1]", tuple(col.shape) == (4, 1), str(tuple(col.shape)))
check("버퍼의 뷰 (그래프가 포인터를 잡고 있다)", col.data_ptr() == ab._scale_vec.data_ptr())
check(
    "버퍼보다 큰 forward 는 스칼라로 후퇴",
    ab._token_scale(CPU, MAX + 1).dim() == 0,
)

print("곱셈 (apply 안에서 일어나는 모양)")
reset()
ab.note_request("on", SP({"ab_scale": 1}))
ab.note_request("no", SP({"ab_scale": 0}))
ab.fill_token_scales(["on", "no"], np.array([2, 2], dtype=np.int32), num_padded=4)
delta = torch.ones(4, 24, dtype=torch.bfloat16)
scaled = delta * ab._token_scale(CPU, 4).to(delta.dtype)
rows = scaled.sum(dim=1).float().tolist()
check(
    "같은 배치에서 두 요청이 다른 편집을 받는다",
    rows == [24.0, 24.0, 0.0, 0.0],
    str(rows),
)
check("scale=0 행은 정확히 0 (원본과 비트 동일)", bool((scaled[2:] == 0).all()))

print("info")
check("per_token 보고", ab.info()["per_token"] is True)
check("fills 보고", ab.info()["fills"] >= 1)

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
