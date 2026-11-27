"""Read-only HTTP view of the runtime abliteration switch (see ab_switch.py).

The served checkpoint is the original (refusal intact) one and the
abliteration edit is a rank-24 adapter applied at ``o_proj``. A request turns
it on for ITSELF ONLY:

    {"vllm_xargs": {"ab_scale": 1}}     -> this request abliterated
    (absent, or anything invalid)       -> original

    GET  /ab   -> what the workers hold (default, per-request overrides, ...)
    POST /ab   -> 410 Gone

``POST /ab`` used to set the engine-wide default, which every request without
its own scale follows -- so any client could change every other client's model.
Clients are not trusted, so that write path is gone (2026-11-27), including the
worker RPC behind it: not even ``VLLM_SERVER_DEV_MODE``'s ``/collective_rpc``
can reach a setter. The default is ``GLM53_AB_SCALE`` at startup, fixed for the
life of the process.

Under TP the switch lives in every worker process, so reads go through
``collective_rpc``; that is why this ships as a pair: a
``vllm.general_plugins`` entry installs the worker-side read methods, a
``vllm.endpoint_plugins`` entry exposes them over HTTP. Both must be named in
``VLLM_PLUGINS``.
"""

from __future__ import annotations

from argparse import Namespace

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.logger import init_logger

# Under "vllm." on purpose: vLLM puts its handler on the "vllm" logger with
# propagate=False and leaves the root bare, so a logger named
# "glm53_ab_plugin" has no handler on its chain and sits at WARNING -- every
# line below was dropped, including the one that records each flip. Six weeks
# of switch traffic went unlogged. Same fix as ab_switch.py (2026-11-20).
logger = init_logger("vllm." + __name__)

router = APIRouter()


def _worker_get_scale(self) -> float:
    import ab_switch

    return ab_switch.get_scale()


def _worker_info(self) -> dict:
    import ab_switch

    return ab_switch.info()


class Glm53AbWorkerPlugin:
    """`vllm.general_plugins` half: install the worker-side RPC methods."""

    def __call__(self) -> None:
        from vllm.v1.worker.gpu_worker import Worker

        # Read-only on purpose: no setter is installed, so nothing that can
        # call a worker method (our routes, or vLLM's dev /collective_rpc) can
        # change the default for other requests.
        Worker.glm53_ab_get_scale = _worker_get_scale
        Worker.glm53_ab_info = _worker_info
        logger.info("[ab] worker RPC 등록됨 (읽기 전용: get_scale/info)")


def register_worker_methods() -> None:
    Glm53AbWorkerPlugin()()


@router.get("/ab")
async def ab_get(raw_request: Request) -> JSONResponse:
    engine = raw_request.app.state.engine_client
    try:
        values = await engine.collective_rpc("glm53_ab_get_scale")
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        return JSONResponse({"error": repr(exc)}, status_code=500)
    # Diagnostics are additive on purpose: codemason parses `scale` and
    # `consistent` out of this body, so those keys keep their shape.
    details: list | None = None
    try:
        details = await engine.collective_rpc("glm53_ab_info")
    except Exception as exc:  # pragma: no cover - older worker, no info RPC
        logger.warning("[ab] info RPC 실패: %r", exc)
    body = {
        "scale": values[0] if values else None,
        "workers": values,
        "consistent": len(set(values)) <= 1,
        "mode": "abliterated" if values and values[0] else "original",
        "global_write": False,
    }
    if details:
        head = details[0] or {}
        body.update(
            {
                # `fills` climbing is the only proof the per-token buffer is
                # being written every step -- worth exposing, since every [ab]
                # log line was silently dropped for six weeks.
                "per_token": bool(head.get("per_token")),
                "buffer_tokens": head.get("buffer_tokens"),
                "fills": [d.get("fills") for d in details],
                "tracked_requests": [d.get("tracked_requests") for d in details],
                "overrides": head.get("overrides"),
                "rejected": [d.get("rejected") for d in details],
                "details": details,
            }
        )
    return JSONResponse(body)


@router.post("/ab")
async def ab_post(raw_request: Request) -> JSONResponse:
    # Refused without reading the body: there is nothing a client can send
    # here that we would act on. Logged so a stale client is visible.
    client = raw_request.client.host if raw_request.client else "?"
    logger.warning("[ab] POST /ab 거부 (%s) — 전역 스위치 없음, 요청별 vllm_xargs.ab_scale 사용", client)
    return JSONResponse(
        {
            "error": "POST /ab is gone: the engine-wide switch let any client "
            "change every other client's model. Ask for the edit on your own "
            'request instead: {"vllm_xargs": {"ab_scale": 1}}.',
            "global_write": False,
        },
        status_code=410,
    )


class Glm53AbEndpointPlugin:
    """`vllm.endpoint_plugins` half: attach the routes."""

    name = "glm53_ab"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(router)
        logger.info("[ab] /ab 라우트 등록됨")

    async def init_state(
        self, engine_client, state: State, args: Namespace
    ) -> None:
        state.glm53_ab_engine = engine_client
