"""HTTP control for the runtime abliteration switch (see ab_switch.py).

Registers ``GET/POST /ab`` on the API server so a client can read or flip the
switch without restarting: the served checkpoint is the original (refusal
intact) one and the abliteration edit is a rank-24 adapter applied at
``o_proj``, so ``scale`` picks which model answers the next forward pass.

Under TP the switch lives in every worker process, so a flip has to reach all
of them. ``collective_rpc`` is the supported way to do that, which is why this
ships as a pair: a ``vllm.general_plugins`` entry installs the worker-side
method, a ``vllm.endpoint_plugins`` entry exposes it over HTTP. Both must be
named in ``VLLM_PLUGINS``.

    POST /ab {"scale": 1.0}   -> abliterated for subsequent requests
    POST /ab {"scale": 0.0}   -> original (default)
    GET  /ab                  -> {"scale": ..., "workers": [...]}

The route is on the same port as the inference API and so is covered by the
same API key; it is still a behaviour-changing control, so do not expose this
port beyond the hosts that already serve inference.
"""

from __future__ import annotations

from argparse import Namespace

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _worker_set_scale(self, scale: float) -> float:  # bound onto the worker
    import ab_switch

    return ab_switch.set_scale(scale)


def _worker_get_scale(self) -> float:
    import ab_switch

    return ab_switch.get_scale()


class Glm53AbWorkerPlugin:
    """`vllm.general_plugins` half: install the worker-side RPC methods."""

    def __call__(self) -> None:
        from vllm.v1.worker.gpu_worker import Worker

        Worker.glm53_ab_set_scale = _worker_set_scale
        Worker.glm53_ab_get_scale = _worker_get_scale
        logger.info("[ab] worker RPC 등록됨 (glm53_ab_set_scale/get_scale)")


def register_worker_methods() -> None:
    Glm53AbWorkerPlugin()()


@router.get("/ab")
async def ab_get(raw_request: Request) -> JSONResponse:
    engine = raw_request.app.state.engine_client
    try:
        values = await engine.collective_rpc("glm53_ab_get_scale")
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        return JSONResponse({"error": repr(exc)}, status_code=500)
    return JSONResponse(
        {
            "scale": values[0] if values else None,
            "workers": values,
            "consistent": len(set(values)) <= 1,
            "mode": "abliterated" if values and values[0] else "original",
        }
    )


@router.post("/ab")
async def ab_post(raw_request: Request) -> JSONResponse:
    body = await raw_request.json()
    if "scale" in body:
        try:
            scale = float(body["scale"])
        except (TypeError, ValueError):
            return JSONResponse({"error": "scale must be a number"}, status_code=400)
    elif "on" in body:
        scale = 1.0 if body["on"] else 0.0
    else:
        return JSONResponse({"error": "pass scale or on"}, status_code=400)
    if not 0.0 <= scale <= 1.0:
        return JSONResponse({"error": "scale must be in [0, 1]"}, status_code=400)

    engine = raw_request.app.state.engine_client
    try:
        values = await engine.collective_rpc("glm53_ab_set_scale", args=(scale,))
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        return JSONResponse({"error": repr(exc)}, status_code=500)
    logger.info("[ab] scale → %.2f (워커 %d개)", scale, len(values))
    return JSONResponse(
        {
            "scale": values[0] if values else scale,
            "workers": values,
            "consistent": len(set(values)) <= 1,
            "mode": "abliterated" if scale else "original",
        }
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
