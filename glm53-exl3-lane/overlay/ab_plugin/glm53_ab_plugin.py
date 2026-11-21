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

# Under "vllm." on purpose: vLLM puts its handler on the "vllm" logger with
# propagate=False and leaves the root bare, so a logger named
# "glm53_ab_plugin" has no handler on its chain and sits at WARNING -- every
# line below was dropped, including the one that records each flip. Six weeks
# of switch traffic went unlogged. Same fix as ab_switch.py (2026-11-20).
logger = init_logger("vllm." + __name__)

router = APIRouter()


def _worker_set_scale(self, scale: float) -> float:  # bound onto the worker
    import ab_switch

    return ab_switch.set_scale(scale)


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

        Worker.glm53_ab_set_scale = _worker_set_scale
        Worker.glm53_ab_get_scale = _worker_get_scale
        Worker.glm53_ab_info = _worker_info
        logger.info("[ab] worker RPC 등록됨 (glm53_ab_set_scale/get_scale/info)")


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
                "details": details,
            }
        )
    return JSONResponse(body)


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
