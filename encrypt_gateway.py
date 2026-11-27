#!/usr/bin/env python3
"""Decrypt gateway that sits in front of a stock vLLM (or any OpenAI-compatible)
server, so codemason can talk to it with end-to-end encrypted request/response
BODIES. vLLM itself is never modified — run it on localhost, run this in front.

    codemason ──(AES-GCM body)──► [this gateway] ──(plaintext, localhost)──► vLLM
              ◄─(AES-GCM chunks)──┘                ◄──(plaintext SSE)────────┘

Only the request/response BODY is encrypted, with a pre-shared symmetric key that
never crosses the network (it is the AES key, not a bearer token). That defeats a
MITM even when the environment can present a trusted-CA certificate (e.g. a stolen
internal root CA doing TLS interception): the interceptor sees only ciphertext and
holds no key. Wire format mirrors codemason/utils/body_crypto.py exactly:

    base64( nonce[12] || AES-256-GCM ciphertext || tag[16] )

Run (same box as vLLM):
    pip install fastapi uvicorn httpx cryptography
    export LLAMACPP_AES_KEY=<same 32-byte base64/hex key as the client>
    export VLLM_UPSTREAM=http://127.0.0.1:8000      # stock vLLM, localhost only
    uvicorn encrypt_gateway:app --host 0.0.0.0 --port 8443

Then point codemason at it:
    LLAMACPP_BASE_URL=http://<gateway-host>:8443
    LLAMACPP_AES_KEY=<same key>
(HTTP is fine here — the body is already encrypted — but TLS on top is welcome.)
"""

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re

import httpx
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

NONCE_LEN = 12
UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
_HTTP_TIMEOUT = float(os.environ.get("GATEWAY_HTTP_TIMEOUT", "1800"))


def _load_key() -> bytes:
    s = (os.environ.get("LLAMACPP_AES_KEY") or "").strip()
    if not s:
        raise SystemExit("LLAMACPP_AES_KEY is required (32-byte base64 or hex)")
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        key = bytes.fromhex(s)
    else:
        key = base64.b64decode(s + "=" * (-len(s) % 4))
    if len(key) != 32:
        raise SystemExit(f"LLAMACPP_AES_KEY must decode to 32 bytes, got {len(key)}")
    return key


KEY = _load_key()

# vLLM now requires this key on EVERY path (lane overlay/server_utils.py), so it
# became the one secret protecting everything — and :8443 is plain HTTP, which
# means a client-sent `Authorization` header would travel the internet in the
# clear while the body next to it is encrypted. Instead the gateway holds the
# key and attaches it itself, but ONLY after a body decrypts: producing valid
# AES-GCM ciphertext already proves possession of the shared key, so that is the
# credential, and it is never transmitted. Fail closed if it is missing.
UPSTREAM_KEY = (os.environ.get("VLLM_API_KEY") or "").strip()
if not UPSTREAM_KEY:
    raise SystemExit("VLLM_API_KEY is required (attached to decrypted requests)")


def encrypt(plaintext) -> str:
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(KEY).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt(blob) -> bytes:
    if isinstance(blob, str):
        blob = blob.encode("ascii")
    raw = base64.b64decode(blob)
    if len(raw) < NONCE_LEN + 16:
        raise InvalidTag()
    return AESGCM(KEY).decrypt(raw[:NONCE_LEN], raw[NONCE_LEN:], None)


def _upstream_down(exc: Exception) -> Response:
    """503 with a readable JSON body instead of a 500 + traceback."""
    body = {"error": {
        "message": (f"upstream vLLM ({UPSTREAM}) is not reachable — "
                    f"still loading or down? ({type(exc).__name__})"),
        "type": "upstream_unavailable",
        "code": 503,
    }}
    return Response(status_code=503, content=json.dumps(body).encode(),
                    media_type="application/json")


# docs/redoc/openapi are served by THIS app, not proxied, so they answered 200
# with no key at all on the public :8443 — verified from off-box 2026-11-13.
# Nothing here is worth documenting publicly: the whole API is one catch-all.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# --- per-session prefix-cache isolation (2026-11-27) --------------------------
# vLLM's prefix cache is shared by everyone who sends the same tokens, so with
# no salt one client can time another's prefix (TTFT tells whether it was
# cached). Every decrypted generation request therefore gets a cache_salt that
# THIS gateway derives; whatever the client put in `cache_salt` is overwritten
# (clients are not trusted). The client may name its session with
# `cache_session` inside the ENCRYPTED body (never a plaintext header -- the
# threat model here is a TLS interceptor); it is removed before forwarding.
# Without one, the salt is per client address. A client that opts in with
# vllm_xargs.cache_share_system_prefix still shares its leading public system
# blocks (tool schema, static instructions) -- the lane decides that boundary
# from the tokens (overlay/v1core/kv_cache_utils.py).
SESSION_SALT = os.environ.get("GATEWAY_SESSION_SALT", "1") != "0"
# The salt key is a SERVER-ONLY secret, deliberately not derived from the AES
# key: the AES key is a credential that clients hold (and any key shipped in a
# browser/WASM build is public), and whoever can compute salts could aim at
# someone else's cache namespace. Fail closed when it is missing.
_SALT_SECRET = (os.environ.get("GATEWAY_CACHE_SALT_SECRET") or "").strip()
if SESSION_SALT and len(_SALT_SECRET) < 32:
    raise SystemExit("GATEWAY_CACHE_SALT_SECRET (>=32 chars, server-only) is required "
                     "while GATEWAY_SESSION_SALT is on")
_SALT_KEY = hashlib.sha256(b"glm53-cache-salt-key\0" + _SALT_SECRET.encode()).digest()
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_GEN_PATHS = ("v1/chat/completions", "v1/completions", "v1/responses")


def _client_address(request: Request) -> str:
    peer = request.client.host if request.client else ""
    try:
        trusted_proxy = ipaddress.ip_address(peer).is_private or ipaddress.ip_address(peer).is_loopback
    except ValueError:
        trusted_proxy = False
    if trusted_proxy:  # NPM in front: the real address is in its headers
        fwd = request.headers.get("x-real-ip") or (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        )
        if fwd:
            return fwd
    return peer or "unknown"


def _session_salt(ident: str) -> str:
    return hmac.new(_SALT_KEY, ident.encode(), hashlib.sha256).hexdigest()[:32]


def _apply_session_salt(path: str, request: Request, plaintext: bytes) -> bytes:
    """Force this session's cache_salt into a generation request body."""
    if not SESSION_SALT or path.strip("/") not in _GEN_PATHS:
        return plaintext
    try:
        body = json.loads(plaintext)
    except ValueError:
        return plaintext
    if not isinstance(body, dict):
        return plaintext
    session = body.pop("cache_session", None)
    if isinstance(session, str) and _SESSION_RE.match(session):
        ident = "session:" + session
    else:
        ident = "addr:" + _client_address(request)
    body["cache_salt"] = _session_salt(ident)
    return json.dumps(body, ensure_ascii=False).encode()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    encrypted = request.headers.get("X-Body-Encrypted", "").lower() == "aesgcm"

    # Forward non-encrypted requests untouched (e.g. GET /v1/models, /slots probes).
    if not encrypted:
        return await _passthrough(request, path, body)

    # Encrypted POST: decrypt body -> forward plaintext JSON to vLLM.
    try:
        plaintext = decrypt(body)
    except (InvalidTag, ValueError, Exception):  # noqa: BLE001
        return Response(status_code=400, content=b"bad ciphertext or wrong key")

    plaintext = _apply_session_salt(path, request, plaintext)

    # Reached only once decrypt() succeeded above -> the caller holds the AES key.
    # Attach our own bearer token; ignore any client-sent one. Deliberately NOT
    # done in _passthrough: injecting there would let any unencrypted request
    # bypass vLLM's authentication entirely.
    upstream_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {UPSTREAM_KEY}",
    }

    is_stream = b'"stream"' in plaintext and b'"stream": false' not in plaintext \
        and b'"stream":false' not in plaintext

    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    if is_stream:
        return StreamingResponse(
            _encrypt_stream(url, upstream_headers, plaintext),
            media_type="text/event-stream",
        )

    # Non-streaming: encrypt the whole response body.
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            r = await client.post(url, content=plaintext, headers=upstream_headers)
        except httpx.TransportError as exc:
            return _upstream_down(exc)
        if r.status_code != 200:
            return Response(status_code=r.status_code, content=r.content)
        return Response(content=encrypt(r.content).encode("ascii"),
                        media_type="application/octet-stream")


async def _encrypt_stream(url, headers, plaintext):
    """Stream vLLM's SSE, re-emitting each event with its body AES-GCM sealed."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            async with client.stream("POST", url, content=plaintext, headers=headers) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    # errors go back in the clear so the client can surface the status
                    yield f"data: {encrypt(err.decode('utf-8', 'replace'))}\n\n".encode()
                    return
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield b"data: [DONE]\n\n"
                    else:
                        yield f"data: {encrypt(payload)}\n\n".encode()
        except httpx.TransportError as exc:
            err = json.dumps({"error": {
                "message": (f"upstream vLLM ({UPSTREAM}) is not reachable — "
                            f"still loading or down? ({type(exc).__name__})"),
                "type": "upstream_unavailable", "code": 503,
            }})
            yield f"data: {encrypt(err)}\n\n".encode()
            yield b"data: [DONE]\n\n"


async def _passthrough(request: Request, path: str, body: bytes):
    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    fwd = {k: v for k, v in request.headers.items()
           if k.lower() not in ("host", "content-length")}
    # Stream the upstream response through instead of buffering it: a plaintext
    # (unencrypted) SSE request — any client that skips body encryption — was
    # otherwise held until generation finished, so TTFT == total time and
    # token-by-token output never appeared. Non-streaming replies are
    # unaffected (single chunk).
    client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
    try:
        r = await client.send(
            client.build_request(request.method, url, content=body or None, headers=fwd),
            stream=True,
        )
    except httpx.TransportError as exc:
        await client.aclose()
        return _upstream_down(exc)

    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "connection")}

    async def _relay():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(_relay(), status_code=r.status_code, headers=resp_headers)
