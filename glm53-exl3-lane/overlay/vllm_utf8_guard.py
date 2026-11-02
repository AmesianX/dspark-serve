# SPDX-License-Identifier: Apache-2.0
"""UTF-8 byte guard: never let the sampler leave a character half-written.

THE BUG THIS EXISTS FOR (measured on this deployment, 2026-09-14)
----------------------------------------------------------------
GLM-5.3 uses a byte-level BPE, so a Korean syllable is several tokens and the
vocabulary contains bare byte pieces. Writing "베르너 하이젠베르크" the model
emitted the LEAD byte 0xEB — a perfectly good start for 베 — and the next step
was a near coin-flip between completing it ('베', logprob -0.828) and switching
script to continue the Latin spelling ('erner', -0.578). It took 'erner'. The
0xEB can now never become a character: the detokenizer holds it, finds it
unfinishable, and emits U+FFFD. The answer reaches the user as "**�erner
하이젠베르크".

Nothing upstream is wrong except that the second choice was ever offered. The
byte stream left the UTF-8 automaton and no layer was watching. Temperature
changes how often the model lands on that flip (vendor 1.0 -> 7 of 8 answers
broken, 0.6 -> 1 of 8) but cannot remove it, because the flip is legal as far
as the sampler is concerned.

WHAT THIS DOES
--------------
Tracks each request's UTF-8 decoder state (Björn Höhrmann's DFA) over the tokens
it has actually produced. While a character is incomplete, every token that
cannot continue it is masked to -inf. At a character boundary — which is where
generation spends nearly all of its time — nothing is masked and nothing is
copied, so the cost is a tensor comparison per step.

It is a constraint on WELL-FORMEDNESS, not on content: at a boundary every token
is allowed, and mid-character exactly the continuations that finish the
character are allowed. No reachable text is forbidden; only byte sequences that
could not be decoded at all.

SPECULATIVE DECODING
--------------------
Under MTP/DFlash the accepted tokens come from the rejection sampler, which
applies logits processors to a flattened [sum(num_draft_tokens), vocab] tensor —
one row per draft position — and in stock vLLM applies only MinTokens there.
`apply_with_spec_decode_tokens` below handles that layout: the entry state for
draft position j of a request is its committed state advanced through that
request's first j draft tokens. A one-line companion patch to
`rejection_sampler.apply_logits_processors` routes the call here (see
rejection_sampler.py in this overlay).

Enable: --logits-processors vllm_utf8_guard.Utf8ByteGuard
Disable without editing the launcher: VLLM_UTF8_GUARD=0
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# --- Höhrmann's UTF-8 DFA -------------------------------------------------
# utf8d[byte] = character class; utf8d[256 + state + class] = next state.
# States are pre-multiplied by 12. ACCEPT = 0 (a character boundary),
# REJECT = 12 (the byte cannot appear here at all).
ACCEPT = 0
REJECT = 12

_UTF8D = (
    # byte -> class
    (0,) * 128
    + (1,) * 16 + (9,) * 16 + (7,) * 32
    + (8,) * 2 + (2,) * 30
    + (10,) + (3,) * 12 + (4,) + (3,) * 2
    + (11,) + (6,) * 3 + (5,) + (8,) * 11
    # state transitions (9 states x 12 classes)
    + (0, 12, 24, 36, 60, 96, 84, 12, 12, 12, 48, 72)
    + (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12)
    + (12, 0, 12, 12, 12, 12, 12, 0, 12, 0, 12, 12)
    + (12, 24, 12, 12, 12, 12, 12, 24, 12, 24, 12, 12)
    + (12, 12, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12)
    + (12, 24, 12, 12, 12, 12, 12, 12, 12, 24, 12, 12)
    + (12, 12, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12)
    + (12, 36, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12)
    + (12, 36, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12)
)

# Every non-REJECT state, in table order. Index in this tuple is the row a
# state occupies in the mask tensors.
_STATES = (0, 24, 36, 48, 60, 72, 84, 96)
_STATE_ROW = {s: i for i, s in enumerate(_STATES)}


def _step(state: int, byte: int) -> int:
    return _UTF8D[256 + state + _UTF8D[byte]]


def _run(state: int, data: bytes) -> int:
    for b in data:
        state = _UTF8D[256 + state + _UTF8D[b]]
        if state == REJECT:
            return REJECT
    return state


# --- vocabulary bytes -----------------------------------------------------

def _byte_alphabet() -> dict[str, int]:
    """GPT-2 byte-level alphabet, inverted: printable char -> original byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _vocab_bytes(tokenizer_path: str, vocab_size: int) -> list[bytes]:
    """The raw bytes each token id contributes to the output stream.

    Read straight out of tokenizer.json rather than through a tokenizer object.
    The vLLM helper for loading one has moved between releases, and this file is
    bind-mounted over a specific image — depending on an internal import is how
    an overlay silently dies on the next image. A vocabulary file is stable.

    Tokens with no byte-level spelling (added/special tokens) get b"": they
    contribute no bytes, and `_build_tables` allows them only at a character
    boundary, so the model cannot stop in the middle of a character either.
    """
    path = Path(tokenizer_path)
    candidates = [path] if path.is_file() else [path / "tokenizer.json"]
    data = None
    for cand in candidates:
        if cand.exists():
            import json

            with open(cand, encoding="utf-8") as f:
                data = json.load(f)
            break
    if data is None:
        raise FileNotFoundError(f"utf8_guard: no tokenizer.json under {tokenizer_path}")

    vocab_map = data["model"]["vocab"]
    if isinstance(vocab_map, list):  # some serialisations use [token, score] pairs
        vocab_map = {tok: i for i, (tok, _score) in enumerate(vocab_map)}
    decoder = _byte_alphabet()
    size = max(vocab_size, max(vocab_map.values(), default=0) + 1)
    out: list[bytes] = [b""] * size
    outside = 0
    for tok, tid in vocab_map.items():
        if tid >= size:
            continue
        try:
            out[tid] = bytes(decoder[c] for c in tok)
        except KeyError:
            # Not byte-level spelled: treat as contributing no bytes rather than
            # guessing, so it is simply confined to character boundaries.
            outside += 1
    if outside:
        logger.warning("utf8_guard: %d tokens are not byte-level spelled", outside)
    return out


def _build_tables(vocab: list[bytes], vocab_size: int):
    """bad[state_row, token] = masking this token would be required.

    Also next_state[state_row, token], used to advance a request's state and to
    walk draft prefixes under speculative decoding.
    """
    n_states = len(_STATES)
    bad = torch.ones((n_states, vocab_size), dtype=torch.bool)
    nxt = torch.full((n_states, vocab_size), REJECT, dtype=torch.uint8)

    for tid in range(min(len(vocab), vocab_size)):
        data = vocab[tid]
        if not data:
            # Contributes no bytes (added/control tokens, EOS among them).
            # Allowed at a boundary only: stopping mid-character would strand
            # the bytes already emitted, which is the same defect by another
            # route.
            bad[0, tid] = False
            nxt[0, tid] = ACCEPT
            continue
        # From a boundary every well-formed token is allowed.
        end = _run(ACCEPT, data)
        if end != REJECT:
            bad[0, tid] = False
            nxt[0, tid] = end
        # Mid-character, only a token that opens with a continuation byte can
        # possibly be allowed — skip the rest without running the DFA.
        if _UTF8D[data[0]] in (1, 7, 9):
            for row in range(1, n_states):
                end = _run(_STATES[row], data)
                if end != REJECT:
                    bad[row, tid] = False
                    nxt[row, tid] = end
    return bad, nxt


def _tokenizer_digest(tokenizer_path: str) -> str:
    """Content hash of the tokenizer.json the tables are built from."""
    path = Path(tokenizer_path)
    cand = path if path.is_file() else path / "tokenizer.json"
    h = hashlib.sha256()
    with open(cand, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(key: str) -> Path:
    root = Path(os.environ.get("VLLM_CACHE_ROOT", "/root/.cache/vllm")) / "utf8_guard"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.pkl"


class Utf8ByteGuard:
    """Mask tokens that would leave the byte stream mid-character.

    Deliberately NOT a vllm LogitsProcessor. Registering one makes vLLM fall
    back from the V2 model runner to V1 ("Model Runner V2 does not yet support
    custom logits processors"), and this lane's model does not survive that
    path: the V1 spec-decode proposer raises on a config attribute, and past
    that the V1 KV-cache grouping asserts on the model's MLA pages. Both are
    upstream gaps unrelated to this guard, and chasing them one by one means
    running the lane on a code path nobody else runs.

    So the guard is a plain object that the sampler calls directly (see the
    companion patches in sampler.py and rejection_sampler.py). The runner stays
    on V2, and the only state needed — each request's emitted tokens — is
    already carried on SamplingMetadata.output_token_ids.
    """

    def __init__(self, tokenizer_path: str, vocab_size: int, device: torch.device):
        self.device = device
        self.enabled = os.environ.get("VLLM_UTF8_GUARD", "1") not in ("0", "false", "no")
        # Live references to each row's output token list, plus how far this
        # processor has already consumed them.
        self._out: list[list[int] | None] = []
        self._seen: list[int] = []
        self._state: list[int] = []
        if not self.enabled:
            logger.warning("utf8_guard: disabled by VLLM_UTF8_GUARD")
            return

        # The path alone is not an identity: the launcher mounts every snapshot
        # at the same /models path, so a different tokenizer with the same
        # vocab width would silently reuse stale tables. Key on the content.
        key = hashlib.sha256(
            f"{tokenizer_path}|{vocab_size}|{_tokenizer_digest(tokenizer_path)}|v3".encode()
        ).hexdigest()[:16]
        path = _cache_path(key)
        bad = nxt = vocab = None
        if path.exists():
            try:
                with open(path, "rb") as f:
                    bad, nxt, vocab = pickle.load(f)
            except Exception as e:
                logger.warning("utf8_guard: cache unreadable (%s); rebuilding", e)
                bad = nxt = vocab = None
        if bad is None:
            vocab = _vocab_bytes(tokenizer_path, vocab_size)
            bad, nxt = _build_tables(vocab, vocab_size)
            try:
                with open(path, "wb") as f:
                    pickle.dump((bad, nxt, vocab), f)
            except Exception as e:
                logger.warning("utf8_guard: cache not written (%s)", e)

        self.next_state = nxt  # kept on CPU: indexed one token at a time
        self.vocab = vocab
        # Fast path (see _tail_state): tokens that land on a character boundary
        # when started from one. Held as bytes so the common check is a single
        # C-level index rather than a tensor read.
        self._ends_clean = bytes(
            1 if int(nxt[0, t]) == ACCEPT else 0 for t in range(nxt.shape[1])
        )
        # GPU path (V2 sampler). One row per DFA state plus an extra all-allow
        # row: the state after a token is read straight off the token itself,
        # and a token that cannot be read from a boundary (it opens with a
        # continuation byte, i.e. the character began two tokens back) maps to
        # the permissive row rather than to a guess.
        import torch as _t

        allow_all = _t.zeros((1, nxt.shape[1]), dtype=_t.bool)
        self.bad = _t.cat([bad, allow_all], dim=0).to(device)
        self.unknown_row = bad.shape[0]
        rows = [0] * nxt.shape[1]
        for t in range(nxt.shape[1]):
            st = int(nxt[0, t])
            rows[t] = _STATE_ROW[st] if st != REJECT else self.unknown_row
        self.state_row_gpu = _t.tensor(rows, dtype=_t.long, device=device)
        self.vocab_size_gpu = nxt.shape[1]
        allowed_mid = int((~bad[1:]).sum())
        logger.info(
            "utf8_guard: armed over %d tokens, %d continuation tokens reachable "
            "mid-character",
            vocab_size,
            allowed_mid,
        )

    # --- state -------------------------------------------------------------

    def _tail_state(self, tokens) -> int:
        """The decoder state after `tokens`, read off their tail.

        Only the bytes since the last character START byte can matter: a start
        byte begins a fresh character, so everything before it is already
        decoded. Walking back a few tokens is therefore exact, and it costs the
        same whether the request has produced ten tokens or sixty thousand —
        which also means no per-request cache to invalidate when the batch is
        reordered underneath us.
        """
        if not tokens:
            return ACCEPT
        last = tokens[-1]
        # Overwhelmingly the common case: the last token ended a character. True
        # whether the state before it was a boundary or mid-character, since a
        # token that lands on ACCEPT from ACCEPT also lands there having
        # completed a pending character.
        if 0 <= last < len(self._ends_clean) and self._ends_clean[last]:
            return ACCEPT
        buf = b""
        for tid in reversed(tokens[-8:]):
            if 0 <= tid < len(self.vocab):
                buf = self.vocab[tid] + buf
            start = -1
            for i in range(len(buf) - 1, -1, -1):
                if _UTF8D[buf[i]] not in (1, 7, 9):  # not a continuation byte
                    start = i
                    break
            if start >= 0:
                state = _run(ACCEPT, buf[start:])
                return ACCEPT if state == REJECT else state
        # Eight tokens of pure continuation bytes is not a real generation;
        # treat it as a boundary rather than masking on a guess.
        return ACCEPT

    # --- masking -----------------------------------------------------------

    def _mask_rows(self, logits: torch.Tensor, rows: list[int], state_rows: list[int]):
        if not rows:
            return logits
        idx = torch.tensor(rows, device=logits.device)
        mask = self.bad[torch.tensor(state_rows, device=logits.device)]
        logits[idx] = logits[idx].masked_fill(mask, float("-inf"))
        return logits

    def apply(self, logits: torch.Tensor, output_token_ids) -> torch.Tensor:
        """One row per request (ordinary decode, and the bonus token)."""
        if not self.enabled or not output_token_ids:
            return logits
        rows: list[int] = []
        state_rows: list[int] = []
        for i in range(min(logits.shape[0], len(output_token_ids))):
            state = self._tail_state(output_token_ids[i] or ())
            if state != ACCEPT:
                rows.append(i)
                state_rows.append(_STATE_ROW[state])
        return self._mask_rows(logits, rows, state_rows)

    def apply_with_spec_decode(
        self, logits: torch.Tensor, num_draft_tokens, spec_token_ids, output_token_ids
    ) -> torch.Tensor:
        """The flattened draft-verification logits.

        Row layout is request-major: request r contributes num_draft_tokens[r]
        consecutive rows, and row j of that block is sampled with the request's
        committed context PLUS its first j draft tokens — so the entry state
        walks the draft prefix.
        """
        if not self.enabled or not num_draft_tokens:
            return logits
        rows: list[int] = []
        state_rows: list[int] = []
        pos = 0
        nxt = self.next_state
        for r, n_draft in enumerate(num_draft_tokens):
            if n_draft <= 0:
                continue
            committed = output_token_ids[r] if output_token_ids and r < len(output_token_ids) else ()
            state = self._tail_state(committed or ())
            drafts = spec_token_ids[r] if spec_token_ids and r < len(spec_token_ids) else ()
            for j in range(n_draft):
                if state != ACCEPT:
                    rows.append(pos + j)
                    state_rows.append(_STATE_ROW[state])
                if j < len(drafts):
                    tid = drafts[j]
                    state = int(nxt[_STATE_ROW[state], tid]) if 0 <= tid < self.bad.shape[1] else ACCEPT
                    if state == REJECT:
                        state = ACCEPT
            pos += n_draft
        return self._mask_rows(logits, rows, state_rows)


    def apply_v2(self, logits: torch.Tensor, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Mask using each row's PRECEDING token. Pure GPU, no host sync.

        The V2 sampler hands every row the token that precedes it — for a draft
        position that is the draft token before it — which is all the guard
        needs in the shape this bug takes: a lead byte emitted as its own token,
        then a token from another script. The state after a single token is a
        table lookup, so this costs one gather and one masked_fill per step and
        never stalls the stream on a device-to-host copy.

        A character split across three or more tokens leaves a narrower hole:
        after the middle piece the preceding token opens with a continuation
        byte, the state cannot be read from it alone, and that row maps to the
        permissive row. The observed failure is the two-token case.
        """
        if not self.enabled or logits.numel() == 0:
            return logits
        ids = prev_token_ids.long()
        valid = (ids >= 0) & (ids < self.vocab_size_gpu)
        safe = torch.where(valid, ids, torch.zeros_like(ids))
        rows = self.state_row_gpu[safe]
        rows = torch.where(
            valid, rows, torch.full_like(rows, self.unknown_row)
        )
        logits.masked_fill_(self.bad[rows], float("-inf"))
        return logits


# --- module singleton -----------------------------------------------------
# The sampler has no place to hold one, and building the tables twice would be
# waste; the guard is stateless between steps by design (see _tail_state), so a
# process-wide instance is safe.
_GUARD: "Utf8ByteGuard | None" = None
_GUARD_FAILED = False


def get_guard(device, vocab_size: int) -> "Utf8ByteGuard | None":
    """Process-wide guard, built on first use.

    Takes what it needs from the call site rather than from the engine config:
    the vocabulary width comes from the logits themselves, and the tokenizer
    directory from VLLM_UTF8_GUARD_TOKENIZER (the launcher passes the model
    path). Sampling happens outside any set_current_vllm_config() context, so
    reaching for the global config here fails — which is exactly what the first
    deployment of this guard hit.
    """
    global _GUARD, _GUARD_FAILED
    if _GUARD is not None or _GUARD_FAILED:
        return _GUARD
    try:
        path = os.environ.get("VLLM_UTF8_GUARD_TOKENIZER", "").strip()
        if not path:
            raise RuntimeError("VLLM_UTF8_GUARD_TOKENIZER is not set")
        _GUARD = Utf8ByteGuard(path, int(vocab_size), device)
    except Exception as e:  # never take the server down over a guard
        _GUARD_FAILED = True
        logger.error("utf8_guard: disabled, failed to initialise: %s", e)
        return None
    return _GUARD
