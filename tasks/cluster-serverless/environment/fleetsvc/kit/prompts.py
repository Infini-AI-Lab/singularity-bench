"""Deterministic prompt text with genuine prefix structure.

The length marker carried in a prompt is AUTHENTICATED, not merely masked -- see `with_control` for
what that buys and what a mask-only encoding would not.


WHY THIS EXISTS. No real API carries KV block ids -- paging is an internal of the serving engine, and a
client sends tokens -- and handing a plane a ready-made prefix identity would remove the part of the job
that is actually work. A plane has to notice, from content alone, that two requests share a prefix.

So prompts are real token sequences. Turn N+1's prompt LITERALLY extends turn N's wire prompt and the
answer it received, token for token, so:

  * the engine derives its own blocks by hashing the content, as a real engine does;
  * a plane can do longest-prefix matching over content, and must, because nothing tells it which
    requests are related;
  * a cache hit happens because the bytes match, not because the trace declared a relationship.

The trace therefore carries a `stream_id` and a token count per turn -- not text. 32M prompt tokens per
run is 160 MB of text; regenerating it deterministically at send time is far cheaper than shipping it,
and keeps the trace diffable.

PREFIX CONSISTENCY IS THE WHOLE CONTRACT: `tokens(s, m)` must be a prefix of `tokens(s, n)` for m < n.
`random.Random(seed).choices()` draws sequentially from a fresh state, so the first m draws of an
n-draw call are identical to an m-draw call.
"""

from __future__ import annotations

import array
import hashlib
import hmac
import random
import sys

# TOKEN IDS, not words. A prompt of `n` ids is exactly `n` tokens to the engine, which is what ServeGen's
# `input_tokens` means. Word-shaped text would be off by the tokenizer's own ratio, and chat completions
# would add a model-dependent template around every prompt; sending ids to `/v1/completions` avoids both.
#
# THE BAND. Ids must be < vocab_size for EVERY model in the catalogue, and must avoid special tokens --
# an id that is a BOS, a pad, or (for the multimodal entries) an image placeholder is not just odd, it
# can make the engine expect inputs that were never sent. Read from every checkpoint's config.json:
# the minimum vocab_size across the catalogue is 32,000 (MythoMax-L2-13b), and every special or modality
# id observed anywhere is either below 1000 (0, 1, 2, 5, 10, 11, 106) or above 100,000. So [1000, 32000)
# is clear for all 83 checkpoints in the frozen architecture metadata -- re-run that scan if the
# catalogue gains a model.
BAND_LO, BAND_HI = 1000, 32000
BAND = BAND_HI - BAND_LO                                                        # 31,000 usable ids
_BAND_POP = range(BAND_LO, BAND_HI)


def tokens(stream_id: int, n: int) -> list[int]:
    """The first `n` token ids of stream `stream_id`. Prefix-stable in `n`."""
    if n <= 0:
        return []
    # `choices` over the band, not a per-token randrange loop: it draws sequentially from a fresh state,
    # so the first m of an n-draw are identical -- which is what makes the sequence prefix-stable -- and
    # it runs in C, about 2.5x faster on long prompts.
    return random.Random(stream_id).choices(_BAND_POP, k=n)


# 4 bytes per id, little-endian, fixed width. Asserted rather than assumed: `array('i')` is 4 bytes on
# every platform this runs on, but a receipt that hashed a different width on the driver than on the
# worker would fail verification for every request, with nothing in the message to say why.
assert array.array("i").itemsize == 4, "canon() assumes a 4-byte 'i'; this platform disagrees"


def canon(ids: list[int]) -> bytes:
    """The canonical form of an id sequence, for hashing.

    Receipts bind the prompt by hash, so the driver, the worker and the emulator must serialise it
    identically. PACKED, not a comma-joined string: the string form costs an int->str conversion per id
    and measured 3.5x slower on a 32k-id prompt -- about 10 s of CPU per graded cell, inside the driver's
    single event loop, which delays the dispatches whose latency it is measuring.
    """
    packed = array.array("i", ids)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def response(key: bytes, model: str, prompt: list[int], n: int) -> list[int]:
    """THE correct response to this exact prompt: a keyed function of the prompt's content.

    This is what makes a response verifiable. The worker computes it from the prompt it received; the
    client computes it from the prompt it sent; they agree only if the plane forwarded the prompt
    unaltered and returned the answer unaltered.

    It is strictly stronger than signing the response alone. A signature proves *provenance* -- some
    worker produced these bytes -- but says nothing about whether the prompt was the right one, so it
    would let a plane truncate a 100k-token prompt to 1k, do 1% of the prefill, and still return a
    perfectly valid signed response, or substitute a prompt it knew was already warm somewhere and
    collect a cache hit it never earned. Binding the response to the prompt closes both, because either
    tamper changes what comes back.

    THE MODEL IS PART OF THE KEY. Without it a plane could quietly serve an expensive model's traffic
    from a cheap one, forward the same prompt, and return an answer that verified perfectly -- the most
    profitable substitution available. WHICH worker answered is NOT part of it: the client does not know
    where the plane routed, and must not need to.

    The key lives only in the workers and the client. A plane holding it could generate the expected
    answer itself and never call a worker at all.
    """
    msg = hashlib.sha256(model.encode() + b"\x00" + canon(prompt)).digest()
    seed = hmac.new(key, msg, hashlib.sha256).digest()
    return random.Random(int.from_bytes(seed[:16], "big")).choices(_BAND_POP, k=max(0, n))


# --------------------------------------------------------------------------------------------------
# How long an answer should be is a property of the PROMPT, not of the request.
#
# A real client sets `max_tokens` as a generous CAP -- it does not know how long the answer will be --
# and the model stops at EOS. Sending max_tokens equal to the exact expected length would be both
# unrealistic and a per-request oracle: a plane would learn precisely how much decode each request was
# about to cost before placing it, deleting the output-length uncertainty that is a real part of the
# problem.
#
# The serving side still needs that length (engines are run to an exact length, and the emulator has no
# model to stop on its own), so it has to reach the worker somehow. The faithful channel is the prompt,
# because that is what determines it in reality. A short keyed suffix carries it: only a holder of the
# key can write or read it, and it sits at the END so the shared prefix with the previous turn is
# untouched.
_CTRL_VOCAB = BAND          # the marker is written in the same id band as the body


# THE LENGTH MARKER: a few tokens on the end of every prompt carrying how long the answer should be.
#
# WHY IT EXISTS. The platform forces each answer to an exact length, so that how much work a request is
# does not depend on what a model happens to generate. But if the CONTROL PLANE could see that length, it
# would know each request's decode time before routing it, which real serving never tells you. So the
# number travels with the request, encrypted and authenticated, and only the worker can read it.
#
# TWO PROPERTIES, and both are needed:
#
#   confidentiality  the count is offset by a keyed mask (addition mod the count space), so the plane
#                    cannot READ it. Without this the encoding is public the moment anyone sees VOCAB.
#   authenticity     a keyed tag over (body, count), so the worker can tell a prompt WE issued from one
#                    the plane made up. Without this the worker cannot distinguish a real marker from a
#                    lucky tail, so treating unmarked prompts differently has no reliable trigger. A
#                    masked-only marker rejects garbage merely by implausibility, at about 1 in 1.17
#                    million.
#
# The tag also makes tampering fail safe: a plane that edits the prompt changes `body`, the tag stops
# verifying, and the worker treats the prompt as unmarked instead of trusting a corrupted number.
COUNT_TOKENS = 2                       # 31000^2 = 9.6e8 distinct counts -- far more than any context
TAG_TOKENS = 4                         # 31000^4 = 9.2e17, so a blind forgery lands with p = 1.1e-18
CTRL_TOKENS = COUNT_TOKENS + TAG_TOKENS
_COUNT_SPACE = _CTRL_VOCAB ** COUNT_TOKENS
_TAG_SPACE = _CTRL_VOCAB ** TAG_TOKENS


def conversation_body(turn: dict, previous_prompt: list[int], previous_answer: list[int],
                      fresh: list[int] | None = None) -> list[int]:
    """Retain the exact preceding wire prompt and generated answer when history accumulates.

    Old control markers remain inside the retained history. Only the last marker
    describes the current request; its six tokens are budgeted by the generator.
    A history reset uses a new stream and retains nothing.
    """
    prefix = previous_prompt + previous_answer if turn.get("prefix_of_prev_tokens", 0) else []
    if len(prefix) != int(turn.get("prefix_of_prev_tokens", 0)):
        raise ValueError("conversation history length differs from the frozen trace")
    size = int(turn["prompt_tokens"])
    if len(prefix) + int(turn["new_user_tokens"]) != size:
        raise ValueError("conversation prefix and new-user lengths do not match")
    # The deterministic filler can be generated during verifier setup. Actual
    # history still comes exclusively from the preceding request and answer.
    if fresh is None:
        fresh = tokens(turn["stream_id"], size)
    if len(fresh) != size:
        raise ValueError("prepared conversation filler has the wrong length")
    return prefix + fresh[len(prefix):]


def _digits(v: int, n: int) -> list[int]:
    out = []
    for _ in range(n):
        out.append(BAND_LO + v % _CTRL_VOCAB)
        v //= _CTRL_VOCAB
    return out


def _value(toks: list[int]) -> int | None:
    v = 0
    for t in reversed(toks):
        if not BAND_LO <= t < BAND_HI:
            return None
        v = v * _CTRL_VOCAB + (t - BAND_LO)
    return v


def _mask_of(key: bytes, body: list[int]) -> int:
    return int.from_bytes(hmac.new(key, b"mask|" + hashlib.sha256(canon(body)).digest(),
                                   hashlib.sha256).digest()[:8], "big")


def _tag_of(key: bytes, body: list[int], n_out: int) -> int:
    msg = b"tag|" + hashlib.sha256(canon(body)).digest() + int(n_out).to_bytes(8, "big")
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], "big") % _TAG_SPACE


def with_control(key: bytes, body: list[int], n_out: int) -> list[int]:
    """Append the encrypted, authenticated length marker. Returns the prompt as it goes on the wire."""
    n = int(n_out)
    if not 0 <= n < _COUNT_SPACE:
        raise ValueError(f"n_out {n} does not fit in {COUNT_TOKENS} control tokens")
    # ADDITION mod the space, not XOR. `(n ^ mask) % SPACE` is not invertible: the XOR runs over the
    # full 64-bit mask and the modulo then wraps it, so `masked ^ (mask % SPACE)` does not recover n
    # for every value.
    masked = (n + _mask_of(key, body)) % _COUNT_SPACE
    return [*body, *_digits(masked, COUNT_TOKENS), *_digits(_tag_of(key, body, n), TAG_TOKENS)]


def read_control(key: bytes, prompt: list[int]) -> int | None:
    """Recover the intended answer length, or None if this prompt did not come from us.

    None means the prompt carries no marker of ours: it is somebody probing the fleet with their own
    tokens, and the worker serves it as such (a length sampled within the caller's own cap) rather than
    with a length read from a marker.
    """
    toks = list(prompt or [])
    if len(toks) < CTRL_TOKENS:
        return None
    body = toks[:-CTRL_TOKENS]
    masked = _value(toks[-CTRL_TOKENS:-TAG_TOKENS])
    tag = _value(toks[-TAG_TOKENS:])
    if masked is None or tag is None:
        return None
    n = (masked - _mask_of(key, body)) % _COUNT_SPACE
    if not 0 <= n < _COUNT_SPACE:
        return None
    # Constant time: the comparison is on a secret-derived value, and a timing side channel here would
    # hand back exactly the oracle the tag exists to deny.
    if not hmac.compare_digest(_tag_of(key, body, n).to_bytes(8, "big"), tag.to_bytes(8, "big")):
        return None
    return n


# BUCKETS the plane is allowed to see. `max_tokens` on the wire is rounded UP to one of these, so the
# plane gets a realistic upper bound -- which is what a real client sends -- while the exact figure stays
# in the marker. Setting it to the model's full context instead would be worse than useless: KV is
# reserved against the bound, so a 32k bound collapses admissible concurrency on every model and reaches
# zero on the largest.
BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def bucket_for(n_out: int, rng, limit: int) -> int:
    """A plausible client-stated cap: a round number at or above `n_out`, never below it.

    TWO THINGS THIS HAS TO GET RIGHT.

    It must stay TIGHT. Choosing among more buckets would declare bounds far above the true length -- a
    request wanting 16 tokens declaring 1024. That is not what a client sends, and it wrecks scheduling:
    KV is reserved against the bound, and an over-large bound collapses admissible concurrency to single
    digits. Two candidates keep the median near 2x while still denying an exact inversion.

    It must never be the COUNT ITSELF. When no bucket fits under the context limit, the fallback is
    `limit`, which the plane can already compute from the model's context and the prompt it holds;
    returning `n_out` would put a non-round `max_tokens` on the wire and announce the exact figure it is
    supposed to hide.

    That fallback is not free of information, and it is worth being exact about what it costs: a bound
    equal to the limit tells the plane the true count exceeds the largest bucket beneath it -- about the
    same resolution the bucket ladder gives everywhere else, and it fires on ~0.1% of units, all in the
    long-output tail. The alternative, clamping those outputs down so a bucket always fits, would distort
    the length distribution to hide less than the ladder already does.
    """
    fits = [b for b in BUCKETS if b >= n_out and b <= limit]
    if not fits:
        return max(1, int(limit))
    return rng.choice(fits[:2])
