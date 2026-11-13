"""HMAC receipts bind work to a fresh verifier-generated request-attempt nonce.

Workers sign model, prompt hash, wire token cap, output-ID hash, finish reason and
request_nonce. The nonce must pass through unchanged, even if a gateway remaps
request_id. Identical prompts, retries and repeated cells cannot reuse evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
MAC_LEN = 32


def sign(key: bytes, fields: dict) -> str:
    """MAC over a canonical rendering of `fields`. Key order and spacing cannot change the result."""
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hmac.new(key, body.encode(), hashlib.sha256).hexdigest()[:MAC_LEN]


def check(key: bytes, fields: dict, mac: str | None) -> bool:
    """True iff `mac` was produced by `sign(key, fields)`. Constant-time on the comparison."""
    if not isinstance(mac, str) or len(mac) != MAC_LEN or not mac.isascii():
        return False
    return hmac.compare_digest(sign(key, fields), mac)


def sha256_hex(s: str | bytes) -> str:
    """Hex sha256 of a string or an already-packed byte sequence.

    Bytes are accepted because prompts and answers are token ids, packed by `prompts.canon`; encoding
    them through a string cost an int->str per id for no benefit."""
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode()).hexdigest()
