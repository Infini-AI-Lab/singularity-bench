"""Completion request contract shared by real and simulated workers."""
from __future__ import annotations

FIELDS = {"model", "prompt", "max_tokens", "stream", "routing"}


def validate(body: dict) -> str | None:
    if not isinstance(body, dict):
        return "request must be a JSON object"
    if extra := set(body) - FIELDS:
        return f"unsupported completion fields: {', '.join(sorted(extra))}"
    if body.get("stream") is not True:
        return "stream must be true"
    prompt = body.get("prompt")
    if (not isinstance(prompt, list) or not prompt
            or not all(type(t) is int and 0 <= t < 2**31 for t in prompt)):
        return "prompt must be a nonempty list of nonnegative token IDs"
    if type(body.get("max_tokens")) is not int or body["max_tokens"] < 1:
        return "max_tokens must be a positive integer"
    routing = body.get("routing", {})
    if not isinstance(routing, dict):
        return "routing must be an object"
    nonce = routing.get("request_nonce", "")
    if not isinstance(nonce, str) or len(nonce) > 128:
        return "routing.request_nonce must be a string of at most 128 characters"
    return None
