"""Bound HTTP bodies before parsing, using only public request dimensions.

vLLM returns the entire prompt-token array in its first streaming frame. That
frame can legitimately exceed aiohttp's implicit readline limit. Read chunks
instead, with space for that array and all output IDs, but no unbounded lines,
ignored-frame floods or compressed-body expansion.
"""
from __future__ import annotations

import asyncio

BASE_BYTES = 64 * 1024
ID_BYTES = 16
OUTPUT_FRAME_BYTES = 1024


class ResponseProtocolError(ValueError):
    pass


async def small_body(response) -> bytes:
    """Consume a small body for keep-alive; close oversized bodies after 64 KiB."""
    try:
        body = await response.content.readexactly(BASE_BYTES + 1)
    except asyncio.IncompleteReadError as exc:
        return exc.partial
    response.close()
    return body[:BASE_BYTES]


async def response_lines(content, input_tokens: int, wire_max: int):
    """Yield original SSE lines within public frame and whole-stream budgets.

    Input length includes the public six-token control-marker overhead. Bounds
    never use the private requested output length: early disconnects must not
    expose it. 16 bytes cover an ID and its JSON delimiter; 1024 per output
    token also permits native per-frame metadata and individual-token chunks.
    Callers disable automatic HTTP decompression before passing the stream.
    """
    frame_max = BASE_BYTES + ID_BYTES * (input_tokens + wire_max)
    remaining = BASE_BYTES + ID_BYTES * input_tokens + OUTPUT_FRAME_BYTES * wire_max
    buffer = bytearray()
    async for chunk in content.iter_chunked(8192):
        remaining -= len(chunk)
        if remaining < 0:
            raise ResponseProtocolError("response exceeds public byte budget")
        buffer.extend(chunk)
        while (end := buffer.find(b"\n")) >= 0:
            if end > frame_max:
                raise ResponseProtocolError("response frame exceeds public byte limit")
            yield bytes(buffer[:end + 1])
            del buffer[:end + 1]
        if len(buffer) > frame_max:
            raise ResponseProtocolError("response frame exceeds public byte limit")
    if buffer:
        yield bytes(buffer)
