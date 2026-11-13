"""Read-only bounded HTTP bridge for native KV events and forward-pass metrics.

Private ZMQ endpoints never accept agent input. KV bytes are unchanged; sequence
gaps and expired HTTP cursors are explicit, not silently interpreted as hits.
FPM is a sample stream (as in upstream); its bounded history reports omissions.
"""
from __future__ import annotations
import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import time

from aiohttp import web

MAX_BYTES = 32 * 1024 * 1024
PAGE_BYTES = 2 * 1024 * 1024


class Buffer:
    def __init__(self, limit=MAX_BYTES):
        self.limit = limit
        self.rows = deque()
        self.size = self.cursor = self.source_gaps = 0
        self.last_seq = None

    def append(self, frames):
        if len(frames) != 3 or len(frames[1]) != 8:
            raise ValueError('invalid native telemetry envelope')
        seq = int.from_bytes(frames[1], 'big')
        if self.last_seq is not None and seq != self.last_seq + 1:
            self.source_gaps += 1
        self.last_seq = seq
        self.cursor += 1
        row = {'cursor': self.cursor, 'frames': [base64.b64encode(f).decode() for f in frames]}
        size = sum(len(f) for f in row['frames']) + 64
        if size > PAGE_BYTES:
            self.source_gaps += 1
            return
        self.rows.append((row, size))
        self.size += size
        while self.size > self.limit or len(self.rows) > 20000:
            _, old = self.rows.popleft()
            self.size -= old

    def page(self, after):
        first = self.rows[0][0]['cursor'] if self.rows else self.cursor + 1
        if after < first - 1 or after > self.cursor:
            raise web.HTTPConflict(text='telemetry cursor expired; rebuild index from an empty worker')
        rows, size = [], 0
        for row, cost in self.rows:
            if row['cursor'] <= after:
                continue
            if size + cost > PAGE_BYTES:
                break
            rows.append(row)
            size += cost
        return {'events': rows, 'next_cursor': rows[-1]['cursor'] if rows else after,
                'latest_cursor': self.cursor, 'source_gaps': self.source_gaps}


class Telemetry:
    def __init__(self, kv_endpoint, fpm_endpoint, capabilities_file):
        self.endpoints = {'kv': kv_endpoint, 'fpm': fpm_endpoint}
        self.buffers = {kind: Buffer() for kind in self.endpoints}
        self.capabilities = json.loads(Path(capabilities_file).read_text())
        self.last_seen = dict.fromkeys(self.endpoints, None)
        self.tasks = []
        self.failure = None

    async def consume(self, kind):
        import zmq
        import zmq.asyncio
        sock = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.setsockopt(zmq.RCVHWM, 20000)
        sock.connect(self.endpoints[kind])
        try:
            while True:
                frames = await sock.recv_multipart()
                self.buffers[kind].append(frames)
                self.last_seen[kind] = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = f'{kind}: {type(exc).__name__}: {exc}'
        finally:
            sock.close(0)

    async def context(self, app):
        self.tasks = [asyncio.create_task(self.consume(k)) for k in self.endpoints]
        yield
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def handle(self, request):
        kind = request.query.get('kind', 'fpm')
        if kind not in self.buffers:
            raise web.HTTPBadRequest(text='kind must be kv or fpm')
        try:
            after = int(request.query.get('after', '0'))
            if after < 0:
                raise ValueError()
        except ValueError:
            raise web.HTTPBadRequest(text='after must be a nonnegative cursor')
        if self.failure:
            raise web.HTTPServiceUnavailable(text=self.failure)
        return web.json_response(self.buffers[kind].page(after) | {
            'schema_version': 1, 'capabilities': self.capabilities,
            'age_s': None if self.last_seen[kind] is None else time.monotonic()-self.last_seen[kind]})
