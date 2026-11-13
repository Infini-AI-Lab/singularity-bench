"""Shared GPU service clocks for a simulator spread over manager processes.

The first manager owns the resource ledger. Managers exchange one batched clock
snapshot per tick, independent of token count. Prefill and decode share GPU time
across engines; each engine advances at its slowest rank's service rate. This
prevents a remote rank from appearing idle merely because its HTTP head is on
another tray. This endpoint is protected by the normal manager credential.
"""
import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import time

import aiohttp


@dataclass
class Engine:
    parts: dict = field(default_factory=dict)
    decoding: bool = False
    prefilling: bool = False
    credit: float = 0.
    work_credit: float = 0.
    card_set: set = field(default_factory=set)
    rate: float = 0.
    changed: object = field(default_factory=asyncio.Event)

    @property
    def cards(self):
        return self.card_set


class Fabric:
    def __init__(self):
        self.engines = {}
        self.busy = set()
        self.locks = {}
        self.previous = time.monotonic()

    def advance(self, now=None):
        now = time.monotonic() if now is None else now
        elapsed = max(0., now-self.previous)
        self.previous = now
        for e in self.engines.values():
            if e.rate:
                e.work_credit += elapsed*e.rate
                if not e.prefilling:
                    e.credit += elapsed*e.rate

    def notify(self, changed_cards):
        self.busy = {card for e in self.engines.values() if e.prefilling for card in e.cards}
        active = [x for x in self.engines.values() if x.decoding or x.prefilling]
        sharing = {}
        for e in active:
            for card in e.cards:
                sharing[card] = sharing.get(card, 0)+1
        for e in self.engines.values():
            e.rate = (min(speed/sharing[card] for cards,speed in e.parts.values() for card in cards)
                      if e.cards and (e.decoding or e.prefilling) else 0.)
            if e.cards & changed_cards:
                # Wake only overlapping forwards. Broadcasting each GPU event
                # to every tray turns a large burst into quadratic CPU work.
                e.changed.set()
                e.changed = asyncio.Event()

    def register(self, wid, node, cards, speed):
        self.advance()
        e = self.engines.setdefault(wid, Engine())
        e.parts[node] = (tuple((node, int(c)) for c in cards), float(speed))
        e.card_set = {card for cards,_ in e.parts.values() for card in cards}
        self.notify(e.cards)

    def remove(self, wid):
        self.advance()
        e = self.engines.pop(wid, None)
        self.locks.pop(wid, None)
        if e:
            e.changed.set()
            self.notify(e.cards)

    def snapshot(self, states):
        self.advance()
        changed = set()
        for wid, active in states.items():
            if wid in self.engines:
                if self.engines[wid].decoding != bool(active):
                    changed.update(self.engines[wid].cards)
                self.engines[wid].decoding = bool(active)
        if changed:
            self.notify(changed)
        return {wid: self.engines[wid].credit for wid in states if wid in self.engines}

    async def prefill(self, wid, seconds):
        e = self.engines[wid]
        # One engine iteration at a time, but independent engines really do
        # run concurrently. A whole-chunk GPU lock would let one co-tenant
        # finish at its solo latency while the other waited.
        lock = self.locks.setdefault(wid, asyncio.Lock())
        await lock.acquire()
        try:
            # Teardown or reset can retire a queued prefill.
            if self.engines.get(wid) is not e:
                raise ValueError('retired simulated engine')
            self.advance()
            e.prefilling = True
            self.notify(e.cards)
            try:
                finish = e.work_credit+max(0., seconds)
                while True:
                    self.advance()
                    if self.engines.get(wid) is not e:
                        raise ValueError('retired simulated engine')
                    remaining = finish-e.work_credit
                    if remaining <= 0:
                        break
                    changed = e.changed
                    try:
                        await asyncio.wait_for(changed.wait(), remaining/e.rate)
                    except asyncio.TimeoutError:
                        pass
            finally:
                self.advance()
                e.prefilling = False
                self.notify(e.cards)
        finally:
            lock.release()


class Client:
    def __init__(self, node, url, token, tick, local=None):
        self.node, self.url, self.token, self.tick = node, url, token, tick
        self.local = local
        self.session = None
        self.task = None
        self.credits = {}
        self.error = None
        self.connected = False
        self.failure_count = 0
        self.last_failure = None
        self.snapshot_lock = asyncio.Lock()

    async def start(self):
        self.session = aiohttp.ClientSession(headers={'X-Grader-Token': self.token},
            timeout=aiohttp.ClientTimeout(total=1200))
        self.task = asyncio.create_task(self.poll())

    async def call(self, operation, **data):
        if self.local:
            return await apply(self.local, {'operation': operation, **data})
        try:
            async with self.session.post(self.url+'/sim/fabric', json={'operation': operation, **data}) as r:
                result = await r.json()
                if r.status != 200:
                    # A worker retired during teardown is a normal cancellation.
                    # Server errors and transport failures invalidate the replay.
                    if r.status >= 500:
                        self.failed(f'HTTP {r.status}')
                    raise RuntimeError(f'simulator fabric: {r.status}: {result}')
                self.connected = True
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self.failed(f'{type(exc).__name__}: {exc}')
            raise

    def failed(self, message):
        # Managers may start before the coordinator opens its listener. Once a
        # connection has succeeded, retain any outage even after it recovers.
        if self.connected:
            self.failure_count += 1
            self.last_failure = message

    def measurement(self):
        return {'fabric_failures': self.failure_count, 'fabric_last_failure': self.last_failure}

    async def poll(self):
        while True:
            try:
                states = {wid: any(r.get('decoding') and not r['aborted'] for r in e.active.values())
                          for wid, e in self.node.engines.items() if e.is_head and e.state == 'awake'}
                async with self.snapshot_lock:
                    # Re-read after waiting for an immediate activity update.
                    states = {wid: any(r.get('decoding') and not r['aborted'] for r in e.active.values())
                              for wid, e in self.node.engines.items() if e.is_head and e.state == 'awake'}
                    self.credits.update(await self.call('snapshot', states=states))
                    self.credits = {wid: value for wid, value in self.credits.items() if wid in self.node.engines}
                self.error = None
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
                # Fail measured work if the clock source is unavailable; never
                # let disconnected peers earn fabricated compute time.
                self.error = str(exc)
            await asyncio.sleep(self.tick)

    async def activity(self, wid, active):
        async with self.snapshot_lock:
            # Another request may have arrived while this coroutine waited.
            engine = self.node.engines.get(wid)
            active = engine is not None and any(r.get('decoding') and not r['aborted'] for r in engine.active.values())
            self.credits.update(await self.call('snapshot', states={wid: active}))

    def credit(self, wid):
        if self.error:
            raise RuntimeError(self.error)
        return self.credits.get(wid, 0.)

    async def close(self):
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        await self.session.close()


async def apply(fabric, body):
    op = body['operation']
    if op == 'register':
        fabric.register(body['worker'], body['node'], body['cards'], body['speed'])
    elif op == 'remove':
        fabric.remove(body['worker'])
    elif op == 'snapshot':
        return fabric.snapshot(body['states'])
    elif op == 'prefill':
        seconds = float(body['seconds'])
        if not 0 <= seconds <= 1200:
            raise ValueError('invalid simulated prefill duration')
        await fabric.prefill(body['worker'], seconds)
    else:
        raise ValueError('unknown simulated fabric operation')
    return {'ok': True}
