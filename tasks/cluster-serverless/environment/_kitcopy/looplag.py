"""Measure scheduler delay in fleet milliseconds, without inventing CPU capacity."""
from __future__ import annotations

import asyncio
import contextlib
import gc
import heapq
import math
import os
import time
from collections import deque


class _P99:
    """Exact floor(0.99*n) order statistic, with O(log n) insertion.

    Stats requests must not repeatedly sort an ever-growing history on the
    serving event loop. The lower max-heap includes the selected order statistic;
    the upper min-heap holds the remaining values. No lag samples are discarded.
    """
    def __init__(self):
        self.lower, self.upper = [], []

    def add(self, value):
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)
        target = int(.99 * len(self)) + 1
        while len(self.lower) > target:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        while len(self.lower) < target:
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def __len__(self):
        return len(self.lower) + len(self.upper)

    def value(self):
        return -self.lower[0] if self.lower else 0.


class Meter:
    def __init__(self, scale: float = 1., interval: float = .05):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("time scale must be finite and positive")
        self.scale, self.interval = scale, interval
        self.samples = _P99()
        self.cpu_max = 0.
        self.task = None
        self.gc_started = None
        self.gc_count, self.gc_max = 0, 0.
        self.started = None
        self.worst_event = {}
        self.cpu_history = deque()

    def _gc(self, phase, info):
        if phase == 'start':
            self.gc_started = time.monotonic()
        elif self.gc_started is not None:
            self.gc_count += 1
            self.gc_max = max(self.gc_max, (time.monotonic()-self.gc_started)*1000*self.scale)
            self.gc_started = None

    def reset(self):
        self.samples = _P99()
        self.cpu_max = 0.
        self.gc_count, self.gc_max = 0, 0.
        self.worst_event = {}
        self.started = time.monotonic()
        self.cpu_history.clear()
        self.cpu_history.append((self.started, time.process_time()))

    async def _run(self):
        history=self.cpu_history
        history.append((time.monotonic(),time.process_time()))
        while True:
            due = time.monotonic() + self.interval
            await asyncio.sleep(self.interval)
            delay = max(0., time.monotonic() - due) * 1000 * self.scale
            self.samples.add(delay)
            now,cpu=time.monotonic(),time.process_time()
            if delay > self.worst_event.get('lag_ms', -1.):
                self.worst_event = {'lag_ms': delay,
                                    'wall_s_since_meter_start': now-self.started,
                                    'sampling_interval_wall_ms': 1000*(now-history[-1][0]),
                                    'sampling_interval_cpu_ms': 1000*(cpu-history[-1][1])}
            history.append((now,cpu))
            while len(history)>2 and now-history[1][0]>=1.:
                history.popleft()
            if now-history[0][0]>=1.:
                self.cpu_max = max(self.cpu_max, 100*(cpu-history[0][1])/(now-history[0][0]))

    async def __aenter__(self):
        self.started = time.monotonic()
        gc.callbacks.append(self._gc)
        self.task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc):
        gc.callbacks.remove(self._gc)
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    def snapshot(self):
        return {"time_scale": self.scale, "lag_samples": len(self.samples),
                "lag_max_ms": self.worst_event.get('lag_ms', 0.),
                "cpu_max_1s_pct": self.cpu_max,
                "gc_collections": self.gc_count,
                "gc_max_ms": self.gc_max,
                "worst_event": dict(self.worst_event),
                "lag_p99_ms": self.samples.value()}


def check_acceleration(scale: float, tolerance_ms: float, driver: dict, nodes: list[dict],
                       expected_nodes=None) -> None:
    """Fail the measurement if clocks disagree or declared error bounds are exceeded."""
    if (scale != 1 or tolerance_ms > 0) and expected_nodes is not None:
        observed = {n.get('node') for n in nodes}
        if observed != set(expected_nodes):
            raise RuntimeError('simulated replay has incomplete node timing evidence')
    for node in nodes:
        measured = node.get("simulation")
        actual = measured["time_scale"] if measured else 1.
        if actual != scale:
            raise RuntimeError(f"driver/fleet clock mismatch: {scale} vs {actual}")
        if measured and measured.get('fabric_failures', 0):
            raise RuntimeError(f"invalid simulated measurement: GPU service coordinator failed: {measured}")
    if scale == 1 and tolerance_ms <= 0:
        return
    if not math.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("accelerated replay requires a positive CLUSTER_SIM_MAX_LAG_MS")
    for name, sample in [("driver", driver)] + [(n.get("node", "node"), n.get("simulation", {})) for n in nodes]:
        if not sample.get("lag_samples") or sample["lag_max_ms"] > tolerance_ms:
            raise RuntimeError(f"invalid accelerated measurement: {name} scheduler lag exceeds "
                               f"{tolerance_ms} fleet ms, or no samples were collected: {sample}")
        p99_limit=float(os.environ.get('CLUSTER_SIM_P99_LAG_MS','0'))
        cpu_limit=float(os.environ.get('CLUSTER_SIM_MAX_CPU_PCT','0'))
        if p99_limit>0 and sample.get('lag_p99_ms',float('inf'))>p99_limit:
            raise RuntimeError(f'invalid simulated measurement: {name} p99 scheduler lag exceeds {p99_limit} ms')
        if cpu_limit>0 and sample.get('cpu_max_1s_pct',float('inf'))>cpu_limit:
            raise RuntimeError(f'invalid simulated measurement: {name} lacks CPU headroom ({sample})')
