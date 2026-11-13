"""Verifier-owned load driver. Replays a trace against a control plane and records what came back.

It is the only thing that decides whether a request was served, and it decides from evidence the control
plane cannot author: a receipt minted inside a worker, which binds **the model**, the prompt, the token
cap, the exact output token IDs, the finish reason and a fresh per-attempt nonce. A control plane that
answers from a cache, invents a response, or quietly serves a cheaper resident model fails
arithmetically rather than statistically.

Open loop: requests fire at their trace arrival times whatever the system is doing, so a slow control
plane accumulates a backlog instead of being politely throttled by its own client.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
import queue
from pathlib import Path
import secrets
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import aiohttp
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts
import receipts
import traceunits
import looplag
from response_io import ResponseProtocolError, response_lines, small_body

# The deadline rule lives in traceunits, so the driver and the scorer cannot disagree about it: the
# driver decides when to abandon a request and the scorer decides how late a late answer was, and two
# copies of the arithmetic would be two different rules.
from traceunits import HARD_DEADLINE_SLACK_S, e2e_budget_ms  # noqa: E402,F401

# The setup window, in seconds: how long the driver holds traffic before the trace starts. A published
# constant of the task. How it was sized, and what follows from it, are NOT recorded here: what to stage
# inside the window is a decision to make, not a fact to be read.
SETUP_S = float(os.environ.get("CLUSTER_SETUP_S", "900"))
# ONE CLOCK FOR THE WHOLE RUN. If the fleet's clock is ever scaled, the driver has to scale with it or
# the two describe different worlds -- arrivals at wall speed against a fleet running N times faster is
# an N-times overprovisioned fleet, not a faster run of the same experiment. Every duration below is
# therefore divided by the scale and every measured latency multiplied back, so a cell scores the same
# whatever the scale.
#
# AGAINST REAL HARDWARE THIS IS THE IDENTITY. The graded fleet runs at 1 and cannot be told otherwise;
# the knob exists so the driver stays correct if it is ever pointed at something that can be.
TIME_SCALE = float(os.environ.get("CLUSTER_TIME_SCALE", "1"))

def wall(scaled_s: float) -> float:
    """Wall seconds to wait for `scaled_s` of fleet time. Identity at scale 1."""
    return scaled_s / TIME_SCALE


def scaled_ms(wall_s: float) -> float:
    """Wall seconds elapsed -> the fleet-time milliseconds they stand for. Identity at scale 1."""
    return wall_s * 1000.0 * TIME_SCALE
# How many times a client re-asks a conversational turn that came back empty before giving up on the
# conversation. Only conversational traffic retries: a re-sent turn REPLACES work that would otherwise
# never happen, so the trace's request set is unchanged, while retrying isolated requests would ADD load
# that is not in the trace and make the offered load a function of how badly the plane is doing.
#
# TWO. Every attempt shares the turn's one patience budget, anchored at its first arrival, so a retry
# only fires after a failure that returns before abandonment.
SESSION_RETRIES = int(os.environ.get("CLUSTER_SESSION_RETRIES", "2"))


class ReplayJournal:
    """Write the request ledger without putting filesystem waits on the clock.

    The bounded writer queue retains ordering, including unsuccessful retries
    and integrity failures. A full queue or any write/flush/close error fails
    the driver. Successful exit requires draining and closing the entire ledger;
    failed or interrupted measurements are ineligible for scoring.
    """
    def __init__(self, stream):
        self.stream = stream
        self.pending = queue.Queue(maxsize=8192)
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while (line := self.pending.get()) is not None:
                if self.error is None:
                    try:
                        self.stream.write(line)
                        self.stream.flush()
                    except Exception as exc:
                        self.error = exc
                # Drain after an error so close can always signal completion.
                # The error remains sticky and prohibits a successful exit.
        finally:
            try:
                self.stream.close()
            except Exception as exc:
                self.error = self.error or exc

    def append(self, line):
        if self.closed:
            raise RuntimeError('request journal is closed')
        if self.error is not None:
            raise RuntimeError('request journal failed') from self.error
        try:
            self.pending.put_nowait(line)
        except queue.Full as exc:
            self.error = RuntimeError('request journal cannot keep up; measurement invalid')
            raise self.error from exc

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join()
        if self.error is not None:
            raise RuntimeError('request journal failed') from self.error


def wire_signature(req):
    return (req['model'], req['stream_id'], req['input_tokens'], int(req['max_tokens']),
            int(req.get('wire_max_tokens') or req['max_tokens']),
            req.get('context_len', 32768), req.get('service_tier', 'standard'))


def prepare_wire(req, key):
    """Prepare trusted request bytes before their open-loop arrival time.

    Receipt fields bind the exact bytes being sent. Only static requests use the
    setup cache; later conversation turns retain the actual preceding response.
    """
    content = req.get('prompt_ids')
    if content is None:
        content = prompts.tokens(req['stream_id'], req['input_tokens'])
    if len(content) != req['input_tokens']:
        raise ValueError('wire prompt length differs from trace; silent input clamping is forbidden')
    count = int(req['max_tokens'])
    maximum = int(req.get('wire_max_tokens') or count)
    prompt = prompts.with_control(key, content, count)
    if len(prompt)+maximum > req.get('context_len', 32768):
        raise ValueError('issued request exceeds its model context')
    nonce = secrets.token_hex(32)
    body = {'model': req['model'], 'max_tokens': maximum, 'stream': True, 'prompt': prompt,
            'routing': {'request_id': req['request_id'], 'request_nonce': nonce,
                        'service_tier': req.get('service_tier', 'standard')}}
    return {'wire': json.dumps(body, separators=(',', ':')).encode(), 'nonce': nonce,
            'prompt_sha256': receipts.sha256_hex(prompts.canon(prompt)),
            'signature': wire_signature(req)}


class Run:
    def __init__(self, target: str, key: bytes, grader_token: str, fleet: dict, journal=None):
        self.target = target.rstrip("/")
        self.key, self.grader_token, self.fleet = key, grader_token, fleet
        self.records: list[dict] = []
        self.journal = journal
        self.record_positions: dict[str, int] = {}
        self.dispatch_lag_ms: list[float] = []
        self.prepared: dict[str, dict] = {}
        self.conversation_filler: dict[str, list[int]] = {}
        self.trace_origin = None
        self.timing_meter = None
        self.setup_timing = {}

    def prepare_trace(self, trace) -> None:
        for unit in traceunits.units(trace):
            unit = dict(unit, max_tokens=unit.get('out_tokens', unit.get('max_tokens')))
            if unit.get('class') == 'conversational':
                filler = prompts.tokens(unit['stream_id'], unit['input_tokens'])
                self.conversation_filler[unit['request_id']] = filler
                if unit['turn'] != 0:
                    continue
                unit['prompt_ids'] = filler
            self.prepared[unit['request_id']] = prepare_wire(unit, self.key)

    def record(self, rec: dict) -> None:
        # Only a live conversation needs native IDs after receipt verification.
        # Keeping every output array for the whole cell would grow memory use
        # and lengthen cyclic-GC scans during simultaneous batch arrivals.
        # The authenticated output hash and token count remain in the journal.
        rec = {k: v for k, v in rec.items() if k != 'out_ids'}
        rid = rec["request_id"]
        if rid in self.record_positions:
            self.records[self.record_positions[rid]] = rec
        else:
            self.record_positions[rid] = len(self.records)
            self.records.append(rec)
        self.write_journal(rec)

    def write_journal(self, event: dict) -> None:
        if self.journal is not None:
            line = json.dumps(event) + "\n"
            if isinstance(self.journal, ReplayJournal):
                self.journal.append(line)
            else:
                self.journal.write(line)
                self.journal.flush()

    async def one(self, sess: aiohttp.ClientSession, req: dict, record: bool = True,
                  deadline_s: float | None = None, origin: float | None = None) -> dict:
        rid = req["request_id"]
        prepared = self.prepared.pop(rid, None)
        if prepared is None:
            prepared = prepare_wire(req, self.key)
        elif prepared['signature'] != wire_signature(req):
            raise ValueError('prepared request differs from the issued trace unit')
        true_out = int(req['max_tokens'])
        wire_max = int(req.get('wire_max_tokens') or true_out)
        # The abandon timer is WALL time; the budget it comes from is in fleet time. `deadline_s`, when the
        # session driver passes one, is already wall -- it is what is left of that turn's patience.
        # PER REQUEST, from the trace -- `traceunits.deadline_ms` reads the caller's own patience and
        # only falls back to the published constant for traces without the field. Computing it from
        # the constant here would make the per-request jitter inert: every caller would give up at
        # the same, predictable moment.
        deadline = (wall(traceunits.deadline_ms(req) / 1000.0)
                    if deadline_s is None else max(0.0, deadline_s))
        rec = {"request_id": rid, "model": req["model"], "class": req["class"], "tier": req["tier"],
               "weight": req["weight"], "status": 0, "ttft_ms": None, "e2e_ms": None,
               "tpot_ms": None, "p95_tbt_ms": None, "out_tok": 0, "worker": None, "receipt_ok": False,
               "complete": False, "stream_done": False, "served": False, "tampered": False, "note": "",
               "finish_reason": ""}
        started = time.monotonic()
        scheduled = req.get("_scheduled_wall")
        if scheduled is not None:
            self.dispatch_lag_ms.append(max(0., started - scheduled) * TIME_SCALE * 1000)
        # TTFT and e2e are measured from `origin` when one is given -- the moment the CALLER first asked,
        # not the moment this attempt went out. A conversation turn that is re-sent has been waiting
        # since its first attempt, and charging it only the last attempt's latency would credit the plane
        # for a wait the caller actually experienced.
        origin = (scheduled if scheduled is not None else started) if origin is None else origin
        if deadline_s is None and scheduled is not None:
            deadline = max(.001, deadline-max(0., started-scheduled))
        if self.trace_origin is not None:
            rec["issued_at_s"] = (started-self.trace_origin)*TIME_SCALE
        out_ids, gaps, receipt, last = [], [], None, None
        malformed = False
        try:
            async with sess.post(f"{self.target}/v1/completions", data=prepared["wire"],
                                 headers={"Content-Type": "application/json", "Accept-Encoding":"identity"},
                                 auto_decompress=False,
                                 # aiohttp normally rounds long timeouts up to a
                                 # whole wall second. Keep the trace's exact
                                 # patience budget at every simulator scale.
                                 timeout=aiohttp.ClientTimeout(total=deadline, ceil_threshold=float('inf'))) as r:
                rec["status"] = r.status
                if r.status != 200:
                    await small_body(r)
                else:
                    async for raw in response_lines(r.content, req['input_tokens'] + prompts.CTRL_TOKENS, wire_max):
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            rec["stream_done"] = True
                            break
                        try:
                            obj = json.loads(chunk)
                        except (ValueError, RecursionError):
                            rec["note"] = "malformed SSE chunk"
                            malformed = True
                            continue
                        if not isinstance(obj, dict) or not isinstance(obj.get("choices", []), list):
                            malformed = True
                            continue
                        if not all(isinstance(ch, dict) for ch in obj.get("choices", [])):
                            malformed = True
                            continue
                        for ch in (obj.get("choices") or []):
                            reason = ch.get("finish_reason")
                            if reason is not None:
                                if not isinstance(reason, str) or len(reason) > 64:
                                    raise ResponseProtocolError('invalid finish reason')
                                rec["finish_reason"] = reason
                        if "cluster_receipt" in obj:
                            malformed |= receipt is not None
                            receipt = obj["cluster_receipt"]
                            continue
                        for ch in obj.get("choices", []):
                            # /v1/completions streams `text`; with return_token_ids it also streams the
                            # delta's `token_ids`. Prefer the ids: they are what the engine actually
                            # emitted, they are what the next turn of a conversation must replay to hit
                            # the prefix cache, and counting them needs no tokenizer.
                            ids = ch.get("token_ids")
                            if ids is None:
                                ids = []
                            if not isinstance(ids, list) or not all(type(t) is int and 0 <= t < 2**31 for t in ids):
                                malformed = True
                                continue
                            # Text is not authenticated. It must never start the latency
                            # clock or add samples to the token-time denominator.
                            if not ids:
                                continue
                            now = time.monotonic()
                            n = len(ids)
                            if len(out_ids) + n > wire_max:
                                raise ResponseProtocolError('response exceeds public output-token cap')
                            out_ids.extend(ids)
                            if rec["ttft_ms"] is None:
                                rec["ttft_ms"] = round(scaled_ms(now - origin), 2)
                                # All tokens in the first frame arrived together.
                                # Include their zero-length intervals so mean
                                # TPOT is (last-first)/(total_tokens-1), regardless
                                # of how many IDs the first SSE frame carries.
                                gaps.extend([0.] * (n - 1))
                            elif last is not None and n:
                                # Per-token gap, not per-chunk: a server that batches four tokens into
                                # one chunk is not four times slower than one that streams them singly.
                                # In fleet time, like every other latency here. These gaps feed `tpot_ms`,
                                # which `on_time` compares against the SLO, so on a scaled clock they must
                                # be scaled too or late requests would count as on time.
                                gaps.extend([scaled_ms(now - last) / n] * n)
                            last = now
        except asyncio.TimeoutError:
            rec["note"] = "hard deadline exceeded"
        except aiohttp.ClientError as e:
            rec["note"] = f"transport: {type(e).__name__}"
        except (ValueError, RecursionError) as e:
            malformed = True
            rec["note"] = f"invalid response: {type(e).__name__}"

        rec["e2e_ms"] = round(scaled_ms(time.monotonic() - origin), 2)
        if self.trace_origin is not None:
            rec["completed_at_s"] = (time.monotonic()-self.trace_origin)*TIME_SCALE
        # Receipts authenticate token IDs, not display text. Keep the exact IDs
        # for verification and diagnostics.
        rec["out_ids"] = out_ids
        rec["out_tok"] = len(out_ids)
        if gaps:
            # TPOT -- mean inter-token time, the standard decode SLO in the serving literature (and what
            # Aegaeon's "TBT <= 20 ms" means). NOT p95-of-gaps: on a short stream that is one noisy
            # scheduler hiccup, so a five-token reply delivered in 42 ms would fail a 20 ms/token SLO it
            # actually beat by half. p95 is still recorded, but as a diagnostic, not a verdict.
            rec["tpot_ms"] = round(statistics.fmean(gaps), 3)
            rec["p95_tbt_ms"] = round(statistics.quantiles(gaps, n=20)[-1] if len(gaps) > 1 else gaps[0], 3)

        # EVERY field is one we know independently: we sent the model, the prompt and max_tokens, and we
        # received the text and finish_reason. So nothing is carried in the receipt and taken on trust --
        # it is the MAC alone, and reconstructing the fields IS the check.
        #
        # Completeness needs no token count. If the MAC verifies, the text we hold is byte-identical to
        # what the worker produced (so nothing was dropped in transit), and `finish_reason` says whether
        # the engine ran to the end -- "length" when it produced everything it was asked for.
        #
        # Deliberately says nothing about how output length is normalised upstream: that is the one
        # detail a worker keeps to itself, and it is not derivable from anything here.
        expect = {"model": req["model"],
                  "prompt_sha256": prepared["prompt_sha256"],
                  # The receipt binds what the worker RECEIVED -- the bucket -- so a plane that
                  # rewrites the budget in either direction is still caught.
                  "max_tokens": wire_max,
                  "text_sha256": receipts.sha256_hex(prompts.canon(out_ids)),
                  "finish_reason": rec["finish_reason"],
                  "request_nonce": prepared["nonce"]}
        rec["receipt_ok"] = not malformed and receipts.check(self.key, expect, receipt)
        rec["out_token_sha256"] = expect["text_sha256"]
        if rec["receipt_ok"]:
            rec["complete"] = (rec["stream_done"] and rec["finish_reason"] == "length"
                               and len(out_ids) == true_out
                               and rec["e2e_ms"] <= traceunits.deadline_ms(req))
            if not rec["complete"]:
                rec["note"] = f"did not run to completion (finish_reason={rec['finish_reason']!r})"
        elif rec["status"] == 200 and rec["stream_done"]:
            # TAMPERING, and the whole run is void. The client saw a well-formed stream terminated by
            # [DONE] that carries no signature any worker would have produced for this exact model,
            # prompt and budget. There is no innocent path to that state: an engine that dies mid-answer
            # never sends [DONE], a plane that gives up returns an error code, and a request that is
            # simply late is abandoned without one. What is left is the answer being invented, truncated,
            # served by a different model, or altered on the way through -- and unlike a dropped request,
            # that is not a failure to serve, it is a false claim to have served.
            rec["tampered"] = True
            rec["note"] = rec["note"] or "response does not match a valid receipt"

        on_time = (rec["ttft_ms"] is not None and rec["ttft_ms"] <= req["slo_ttft_ms"]
                   and (rec["tpot_ms"] is None or rec["tpot_ms"] <= req["slo_tbt_ms"]))
        rec["on_time"] = bool(on_time)
        rec["served"] = bool(rec["receipt_ok"] and rec["complete"] and on_time)
        # The session driver journals each attempt and scores its last outcome.
        if record:
            self.record(rec)
        return rec

    def host_of(self, nd: dict) -> str:
        return nd.get("host") or self.fleet.get("host") or "127.0.0.1"

    async def fleet_admin(self, sess: aiohttp.ClientSession, path: str, payload: dict) -> list[dict]:
        """Send an admin call to EVERY node. `reset` and `rekey` propagate to siblings within a process,
        and there is one process per node -- so on a cluster, sending to one node leaves the rest holding
        stale state and, for a rekey, an old signing key whose receipts no longer verify."""
        out = []
        for nd in self.fleet["nodes"]:
            with contextlib.suppress(Exception):
                async with sess.post(f"http://{self.host_of(nd)}:{nd['port']}{path}", json=payload,
                                     headers={"X-Grader-Token": self.grader_token}) as r:
                    out.append(await r.json())
        return out

    async def worker_stats(self, sess: aiohttp.ClientSession) -> list[dict]:
        """Every machine, through the controller -- which is also the only thing that knows what the
        plane deployed, since workers are created by a deploy rather than enumerated up front."""
        out = []
        ctl = self.fleet.get("controller") or {}
        base = f"http://{ctl.get('host') or self.fleet.get('host')}:{ctl.get('port', 9000)}"
        for path in ("/nodes", "/workers"):
            try:
                async with sess.get(f"{base}{path}", timeout=aiohttp.ClientTimeout(total=60)) as r:
                    out.append({path.strip("/"): await r.json()})
            except Exception as e:                                    # noqa: BLE001
                out.append({path.strip("/"): [], "error": type(e).__name__})
        return out


class PendingRequests:
    """Retain in-flight tasks and failures, without retaining completed outputs."""
    def __init__(self):
        self.pending = set()
        self.errors = []

    def append(self, task):
        self.pending.add(task)
        def finished(done):
            self.pending.discard(done)
            if not done.cancelled() and (exc := done.exception()) is not None:
                self.errors.append(exc)
        task.add_done_callback(finished)

    async def finish(self):
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        if self.errors:
            raise self.errors[0]


async def setup_window(sess: aiohttp.ClientSession, run: Run, seconds: float) -> dict:
    """Hold traffic for a fixed window while the plane deploys whatever it wants warm.

    The plane deploys for itself through the controller -- that is what a control plane does, and
    workers are created by a deploy rather than enumerated up front.

    THE WINDOW IS A FIXED LENGTH, held whether or not the plane is ready early. Deriving it from the
    plane's own readiness would let a plane that comes up instantly and deploys nothing get the same
    head start as one that spent the time loading weights, and a plane could game the clock by delaying
    its own boot. A plane may bring its endpoint up first and deploy after, or the reverse; it costs the
    same either way.

    A plane that is never healthy is a scored ZERO, not a crashed run: it delivered nothing, which is a
    result, and losing the trial would hide it.
    """
    end = time.monotonic() + seconds
    healthy_at = None
    while time.monotonic() < end:
        with contextlib.suppress(Exception):
            async with sess.get(f"{run.target}/health",
                                headers={"Accept-Encoding":"identity"}, auto_decompress=False,
                                timeout=aiohttp.ClientTimeout(total=max(.001, min(10., end - time.monotonic())))) as r:
                if r.status == 200 and time.monotonic() <= end:
                    healthy_at = seconds - (end - time.monotonic())
                    break
        await asyncio.sleep(min(.1, max(0., end - time.monotonic())))
    if healthy_at is None:
        return {"status": "never healthy", "healthy": False, "window_s": seconds}
    # Establish a pool near the END of setup: an HTTP server can close idle
    # sockets long before the 900-second setup window expires.
    remaining = end-time.monotonic()
    if remaining > 2.:
        await asyncio.sleep(remaining-2.)
    # Otherwise a simultaneous bulk job
    # spends its arrival budget serially constructing many TCP connections in
    # Python, even though the verifier has ample CPU headroom.
    async def prime_connection():
        with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError):
            async with sess.get(f'{run.target}/health', timeout=aiohttp.ClientTimeout(
                    total=max(.001, min(2., end-time.monotonic()))),
                    headers={"Accept-Encoding":"identity"}, auto_decompress=False) as response:
                await small_body(response)
    if end-time.monotonic() > .1:
        await asyncio.gather(*(prime_connection() for _ in range(128)))
    # Hold the REST of the window regardless: the plane is still deploying, and cutting the window short
    # because it answered early would punish bringing the endpoint up first.
    remaining = end - time.monotonic()
    if remaining > 0:
        await asyncio.sleep(remaining)
    return {"status": "ready", "healthy": True, "healthy_after_s": round(healthy_at, 1),
            "window_s": seconds}

def _unit(model: str, tier: str, cls: str, rid: str, prompt_tokens: int, out_tokens: int,
          stream_id: int, service_tier: str, ttft: float, tbt: float, wire_max: int = 0,
          slack_s: float | None = None) -> dict:
    """The flat shape `Run.one` issues, whatever item it came from.

    `max_tokens` is the TRUE length -- what the marker carries and what the deadline is computed from.
    `wire_max_tokens` is the bucket the plane is shown. Keeping both is the point: scoring is against the
    work actually asked for, while the plane plans against a bound, as it would in reality.

    `slack_s` MUST be carried. Without it every session turn and batch request would be driven against
    `traceunits.HARD_DEADLINE_SLACK_S` -- a flat fallback -- while the SCORER judges the same request
    against the jittered value in the trace, so requests would be abandoned on one clock and scored on
    another. The per-request jitter exists precisely so a plane cannot compute one deadline for
    everything.
    """
    return {"request_id": rid, "model": model, "tier": tier, "class": cls, "weight": 1.0,
            "input_tokens": int(prompt_tokens), "max_tokens": int(out_tokens),
            "wire_max_tokens": int(wire_max or out_tokens),
            "stream_id": int(stream_id), "service_tier": service_tier,
            "slo_ttft_ms": ttft, "slo_tbt_ms": tbt,
            **({} if slack_s is None else {"deadline_slack_s": float(slack_s)})}


async def _session(sess, run: Run, item: dict, t0: float, tasks: list) -> None:
    """One conversation, CLOSED-LOOP after the first turn.

    The start is open-loop -- a user opens a chat independently of our latency -- but turn N+1 is sent
    `think_after_s` after turn N COMPLETES, not at a fixed offset. That is what makes a slow fleet
    genuinely delay the conversation instead of letting requests pile up, and it is why this cannot be
    expressed as a list of arrival times.
    """
    delay = wall(item["start_s"]) - (time.monotonic() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    previous_prompt, previous_answer = [], []
    for k, t in enumerate(item["turns"]):
        req = _unit(item["model"], item["tier"], "conversational",
                    f"{item['id']}-t{k}", t["prompt_tokens"], t["out_tokens"], t["stream_id"],
                    t["service_tier"], t["slo_ttft_ms"], t["slo_tbt_ms"],
                    t.get("wire_max_tokens", 0), t.get("deadline_slack_s"))
        req["context_len"] = item.get("context_len", 32768)
        if k == 0:
            req["_scheduled_wall"] = t0 + wall(item["start_s"])
        if "new_user_tokens" in t:
            req["prompt_ids"] = prompts.conversation_body(
                t, previous_prompt, previous_answer,
                fresh=run.conversation_filler.pop(req['request_id'], None))
        # One logical arrival, one patience budget. Only failures that return
        # before abandonment can be retried. Every attempt uses a fresh nonce;
        # latency remains anchored at the first arrival. Unissued descendants
        # remain in the scoring denominator.
        rec = None
        tampered_attempts = 0
        turn_start = req.get('_scheduled_wall', time.monotonic())
        turn_deadline_s = wall(traceunits.deadline_ms(req) / 1000.0)
        for attempt in range(1 + SESSION_RETRIES):
            if attempt:
                req.pop("_scheduled_wall", None)
            attempt_deadline = turn_deadline_s - (time.monotonic() - turn_start)
            if attempt_deadline <= 0:
                break
            task = asyncio.create_task(
                run.one(sess, req, record=False, deadline_s=attempt_deadline, origin=turn_start))
            tasks.append(task)
            rec = await task                        # the user is waiting; nothing else in this session
            tampered_attempts += bool(rec and rec.get("tampered"))
            if rec is not None:
                rec["tampered_attempts"] = tampered_attempts
                rec["tampered"] = bool(tampered_attempts)
                rec["attempts"] = attempt + 1
                # Journal each outcome immediately. summarize uses the last record
                # per logical request, while integrity remains sticky across attempts.
                run.record(rec)
            if rec and rec.get("complete"):
                break
        if not (rec and rec.get("complete")):
            return                                  # never answered: the conversation stops here
        if "prompt_ids" in req:
            previous_prompt = prompts.with_control(run.key, req["prompt_ids"], req["max_tokens"])
            previous_answer = rec["out_ids"]
        think = t.get("think_after_s")
        if think is None:
            return
        await asyncio.sleep(wall(think))


async def _batch(sess, run: Run, item: dict, t0: float, tasks: list) -> None:
    """A bulk job: every sub-request submitted at once, then left to the fleet to schedule."""
    delay = wall(item["submit_s"]) - (time.monotonic() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    for r in item["requests"]:
        tasks.append(asyncio.create_task(run.one(sess, _unit(
            item["model"], item["tier"], "batch", r["id"], r["prompt_tokens"], r["out_tokens"],
            r["stream_id"], item["service_tier"], r.get("slo_ttft_ms", item["slo_ttft_ms"]),
            r.get("slo_tbt_ms", item["slo_tbt_ms"]),
            r.get("wire_max_tokens", 0), r.get("deadline_slack_s"))
            | {"context_len": item.get("context_len", 32768),
               "_scheduled_wall": t0 + wall(item["submit_s"])})))


async def replay(trace: list[dict], run: Run, warmup_s: float = 0.0,
                 skip_plan: bool = False) -> list[dict]:
    """A WARM-UP window runs before the trace with no traffic on it.

    Without one, every plane is judged on how fast it can cold-load a catalogue -- loads serialise per
    GPU, and a request that needs one is lost regardless of policy, so cold loads would be a blunt tax
    rather than a decision. With one, the fleet's starting state is the plane's own
    choice, and a cold load during the trace is a consequence of that choice.
    """
    conn = aiohttp.TCPConnector(limit=0, keepalive_timeout=max(3600., wall(SETUP_S)+60.))
    async with aiohttp.ClientSession(connector=conn) as sess:
        if skip_plan:
            # A DEVELOPMENT SHORTCUT, and nothing else: run a trace against a fleet that is already
            # loaded, without sitting out the setup window first. GRADING NEVER USES IT -- every graded
            # cell is an independent measurement that wipes the fleet, starts a fresh plane and lets it
            # plan again, so a plane cannot be handed residency it did not choose for this catalogue,
            # and no cell's score depends on an earlier cell.
            plan = {"status": "skipped (running against the fleet as it stands)", "healthy": True}
        else:
            started_at = float(os.environ.get("CLUSTER_CELL_STARTED_AT", time.monotonic()))
            remaining = wall(SETUP_S) - (time.monotonic() - started_at)
            if remaining <= 0:
                raise RuntimeError("driver startup consumed the entire setup window")
            plan = await setup_window(sess, run, remaining)
        print(f"[loadgen] setup window: {plan}", flush=True)
        if not plan.get("healthy", True):
            # Failed service still scores zero. Collect trusted timing evidence
            # so an accelerated run can distinguish it from simulator overload.
            event = {"aborted": "the control plane never answered /health inside the setup window"}
            run.write_journal(event)
            return [event, *await run.worker_stats(sess)]
        # Setup has no request SLOs. Keep its timing evidence separate from
        # replay instead of rejecting a run for connection priming that finished
        # before the fixed arrival clock. Lateness at the actual boundary is
        # still charged to every affected request and checked below.
        if run.timing_meter is not None:
            run.setup_timing = run.timing_meter.snapshot()
            run.timing_meter.reset()
        t0 = float(os.environ.get('CLUSTER_TRACE_STARTED_AT', time.monotonic()))
        run.trace_origin = t0
        tasks = PendingRequests()
        # Sessions and batch jobs run as their OWN tasks: a session has to wait for its own turn to come
        # back before sending the next, and a single sequential loop over the trace cannot express that
        # without stalling every other item behind it.
        drivers = [asyncio.create_task(_session(sess, run, it, t0, tasks))
                   for it in trace if it.get("kind") == "session"]
        drivers += [asyncio.create_task(_batch(sess, run, it, t0, tasks))
                    for it in trace if it.get("kind") == "batch"]
        for req in [it for it in trace if it.get("kind", "request") == "request"]:
            delay = wall(req["arrival_s"]) - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            req = dict(req, max_tokens=req.get("out_tokens", req.get("max_tokens")),
                       _scheduled_wall=t0 + wall(req["arrival_s"]))
            tasks.append(asyncio.create_task(run.one(sess, req)))
        # The open-loop stream is exhausted, but sessions still have turns to send.
        if drivers:
            await asyncio.gather(*drivers)
        # AWAIT EVERY REQUEST. There is no drain window and nothing is cancelled from outside.
        #
        # The per-request hard deadline above already bounds the tail -- every task abandons itself at
        # its own budget plus HARD_DEADLINE_SLACK_S. A second, global bound could cancel a request
        # still inside its own deadline because the wall clock ran out, making its fate depend on when
        # in the trace it arrived rather than on the plane. Waiting for every task is bounded by the
        # same arithmetic and scores each request on its own terms.
        pending = tasks.pending
        if pending:
            print(f"[loadgen] arrivals exhausted; awaiting {len(pending)} in-flight request(s)", flush=True)
        await tasks.finish()
        return await run.worker_stats(sess)


def partition_trace(trace, count):
    """Keep conversations together; distribute simultaneous batch requests.

    Logical IDs, job submission times, budgets and the assigned denominator are
    unchanged. This is driver parallelism, never admission or trace thinning.
    """
    parts = [[] for _ in range(count)]
    cursor = 0
    for item in trace:
        if item.get('kind') == 'batch':
            groups = [[] for _ in parts]
            for req in item['requests']:
                groups[cursor % count].append(req)
                cursor += 1
            for part, requests in zip(parts, groups):
                if requests:
                    part.append(item | {'requests': requests})
        else:
            parts[cursor % count].append(item)
            cursor += 1
    return [part for part in parts if part]


def sharded_main(args):
    """Use the verifier's CPU allocation without changing its single trace clock."""
    started = time.monotonic()
    env = dict(os.environ, CLUSTER_RECEIPT_KEY=args.key, CLUSTER_GRADER_TOKEN=args.grader_token,
               CLUSTER_DRIVER_SHARD='1')
    env.setdefault('CLUSTER_CELL_STARTED_AT', str(started))
    origin = float(env['CLUSTER_CELL_STARTED_AT']) + wall(SETUP_S)
    if args.skip_plan:
        raise ValueError('--skip-plan requires --shards 1; a shared setup clock is required')
    env['CLUSTER_TRACE_STARTED_AT'] = str(origin)
    trace = [json.loads(line) for line in open(args.trace)]
    traceunits.validate_wire_context(trace)
    parts = partition_trace(trace, args.shards)
    jobs, samples = [], []
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(*unused):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    try:
        with tempfile.TemporaryDirectory(prefix='.loadgen-', dir=Path(args.out).parent) as name:
            root = Path(name)
            try:
                for i, part in enumerate(parts):
                    (root/f'{i}.trace').write_text(''.join(json.dumps(item)+'\n' for item in part))
                    cmd = [sys.executable, str(Path(__file__).resolve()), '--shards', '1',
                           '--trace', str(root/f'{i}.trace'), '--fleet', args.fleet,
                           '--target', args.target, '--scale', str(args.scale), '--warmup', str(args.warmup),
                           '--out', str(root/f'{i}.records'), '--stats-out', str(root/f'{i}.stats')]
                    jobs.append(subprocess.Popen(cmd, env=env))
                codes = [job.wait() for job in jobs]
                with open(args.out, 'wb') as output:
                    for i in range(len(parts)):
                        with (root/f'{i}.records').open('rb') as source:
                            shutil.copyfileobj(source, output)
                        samples.append(json.loads((root/f'{i}.stats').read_text()))
                # Each shard observes the same fleet. Keep the final snapshot;
                # summing snapshots would count each real completion repeatedly.
                last = max(samples, key=lambda s: s['finished_monotonic_s'])
                timing = {'time_scale': TIME_SCALE,
                          'lag_samples': sum(s['driver_timing']['lag_samples'] for s in samples)}
                for key in ('lag_max_ms', 'lag_p99_ms', 'cpu_max_1s_pct', 'gc_max_ms'):
                    timing[key] = max(s['driver_timing'][key] for s in samples)
                timing['gc_collections'] = sum(s['driver_timing']['gc_collections'] for s in samples)
                arrivals = sorted(x for s in samples for x in s['raw_dispatch_lag_ms'])
                dispatch = {'samples': len(arrivals), 'max_ms': max(arrivals, default=0.),
                            'p99_ms': arrivals[min(len(arrivals)-1, int(.99*len(arrivals)))] if arrivals else 0.}
                result = last | {'wall_s': time.monotonic()-started,
                    'n_requests': len(traceunits.units(trace)), 'driver_timing': timing,
                    'dispatch_timing': dispatch, 'driver_shards': len(parts),
                    'issuer_peak_rss_sum_mib': sum(s['peak_rss_mib'] for s in samples),
                    'shard_returncodes': codes, 'shard_timing': [{k:s[k] for k in
                        ('driver_timing','driver_setup_timing','dispatch_timing','n_requests')} for s in samples],
                    'percentile_aggregation': 'pooled request dispatch p99; maximum per-process scheduler p99'}
                result.pop('raw_dispatch_lag_ms', None)
                Path(args.stats_out).write_text(json.dumps(result, indent=1)+'\n')
                limit = float(env.get('CLUSTER_SIM_P99_LAG_MS', '0'))
                if limit > 0 and dispatch['p99_ms'] > limit:
                    raise RuntimeError('invalid simulated measurement: pooled p99 dispatch exceeds declared budget')
                return 0 if all(code == 0 for code in codes) else 1
            finally:
                for job in jobs:
                    if job.poll() is None:
                        job.terminate()
                for job in jobs:
                    try:
                        job.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        job.kill(); job.wait()
                # Preserve already-journaled outcomes even if the supervisor is
                # interrupted. In particular, a prior integrity failure cannot
                # disappear merely because a later request or shard failed.
                with open(args.out, 'wb') as output:
                    for i in range(len(parts)):
                        path = root/f'{i}.records'
                        if path.exists():
                            with path.open('rb') as source:
                                shutil.copyfileobj(source, output)
    finally:
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--target", default="http://localhost:8000")
    # NOT required, and preferably NOT given: `--key` puts the receipt key on a command line, and
    # /proc/<pid>/cmdline is world-readable at any uid. It is passed in CLUSTER_RECEIPT_KEY instead,
    # which is -r-------- and therefore private once the reader's uid differs. The flag stays for
    # interactive use, where the key is a local one and secrecy is not the point.
    ap.add_argument("--key", default=os.environ.get("CLUSTER_RECEIPT_KEY", ""))
    ap.add_argument("--grader-token", default=os.environ.get("CLUSTER_GRADER_TOKEN", ""))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--skip-plan", action="store_true",
                    help="DEV ONLY: skip the setup window and run against the fleet as it stands")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument('--shards', type=int, default=int(os.environ.get('CLUSTER_DRIVER_SHARDS', '8')))
    a = ap.parse_args()
    if not 1 <= a.shards <= 8:
        ap.error('driver shards must be between 1 and the eight allocated verifier CPUs')
    if a.shards > 1:
        raise SystemExit(sharded_main(a))
    tolerance_ms = float(os.environ.get("CLUSTER_SIM_MAX_LAG_MS", "0"))
    if TIME_SCALE != 1 and tolerance_ms <= 0:
        raise ValueError("accelerated replay requires a predeclared CLUSTER_SIM_MAX_LAG_MS")

    fleet = json.load(open(a.fleet))
    trace = [json.loads(l) for l in open(a.trace)]
    traceunits.validate_wire_context(trace)
    if a.scale < 1.0:
        cut = max(r["arrival_s"] for r in trace) * a.scale
        trace = [r for r in trace if r["arrival_s"] <= cut]

    if not a.key:
        raise SystemExit("loadgen: no receipt key. Pass --key, or set CLUSTER_RECEIPT_KEY. An empty key "
                         "signs nothing and every request would score incomplete, which is a confusing "
                         "way to discover a missing environment variable.")
    journal = ReplayJournal(open(a.out, "w"))
    run = Run(a.target, a.key.encode(), a.grader_token, fleet, journal=journal)
    t = time.monotonic()
    os.environ.setdefault('CLUSTER_CELL_STARTED_AT', str(t))
    stats = None
    timing = {}
    async def measured_replay():
        nonlocal timing
        async with looplag.Meter(TIME_SCALE) as meter:
            run.timing_meter = meter
            try:
                return await replay(trace, run, a.warmup, a.skip_plan)
            finally:
                timing = meter.snapshot()
    try:
        # Preparation consumes the fixed setup window, including across shards.
        # Conversation history continues to use the preceding actual answer.
        run.prepare_trace(trace)
        # Each issuer is a bounded one-cell process. Cyclic GC pauses of several
        # milliseconds would delay dispatch even after freezing static inputs, so
        # it is deferred until replay ends; reference counting and explicit HTTP
        # cleanup remain active. Completed request tasks are released promptly;
        # peak RSS is reported for validation.
        gc.collect()
        gc.freeze()
        print(f'[loadgen] prepared {len(run.prepared)} static requests in {time.monotonic()-t:.2f}s', flush=True)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            stats = asyncio.run(measured_replay(), loop_factory=uvloop.new_event_loop)
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
    finally:
        # Always write what was collected, even on a timeout, a Ctrl-C or a crash. A run that dies with
        # no records file is scored as a failed trial rather than as the bad run it was, and the
        # difference matters: one is a lost measurement, the other is a result.
        wall = time.monotonic() - t
        journal.close()
        # UNITS, not lines. A session line is one conversation and several turns, a batch line one job
        # and hundreds of requests, so `len(trace)` would undercount the requests and inflate the served
        # fraction printed below.
        json.dump({"wall_s": round(wall, 2), "finished_monotonic_s": time.monotonic(),
                   "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                   "n_requests": len(traceunits.units(trace)),
                   "worker_stats": stats, "driver_timing": timing,
                   "driver_setup_timing": run.setup_timing,
                   "driver_timing_scope": "replay and final statistics; fixed setup reported separately",
                   "dispatch_timing": {"samples": len(run.dispatch_lag_ms),
                       "max_ms": max(run.dispatch_lag_ms, default=0.),
                       "p99_ms": (sorted(run.dispatch_lag_ms)[min(len(run.dispatch_lag_ms)-1, int(.99 * len(run.dispatch_lag_ms)))]
                                  if run.dispatch_lag_ms else 0.)},
                   "raw_dispatch_lag_ms": run.dispatch_lag_ms if os.environ.get('CLUSTER_DRIVER_SHARD') else [],
                   "sim_max_lag_ms": tolerance_ms},
                  open(a.stats_out, "w"), indent=1)
    nodes = [n for group in stats or [] for n in group.get("nodes", [])]
    if TIME_SCALE != 1 and not nodes:
        raise RuntimeError("accelerated replay has no node timing evidence")
    looplag.check_acceleration(TIME_SCALE, tolerance_ms, timing, nodes,
                              expected_nodes=[n['name'] for n in fleet['nodes']])
    if tolerance_ms > 0 and max(run.dispatch_lag_ms, default=0.) > tolerance_ms:
        raise RuntimeError("invalid simulation measurement: arrival dispatch lag exceeds declared budget")
    p99_limit=float(os.environ.get('CLUSTER_SIM_P99_LAG_MS','0'))
    if p99_limit>0 and run.dispatch_lag_ms and not os.environ.get('CLUSTER_DRIVER_SHARD'):
        p99=sorted(run.dispatch_lag_ms)[min(len(run.dispatch_lag_ms)-1,int(.99*len(run.dispatch_lag_ms)))]
        if p99>p99_limit:
            raise RuntimeError('invalid simulation measurement: p99 dispatch lag exceeds declared budget')
    served = sum(1 for r in run.records if r["served"])
    n_units = len(traceunits.units(trace))
    print(f"[loadgen] {n_units} requests in {wall:.1f}s   served {served} ({served/max(1, n_units):.1%})")


if __name__ == "__main__":
    main()
