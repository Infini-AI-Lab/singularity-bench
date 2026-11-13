#!/usr/bin/env python3
"""simulator.py — the serving-traffic simulator that SCORES your server.

It does two things, and nothing else:

  1. IT SIMULATES SERVING TRAFFIC. Recorded agentic coding sessions (the AgentX corpus:
     393 Claude Code sessions, SemiAnalysis `cc-traces-weka-062126-256k`) are replayed
     against your server on a FIXED number of concurrent session lanes for a FIXED
     window -- a stand-in for a coding-agent product's real load.

  2. IT SCORES ANY SERVER. It speaks only the OpenAI `/v1/completions` HTTP contract,
     so it is completely AGNOSTIC to how your server is built: what framework, what
     parallelism, what kernels, what language. Point it at a base URL and it produces
     a number. The grader runs its own copy of this same file against your server.

THE NUMBER IS MONEY. Every token you serve is billed at the prices below, and the score
is what the machine EARNED per GPU-hour:

    income = (input - cached) x PRICE.input + cached x PRICE.cached_input
             + output x PRICE.output                                  (USD per 1M tokens)
             summed over every request that COMPLETED SUCCESSFULLY (see below)
    score  = income / (GPU_COUNT x DURATION_S in hours)

  The denominator is the WINDOW, a constant: requests are only started inside it, and a
  request in flight when it closes still pays if it completes within DRAIN_S. So nothing
  about the denominator is yours to move, and completing a request can only raise your
  score -- never lower it.

  * input tokens are counted by THIS client (it sends token ids);
  * cached tokens are what your server reports in usage.prompt_tokens_details.cached_tokens
    (capped at the input). They are REPORTED but priced like input, so the split cannot
    change your income;
  * output tokens are what your server reports in usage.completion_tokens, capped at
    max_tokens and at the UTF-8 bytes of the text actually received + OUTPUT_SLACK (every
    token of a byte-level BPE vocabulary decodes to at least one byte, so a server cannot
    claim more tokens than the text it sent could hold).

THE CONFIGURATION BELOW IS THE GRADED CONFIGURATION. The load is not yours to choose:
CONCURRENCY and DURATION_S are fixed. Read the constants, read `Replayer`, and see for
yourself exactly what traffic your server will face.

THE ONE THING THE GRADER CHANGES IS THE SEED. It runs this same file with a seed you are
not given. The seed decides which sessions are replayed, where each one starts, and the
content of every prompt -- so the corpus, its statistics and the replay rules are exactly
what you see here, but no prompt you see in development is ever sent at grade time.

YOU ARE EXPECTED TO SPECIALIZE FOR THIS TRAFFIC. Tuning your server for the shape you
find here is the intended skill, not cheating -- it cannot help you pass the separate
accuracy gate, which runs on real prompts.

HOW THE LOAD WORKS (the code below is the authority):
  * CONCURRENCY lanes. Each lane replays one session at a time, taken in order from a
    seeded queue, starting at a seeded point inside the session. The lane's first request
    is a PRIMER (max_tokens 1, not billed) carrying the main agent's context at that
    point: it stands for the conversation the lane joins part-way through, so its prefill
    is inside the clock and is not paid. The lane then replays every later request of
    the session, in ALL its streams (main agent, subagents, parallel workers), and moves
    to the next queue entry when done. A LANE IS A SESSION, NOT A REQUEST: its streams
    run in parallel, so the number of requests in flight is not bounded by CONCURRENCY.
  * A request is sent min(gap_s, MAX_GAP_S) after the request it follows finishes, and
    after every stream it waits for (a parent waiting for its subagents) has finished.
  * WHICH PROMPTS ARRIVE COLD. Most requests extend a prompt sent earlier in the same
    session. Two kinds do not: (a) the FIRST request of a newly spawned stream (`base`
    null in the session file: a subagent starting up) shares nothing with anything sent
    before but the chat header and the session marker; (b) a request whose predecessor
    lies BEFORE the lane's start point (a subagent already running when the lane joined)
    shares only what its prompt has in common with the primer's. Kind (b) is sent at its
    recorded offset from the lane's start request, capped at MAX_GAP_S, and never before
    the primer has finished. Both are billed like any other request.
  * If nothing is in flight anywhere, pending start times shift so the next request
    goes out within SYSTEM_IDLE_CAP_S.
  * Requests START during DURATION_S (the clock starts when the primers go out, so their
    prefill is inside it). Those in flight when it closes may finish within DRAIN_S; one
    still unfinished then is cancelled: it pays nothing and is not a failure either.
  * Consecutive requests of a session re-send the conversation so far: a request's prompt
    is the first `keep` 64-token blocks of an earlier request's prompt plus `add` fresh
    blocks. Block content is code sliced from agentx/code.txt at seed-derived offsets, so
    blocks shared in the recording are identical here.
  * Each request: POST /v1/completions, `prompt` = TOKEN IDS, stream=true with
    stream_options.include_usage, temperature 0, ignore_eos=true,
    skip_special_tokens=false, max_tokens = the recorded output length (capped at
    MAX_OUTPUT_TOKENS and at what CONTEXT_LIMIT still holds: see max_tokens_for).
  * A request SUCCEEDS if it returns HTTP 200, reports usage.completion_tokens, and its
    billed output reaches max_tokens - OUTPUT_SLACK. So the server MUST honor ignore_eos
    (generate the full length)
    ... and honor it HONESTLY: the tokens past EOS must come from the same code path as
    the tokens before it -- EOS fed back as an ordinary token, no branch on it except to
    keep going -- never padding, a repeated token or a cheaper path used only past EOS.
    This file cannot tell the difference; the delivered /app is read for it after
    grading, the verdict is recorded with the grade, and a server found doing it scores 0.
    The text is checked only through the byte cap above --
    a ceiling on the tokens that can be claimed (at most one per byte of text received),
    not a verification of the text: dropping special tokens (ignoring
    skip_special_tokens=false) fails a request only when the text that is left is too
    short to cover max_tokens - OUTPUT_SLACK. Anything else that finishes is a FAILURE
    and pays nothing.

GATE (inside this file): let finished = succeeded + failed, over the non-primer requests
(one cut off at the drain deadline is neither). The run passes if at least one succeeded
and failed <= max(1, floor((1 - COMPLETION_MIN) x finished)) -- e.g. 1 failure allowed
at 60 finished, 4 at 450. Otherwise the score is 0.

Run it against your own server:
    bash /app/start_server.sh
    python3 /app/simulator.py                  # the graded shape, on the development seed
    python3 /app/simulator.py --duration 120   # a shorter window, for development only
    python3 /app/simulator.py --seed 7         # a different draw of sessions and content

By default it targets http://127.0.0.1:8000 (where your server must listen). Override
the ADDRESS with SIM_BASE=http://host:port. Results: $SIM_OUT (default /tmp/simulator)
/result.json and /details.json (one record per request).
"""
import argparse, asyncio, gzip, hashlib, heapq, json, os, random, sys, time
from array import array
from pathlib import Path

# ===========================================================================
# THE GRADED CONFIGURATION.
# ===========================================================================
CONCURRENCY = 16          # session lanes replayed in parallel -- FIXED, not yours to choose
DURATION_S = 900          # window in which requests START
DRAIN_S = 300             # in-flight requests may finish within this long after the window
SEED = 20260921           # DEVELOPMENT seed; the grader runs this file with a different one
GPU_COUNT = 4             # the denominator: all 4 GPUs count, whether or not you use them

# USD per 1M tokens: OpenRouter's listing for z-ai/glm-5.3-flash on 2026-09-21
# ($0.09 in, $0.30 out).
# Cached input is priced like input ON PURPOSE: cached tokens are server-reported, and
# equal prices remove any incentive to misreport them.
PRICE = {"input": 0.09, "cached_input": 0.09, "output": 0.30}

MAX_GAP_S = 60            # recorded think-time between requests is capped at this
SYSTEM_IDLE_CAP_S = 10    # with nothing in flight, the next request goes out within this
START_MAX_FRACTION = 0.9  # a lane starts within the first 90% of a session's main turns
MAX_OUTPUT_TOKENS = 8192  # recorded output lengths are capped at this
CONTEXT_LIMIT = 262144    # the corpus is the 256k-context drop, which this model's context
                          # holds in full: ALL 393 sessions are replayed (longest prompt
                          # 255,836 tokens), and prompt + output <= this
REQUEST_TIMEOUT_S = 1500  # per request; in the graded shape the drain deadline always comes first
COMPLETION_MIN = 0.99     # GATE: fraction of finished requests that must succeed (see score_records)
# ===========================================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/GLM-5.3-Flash")
DATA_DIR = Path(__file__).resolve().parent / "agentx"
BLOCK = 64                # tokens per content block (the corpus's block size)
MARKER_TOKENS = 16        # per-trajectory marker: every replayed session is a fresh prefix
OUTPUT_SLACK = 4          # tokens: a character cut off at the very end may not reach the text


def log(msg):
    print(f"[sim] {msg}", flush=True)


def h64(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _load_tokenizer(path=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path or MODEL_PATH)


def _chat_frame(tok):
    """The model's own chat framing around one user message, as (head, tail) strings."""
    try:
        rendered = tok.apply_chat_template([{"role": "user", "content": "\x01"}], tokenize=False,
                                           add_generation_prompt=True)
        head, tail = rendered.split("\x01")
    except Exception:
        head, tail = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>", "<|assistant|><think>"
    return head, tail


# ------------------------------------------------------------------ content

class Content:
    """Deterministic prompt synthesis from the seed: chat framing + per-trajectory marker + 64-token code blocks."""

    def __init__(self, tok, seed, salt=""):
        self.seed, self.salt = seed, salt
        tok.model_max_length = 10**9    # the code pool is tokenized in one go; silence the length warning
        text = (DATA_DIR / "code.txt").read_text()
        special = set(tok.all_special_ids)
        ids = [t for t in tok(text, add_special_tokens=False).input_ids if t not in special]
        self.pool = array("l", ids)
        head, tail = _chat_frame(tok)
        self.head = tok(head, add_special_tokens=False).input_ids
        self.tail = tok(tail, add_special_tokens=False).input_ids

    def block(self, traj, key):
        off = h64(self.seed, traj, key) % (len(self.pool) - BLOCK)
        return self.pool[off:off + BLOCK]

    def marker(self, traj):
        return [self.pool[h64(self.seed, self.salt, "marker", traj, i) % len(self.pool)]
                for i in range(MARKER_TOKENS)]


def request_block_keys(session, idx, memo):
    """Block keys of request idx's prompt: the first `keep` keys of its base, then `add` fresh keys."""
    chain, i = [], idx
    while i is not None and i not in memo:
        chain.append(i)
        i = session["requests"][i]["base"]
    for j in reversed(chain):
        r = session["requests"][j]
        prefix = memo[r["base"]][:r["keep"]] if r["base"] is not None else array("q")
        memo[j] = prefix + array("q", (j * 100_000 + k for k in range(r["add"])))
    return memo[idx]


def prompt_ids(content, session, idx, traj, memo):
    ids = list(content.head) + content.marker(traj)
    for key in request_block_keys(session, idx, memo):
        ids.extend(content.block(traj, key))
    ids.extend(content.tail)
    return ids


def load_sessions(frame_tokens=0):
    """The recorded sessions that are replayed: a session is kept only if its LONGEST prompt
    (chat framing + marker + blocks) fits in CONTEXT_LIMIT, as AgentX drops traces a
    server's context cannot hold. Whether that drops any is said at CONTEXT_LIMIT.
    Deterministic, so every seed draws from the same list."""
    with gzip.open(DATA_DIR / "sessions.jsonl.gz", "rt") as f:
        sessions = [json.loads(line) for line in f]
    room = CONTEXT_LIMIT - frame_tokens - MARKER_TOKENS - 1      # -1: at least one output token
    return [s for s in sessions if max((r["keep"] + r["add"]) * BLOCK for r in s["requests"]) <= room]


def plan_trajectory(sessions, seed, position):
    """Queue position -> (session index, start request index). Deterministic in (seed, position)."""
    rng = random.Random(h64(seed, "queue", position))
    s = rng.randrange(len(sessions))
    reqs = sessions[s]["requests"]
    main = [i for i, r in enumerate(reqs) if sessions[s]["streams"][r["stream"]]["kind"] == "main"]
    horizon = max(1, int(len(main) * START_MAX_FRACTION))
    return s, main[rng.randrange(horizon)]


def max_tokens_for(session, idx, n_prompt):
    out = min(session["requests"][idx]["out"], MAX_OUTPUT_TOKENS)
    return max(1, min(out, CONTEXT_LIMIT - n_prompt))


# ------------------------------------------------------------------ replay

class Trajectory:
    def __init__(self, position, session_index, session, start, t0):
        self.position, self.si, self.session, self.start, self.t0 = position, session_index, session, start, t0
        reqs = session["requests"]
        self.members = [i for i in range(start + 1, len(reqs))]
        self.done = {}                       # request idx -> completion time
        self.last_of_stream = {}
        for i in self.members:
            self.last_of_stream[reqs[i]["stream"]] = i
        self.remaining = set(self.members)
        self.scheduled = set()               # requests already queued or sent
        self.memo = {}
        self.lane = None


class Replayer:
    def __init__(self, url, model, content, sessions, duration_s, progress_s=60):
        self.url, self.model, self.content, self.sessions = url, model, content, sessions
        self.duration_s, self.progress_s = duration_s, progress_s
        self.heap, self.seq, self.inflight = [], 0, 0
        self.records, self.next_position = [], 0
        self.lanes = {}
        self.stop_dispatch = False
        self.tasks = set()

    def push(self, when, traj, idx, primer=False):
        self.seq += 1
        traj.scheduled.add(idx)
        heapq.heappush(self.heap, (when, self.seq, traj, idx, primer))

    def start_lane(self, lane, now):
        position = self.next_position
        self.next_position += 1
        si, start = plan_trajectory(self.sessions, self.content.seed, position)
        traj = Trajectory(position, si, self.sessions[si], start, now)
        traj.lane = lane
        self.lanes[lane] = traj
        self.push(now, traj, start, primer=True)

    def ready_time(self, traj, idx):
        """None if dependencies are pending, else the time the request may be sent."""
        r = traj.session["requests"][idx]
        after = r["after"]
        if after is not None and after >= traj.start:
            if after not in traj.done:
                return None
            when = traj.done[after] + min(r["gap_s"], MAX_GAP_S)
        else:
            start_s = traj.session["requests"][traj.start]["start_s"]
            when = traj.t0 + min(max(0.0, r["start_s"] - start_s), MAX_GAP_S)
        for stream in r["waits_for"]:
            last = traj.last_of_stream.get(stream)
            if last is not None and last < idx:
                if last not in traj.done:
                    return None
                when = max(when, traj.done[last])
        return when

    def on_complete(self, traj, idx, now):
        traj.done[idx] = now
        traj.remaining.discard(idx)
        if self.stop_dispatch:
            return
        for j in list(traj.remaining):
            if j in traj.scheduled:
                continue
            when = self.ready_time(traj, j)
            if when is not None:
                self.push(when, traj, j)
        if not traj.remaining and self.lanes.get(traj.lane) is traj:
            self.start_lane(traj.lane, now)

    @staticmethod
    def _on_event(rec, line):
        """Fold one SSE line into the request's record."""
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        obj = json.loads(data)
        for choice in obj.get("choices") or []:
            piece = choice.get("text") or ""
            if piece:
                now = time.monotonic()
                rec["t_first"] = rec["t_first"] or now
                rec["t_last"] = now
                rec["text"] += piece
        usage = obj.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            rec["cached_tokens"] = int(details["cached_tokens"])
        if usage.get("completion_tokens") is not None:
            rec["reported_output_tokens"] = int(usage["completion_tokens"])

    async def send(self, http, traj, idx, primer):
        import aiohttp
        rec = {"position": traj.position, "session": traj.si, "request": idx, "primer": primer, "lane": traj.lane,
               "input_tokens": 0, "max_tokens": 1, "t_send": time.monotonic(), "t_first": None, "t_last": None,
               "text": "", "cached_tokens": 0, "reported_output_tokens": None, "ok": False, "error": None}
        self.inflight += 1
        try:
            # Everything that can raise is inside the try: a request must always leave a
            # record and release its lane, whatever went wrong.
            ids = prompt_ids(self.content, traj.session, idx, traj.position, traj.memo)
            mt = 1 if primer else max_tokens_for(traj.session, idx, len(ids))
            rec.update(input_tokens=len(ids), max_tokens=mt, t_send=time.monotonic())
            body = {"model": self.model, "prompt": ids, "max_tokens": mt, "temperature": 0.0, "ignore_eos": True,
                    "skip_special_tokens": False, "stream": True, "stream_options": {"include_usage": True}}
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            async with http.post(self.url + "/v1/completions", json=body, timeout=timeout) as resp:
                if resp.status != 200:
                    rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                else:
                    buf = b""
                    async for chunk in resp.content.iter_any():
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._on_event(rec, line)
                    self._on_event(rec, buf)      # a last event with no trailing newline still counts
                    rec["ok"] = True
        except asyncio.CancelledError:
            rec["aborted"] = True
            rec["error"] = "aborted at the drain deadline"
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            self.inflight -= 1
            rec["t_done"] = time.monotonic()
            self.records.append(rec)
            self.on_complete(traj, idx, rec["t_done"])

    def _progress(self, t_start, now):
        done = [r for r in self.records if not r["primer"] and r["ok"]]
        in_tok = sum(r["input_tokens"] for r in done)
        log(f"  t={now - t_start:6.0f}s  finished={len(done)}  in flight={self.inflight}  "
            f"sessions started={self.next_position}  input tokens served={in_tok:,}")

    async def run(self):
        import aiohttp

        # One connection per request: reusing an idle keep-alive connection races the
        # server closing it (seen as ServerDisconnectedError against real engines).
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as http:
            t_start = time.monotonic()
            for lane in range(CONCURRENCY):
                self.start_lane(lane, t_start)
            end = t_start + self.duration_s
            next_progress = t_start + self.progress_s
            while time.monotonic() < end:
                now = time.monotonic()
                if self.heap and self.inflight == 0 and self.heap[0][0] > now + SYSTEM_IDLE_CAP_S:
                    shift = self.heap[0][0] - (now + SYSTEM_IDLE_CAP_S)
                    self.heap = [(w - shift, *rest) for (w, *rest) in self.heap]
                    heapq.heapify(self.heap)
                while self.heap and self.heap[0][0] <= now:
                    _, _, traj, idx, primer = heapq.heappop(self.heap)
                    task = asyncio.create_task(self.send(http, traj, idx, primer))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                if self.progress_s and now >= next_progress:
                    self._progress(t_start, now)
                    next_progress += self.progress_s
                wait = min(0.05, max(0.0, (self.heap[0][0] - now) if self.heap else 0.05))
                await asyncio.sleep(wait if wait > 0 else 0.001)
            self.stop_dispatch = True
            t_window_end = time.monotonic()
            if self.tasks:
                log(f"window closed; draining {len(self.tasks)} in-flight request(s) for up to {DRAIN_S}s")
                done, pending = await asyncio.wait(self.tasks, timeout=DRAIN_S)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            return t_start, t_window_end


# ------------------------------------------------------------------ scoring

def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def text_bytes(text):
    """UTF-8 size of streamed text. `surrogatepass`: JSON can carry a lone surrogate (a
    detokenizer cut mid-character), and a plain encode() would raise on it."""
    return len(text.encode("utf-8", "surrogatepass"))


def output_tokens(rec):
    """Billed output tokens of a completed request (see the module docstring), or None if usage was not reported."""
    if rec.get("reported_output_tokens") is None:
        return None
    return max(0, min(rec["reported_output_tokens"], rec["max_tokens"], text_bytes(rec["text"]) + OUTPUT_SLACK))


def score_records(records, t_start, window_s):
    """Price what was served and apply the completion gate. Returns the metrics dict.

    The numerator is every request that succeeded, including those that finished during
    the drain; the denominator is the window, a constant. Nothing about the denominator is
    the server's to move, so completing a request can never lower the score."""
    counted = [r for r in records if not r["primer"]]
    for r in counted:
        if r["ok"]:
            r["output_tokens"] = output_tokens(r)
            if r["output_tokens"] is None:
                r["ok"], r["error"] = False, "no usage.completion_tokens in the stream"
            elif r["output_tokens"] < r["max_tokens"] - OUTPUT_SLACK:
                r["ok"], r["error"] = False, (
                    f"billed output {r['output_tokens']} of max_tokens {r['max_tokens']} (usage reported "
                    f"{r['reported_output_tokens']}, text received {text_bytes(r['text'])} bytes): the request must "
                    f"generate max_tokens tokens (ignore_eos) and return all their text (skip_special_tokens=false)")
    ok = [r for r in counted if r["ok"]]
    # Closed loop: the few requests still in flight when the drain window ends are
    # neither successes nor failures; everything that FINISHED counts toward the gate.
    failed = [r for r in counted if not r["ok"] and not r.get("aborted")]
    in_tok = cached = out_tok = 0
    ttfts, tps = [], []
    for r in ok:
        n_out = r["output_tokens"]
        c = min(max(0, r["cached_tokens"]), r["input_tokens"])
        in_tok += r["input_tokens"]
        cached += c
        out_tok += n_out
        if r["t_first"] is not None:
            ttfts.append(r["t_first"] - r["t_send"])
            if n_out >= 2 and r["t_last"] > r["t_first"]:
                tps.append((n_out - 1) / (r["t_last"] - r["t_first"]))
    hours = max(window_s, 1e-9) / 3600.0
    income = ((in_tok - cached) * PRICE["input"] + cached * PRICE["cached_input"] + out_tok * PRICE["output"]) / 1e6
    finished = len(ok) + len(failed)
    completion = len(ok) / max(1, finished)
    # One failure is always tolerated: below 100 finished requests a bare 0.99 would turn a
    # single transient error into a zero, and a long-context server may finish fewer.
    allowed_failures = max(1, int((1.0 - COMPLETION_MIN) * finished + 1e-9))
    primers_failed = sum(1 for r in records if r["primer"] and not r["ok"] and not r.get("aborted"))
    errors = []
    for r in failed:
        if r["error"] and r["error"] not in errors:
            errors.append(r["error"])
    return {
        "income_per_gpu_hour": round(income / (GPU_COUNT * hours), 4),
        "income_usd": round(income, 6),
        "gpu_hours": round(GPU_COUNT * hours, 5),
        "window_seconds": round(window_s, 1),
        "last_completion_seconds": round(max([r["t_done"] for r in ok], default=t_start) - t_start, 1),
        "requests_completed": len(ok),
        "requests_failed": len(failed),
        "requests_aborted_at_drain": sum(1 for r in counted if r.get("aborted")),
        "primers_failed": primers_failed,
        "completion_rate": round(completion, 4),
        "failures_allowed": allowed_failures,
        "completion_gate": bool(ok) and len(failed) <= allowed_failures,
        "input_tokens": in_tok,
        "cached_tokens_reported": cached,
        "cache_hit": round(cached / in_tok, 4) if in_tok else None,
        "output_tokens": out_tok,
        **{f"ttft_p{q}_s": round(percentile(ttfts, q), 3) if ttfts else None for q in (50, 90, 99)},
        **{f"stream_tps_p{q}": round(percentile(tps, q), 2) if tps else None for q in (10, 50, 90)},
        "sessions_started": len({r["position"] for r in records}),
        "errors": errors[:5],
    }


def _discover_model(base, timeout=30):
    """Server-agnostic: ask the server what it calls itself; fall back to the path."""
    import urllib.request
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return MODEL_PATH


_PREPARED = {}


def _prepare(tokenizer, seed, salt):
    """Tokenizing the code pool takes a while; do it once per (seed, salt) in a process."""
    key = (seed, salt, MODEL_PATH)
    if key not in _PREPARED:
        tok = tokenizer if tokenizer is not None else _load_tokenizer()
        content = Content(tok, seed, salt)
        _PREPARED[key] = (content, load_sessions(len(content.head) + len(content.tail)))
    return _PREPARED[key]


def run(base=None, seed=None, duration_s=None, tokenizer=None, salt="", progress_s=60, return_records=False):
    """Replay the traffic against ANY OpenAI-compatible server and return its metrics.

    `seed` and `duration_s` default to the constants above. `salt` changes only the
    per-trajectory marker, so a repeated pass on one seed does not hit the previous
    pass's prefix cache."""
    base = (base or os.environ.get("SIM_BASE", "http://127.0.0.1:8000")).rstrip("/")
    seed = SEED if seed is None else int(seed)
    duration_s = float(DURATION_S if duration_s is None else duration_s)
    content, sessions = _prepare(tokenizer, seed, salt)
    replayer = Replayer(base, _discover_model(base), content, sessions, duration_s, progress_s)
    t_start, _t_window_end = asyncio.run(replayer.run())
    metrics = score_records(replayer.records, t_start, duration_s)
    metrics["config"] = {"concurrency": CONCURRENCY, "duration_s": duration_s, "drain_s": DRAIN_S,
                         "gpus": GPU_COUNT, "price_usd_per_million_tokens": PRICE,
                         "graded_shape": duration_s == DURATION_S}
    if return_records:
        return metrics, [{k: v for k, v in r.items() if k != "text"} for r in replayer.records]
    return metrics


def score(metrics):
    """THE NUMBER. Whatever a simulator measures, this is the single scalar the grader
    takes from it. Here: USD earned per GPU-hour, or 0 if the completion gate fails."""
    return metrics["income_per_gpu_hour"] if metrics["completion_gate"] else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay the graded traffic against a server. The options exist for "
                                             "DEVELOPMENT; the grader uses the constants in this file and its own seed.")
    ap.add_argument("--duration", type=float, default=None, help=f"window in seconds (graded: {DURATION_S})")
    ap.add_argument("--seed", type=int, default=None, help=f"traffic seed (development default: {SEED})")
    args = ap.parse_args()
    base = os.environ.get("SIM_BASE", "http://127.0.0.1:8000")
    out_dir = Path(os.environ.get("SIM_OUT", "/tmp/simulator"))
    log(f"traffic: {CONCURRENCY} session lanes for {args.duration or DURATION_S:.0f}s (+{DRAIN_S}s drain) | "
        f"seed {SEED if args.seed is None else args.seed} | target {base}")
    m, records = run(base, seed=args.seed, duration_s=args.duration, return_records=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(m, indent=2))
    (out_dir / "details.json").write_text(json.dumps({"metrics": m, "records": records}, indent=2))
    print(json.dumps(m, indent=2))
    print(f"\nSCORE (USD earned per GPU-hour): {score(m)}")
    if not m["config"]["graded_shape"]:
        print(f"[sim] NOTE: a {m['config']['duration_s']:.0f}s window is a development run; the graded window is "
              f"{DURATION_S}s", file=sys.stderr)
    if not m["completion_gate"]:
        finished = m["requests_completed"] + m["requests_failed"]
        why = (f"no request completed successfully ({m['requests_failed']} failed, "
               f"{m['requests_aborted_at_drain']} cut off at the drain deadline)" if not m["requests_completed"]
               else f"{m['requests_failed']} of {finished} finished request(s) failed ({m['failures_allowed']} allowed)")
        print(f"[sim] COMPLETION GATE FAILED: {why}; first errors: {m['errors']}", file=sys.stderr)
