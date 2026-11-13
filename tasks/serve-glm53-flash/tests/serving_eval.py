"""Evaluation core of the serve-glm53-flash task: the server lifecycle (GPUs and port
cleared -> launch -> ready -> measure -> clean shutdown), the income measurement and the
accuracy gate.

INCOME (the reward) -- simulator.py, the agent-visible file at /app/simulator.py:
  * TRAFFIC: recorded agentic coding sessions (the AgentX corpus) replayed on a FIXED
    number of session lanes (simulator.CONCURRENCY) for a FIXED window
    (simulator.DURATION_S + up to simulator.DRAIN_S of drain). The load is closed-loop:
    a faster server is sent more requests, and is paid for more tokens.
  * METRIC: USD earned per GPU-hour at the prices in simulator.PRICE: the income of every
    request that completed, divided by GPU_COUNT x the WINDOW (a constant -- nothing
    about the denominator is the server's to move). Input tokens are counted by the
    client; cached and output tokens are the server's usage report, with output capped
    at max_tokens and at the bytes of text actually received.
  * SEED: the grader replays the SAME file with the seed passed to run_grader(), which
    the agent never sees (neither tests/ nor the seed is in the image). The seed decides
    which sessions run, where they start and every prompt's content, so nothing seen in
    development is sent here and no prefix cache can be built ahead of time.
  * PROTOCOL: launch -> poll /v1/models ready -> TIMED_PASSES pass(es) of the traffic
    (median; a repeated pass gets its own marker salt, so it cannot hit an earlier
    pass's prefix cache) -> accuracy pass -> clean shutdown. There is NO separate
    warmup pass: every lane opens with an unbilled primer request, and a server is
    expected to finish its own warmup before it reports ready.
  * GATE (traffic): of the requests that finished, at most 1 - simulator.COMPLETION_MIN
    may have failed (one failure is always tolerated), with ignore_eos enforced (a
    request that falls short of max_tokens is a failure).

CORRECTNESS (fixed full held-out set, ONE pass, after the traffic):
  * Runs on the FIXED hidden_eval.json + canaries.json, loaded WHOLE --
    never sampled. Scored by score.py. PASS iff
        accuracy >= ACC_RATIO * REFERENCE_ACCURACY      <- the accuracy gate
    canary accuracy is computed and REPORTED, but does not fail a run.

REWARD = median income per GPU-hour, iff BOTH gates pass; else 0.0.

WHAT IS NOT CHECKED: the CONTENT of what the server streams during the traffic. The
traffic runs with ignore_eos on synthesized prompts, and no reference generations are
shipped for it, so a server that answered the traffic with garbage would still be paid.
It is caught only if the same defect also breaks the accuracy pass. Latency (TTFT,
per-stream decode speed) is REPORTED and not gated. The integrity review reads the
delivered /app for both.

RAW OUTPUTS (Breakwater): in the intermediate stage (BREAKWATER_STAGE=intermediate) each
traffic pass's complete per-request records, streamed text included, are written to
traffic-pass-<k>.json and the accuracy generations to accuracy.generations.json in
REWARD_DIR, so the verifier can recompute the reward without running anything.
"""
import os, sys, json, re, time, signal, subprocess, statistics, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_json  # noqa: E402
from score import score_candidate  # noqa: E402
import simulator  # noqa: E402  (THE scored traffic; agent-visible at /app/simulator.py)

# ---- launch commands (set by run_grader / the caller / env) ----
STOCK_CMD = os.environ.get("STOCK_CMD", "")          # set by run_grader()
AGENT_CMD = os.environ.get("AGENT_SERVE_CMD", "bash /app/serve.sh")
# TIME LIMIT of an ACCURACY-pass request (instruction.md tells the agent: one hour, and an
# unanswered request is a wrong answer). The pass fires every held-out request at once at
# max_tokens=16384, and generations are heavy-tailed (a few run to the full cap), so the
# last request to be served waits roughly the whole pass.
# urlopen's timeout is per socket read, not a total, so run_batch ALSO enforces this as a
# deadline on the whole pass: a server that drips bytes cannot hold the grader forever.
CORR_TIMEOUT = float(os.environ.get("CORR_TIMEOUT", "3600"))

# ---- standardized, engine-INDEPENDENT evaluation constants (identical everywhere) ----
HOST = os.environ.get("SERVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
MAX_TOKENS = 16384           # correctness-pass cap. GLM-5.3-Flash is a REASONING model that emits
                             # long <think> blocks; a small cap truncates mid-thought and undercounts
                             # accuracy.
# The model's tokenizer. The simulator builds its prompts with it; the accuracy pass only LOGS a
# re-tokenized count (income bills the server's own usage, capped -- see simulator.py).
TOK_PATH = os.environ.get("TOK_PATH", "/models/GLM-5.3-Flash")
# weight load + the server's OWN warmup (compile, CUDA-graph capture, autotune). There is no
# discarded warmup pass, so a server is told to warm itself up before reporting ready, and it
# is given the time to do it.
READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "1800"))
# Passes of the traffic; the reward is their median. ONE by default: a pass is
# DURATION_S + up to DRAIN_S (~20 min), and the traffic is a fixed window rather than a
# fixed amount of work, so its cost does not shrink for a fast server.
TIMED_PASSES = int(os.environ.get("TIMED_PASSES", "1"))
SHUTDOWN_TIMEOUT = float(os.environ.get("SHUTDOWN_TIMEOUT", "180"))  # wait for proc+port+GPU to be free
GPU_FREE_MIB = int(os.environ.get("GPU_FREE_MIB", "3000"))     # consider GPU free below this used-MiB
OUTDIR = os.environ.get("REWARD_DIR", "/logs/verifier")
# Also measure the unmodified starter server (default OFF; Breakwater grading sets it to
# False). The reward is the ABSOLUTE income of the agent's server, so runs form a
# leaderboard. MEASURE_STOCK=1 also launches STOCK_CMD (a command that starts the unmodified
# starter server) and REPORTS its income; it never enters the reward.
MEASURE_STOCK = os.environ.get("MEASURE_STOCK", "0") == "1"


class GpuNotClearError(RuntimeError):
    """Raised by clear_gpu when the GPU cannot be emptied before a measured launch.
    Grading on a partially-occupied GPU would unfairly starve/OOM the fresh server, so
    we fail the grade cleanly (reward 0 with error=gpu_not_clear) instead."""


def log(*a):
    print("[grader]", *a, flush=True)


# ---------------------------------------------------------------------------
# The model's tokenizer, loaded once. The simulator needs it to build prompts; the
# accuracy pass uses it only to LOG how many tokens came back (re-tokenized text). Income
# is NOT computed from it: simulator.py bills the server's own usage, capped -- see there.
# ---------------------------------------------------------------------------
_TOK = None
_TOK_METHOD = None


def _load_tokenizer():
    global _TOK, _TOK_METHOD
    if _TOK is not None or _TOK_METHOD is not None:
        return
    try:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOK_PATH, local_files_only=True)
        _TOK_METHOD = "glm_retokenize"
        log(f"output-token counter: re-tokenizing with GLM tokenizer @ {TOK_PATH}")
    except Exception as e:
        _TOK = None
        _TOK_METHOD = "whitespace_fallback"
        log(f"WARNING: could not load GLM tokenizer ({e}); FALLING BACK to whitespace token count")


def count_output_tokens(text):
    _load_tokenizer()
    if not text:
        return 0
    if _TOK is not None:
        return len(_TOK(text, add_special_tokens=False).input_ids)
    return len(text.split())


def _get(base, path, timeout=5):
    req = urllib.request.Request(base + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def wait_ready(proc, base, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode}) before becoming ready")
        for path in ("/v1/models", "/health"):
            try:
                st, _ = _get(base, path, timeout=5)
                if st == 200:
                    log(f"server ready on {path} after {time.time()-t0:.1f}s")
                    return True
            except Exception:
                pass
        time.sleep(2)
    return False


def discover_model(base):
    """Use the model id the server advertises (robust across vLLM/SGLang/custom)."""
    try:
        st, body = _get(base, "/v1/models", timeout=10)
        data = json.loads(body)
        mid = data["data"][0]["id"]
        log("served model id:", mid)
        return mid
    except Exception as e:
        log("could not read /v1/models, falling back to model path:", e)
        return os.environ.get("MODEL_NAME", "/models/GLM-5.3-Flash")


def complete_one(base, model, item, timeout=None):
    payload = json.dumps({
        "model": model,
        "prompt": item["prompt"],
        "temperature": 0.0,
        "max_tokens": int(item.get("max_tokens") or MAX_TOKENS),
        "stream": False,
    }).encode()
    req = urllib.request.Request(base + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=(timeout or CORR_TIMEOUT)) as r:
        raw_response = r.read()
        resp = json.loads(raw_response)
    text = resp["choices"][0]["text"]
    # Anything but a string would blow up answer extraction later, OUTSIDE any try, and
    # take the whole grade down with it. It is simply not an answer.
    return item["id"], item["benchmark"], text if isinstance(text, str) else ""


def run_batch(base, model, items, timeout=None):
    """Fire all requests concurrently; return (candidate_list, total_out_tokens,
    wall_seconds). A request that fails, or is not answered within the time limit, comes
    back as an empty generation (a wrong answer). The token total is re-tokenized text,
    for the log only."""
    limit = timeout or CORR_TIMEOUT
    results = {}
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=len(items))     # ALL at once: the time limit below is then every request's own
    futs = {ex.submit(complete_one, base, model, it, limit): it for it in items}
    try:
        for fut in as_completed(futs, timeout=limit + 60):
            it = futs[fut]
            try:
                _id, bench, text = fut.result()
            except Exception as e:
                log(f"request {it['id']} failed: {e}")
                _id, bench, text = it["id"], it["benchmark"], ""
            results[_id] = {"id": _id, "benchmark": bench, "generation": text}
    except FuturesTimeout:
        log(f"accuracy pass: {len(items) - len(results)} request(s) unanswered after {limit + 60:.0f}s; "
            f"scored as wrong")
    # Do not wait for stragglers: a thread stuck on a dripping socket must not hold the grade.
    ex.shutdown(wait=False, cancel_futures=True)
    for it in items:
        results.setdefault(it["id"], {"id": it["id"], "benchmark": it["benchmark"], "generation": ""})
    wall = time.time() - t0
    cand = [results[it["id"]] for it in items]
    total = 0
    for it in items:
        try:        # a log-only statistic: it must never take the grade down
            total += count_output_tokens(results[it["id"]]["generation"])
        except Exception as e:
            log(f"could not count the tokens of {it['id']}: {type(e).__name__}")
    return cand, total, wall


def _visible_gpu_selector():
    """nvidia-smi `-i <ids>` selector for THIS process's visible GPUs. On a shared host
    the grader runs with CUDA_VISIBLE_DEVICES=<its ids>, but nvidia-smi ignores that env
    and reports ALL GPUs -- so we must scope queries to the visible devices or we'd read
    another trial's memory (false 'GPU not clear') and, worse, try to kill another
    trial's process. ALL visible ids are passed: a multi-GPU server holds memory on every
    one of them, and checking only the first would call the machine clear while the
    other GPUs are still occupied. Returns ['-i', 'a,b,c,d'] or []."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
    return ["-i", ",".join(ids)] if ids else []   # nvidia-smi -i accepts indices or GPU UUIDs


def _gpu_used_mib():
    """MAX used MiB over the VISIBLE GPUs, or None if unavailable. Scoped with `-i` so on
    a shared host we never read another trial's GPU (see selector)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", *_visible_gpu_selector(),
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=15,
        ).decode().strip().splitlines()
        return max(int(x.strip()) for x in out if x.strip())
    except Exception:
        return None


def shutdown(proc, base, timeout):
    """Kill the server's whole process group, then wait until the process is gone,
    the port stops answering, AND the GPU memory is released — so the next launch
    gets a clean port 8000 and the full GPU."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < 60:
        if proc.poll() is not None:
            break
        time.sleep(1)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass

    # port free?
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _get(base, "/v1/models", timeout=2)
            time.sleep(1)  # still answering -> old server not fully down
        except Exception:
            break          # no longer answering -> port released
    # GPU memory released?
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = _gpu_used_mib()
        if used is None or used < GPU_FREE_MIB:
            log(f"GPU free (used={used} MiB), port released")
            return
        time.sleep(2)
    log(f"WARNING: GPU still shows {_gpu_used_mib()} MiB used after shutdown wait")


def _stale_server_targets():
    """(kind, id) of leftover server processes that may hold the GPU or the serve port
    at grade time. The agent's DEV-LOOP server (launched via /app/start_server.sh) is
    detached and can outlive the agent, so it is still on the GPU when grading starts —
    starving the server under test (OOM, or a KV pool sized against a shrunk budget). Returns process GROUPS (from the
    start_server.sh pidfile) and individual PIDs (port listeners + server-like commands).
    NEVER includes the grader itself."""
    me = os.getpid()
    targets = []  # ("pg", pgid) or ("p", pid)
    # 1) start_server.sh records the server's process GROUP in /tmp/server.pid
    try:
        with open("/tmp/server.pid") as f:
            targets.append(("pg", int(f.read().strip())))
    except Exception:
        pass
    # 2) anything LISTENING on the serve port
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"ss -tlnpH 2>/dev/null | grep -E ':{PORT}\\b' || true"],
            timeout=10).decode()
        for m in re.findall(r"pid=(\d+)", out):
            targets.append(("p", int(m)))
    except Exception:
        pass
    # 3) server-like processes by command (never the grader/this module)
    try:
        out = subprocess.check_output(["bash", "-lc", "ps -eo pid=,args= 2>/dev/null"],
                                      timeout=10).decode()
        for line in out.splitlines():
            sp = line.strip().split(None, 1)
            if len(sp) != 2 or not sp[0].isdigit():
                continue
            pid, args = int(sp[0]), sp[1]
            if pid == me:
                continue
            if re.search(r"server\.py|/serve\.sh|uvicorn|sglang\.launch_server|start_server\.sh", args) \
               and "grader.py" not in args and "serving_eval" not in args:
                targets.append(("p", pid))
    except Exception:
        pass
    # 4) DEFINITIVE — any process ACTUALLY holding GPU memory (by fact, not by name).
    #    Catches a leftover server the agent named something we don't pattern-match
    #    (from-scratch: the launch command can be anything) or an orphaned CUDA process
    #    that is no longer listening on the port. clear_gpu runs only when NO graded
    #    server should be alive (it launches the fresh one AFTER), and the grader itself
    #    is CPU-only (HTTP client + tokenizer, never on the GPU) — so every compute-app
    #    PID here is a stale model process that is safe to kill. Skip our own PID.
    # CRITICAL: scope with `-i` to the VISIBLE GPU -- nvidia-smi otherwise lists compute
    # PIDs on ALL GPUs, and on a multi-GPU host that would target OTHER trials' servers.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", *_visible_gpu_selector(),
             "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            timeout=15).decode()
        for line in out.splitlines():
            s = line.strip()
            if s.isdigit() and int(s) != me:
                targets.append(("p", int(s)))
    except Exception:
        pass
    return targets


def clear_gpu(timeout):
    """Free the GPU + serve port BEFORE launching a measured server. A leftover agent
    dev-server can still be holding the GPU at grade time and starve the server under
    test. Kill every stale server
    (by pidfile process-group, by port listener, and by command pattern — never the
    grader), then WAIT until the GPU memory is actually released."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for kind, pid in _stale_server_targets():
            try:
                os.killpg(pid, sig) if kind == "pg" else os.kill(pid, sig)
            except Exception:
                pass
        time.sleep(1.5)
    try:
        os.remove("/tmp/server.pid")
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = _gpu_used_mib()
        if used is None or used < GPU_FREE_MIB:
            log(f"clear_gpu: GPU free (used={used} MiB)")
            return
        for kind, pid in _stale_server_targets():  # SIGKILL stragglers each pass
            try:
                os.killpg(pid, signal.SIGKILL) if kind == "pg" else os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(2)
    # HARD GATE: the GPU MUST be empty before we launch a measured server. Proceeding
    # onto a partially-occupied GPU sizes the fresh server's KV pool against a shrunk
    # HBM budget -> it can exhaust/OOM under load and score 0 through no fault of the
    # agent's code. If we cannot clear it after `timeout`s of killing stragglers, FAIL
    # the grade cleanly (GpuNotClearError) rather than silently measure on a dirty GPU.
    still = _gpu_used_mib()
    msg = (f"GPU still {still} MiB used after {timeout}s clear attempt "
           f"(threshold {GPU_FREE_MIB} MiB); refusing to grade on a non-empty GPU")
    log(f"clear_gpu: ERROR {msg}")
    raise GpuNotClearError(msg)


class StageError(RuntimeError):
    """A grade that failed, tagged with WHERE: 'not_ready', 'traffic' or 'accuracy'."""

    def __init__(self, stage, cause):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}" if isinstance(cause, BaseException)
                         else f"{stage}: {cause}")
        self.stage = stage


def measure_income(serve_cmd, port, seed, corr_items=None):
    """Launch serve_cmd, run TIMED_PASSES pass(es) of the simulator's traffic on `seed`,
    and return (median_income_per_gpu_hour, per_pass_scores, per_pass_metrics,
    last_pass_records, corr_candidates, model). Raises StageError naming the stage.

    INCOME and CORRECTNESS are measured SEPARATELY, on the same server instance: the
    traffic first, then (if corr_items is given) ONE pass over the FULL held-out set for
    the accuracy gate. The server is shut down cleanly (port + GPUs freed) before
    returning."""
    base = f"http://{HOST}:{port}"
    clear_gpu(SHUTDOWN_TIMEOUT)   # evict any leftover agent dev-server holding the GPUs/port
    log(f"launching server: {serve_cmd!r}")
    proc = subprocess.Popen([sys.executable, "-I", "/opt/breakwater-runtime/run_worker.py",
                             "/bin/bash", "-c", serve_cmd], start_new_session=True)
    try:
        try:
            if not wait_ready(proc, base, READY_TIMEOUT):
                raise RuntimeError(f"no answer on /v1/models or /health within {READY_TIMEOUT:.0f}s")
        except Exception as e:
            raise StageError("not_ready", e) from e
        model = discover_model(base)

        # ---- INCOME: simulator.py defines the traffic AND the score ----
        # The simulator is server-agnostic (plain OpenAI /v1/completions over HTTP) and
        # carries its own graded configuration -- the grader passes only where to send
        # the traffic, and the seed the agent was never shown.
        log(f"simulator: {simulator.CONCURRENCY} session lanes for {simulator.DURATION_S}s "
            f"(+{simulator.DRAIN_S}s drain), {simulator.GPU_COUNT} GPU(s), prices {simulator.PRICE}")
        scores, passes, records = [], [], None
        try:
            for k in range(TIMED_PASSES):
                # The first pass is the file exactly as the agent runs it, on another seed.
                # A REPEATED pass salts the per-session marker, or it would be served from
                # the prefix cache the previous pass left behind.
                # Under Breakwater the pass's complete records (streamed text included)
                # are kept for the verifier's recomputation; see the module docstring.
                capture = None
                if os.environ.get("BREAKWATER_STAGE") == "intermediate":
                    os.makedirs(OUTDIR, exist_ok=True)
                    capture = os.path.join(OUTDIR, f"traffic-pass-{k}.json")
                m, records = simulator.run(base=base, seed=seed, tokenizer=_TOK, salt=(f"pass{k}" if k else ""),
                                           return_records=True, capture_path=capture)
                passes.append(m)
                scores.append(simulator.score(m))
                log(f"pass {k+1}/{TIMED_PASSES}: ${m['income_per_gpu_hour']}/GPU-h (${m['income_usd']} over a "
                    f"{m['window_seconds']}s window, last completion at {m['last_completion_seconds']}s) | "
                    f"completed={m['requests_completed']} failed={m['requests_failed']} "
                    f"(allowed {m['failures_allowed']}) aborted_at_drain={m['requests_aborted_at_drain']} "
                    f"completion_gate={m['completion_gate']} | cache_hit={m['cache_hit']} "
                    f"ttft_p90={m['ttft_p90_s']}s stream_tps_p10={m['stream_tps_p10']}")
                if m["errors"]:
                    log(f"  first errors: {m['errors']}")
        except Exception as e:
            raise StageError("traffic", e) from e
        med = statistics.median(scores) if scores else 0.0
        log(f"median income = ${med:.4f}/GPU-h (passes={scores})")
        corr_cand = None
        if corr_items is not None:
            try:
                corr_cand, ctot, cwall = run_batch(base, model, corr_items, timeout=CORR_TIMEOUT)
            except Exception as e:
                raise StageError("accuracy", e) from e
            log(f"separate correctness pass ({len(corr_items)} req): {ctot} tok / {cwall:.2f}s "
                f"(not part of the income)")
        return med, scores, passes, records, corr_cand, model
    finally:
        shutdown(proc, base, SHUTDOWN_TIMEOUT)


def write_reward(d):
    """Write the result to OUTDIR as three files: reward.json holds ONLY the numeric
    fields (int/float/bool), the Harbor reward format, which rejects strings, lists and
    dicts; diagnostics.json holds the FULL result (the verifier compares its `reward`
    with its own recomputation); reward.txt holds the scalar reward."""
    os.makedirs(OUTDIR, exist_ok=True)
    numeric = {k: v for k, v in d.items()
               if isinstance(v, (int, float, bool)) and v is not None}
    with open(os.path.join(OUTDIR, "reward.json"), "w") as f:
        json.dump(numeric, f, indent=2)
    with open(os.path.join(OUTDIR, "reward.txt"), "w") as f:
        f.write(f"{d.get('reward', 0.0)}\n")
    with open(os.path.join(OUTDIR, "diagnostics.json"), "w") as f:
        json.dump(d, f, indent=2)
    log("wrote", os.path.join(OUTDIR, "reward.json"), "(numeric-only,",
        f"{len(numeric)} keys) + diagnostics.json")
    log(json.dumps(d, indent=2))


def run_grader(stock_cmd, eval_seed):
    """Grade the server AGENT_CMD launches; called by breakwater/intermediate.py.
    `stock_cmd` launches the unmodified starter server, measured only when MEASURE_STOCK
    is set; `eval_seed` is the hidden traffic seed. The environment variables STOCK_CMD
    and SIM_SEED override them."""
    global STOCK_CMD
    STOCK_CMD = os.environ.get("STOCK_CMD", stock_cmd)
    seed = int(os.environ.get("SIM_SEED", eval_seed))
    _load_tokenizer()  # surface the token-count method early; the simulator reuses it

    eval_items = load_json(os.path.join(HERE, "hidden_eval.json"))
    canaries = load_json(os.path.join(HERE, "canaries.json"))
    # INCOME and ACCURACY are measured SEPARATELY and on DIFFERENT workloads:
    #   * income   -> simulator.py, replayed agentic sessions the agent CAN read and tune
    #     for (that is intended, not cheating), on a seed it cannot see.
    #   * accuracy -> the held-out math_500 prompts below, in their own pass. The agent is
    #     told the gate exists but not when or how it is measured.
    all_items = eval_items + canaries
    log(f"correctness workload: {len(eval_items)} eval + {len(canaries)} canary = "
        f"{len(all_items)} requests (FULL set -- not sampled). traffic: simulator.py, "
        f"{simulator.CONCURRENCY} lanes x {simulator.DURATION_S}s, passes={TIMED_PASSES}, "
        f"token_count={_TOK_METHOD}")

    # ---- 1. (optional) the unmodified starter server's income: REPORTED, never rewarded ----
    stock_income, stock_list = None, None
    if MEASURE_STOCK:
        try:
            stock_income, stock_list, _p, _r, _c, _ = measure_income(STOCK_CMD, PORT, seed)
        except Exception as e:
            log("starter server failed (reported only, grading continues):", e)

    # ---- 2. AGENT: income on the traffic + generations for the accuracy gate ----
    try:
        income, income_list, passes, records, cand, model = measure_income(
            AGENT_CMD, PORT, seed, corr_items=all_items)
    except GpuNotClearError as e:
        # GPU couldn't be emptied before launch -> grading here would be unfair. Fail
        # cleanly with a DISTINCT error so it's diagnosable (infra issue, not the agent).
        log("agent grade aborted:", e)
        write_reward({"reward": 0.0, "correctness_pass": False, "grader_error": 1,
                      "error": f"gpu_not_clear: {e}",
                      "gpu_free_mib_threshold": GPU_FREE_MIB,
                      "ready_timeout_s": READY_TIMEOUT})
        return
    except Exception as e:
        # StageError says where: not_ready (the server never answered), traffic, accuracy.
        log("agent grade failed:", e)
        write_reward({"reward": 0.0, "correctness_pass": False, "grader_error": 1,
                      "error": str(e) if isinstance(e, StageError) else f"grader: {type(e).__name__}: {e}",
                      "ready_timeout_s": READY_TIMEOUT})
        return

    if os.environ.get("BREAKWATER_STAGE") == "intermediate":
        # The accuracy generations, for the verifier's recomputation (module docstring).
        with open(os.path.join(OUTDIR, "accuracy.generations.json"), "x") as raw_stream:
            json.dump(cand, raw_stream, allow_nan=False)

    # ---- 3. gates: accuracy (held-out set) AND completion (inside simulator.score) ----
    try:
        res = score_candidate(cand, eval_items=eval_items, canaries=canaries)
    except Exception as e:      # whatever a server returned, the grade must still be written
        log("scoring the accuracy pass failed:", e)
        write_reward({"reward": 0.0, "correctness_pass": False, "grader_error": 1,
                      "error": f"accuracy_scoring: {type(e).__name__}: {e}"})
        return
    corr_pass = bool(res["PASS"])
    completion_pass = bool(passes) and all(m["completion_gate"] for m in passes)   # EVERY pass, not just the last
    passed = corr_pass and completion_pass

    # ---- 4. reward: ABSOLUTE income per GPU-hour, gated ----
    # `income` is already 0 for a pass that failed the completion gate (simulator.score).
    reward = round(income, 4) if passed else 0.0

    sim = passes[-1] if passes else {}      # diagnostics below describe the LAST pass
    write_reward({
        "reward": reward,                       # USD per GPU-hour; 0.0 if a gate fails
        "income_per_gpu_hour": sim.get("income_per_gpu_hour"),   # last pass, UNGATED
        "income_usd": sim.get("income_usd"),
        "window_seconds": sim.get("window_seconds"),
        "last_completion_seconds": sim.get("last_completion_seconds"),
        "gate_pass": passed,
        "correctness_pass": corr_pass,
        "completion_pass": completion_pass,
        "completion_rate": sim.get("completion_rate"),
        "requests_completed": sim.get("requests_completed"),
        "requests_failed": sim.get("requests_failed"),
        "failures_allowed": sim.get("failures_allowed"),
        "requests_aborted_at_drain": sim.get("requests_aborted_at_drain"),
        "primers_failed": sim.get("primers_failed"),
        "sessions_started": sim.get("sessions_started"),
        "input_tokens": sim.get("input_tokens"),
        "cached_tokens_reported": sim.get("cached_tokens_reported"),
        "cache_hit": sim.get("cache_hit"),
        "output_tokens": sim.get("output_tokens"),
        **{k: sim.get(k) for k in ("ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
                                   "stream_tps_p10", "stream_tps_p50", "stream_tps_p90")},
        "accuracy": res["accuracy"],
        "ref_accuracy": res["reference_accuracy"],
        "canary_accuracy": res["canary_accuracy"],
        "stock_income_per_gpu_hour": stock_income,
        "income_passes": income_list,
        "stock_income_passes": stock_list,
        "traffic_errors": sim.get("errors"),
        "traffic_config": sim.get("config"),
        "gates": {**res["gates"], "completion": completion_pass},
        "thresholds": {**res["thresholds"], "COMPLETION_MIN": simulator.COMPLETION_MIN},
        "per_benchmark": res["per_benchmark"],
        "n_requests": len(all_items),            # correctness workload size
        "token_count_method": _TOK_METHOD,
        "served_model": model,
    })
    # Per-request traffic records (no text): what was sent, when, and what was billed.
    try:
        with open(os.path.join(OUTDIR, "traffic_records.json"), "w") as f:
            json.dump(records, f)
    except Exception as e:
        log("could not write traffic_records.json:", e)

