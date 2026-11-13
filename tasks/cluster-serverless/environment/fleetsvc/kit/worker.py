"""Wrap one trusted vLLM engine with serving, receipts and sleep/wake.

The fixed request schema prevents callers from changing model work. Receipts
bind the model, received prompt IDs, wire cap, output IDs, finish reason and a
fresh request-attempt nonce. Logical routing IDs may be remapped.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import secrets
import sys

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_args  # noqa: E402
import prompts  # noqa: E402
import protocol  # noqa: E402
import receipts  # noqa: E402
from response_io import ResponseProtocolError, response_lines, small_body  # noqa: E402
from inference_audit import AuditUnavailable, InferenceAudit  # noqa: E402

WORKER_ID = os.environ.get("CLUSTER_WORKER_ID", "w000")
MODEL = os.environ.get("CLUSTER_WORKER_MODEL", "")
PORT = int(os.environ.get("CLUSTER_WORKER_PORT", "9200"))
VLLM_PORT = int(os.environ.get("CLUSTER_VLLM_PORT", "21000"))
BACKEND_KEY = os.environ.get("CLUSTER_BACKEND_KEY", "")
NODE = os.environ.get("CLUSTER_NODE", "node0")
KEY = os.environ.get("CLUSTER_RECEIPT_KEY", "insecure-dev-key").encode()
GRADER_TOKEN = os.environ.get("CLUSTER_GRADER_TOKEN", "")
EXTRA_ARGS = tuple(json.loads(os.environ.get("CLUSTER_ENGINE_ARGS", "[]")))
# The manager allows ten seconds before SIGKILL. Leave time for request
# cancellation, client-session cleanup and the final durable journal record.
SERVER_OPTIONS = {"handler_cancellation": True, "shutdown_timeout": 2.0}


def _err(status: int, message: str, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


class State:
    def __init__(self) -> None:
        self.state = "awake"          # awake | asleep
        self.active: dict[str, dict] = {}
        self.served = 0
        self.aborted = 0
        self.admin = asyncio.Lock()


async def _vllm(sess, path: str, method: str = "POST") -> tuple[int, dict]:
    try:
        async with sess.request(method, f"http://127.0.0.1:{VLLM_PORT}{path}",
                                timeout=aiohttp.ClientTimeout(total=600)) as r:
            body = (await small_body(r)).decode(errors="replace")
            return r.status, (json.loads(body) if body.strip().startswith("{") else {"raw": body[:200]})
    except (aiohttp.ClientError, ValueError, RecursionError) as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def _needs_sleep_support() -> web.Response | None:
    """409, never a silent no-op, when the engine has no sleep support.

    Sleep is opt-in: without `--enable-sleep-mode` the engine landed awake and holds its share of the
    card until it is thrown away. Answering "ok" to a sleep it cannot perform would let a plane schedule
    against memory it never got back.
    """
    if not engine_args.has(EXTRA_ARGS, "--enable-sleep-mode"):
        return _err(409, f"{MODEL} was deployed without --enable-sleep-mode, so it cannot sleep or "
                         f"wake. Redeploy it with that flag if you want to evict it later.",
                    sleep_supported=False)
    return None


async def handle_sleep(request: web.Request) -> web.Response:
    s: State = request.app["state"]
    if (deny := _needs_sleep_support()) is not None:
        return deny
    async with s.admin:
        if s.state == "asleep":
            return web.json_response({"status": "already asleep", "aborted_requests": 0})
        # In-flight work dies with the weights. Mark it first so the streams stop writing and mint
        # nothing: an aborted generation produced no complete answer and must be visible as such.
        for rec in s.active.values():
            rec["aborted"] = True
        n = len(s.active)
        st, body = await _vllm(request.app["session"], "/sleep?level=1")
        if st != 200:
            return _err(502, f"{MODEL} would not sleep ({st})", detail=body)
        s.state = "asleep"
    return web.json_response({"status": "asleep", "aborted_requests": n})


async def handle_wake(request: web.Request) -> web.Response:
    s: State = request.app["state"]
    if (deny := _needs_sleep_support()) is not None:
        return deny
    async with s.admin:
        if s.state == "awake":
            return web.json_response({"status": "already awake"})
        st, body = await _vllm(request.app["session"], "/wake_up")
        if st != 200:
            return _err(502, f"{MODEL} would not wake ({st})", detail=body)
        s.state = "awake"
    return web.json_response({"status": "awake"})


async def handle_completions(request: web.Request) -> web.StreamResponse:
    s: State = request.app["state"]
    body = await request.json()
    if problem := protocol.validate(body):
        return _err(400, problem)
    if body.get("model") not in (None, "", MODEL):
        return _err(404, f"{WORKER_ID} serves {MODEL}, not {body.get('model')!r}")
    out_tok = int(body.get("max_tokens") or 0)
    if out_tok <= 0:
        return _err(400, "max_tokens is required")
    if s.state == "asleep":
        return _err(409, f"{MODEL} is asleep on {WORKER_ID}", state="asleep")

    # TOKEN IDS. `prompt` is a list of ints on /v1/completions; exactly `input_tokens` of them, so the
    # engine processes what ServeGen recorded rather than a tokenizer's rendering of synthetic words.
    prompt_ids = [int(x) for x in (body.get("prompt") or [])]
    # Internal ownership must not depend on an ID the caller can reuse.
    rid = object()
    # `include_usage` makes vLLM emit a final chunk carrying completion_tokens -- the only honest source
    # for the count, since the control plane has no tokenizer and a word count is not a token count.
    # HOW LONG THIS ANSWER SHOULD BE, and why the caller is not told.
    #
    # Both fleets must agree on how much work a request is -- the emulator has no model and cannot
    # discover an answer's length by generating it. So the platform fixes the length. But a control
    # plane that could SEE that length would know each request's decode time before routing it, which
    # real serving never tells you, so the figure travels inside the prompt: encrypted so the plane
    # cannot read it, authenticated so the plane cannot forge one (see _kit/prompts.py).
    #
    # NO MARKER -> SAMPLE A LENGTH. An unmarked prompt is somebody sending their own tokens: the agent
    # probing the fleet, or a plane inventing requests instead of forwarding ours.
    #
    # The marker's existence is public (prompts.py ships), and natural stopping on an unmarked prompt is
    # semantic noise: the engine often emits EOS immediately, so a probe would measure almost nothing and
    # read the fleet as far faster than it is. Sampling makes a probe behave like real serving, leaks
    # nothing (the bound is the caller's own max_tokens and the draw is ours), and makes marked and
    # unmarked traffic look alike.
    #
    # Uniform on purpose: drawing from the trace's own length distribution would leak that distribution.
    intended = prompts.read_control(KEY, prompt_ids)
    upstream = {"model": MODEL, "prompt": prompt_ids, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "return_token_ids": True, "detokenize": False}
    if intended is not None:
        upstream["max_tokens"] = max(1, min(intended, out_tok))
    else:
        upstream["max_tokens"] = random.randint(1, max(1, out_tok))
    upstream["ignore_eos"] = True
    audit: InferenceAudit = request.app["audit"]
    audit_id = secrets.token_hex(16)  # never trust a caller's routing ID for correlation
    try:
        audit.record("request", audit_id=audit_id, request=body,
                     engine_max_tokens=upstream["max_tokens"], peer=request.remote)
    except AuditUnavailable:
        return _err(503, "inference audit unavailable")
    rec = {"aborted": False}
    s.active[rid] = rec
    out_ids: list[int] = []
    finish_reason = ""
    outcome = "interrupted"
    up = resp = None
    try:
        sess = request.app["session"]
        try:
            up = await sess.post(f"http://127.0.0.1:{VLLM_PORT}/v1/completions", json=upstream)
        except aiohttp.ClientError as e:
            # The engine died under us. The worker stays up and says so; the manager's watchdog is what
            # notices the process is gone and keeps the post-mortem.
            outcome = "upstream_connection_error"
            return _err(502, f"upstream {type(e).__name__}: {e}", worker=WORKER_ID)
        audit.record("upstream_status", audit_id=audit_id, status=up.status)
        if up.status != 200:
            detail = (await small_body(up)).decode(errors="replace")
            audit.record("upstream_error", audit_id=audit_id, detail=detail)
            outcome = "upstream_error"
            return _err(up.status, detail[:200])

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
        await resp.prepare(request)
        stream_done = False
        try:
            async for raw in response_lines(up.content, len(prompt_ids), out_tok):
                if rec["aborted"]:
                    break
                audit.record("response_bytes", audit_id=audit_id, bytes_base64=base64.b64encode(raw).decode())
                line = raw.decode(errors="replace").strip()
                if line == "data: [DONE]":
                    stream_done = True
                    break
                if line.startswith("data: "):
                    obj = json.loads(line[6:])
                    if not isinstance(obj, dict) or not isinstance(obj.get("choices", []), list):
                        raise ResponseProtocolError("invalid upstream choices")
                    for ch in obj.get("choices", []):
                        if not isinstance(ch, dict):
                            raise ResponseProtocolError("invalid upstream choice")
                        reason = ch.get("finish_reason")
                        if reason is not None:
                            if not isinstance(reason, str) or len(reason) > 64:
                                raise ResponseProtocolError("invalid upstream finish reason")
                            finish_reason = reason
                        ids = ch.get("token_ids")
                        if ids is None:
                            ids = []
                        if not isinstance(ids, list) or not all(type(x) is int and 0 <= x < 2**31 for x in ids):
                            raise ResponseProtocolError("invalid upstream token IDs")
                        if len(out_ids) + len(ids) > out_tok:
                            raise ResponseProtocolError("upstream exceeds public output-token cap")
                        out_ids.extend(ids)
                    audit.record("response", audit_id=audit_id, frame=obj)
                await resp.write(raw)   # the receipt and [DONE] are appended only after completion
        # aiohttp's ClientConnectionResetError is NOT the builtin ConnectionResetError: it descends from
        # ClientError. Listing only the builtin would let a client hanging up mid-stream escape as an
        # unhandled task exception, leaving the abort uncounted.
        except (asyncio.CancelledError, ConnectionResetError, aiohttp.ClientConnectionResetError):
            outcome = "cancelled_or_disconnected"
            s.aborted += 1
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RecursionError):
            outcome = "invalid_or_truncated_response"
            s.aborted += 1
            return resp  # no receipt or [DONE] for malformed or truncated engine output

        if (rec["aborted"] or not stream_done or finish_reason != "length"
                or len(out_ids) != upstream["max_tokens"]):
            outcome = "incomplete"
            s.aborted += 1
            return resp
        receipt = receipts.sign(KEY, {
            "model": MODEL,
            "prompt_sha256": receipts.sha256_hex(prompts.canon(prompt_ids)),
            "max_tokens": out_tok,
            "text_sha256": receipts.sha256_hex(prompts.canon(out_ids)),
            "finish_reason": finish_reason or "",
            "request_nonce": body.get("routing", {}).get("request_nonce", ""),
        })
        await resp.write(b"data: " + json.dumps({"cluster_receipt": receipt}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        s.served += 1
        outcome = "completed"
        return resp
    except AuditUnavailable:
        outcome = "audit_failed"
        s.aborted += 1
        # After headers, terminate the stream without a completion receipt.
        return resp if resp is not None else _err(503, "inference audit unavailable")
    finally:
        if up is not None:
            up.release()
        s.active.pop(rid, None)
        try:
            audit.record("request_finished", audit_id=audit_id, outcome=outcome,
                         output_tokens=len(out_ids), finish_reason=finish_reason)
        except AuditUnavailable:
            # The journal now has an unfinished request and this worker rejects
            # further inference. Do not mask an existing cancellation/exception.
            print("inference audit failed; worker will reject further inference", file=sys.stderr, flush=True)


async def handle_stats(request: web.Request) -> web.Response:
    s: State = request.app["state"]
    return web.json_response({"worker": WORKER_ID, "node": NODE, "model": MODEL, "state": s.state,
                              "in_flight": len(s.active), "served": s.served, "aborted": s.aborted,
                              "sleep_supported": engine_args.has(EXTRA_ARGS, "--enable-sleep-mode")})


async def handle_rekey(request: web.Request) -> web.Response:
    """Retire the key the agent may have observed in development, before anything is graded."""
    global KEY
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    KEY = ((await request.json()).get("key") or "").encode() or KEY
    return web.json_response({"status": "rekeyed", "worker": WORKER_ID})


def build(*, audit_directory=None) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["state"] = State()

    async def _audit(a):
        a["audit"] = InferenceAudit(
            audit_directory if audit_directory is not None else os.environ.get("CLUSTER_AUDIT_DIR", ""),
            {"run_id": os.environ.get("CLUSTER_RUN_ID", ""), "node": NODE,
             "worker": WORKER_ID, "model": MODEL,
             "model_info": json.loads(os.environ.get("CLUSTER_AUDIT_MODEL", "{}"))})
        try:
            yield
        finally:
            a["audit"].close()
    app.cleanup_ctx.append(_audit)
    app.add_routes([
        web.post("/v1/completions", handle_completions),
        web.get("/v1/models", lambda r: web.json_response(
            {"object": "list", "data": [{"id": MODEL, "object": "model"}]})),
        web.post("/admin/sleep", handle_sleep),
        web.post("/admin/wake", handle_wake),
        web.post("/admin/rekey", handle_rekey),
        web.get("/stats", handle_stats),
        web.get("/health", lambda r: web.json_response({"status": "ok", "worker": WORKER_ID})),
    ])

    async def _session(a):
        a["session"] = aiohttp.ClientSession(
            auto_decompress=False, headers={"Accept-Encoding": "identity",
                **({"Authorization": f"Bearer {BACKEND_KEY}"} if BACKEND_KEY else {})})
        yield
        await a["session"].close()
    app.cleanup_ctx.append(_session)
    if os.environ.get('CLUSTER_CAPABILITIES_FILE'):
        from telemetry import Telemetry
        telemetry = Telemetry(os.environ['CLUSTER_KV_ENDPOINT'], os.environ['CLUSTER_FPM_ENDPOINT'],
                              os.environ['CLUSTER_CAPABILITIES_FILE'])
        app['telemetry'] = telemetry
        app.cleanup_ctx.append(telemetry.context)
        app.router.add_get('/telemetry', telemetry.handle)
    return app


if __name__ == "__main__":
    web.run_app(build(), host="0.0.0.0", port=PORT, access_log=None, print=None,
                **SERVER_OPTIONS)
