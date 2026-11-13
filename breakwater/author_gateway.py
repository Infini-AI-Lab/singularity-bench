"""Per-trial Messages or Responses proxy pinned to ONE author model.

The upstream credential lives in this sidecar. Unknown models, routes, redirects
and query parameters fail closed. JSON and SSE are relayed without translation.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import secrets
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException, IncompleteRead

MODEL = os.environ.get("AUTHOR_MODEL", "")
UPSTREAM = os.environ.get("AUTHOR_UPSTREAM_BASE_URL", "https://api.fireworks.ai/inference").rstrip("/")
KEY = os.environ.get("AUTHOR_UPSTREAM_API_KEY", "")
AUTH = os.environ.get("AUTHOR_UPSTREAM_AUTH", "bearer")
PROTOCOL = os.environ.get("AUTHOR_PROTOCOL", "messages")
LOG = os.environ.get("AUTHOR_GATEWAY_LOG", "/logs/author/requests.jsonl")
MAX_BODY = 32 * 1024 * 1024
_log_lock = threading.RLock()
_pending = set()
_closed = False
_faulted = False
_sequence = 0
_previous = "0" * 64

# This is a client-tool transport, not access to the provider's agent runtime.
# New provider capabilities require an explicit review of this contract.
MESSAGES_FIELDS = frozenset({
    'model', 'messages', 'max_tokens', 'system', 'metadata', 'stop_sequences',
    'stream', 'temperature', 'top_p', 'top_k', 'tools', 'tool_choice', 'thinking',
    'output_config', 'context_management', 'service_tier', 'cache_control',
})
MESSAGES_BETAS = frozenset({
    'claude-code-20250219', 'prompt-caching-2024-07-31',
    'token-counting-2024-11-01', 'output-128k-2025-02-19',
    'extended-cache-ttl-2025-04-11', 'interleaved-thinking-2025-05-14',
    'fine-grained-tool-streaming-2025-05-14', 'context-management-2025-06-27',
    'context-1m-2025-08-07', 'structured-outputs-2025-11-13',
    'effort-2025-11-24', 'prompt-caching-scope-2026-01-05',
    'structured-outputs-2025-12-15', 'redact-thinking-2026-02-12',
    'mid-conversation-system-2026-04-07', 'thinking-token-count-2026-05-13',
    'mid-conversation-tool-changes-2026-07-01',
})


def messages_policy(payload, headers):
    """Validate Messages capabilities; ordinary tool arguments remain opaque data."""
    unknown = set(payload) - MESSAGES_FIELDS
    if unknown:
        raise ValueError('Unsupported Messages request fields: ' + ', '.join(sorted(unknown)))
    versions = headers.get_all('anthropic-version', [])
    if versions and versions != ['2023-06-01']:
        raise ValueError('Only anthropic-version 2023-06-01 is supported')
    betas = [item.strip() for value in headers.get_all('anthropic-beta', []) for item in value.split(',')]
    if any(beta not in MESSAGES_BETAS for beta in betas):
        raise ValueError('Unsupported Messages beta capability: ' + ','.join(sorted(set(betas) - MESSAGES_BETAS)))
    tools = payload.get('tools', [])
    if not isinstance(tools, list):
        raise ValueError('Messages tools must be a list of client tool definitions')
    names = set()
    for tool in tools:
        allowed = {'type', 'name', 'description', 'input_schema', 'cache_control',
                   'strict', 'input_examples', 'defer_loading', 'eager_input_streaming', 'allowed_callers'}
        if (not isinstance(tool, dict) or set(tool) - allowed
                or tool.get('type', 'custom') != 'custom'
                or not isinstance(tool.get('name'), str)
                or not isinstance(tool.get('input_schema'), dict)
                or tool.get('allowed_callers', ['direct']) != ['direct']):
            raise ValueError('Only directly called client-side Messages tools are allowed')
        names.add(tool['name'])
    choice = payload.get('tool_choice', {'type': 'auto'})
    if (not isinstance(choice, dict) or set(choice) - {'type', 'name', 'disable_parallel_tool_use'}
            or choice.get('type') not in ('auto', 'any', 'none', 'tool')
            or ('name' in choice and (not isinstance(choice['name'], str) or choice['name'] not in names))
            or (choice.get('type') == 'tool' and choice.get('name') not in names)):
        raise ValueError('Unsupported Messages tool choice')
    # Provider-side URL/file references and server-tool continuation blocks are
    # outside this text/inline-content client transport, including in tool results.
    messages = payload.get('messages', [])
    if not isinstance(messages, list):
        raise ValueError('Messages must be a list')
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {'role', 'content'}
                or message.get('role') not in ('user', 'assistant', 'system')):
            raise ValueError('Unsupported Messages message')
        messages_content(message.get('content'), names)
    if 'system' in payload:
        messages_content(payload['system'], names)
    return ','.join(dict.fromkeys(betas))


def messages_content(content, tool_names=()):
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ValueError('Unsupported Messages content')
    for block in content:
        if not isinstance(block, dict):
            raise ValueError('Unsupported Messages content block')
        kind = block.get('type')
        fields = {
            'text': {'text', 'cache_control'},
            'thinking': {'thinking', 'signature'},
            'redacted_thinking': {'data'},
            'tool_use': {'id', 'name', 'input', 'cache_control', 'caller'},
            'tool_result': {'tool_use_id', 'content', 'is_error', 'cache_control'},
            'tool_addition': {'tool', 'cache_control'},
            'tool_removal': {'tool', 'cache_control'},
            'image': {'source', 'cache_control'},
            'document': {'source', 'title', 'context', 'citations', 'cache_control'},
        }
        if not isinstance(kind, str) or kind not in fields or set(block) - fields[kind] - {'type'}:
            raise ValueError('Only client Messages content blocks are allowed')
        if kind == 'tool_result' and 'content' in block:
            messages_content(block['content'], tool_names)
        if kind == 'tool_use' and block.get('caller', {'type': 'direct'}) != {'type': 'direct'}:
            raise ValueError('Only direct client tool calls are allowed')
        if kind in ('tool_addition', 'tool_removal'):
            # Native CLI context updates may select an already-validated tool.
            # Inline definitions and MCP references must not bypass tools[].
            tool = block.get('tool')
            if (not isinstance(tool, dict) or set(tool) != {'type', 'name'}
                    or tool['type'] != 'tool_reference' or not isinstance(tool['name'], str)
                    or tool['name'] not in tool_names):
                raise ValueError('Context updates may only reference declared client tools')
        if kind in ('image', 'document'):
            source = block.get('source')
            if (not isinstance(source, dict) or source.get('type') not in ('base64', 'text')
                    or set(source) - {'type', 'media_type', 'data'}):
                raise ValueError('Only inline Messages media is allowed')


def local_tools_only(tools):
    """Responses may describe client tools, never provider-hosted execution."""
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            return False
        kind = tool.get('type')
        if kind == 'namespace':
            if not local_tools_only(tool.get('tools')):
                return False
        elif kind not in ('function', 'custom'):
            return False
    return True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def audit(record, raw=None):
    """Private sidecar journal: persist complete bytes before forwarding them."""
    global _faulted
    with _log_lock:
        if _closed:
            raise OSError("Gateway journal is closed")
        try:
            _audit_locked(record, raw)
        except OSError:
            _faulted = True
            raise


def _audit_locked(record, raw):
    global _sequence, _previous
    root = Path(LOG).parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    if raw is not None:
        redactions = raw.count(KEY.encode()) if KEY else 0
        raw = raw.replace(KEY.encode(), b"[REDACTED_PROVIDER_CREDENTIAL]") if KEY else raw
        digest = hashlib.sha256(raw).hexdigest()
        blobs = root / "blobs"
        blobs.mkdir(mode=0o700, exist_ok=True)
        path = blobs / digest
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o600)
        record = {**record, "output": {"path": f"blobs/{digest}", "sha256": digest,
                  "bytes": len(raw), "credential_redactions": redactions}}
    event = {"schema_version": 1, "seq": _sequence,
             "utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
             "source": "author-gateway", "phase": "agent", "trust_level": "externally_observed",
             "previous_sha256": _previous, **record}
    digest = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    event["sha256"] = digest
    with open(LOG, "a") as output:
        output.write(json.dumps(event) + "\n")
        output.flush()
        os.fsync(output.fileno())
    _sequence += 1
    _previous = digest

    if record.get('event') == 'model_request':
        _pending.add(record['call_id'])
    elif record.get('event') == 'model_response_end':
        _pending.discard(record['call_id'])


def finalize_journal():
    """Freeze the journal before exit, explicitly cancelling unfinished calls.

    Bytes already journaled remain reviewable, even if the client disconnected.
    An unreturned upstream response is not claimed to have been captured. Once
    closed, no request thread can journal another model chunk. Already recorded
    bytes may be in transit until process exit; collection waits for that exit.
    """
    global _closed
    with _log_lock:
        if _closed:
            return
        for call_id in sorted(_pending):
            audit({'event': 'model_response_end', 'call_id': call_id, 'status': 503,
                   'outcome': 'gateway_stopped'})
        audit({'event': 'gateway_stopped', 'faulted': _faulted})
        _closed = True


def install_shutdown_handler():
    def shutdown(*_):
        try:
            finalize_journal()
        except BaseException:
            os._exit(1)
        # Request threads may still be blocked in urllib before response headers.
        # Kill them together after fsync; do not wait for provider timeouts.
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)


def relay_response(handler, response, record):
    """Journal before delivery, distinguishing transport outcomes from lost capture."""
    def capture(event, raw=None, **fields):
        try:
            audit({**record, 'event': event, **fields}, raw=raw)
            return True
        except OSError:
            # If even this record fails, the unfinished call still prevents the
            # host checker from accepting the evidence. Never forward its bytes.
            try:
                audit({**record, 'event': 'collector_error', 'complete': False})
            except OSError:
                pass
            return False

    def finish(outcome, **fields):
        capture('model_response_end', status=response.status, outcome=outcome, **fields)

    try:
        handler.send_response(response.status)
        handler.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
        handler.send_header('Connection', 'close')
        handler.end_headers()
    except OSError:
        finish('client_cancelled')
        return
    while True:
        try:
            chunk = response.read1(65536)
        except (OSError, HTTPException) as exc:
            # Chunked HTTP reads can carry bytes received before a truncated
            # frame. Retain those too, even though they were never delivered.
            if isinstance(exc, IncompleteRead) and exc.partial:
                if not capture('model_response_chunk', raw=exc.partial):
                    return
            finish('upstream_error', upstream_error=True, error_type=type(exc).__name__)
            return
        if not chunk:
            if getattr(response, 'length', None) not in (None, 0):
                finish('upstream_error', upstream_error=True, error_type='IncompleteRead')
            else:
                finish('completed')
            return
        if not capture('model_response_chunk', raw=chunk):
            return
        try:
            handler.wfile.write(chunk)
            handler.wfile.flush()
        except OSError:
            finish('client_cancelled')
            return


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def send(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def error(self, status, message):
        self.send(status, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})

    def do_GET(self):
        if self.path == "/health":
            return self.send(200 if MODEL and KEY else 503, {"ok": bool(MODEL and KEY), "model": MODEL, "protocol": PROTOCOL})
        if self.path == "/v1/models":
            return self.send(200, {"data": [{"id": MODEL, "type": "model", "display_name": MODEL}], "has_more": False})
        self.error(404, "Unknown route")

    def do_POST(self):
        self.close_connection = True
        target = urllib.parse.urlsplit(self.path)
        routes, queries = (("/v1/messages", "/v1/messages/count_tokens"), ("", "beta=true")) if PROTOCOL == 'messages' else (
            ("/v1/responses", "/v1/responses/compact"), ("",))
        if target.scheme or target.netloc or target.fragment or target.path not in routes or target.query not in queries:
            return self.error(404, "Unknown route or query")
        if not MODEL or not KEY:
            return self.error(503, "Author gateway is unconfigured")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                return self.error(413, "A bounded Content-Length is required")
            request_bytes = self.rfile.read(size)
            payload = json.loads(request_bytes, object_pairs_hook=_unique)
            if not isinstance(payload, dict) or payload.get("model") != MODEL:
                return self.error(400, "Only the assigned author model is available")
            if PROTOCOL == 'responses':
                allowed = {'model', 'input', 'instructions', 'tools', 'tool_choice', 'parallel_tool_calls',
                           'reasoning', 'text', 'temperature', 'top_p', 'max_output_tokens', 'max_tool_calls',
                           'store', 'stream', 'include', 'prompt_cache_key', 'prompt_cache_retention',
                           'service_tier', 'truncation', 'metadata', 'client_metadata', 'context_management', 'safety_identifier'}
                if set(payload) - allowed or not local_tools_only(payload.get('tools', [])):
                    return self.error(400, 'Only client-side Responses tools and supported request fields are allowed')
                if payload.get('store', False) is not False:
                    return self.error(400, 'Stored author responses are not enabled')
            else:
                try:
                    betas = messages_policy(payload, self.headers)
                except ValueError as error:
                    return self.error(400, str(error))
        except (ValueError, UnicodeError):
            return self.error(400, "Invalid JSON request")
        # Re-encode parsed data: no duplicate model keys or client-selected URLs.
        payload["model"] = MODEL
        headers = {"Content-Type": "application/json", "User-Agent": "breakwater-author-gateway/1.0"}
        if PROTOCOL == 'messages':
            headers['anthropic-version'] = '2023-06-01'
        headers["x-api-key" if AUTH == "anthropic" else "Authorization"] = KEY if AUTH == "anthropic" else f"Bearer {KEY}"
        if PROTOCOL == 'messages' and betas:
            headers['anthropic-beta'] = betas
        request = urllib.request.Request(UPSTREAM + self.path, data=json.dumps(payload).encode(), headers=headers)
        try:
            call_id = secrets.token_hex(16)
            audit({"event": "model_request", "call_id": call_id, "route": target.path,
                   "model": MODEL, "stream": bool(payload.get("stream"))}, raw=request_bytes)
        except OSError:
            return self.error(503, "Author upstream unavailable")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            response = opener.open(request, timeout=600)
        except urllib.error.HTTPError as error:
            # Never reflect a credential-bearing upstream error body.
            detail = error.read()
            error.close()
            try:
                audit({'event': 'model_upstream_error', 'call_id': call_id, 'status': error.code}, raw=detail)
                audit({'event': 'model_response_end', 'call_id': call_id, 'status': error.code, 'upstream_error': True})
            except OSError:
                return self.error(503, 'Author upstream unavailable')
            return self.error(error.code if 400 <= error.code < 600 else 502, "Author upstream rejected the request")
        except (OSError, urllib.error.URLError):
            try:
                audit({'event': 'model_response_end', 'call_id': call_id, 'status': 502, 'upstream_error': True})
            except OSError:
                pass
            return self.error(502, "Author upstream unavailable")
        with response:
            relay_response(self, response, {'call_id': call_id})


class Server(ThreadingHTTPServer):
    # Concurrent model/tool calls must reach admission checks instead of being
    # reset when the default five-connection listener backlog is exhausted.
    request_queue_size = 128
    daemon_threads = True


def main():
    if not MODEL or not KEY or AUTH not in ("bearer", "anthropic") or PROTOCOL not in ('messages', 'responses'):
        raise SystemExit("Author gateway requires a fixed model, upstream key and valid auth mode")
    if PROTOCOL == 'responses' and AUTH != 'bearer':
        raise SystemExit('Responses requires bearer authentication')
    parsed = urllib.parse.urlsplit(UPSTREAM)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        raise SystemExit("Invalid author upstream URL")
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    install_shutdown_handler()
    audit({"event": "gateway_started", "model": MODEL, "protocol": PROTOCOL})
    Server(("0.0.0.0", int(os.environ.get("PORT", "8001"))), Handler).serve_forever()


if __name__ == "__main__":
    main()
