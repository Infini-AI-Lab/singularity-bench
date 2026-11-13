#!/usr/bin/env python3
"""Hash-chained OpenAI-compatible model gateway for harness development and grid cells.

ROLE=development (harness authoring): the development-model contract published in the
coding-agent-harness instruction and spec -- aliases ``chat``/``chat-b`` and ``embed``,
text-only non-streaming chat with one explicit output bound, persistent atomic token
reservations, refunds against reported usage, no silent truncation.

ROLE=driver (one downstream grid cell): the driver model behind two lanes. The
ModelClient lane (``/driver/v1/...``) is used by the trusted cell driver; the utility
lane (``/v1/...``) is the harness's own served endpoint. Both lanes resolve every chat
alias to the cell's driver model and ``embed`` to the fixed embedding model. Cells have
no token budget (wall clock is their only budget).

The upstream key exists only in this sidecar. Every request body is journaled before
it is forwarded and every upstream response is journaled before it is delivered, in a
sequence/hash chain with content-addressed blobs. Unknown routes, models, redirects
and streaming fail closed. Rejections never reach the provider.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROLE = os.environ.get('GATEWAY_ROLE', 'development')
UPSTREAM = os.environ.get('UPSTREAM_BASE_URL', 'https://api.fireworks.ai/inference/v1').rstrip('/')
KEY = os.environ.get('UPSTREAM_API_KEY', '')
LOG = os.environ.get('GATEWAY_LOG', '/logs/gateway/requests.jsonl')
STATE = os.environ.get('GATEWAY_STATE', '/logs/gateway/budget.json')
CHAT_MODELS = json.loads(os.environ.get('CHAT_MODELS', '{}'))
DRIVER_MODEL = os.environ.get('DRIVER_MODEL', '')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'accounts/fireworks/models/qwen3-embedding-8b')
MAX_CHAT_TOKENS = int(os.environ.get('MAX_CHAT_TOKENS', '0'))
MAX_EMBED_TOKENS = int(os.environ.get('MAX_EMBED_TOKENS', '0'))
UPSTREAM_TIMEOUT = float(os.environ.get('UPSTREAM_TIMEOUT', '600'))
PORT = int(os.environ.get('PORT', '8000'))
SOURCE = os.environ.get('GATEWAY_SOURCE', 'development-gateway' if ROLE == 'development' else 'driver-gateway')
MAX_BODY = 32 * 1024 * 1024

CHAT_FIELDS = {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'top_k',
               'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'stop', 'seed',
               'tools', 'tool_choice', 'parallel_tool_calls', 'response_format', 'stream', 'n',
               'logprobs', 'top_logprobs', 'reasoning_effort'}
MESSAGE_FIELDS = {'role', 'content', 'name', 'tool_calls', 'tool_call_id', 'function_call', 'refusal',
                  'reasoning_content'}

_lock = threading.RLock()
_pending = {}
_closed = False
_sequence = 0
_previous = '0' * 64
_state = {'chat_tokens': 0, 'embed_tokens': 0, 'chat_reserved': 0, 'embed_reserved': 0, 'chat_calls': 0,
          'embed_calls': 0, 'denied': 0, 'rejected': 0, 'upstream_errors': 0, 'faulted': False}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def configuration():
    return {'role': ROLE, 'chat_models': CHAT_MODELS if ROLE == 'development' else {'driver': DRIVER_MODEL},
            'embed_model': EMBED_MODEL, 'chat_limit': MAX_CHAT_TOKENS, 'embed_limit': MAX_EMBED_TOKENS,
            'upstream': UPSTREAM}


def audit(record, raw=None):
    """Persist one chained event (and its complete bytes) BEFORE acting on it."""
    global _sequence, _previous
    with _lock:
        if _closed:
            raise OSError('Gateway journal is closed')
        root = Path(LOG).parent
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if raw is not None:
            redactions = raw.count(KEY.encode()) if KEY else 0
            if KEY:
                raw = raw.replace(KEY.encode(), b'[REDACTED_PROVIDER_CREDENTIAL]')
            value = hashlib.sha256(raw).hexdigest()
            blobs = root / 'blobs'
            blobs.mkdir(mode=0o700, exist_ok=True)
            path = blobs / value
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.chmod(0o600)
            record = {**record, 'output': {'path': f'blobs/{value}', 'sha256': value, 'bytes': len(raw),
                                           'credential_redactions': redactions}}
        event = {'schema_version': 1, 'seq': _sequence, 'utc': datetime.now(timezone.utc).isoformat(),
                 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), 'source': SOURCE,
                 'trust_level': 'externally_observed', 'previous_sha256': _previous, **record}
        value = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        event['sha256'] = value
        with open(LOG, 'a') as output:
            output.write(json.dumps(event) + '\n')
            output.flush()
            os.fsync(output.fileno())
        _sequence += 1
        _previous = value


def persist():
    path = Path(STATE)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        json.dump({'state': _state, 'configuration': configuration()}, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def reservation(payload, kind):
    """The development contract's conservative input bound plus the declared output bound."""
    body_bytes = len(json.dumps(payload, ensure_ascii=True).encode())
    if kind == 'embed':
        if set(payload) - {'model', 'input', 'encoding_format', 'dimensions', 'user'}:
            raise ValueError('Unsupported embedding fields')
        value = payload.get('input')
        if isinstance(value, str):
            count = 1
        elif isinstance(value, list) and value:
            if all(isinstance(x, str) for x in value):
                count = len(value)
            elif all(type(x) is int and x >= 0 for x in value):
                count = 1
            elif all(isinstance(x, list) and x and all(type(t) is int and t >= 0 for t in x) for x in value):
                count = len(value)
            else:
                raise ValueError('Embedding input must be text or token IDs')
        else:
            raise ValueError('Embedding input is required')
        return 4 * body_bytes + 2048 * count
    if set(payload) - CHAT_FIELDS:
        raise ValueError('Unsupported chat fields: ' + ', '.join(sorted(set(payload) - CHAT_FIELDS)))
    if payload.get('stream', False) is not False or type(payload.get('n', 1)) is not int or payload.get('n', 1) != 1:
        raise ValueError('Development chat supports non-streaming requests with n=1')
    limits = [payload[key] for key in ('max_tokens', 'max_completion_tokens') if key in payload]
    if len(limits) != 1 or type(limits[0]) is not int or limits[0] <= 0:
        raise ValueError('Specify one positive integer max_tokens or max_completion_tokens')
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    for message in messages:
        if not isinstance(message, dict) or set(message) - MESSAGE_FIELDS:
            raise ValueError('Unsupported message fields')
        content = message.get('content')
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or not all(isinstance(block, dict) and set(block) <= {'type', 'text'}
                                                    and block.get('type') == 'text' and isinstance(block.get('text'), str)
                                                    for block in content):
            raise ValueError('Development chat accepts text content only')
    tools = payload.get('tools', [])
    if not isinstance(tools, list) or not all(isinstance(t, dict) and t.get('type') == 'function' for t in tools):
        raise ValueError('Only client-executed function tools are supported')
    return 4 * body_bytes + 2048 * (1 + len(messages) + len(tools)) + limits[0]


def usage_tokens(reply):
    usage = reply.get('usage') if isinstance(reply, dict) else None
    if not isinstance(usage, dict):
        return None
    total = usage.get('total_tokens')
    if type(total) is int and total > 0:
        return total
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens', 0)
    if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0 and prompt + completion > 0:
        return prompt + completion
    return None


def route(path):
    """-> (lane, kind, upstream route) for the fixed route table, else None."""
    target = urllib.parse.urlsplit(path)
    if target.scheme or target.netloc or target.query or target.fragment:
        return None
    table = {'/v1/chat/completions': ('utility', 'chat'), '/chat/completions': ('utility', 'chat'),
             '/v1/embeddings': ('utility', 'embed'), '/embeddings': ('utility', 'embed')}
    if ROLE == 'driver':
        table['/driver/v1/chat/completions'] = ('driver', 'chat')
    value = table.get(target.path)
    if value is None:
        return None
    return value[0], value[1], '/chat/completions' if value[1] == 'chat' else '/embeddings'


def resolve(alias, kind):
    if ROLE == 'development':
        if kind == 'chat':
            return CHAT_MODELS.get(alias) if isinstance(alias, str) else None
        return EMBED_MODEL if alias in ('embed', EMBED_MODEL) else None
    if kind == 'chat':
        # Every chat alias a harness learned during development resolves to the driver,
        # so the utility model and the ModelClient are one configuration per cell.
        return DRIVER_MODEL if alias in (None, 'chat', 'chat-b', 'driver', DRIVER_MODEL) else None
    return EMBED_MODEL if alias in (None, 'embed', EMBED_MODEL) else None


def driver_policy(payload):
    """Cells forward the harness's generation options; only transport shape is enforced."""
    if payload.get('stream', False) is not False:
        raise ValueError('Streaming is not supported by the evaluation endpoint')
    if 'n' in payload and payload['n'] != 1:
        raise ValueError('Exactly one completion per request')
    return 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def send(self, status, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        if status == 429:
            self.send_header('Retry-After', '1')
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def reject(self, status, kind, message, **fields):
        try:
            with _lock:
                _state['rejected'] += 1
            audit({'event': 'request_rejected', 'status': status, 'error_type': kind, 'reason': message[:500], **fields})
        except OSError:
            pass
        return self.send(status, {'error': {'type': kind, 'message': message}})

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', '/v1/health', '/driver/v1/health'):
            with _lock:
                body = {'ok': bool(KEY) and not _state['faulted'] and not _closed, 'role': ROLE,
                        'chat_models': sorted(CHAT_MODELS) if ROLE == 'development' else ['chat', 'chat-b', 'driver'],
                        'embed_model': 'embed', 'chat_tokens_reserved': _state['chat_reserved'],
                        'embed_tokens_reserved': _state['embed_reserved']}
                if MAX_CHAT_TOKENS:
                    body['chat_tokens_left'] = max(0, MAX_CHAT_TOKENS - _state['chat_tokens'])
                if MAX_EMBED_TOKENS:
                    body['embed_tokens_left'] = max(0, MAX_EMBED_TOKENS - _state['embed_tokens'])
                return self.send(503 if _state['faulted'] else 200, body)
        if self.path.rstrip('/') in ('/v1/models', '/driver/v1/models'):
            names = sorted(CHAT_MODELS) if ROLE == 'development' else ['chat', 'chat-b', 'driver']
            return self.send(200, {'object': 'list', 'data': [{'id': n, 'object': 'model'} for n in names]
                                   + [{'id': 'embed', 'object': 'model'}]})
        return self.reject(404, 'unknown_route', 'Unknown route', method='GET', path=self.path[:200])

    def do_POST(self):
        self.close_connection = True
        selected = route(self.path)
        if selected is None:
            return self.reject(404, 'unknown_route', 'Unknown route', path=self.path[:200])
        lane, kind, upstream_route = selected
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY or self.headers.get('Transfer-Encoding'):
                return self.reject(413, 'invalid_body', 'A bounded Content-Length is required', lane=lane)
            raw = self.rfile.read(size)
            payload = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object')
        except (ValueError, UnicodeError) as error:
            return self.reject(400, 'invalid_body', str(error), lane=lane)
        alias = payload.get('model')
        model = resolve(alias, kind)
        if model is None:
            return self.reject(400, 'unknown_model', 'Model is not available on this endpoint', lane=lane,
                               alias=str(alias)[:120])
        wire = dict(payload, model=model)
        try:
            reserved = reservation(wire, kind) if ROLE == 'development' else driver_policy(wire)
        except ValueError as error:
            return self.reject(400, 'invalid_request', str(error), lane=lane)
        cap = (MAX_CHAT_TOKENS if kind == 'chat' else MAX_EMBED_TOKENS) if ROLE == 'development' else 0
        call_id = secrets.token_hex(16)
        with _lock:
            if not KEY or _state['faulted'] or _closed:
                return self.send(503, {'error': {'type': 'gateway_unavailable'}})
            if cap and _state[kind + '_tokens'] + reserved > cap:
                _state['denied'] += 1
                try:
                    audit({'event': 'request_rejected', 'status': 429, 'error_type': 'budget_unavailable',
                           'reason': 'reservation exceeds the available allowance', 'lane': lane,
                           'reserved': reserved, 'kind': kind})
                except OSError:
                    pass
                return self.send(429, {'error': {'type': 'budget_unavailable', 'message':
                    'Request reservation exceeds the available allowance; retry after in-flight requests finish, '
                    'or use a smaller input/output limit'}})
            try:
                audit({'event': 'model_request', 'call_id': call_id, 'lane': lane, 'kind': kind,
                       'alias': alias if isinstance(alias, str) else None, 'model': model,
                       'reserved': reserved, 'route': upstream_route}, raw=raw)
                _pending[call_id] = (kind, reserved)
                _state[kind + '_tokens'] += reserved
                _state[kind + '_reserved'] += reserved
                _state[kind + '_calls'] += 1
                persist()  # The debit is durable before inference starts.
            except OSError:
                _state['faulted'] = True
                return self.send(503, {'error': {'type': 'accounting_unavailable'}})
        request = urllib.request.Request(UPSTREAM + upstream_route, data=json.dumps(wire).encode(), headers={
            'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}',
            'User-Agent': 'singularity-bench-gateway/2.1'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        used, status, body, outcome = reserved, 502, b'{"error": {"type": "upstream_unavailable"}}', 'upstream_unavailable'
        try:
            with opener.open(request, timeout=UPSTREAM_TIMEOUT) as response:
                body = response.read()
            status, outcome = 200, 'completed'
            measured = usage_tokens(json.loads(body))
            if measured is not None:
                used = measured
            elif ROLE == 'driver':
                used = 0
        except urllib.error.HTTPError as error:
            status = error.code if 400 <= error.code < 600 else 502
            body = json.dumps({'error': {'type': 'upstream_error', 'status': status}}).encode()
            error.close()
            outcome = 'upstream_rejected'
            if status in (400, 401, 403, 404, 413, 422, 429):
                used = 0  # Rejected before inference.
        except (OSError, urllib.error.URLError, ValueError):
            pass  # Ambiguous: keep the development reservation charged.
        with _lock:
            if _closed:
                return
            if ROLE == 'development' and used > reserved:
                _state['faulted'] = True
                status, body, outcome = 502, b'{"error": {"type": "accounting_bound_exceeded"}}', 'accounting_fault'
            if ROLE == 'development':
                _state[kind + '_tokens'] += used - reserved
            else:
                _state[kind + '_tokens'] += used
            _state[kind + '_reserved'] -= reserved
            if status != 200:
                _state['upstream_errors'] += 1
            try:
                persist()
                audit({'event': 'model_response_end', 'call_id': call_id, 'status': status, 'tokens': used,
                       'outcome': outcome, 'lane': lane, 'kind': kind}, raw=body)
                _pending.pop(call_id)
            except OSError:
                _state['faulted'] = True
                status, body = 503, b'{"error": {"type": "accounting_unavailable"}}'
        self.send(status, body)


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def initialize():
    global CHAT_MODELS
    if ROLE not in ('development', 'driver'):
        raise SystemExit('GATEWAY_ROLE must be development or driver')
    if ROLE == 'development':
        CHAT_MODELS = CHAT_MODELS or {'chat': 'accounts/fireworks/models/glm-5p3',
                                      'chat-b': 'accounts/fireworks/models/kimi-k3'}
        if set(CHAT_MODELS) != {'chat', 'chat-b'} or not all(isinstance(v, str) and v for v in CHAT_MODELS.values()):
            raise SystemExit('Development gateway requires exactly chat and chat-b')
        if min(MAX_CHAT_TOKENS, MAX_EMBED_TOKENS) <= 0:
            raise SystemExit('Development gateway requires positive token allowances')
    elif not DRIVER_MODEL:
        raise SystemExit('Driver gateway requires DRIVER_MODEL')
    parsed = urllib.parse.urlsplit(UPSTREAM)
    schemes = ('https', 'http') if os.environ.get('GATEWAY_ALLOW_HTTP_UPSTREAM') == '1' else ('https',)
    if not KEY or parsed.scheme not in schemes or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        raise SystemExit('Gateway requires an upstream key and an HTTPS upstream')
    Path(LOG).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if Path(STATE).exists() or Path(LOG).exists():
        raise SystemExit('Use a fresh gateway journal for every trial')
    persist()
    audit({'event': 'gateway_started', 'configuration': configuration()})


def finalize():
    """Explicitly cancel unfinished calls (reservations stay charged) and freeze the journal."""
    global _closed
    with _lock:
        if _closed:
            return
        for call_id, (kind, reserved) in sorted(_pending.items()):
            audit({'event': 'model_response_end', 'call_id': call_id, 'status': 503, 'tokens': reserved,
                   'outcome': 'gateway_stopped', 'kind': kind})
        _pending.clear()
        _state.update(chat_reserved=0, embed_reserved=0)
        persist()
        audit({'event': 'gateway_stopped', 'faulted': _state['faulted'], 'state': dict(_state)})
        _closed = True


def main():
    def shutdown(*_):
        try:
            finalize()
        except BaseException:
            os._exit(1)
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    initialize()
    Server(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
