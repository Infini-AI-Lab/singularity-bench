#!/usr/bin/env python3
"""A scripted stand-in model for Codex: a Responses-API server that has Codex carry out a fixed list of steps.

    python3 -I scripted_model.py --steps STEPS.json [--port PORT]

STEPS.json is a list of steps, each a shell command (a string), ``{"command": "...", "delay": SECONDS}`` (the command,
asked for only after that many seconds) or ``{"patch": "*** Begin Patch ..."}``. The server prints its base URL
(``http://127.0.0.1:PORT``, for ``run.py --author-upstream``) and serves until it is stopped.

For each request Codex sends (through the author gateway), the next action follows from the conversation so far, so
the server keeps no state: the next step's tool call, a ``write_stdin`` poll while the last command is still running
(Codex's exec_command returns after at most 30 seconds with "Process running with session ID N", so long commands
such as training are polled until they exit), and a final message once every step is done. A command runs with
Codex's exec_command tool; a patch with its apply_patch tool when Codex offers one for the model, else as the
``apply_patch`` command, as for models Codex has no metadata for. If Codex could not run a step at all (it had no
execution environment, or lost its executor), the model stops there with a message that says so, also printed to
stderr, instead of going through the remaining steps.
"""
import argparse
import http.server
import json
from pathlib import Path
import re
import sys
import threading
import time

RUNNING = re.compile(r'Process running with session ID (\d+)')
# Codex's tool output when it cannot run a call: no execution environment, or a lost executor.
REJECTED = re.compile(r'unsupported call|exec-server transport disconnected|Failed to create unified exec process')
STOPPING = 'Stopping: Codex could not run the last step: '
USAGE = {'input_tokens': 10, 'input_tokens_details': None, 'output_tokens': 5, 'output_tokens_details': None,
         'total_tokens': 15}


def next_item(body, steps):
    """The model's next output item for this request."""
    items = [item for item in body.get('input', []) if isinstance(item, dict)]
    calls = [item for item in items if item.get('type') in ('function_call', 'custom_tool_call')]
    outputs = [item for item in items if item.get('type') in ('function_call_output', 'custom_tool_call_output')]
    number = len(calls) + 1
    if outputs:
        output = outputs[-1].get('output')
        text = output if isinstance(output, str) else json.dumps(output)
        running = RUNNING.search(text)
        if running:
            return {'type': 'function_call', 'call_id': f'call_{number}', 'name': 'write_stdin',
                    'arguments': json.dumps({'session_id': int(running.group(1)), 'chars': '', 'yield_time_ms': 30000})}
        if REJECTED.search(text):
            return {'type': 'message', 'role': 'assistant', 'id': f'msg_{number}',
                    'content': [{'type': 'output_text', 'text': STOPPING + text[:500]}]}
    # Polls are calls too; the steps done are the calls that were not polls.
    done = sum(item.get('name') != 'write_stdin' for item in calls)
    if done >= len(steps):
        return {'type': 'message', 'role': 'assistant', 'id': f'msg_{number}',
                'content': [{'type': 'output_text', 'text': 'Every scripted step is done.'}]}
    step = steps[done]
    if isinstance(step, dict) and 'command' in step:
        return {'type': 'function_call', 'call_id': f'call_{number}', 'name': 'exec_command',
                'arguments': json.dumps({'cmd': step['command'], 'yield_time_ms': 30000}), 'delay': step.get('delay', 0)}
    tools = {tool.get('name'): tool.get('type') for tool in body.get('tools', []) if isinstance(tool, dict)}
    if isinstance(step, dict) and tools.get('apply_patch') == 'custom':
        return {'type': 'custom_tool_call', 'call_id': f'call_{number}', 'name': 'apply_patch', 'input': step['patch']}
    if isinstance(step, dict) and tools.get('apply_patch') == 'function':
        return {'type': 'function_call', 'call_id': f'call_{number}', 'name': 'apply_patch',
                'arguments': json.dumps({'input': step['patch']})}
    command = f"apply_patch <<'EOF'\n{step['patch']}EOF" if isinstance(step, dict) else step
    return {'type': 'function_call', 'call_id': f'call_{number}', 'name': 'exec_command',
            'arguments': json.dumps({'cmd': command, 'yield_time_ms': 30000})}


def handler(steps):
    """A request handler class for ``steps``; its ``requests`` list records every request (path and body)."""

    class ScriptedModel(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        requests = []
        lock = threading.Lock()

        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with self.lock:
                self.requests.append({'path': self.path, 'body': body})
                name = f'resp_{len(self.requests)}'
            item = next_item(body, steps)
            if item['type'] == 'message' and item['content'][0]['text'].startswith(STOPPING):
                print('scripted model: ' + item['content'][0]['text'], file=sys.stderr, flush=True)
            time.sleep(item.pop('delay', 0))
            events = [{'type': 'response.created', 'response': {'id': name}},
                      {'type': 'response.output_item.done', 'item': item},
                      {'type': 'response.completed', 'response': {'id': name, 'usage': USAGE}}]
            raw = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return ScriptedModel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    steps = json.loads(args.steps.read_text())
    if not isinstance(steps, list) or not all(
            isinstance(step, str) or (isinstance(step, dict) and (set(step) == {'patch'} or (
                'command' in step and set(step) <= {'command', 'delay'}))) for step in steps):
        raise SystemExit('STEPS.json must be a list of commands, {"command": ..., "delay": ...} and {"patch": ...} steps')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler(steps))
    print(f'http://127.0.0.1:{server.server_address[1]}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
