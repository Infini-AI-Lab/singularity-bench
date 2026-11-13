"""Operator-side HTTP code judge. Each test runs in a fresh isolated Docker container.

Run on a CPU host with Docker, outside the GPU containers. Generated code never sees
the Docker socket or the grading files. RL invocations receive test inputs only;
expanded benchmark test programs run inside the CPU sandbox. Only this trusted
service accesses Docker. Bind to loopback by default; a non-loopback bind requires
MIDTRAIN_CODE_SANDBOX_TOKEN. Keep the service reachable from the evaluation stage only.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import uuid


def equivalent(actual, expected):
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        expected = expected.strip().split()
        actual = actual.strip().split()
        if len(actual) != len(expected):
            return False
        for got, want in zip(actual, expected):
            if got == want:
                continue
            try:
                if not math.isfinite(float(got)) or not math.isfinite(float(want)):
                    return False
                if not math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-6):
                    return False
            except ValueError:
                return False
        return True
    if type(actual) is not type(expected) and not (isinstance(actual, (int, float)) and isinstance(expected, (int, float))):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected


def functional_program(code, fn_name, test_input):
    if isinstance(test_input, str):
        arguments = [json.loads(line) for line in test_input.splitlines() if line.strip()]
    elif isinstance(test_input, list):
        arguments = test_input
    else:
        raise ValueError('functional input must encode a list of arguments')
    wrapper = '''
import json as _judge_json
import sys as _judge_sys
_judge_args = _judge_json.loads(_judge_sys.stdin.read())
_judge_namespace = {"__name__": "candidate", "List": list, "Optional": __import__("typing").Optional}
exec(compile(SOURCE, "candidate.py", "exec"), _judge_namespace)
_judge_target = _judge_namespace["Solution"]() if "Solution" in _judge_namespace else None
_judge_fn = getattr(_judge_target, FUNCTION) if _judge_target is not None else _judge_namespace[FUNCTION]
_judge_result = _judge_fn(*_judge_args)
print(_judge_json.dumps(_judge_result))
'''
    wrapper = wrapper.replace('SOURCE', '_judge_source').replace('FUNCTION', '_judge_function')
    return '_judge_source = ' + repr(code) + '\n_judge_function = ' + repr(fn_name) + '\n' + wrapper, json.dumps(arguments)


def humanevalplus_program(code, label, marker):
    """Run the full published check(candidate) only inside the isolated container.

    Separate namespaces keep candidate helpers from shadowing test helpers.
    The success marker is printed only after every assertion returns normally;
    an early exit(0) or ordinary candidate stdout cannot count as a pass.
    """
    entry = label.get('entry_point')
    tests = label.get('test')
    if (not isinstance(entry, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', entry)
            or not isinstance(tests, str) or not tests.strip()):
        raise ValueError('invalid expanded functional test suite')
    return ('import contextlib, io\n'
            '_candidate = {"__name__": "candidate"}\n'
            '_tests = {"__name__": "tests"}\n'
            'with contextlib.redirect_stdout(io.StringIO()):\n'
            f'    exec(compile({code!r}, "candidate.py", "exec"), _candidate)\n'
            f'    exec(compile({tests!r}, "tests.py", "exec"), _tests)\n'
            f'    _tests["check"](_candidate[{entry!r}])\n'
            f'print({marker!r})\n')


class Judge:
    def __init__(self, image, workers=16, timeout=6.0):
        # Freeze the operator-built CPU image rather than following a mutable tag per test.
        self.image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
        self.timeout = timeout
        self.slots = threading.BoundedSemaphore(workers)

    def execute(self, code, input_text, *, timeout=None):
        with self.slots, tempfile.TemporaryDirectory(prefix='midtrain-code-') as tmp:
            name = 'midtrain-code-' + uuid.uuid4().hex
            command = ['docker', 'run', '--rm', '-i', '--name', name,
                       '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                       '--security-opt', 'no-new-privileges', '--user', '65534:65534',
                       '--pids-limit', '32', '--memory', '512m', '--memory-swap', '512m', '--cpus', '1',
                       '--tmpfs', '/tmp:rw,noexec,nosuid,size=32m',
                       '--ulimit', 'nofile=64:64', '--ulimit', 'fsize=1048576:1048576',
                       '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
                       self.image, 'python3', '-I', '-c',
                       'import io, json, sys; p = json.load(sys.stdin); '
                       'sys.stdin = io.TextIOWrapper(io.BytesIO(p["stdin"].encode())); '
                       'exec(compile(p["code"], "sandbox.py", "exec"))']
            output = Path(tmp) / 'stdout'
            payload = Path(tmp) / 'stdin'
            # Expanded tests can exceed execve's per-argument limit. Transfer code
            # through stdin; preserve the candidate's stdin for ordinary tests.
            payload.write_text(json.dumps({'code': code, 'stdin': input_text}))
            try:
                with payload.open('rb') as stdin, output.open('wb') as stdout:
                    process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL)
                    deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
                    try:
                        while process.poll() is None:
                            if time.monotonic() >= deadline or output.stat().st_size > (1 << 20):
                                process.kill()
                                process.wait()
                                return None
                            time.sleep(0.02)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                    if process.returncode == 125:
                        raise RuntimeError('Docker could not launch the code sandbox')
                    if process.returncode != 0 or output.stat().st_size > (1 << 20):
                        return None
                return output.read_text(errors='replace')
            finally:
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)

    def score(self, code, label):
        if not isinstance(code, str) or len(code) > (1 << 20):
            raise ValueError('invalid code')
        if label.get('testtype') == 'humanevalplus':
            marker = 'passed-' + uuid.uuid4().hex
            program = humanevalplus_program(code, label, marker)
            result = self.execute(program, '', timeout=180.0)
            return result == marker + '\n'
        return self.score_rl(code, label)

    def score_rl(self, code, label):
        if not isinstance(code, str) or len(code) > (1 << 20):
            raise ValueError('invalid code')
        tests = label['tests']
        if not tests or label['testtype'] not in ('stdin', 'functional'):
            raise ValueError('invalid test suite')
        # Evaluate every supplied test, with an upper bound per generated solution.
        deadline = time.monotonic() + 180
        for test in tests:
            if time.monotonic() >= deadline:
                return False
            if label['testtype'] == 'stdin':
                result = self.execute(code, test['input'])
                if result is None or not equivalent(result, test['output']):
                    return False
            else:
                program, input_text = functional_program(code, label['fn_name'], test['input'])
                result = self.execute(program, input_text)
                if result is None:
                    return False
                try:
                    actual = json.loads(result)
                    expected = json.loads(test['output']) if isinstance(test['output'], str) else test['output']
                except (ValueError, TypeError):
                    return False
                if not equivalent(actual, expected):
                    return False
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='midtrain-code-sandbox:local')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    token = os.environ.get('MIDTRAIN_CODE_SANDBOX_TOKEN')
    if args.host not in ('127.0.0.1', 'localhost') and not token:
        parser.error('non-loopback binds require MIDTRAIN_CODE_SANDBOX_TOKEN')
    judge = Judge(args.image, args.workers)
    # Infrastructure check, including an actual container execution.
    if judge.execute('print(2 + 3)', '') != '5\n':
        raise RuntimeError('code sandbox self-check failed')
    if not judge.score('def square(x): return x * x', {
            'testtype': 'humanevalplus', 'entry_point': 'square',
            'test': 'import numpy as np\ndef check(candidate):\n    np.testing.assert_allclose(candidate(3), 9)\n'}):
        raise RuntimeError('Expanded code-test self-check failed; rebuild the CPU sandbox image with NumPy')

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            if token and self.headers.get('Authorization') != 'Bearer ' + token:
                self.respond(401, {'status': 'unauthorized'})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.respond(200 if self.path == '/health' else 404,
                             {'status': 'ok' if self.path == '/health' else 'not_found', 'image': judge.image})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/score':
                self.respond(404, {'status': 'not_found'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (16 << 20):
                    raise ValueError('request must be <=16MiB')
                payload = json.loads(self.rfile.read(size))
                passed = judge.score(payload['code'], payload['label'])
                self.respond(200, {'status': 'ok', 'passed': passed})
            except Exception as error:
                self.respond(500, {'status': 'infrastructure_error', 'error': str(error)})

        def log_message(self, *_):
            pass

    print(f'Code sandbox listening on {args.host}:{args.port}; CPU image {judge.image}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
