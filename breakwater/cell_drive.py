r"""Run a sealed submitted harness against one downstream task (worker UID 1101 only).

The protocol: one run() call by default, wall clock is the only budget,
calls_left()/tokens_left() are sentinels, and run must not raise. There is NO provider
key anywhere in this container. The ModelClient posts
to the driver gateway's ModelClient lane and the harness's own served endpoint
(``HARNESS_LLM_BASE_URL``) is the gateway's utility lane; both resolve to the cell's
driver model, and every request/response is journaled outside this container before
the harness sees it.

The record written here is a convenience claim made inside a process the submission
controls. Call counts, tokens and model outputs are taken from the gateway journal.

    python3 -I -B cell_drive.py <harness_dir> <task_file> <workspace>
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ['BREAKWATER_DRIVER_BASE'].rstrip('/')
WALL_SEC = float(os.environ.get('HARNESS_WALL_SEC', '14400'))
SESSIONS = max(1, int(os.environ.get('HARNESS_SESSIONS', '1')))
RECORD = os.environ.get('HARNESS_RECORD', '/run/breakwater-cell/record/eval_record.json')
SNAPSHOT_SEC = 30
UNBOUNDED = 10 ** 9
HEADERS = {'Content-Type': 'application/json', 'User-Agent': 'singularity-bench-eval/1.0'}


class LiveModel:
    """The model the harness drives (one attempt per call; no retry the harness did not ask for)."""

    def __init__(self) -> None:
        self.calls = 0
        self.errors: list[str] = []
        self.started = time.time()
        self.turns: list[dict] = []

    def complete(self, messages, tools=None, **kw):
        if time.time() - self.started > WALL_SEC:
            raise RuntimeError('wall clock exhausted')
        if 'model' in kw:
            raise ValueError('The driver model is assigned by the evaluator')
        body = {**kw, 'model': 'driver', 'messages': messages}
        if not {'max_tokens', 'max_completion_tokens'} & body.keys():
            body['max_tokens'] = 8192
        if tools:
            body['tools'] = tools
            body.setdefault('tool_choice', 'auto')
        self.calls += 1
        request = urllib.request.Request(BASE + '/driver/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers=HEADERS)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=600) as response:
                out = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read()[:400].decode('utf-8', 'replace')
            self.errors.append(f'HTTP {error.code}: {detail}')
            raise RuntimeError(f'upstream returned {error.code}') from None
        except Exception as error:  # noqa: BLE001
            self.errors.append(f'{type(error).__name__}: {error}')
            raise RuntimeError(str(error)) from None
        message = ((out.get('choices') or [{}])[0].get('message') or {})
        self.turns.append({'n': self.calls, 'sent_messages': len(messages),
                           'tools_offered': [t.get('function', {}).get('name') for t in (tools or [])
                                             if isinstance(t, dict)],
                           'tool_calls': [c.get('function', {}).get('name') for c in (message.get('tool_calls') or [])
                                          if isinstance(c, dict)]})
        return out


class LiveBudget:
    def __init__(self, model: LiveModel) -> None:
        self.model = model

    def seconds_left(self) -> float:
        return max(0.0, WALL_SEC - (time.time() - self.model.started))

    def exhausted(self) -> bool:
        return self.seconds_left() <= 0

    def calls_left(self) -> int:
        return UNBOUNDED

    def tokens_left(self) -> int:
        return UNBOUNDED


def dump(record: dict) -> None:
    temporary = RECORD + '.tmp'
    try:
        with open(temporary, 'w') as stream:
            json.dump(record, stream, indent=1)
        os.replace(temporary, RECORD)
    except OSError:
        pass


def main() -> int:
    harness_dir, task_file, workspace = sys.argv[1], sys.argv[2], sys.argv[3]
    # Published BEFORE the import: harnesses read the endpoint at module scope.
    os.environ['HARNESS_LLM_BASE_URL'] = BASE + '/v1'
    os.environ.setdefault('HARNESS_CHAT_MODEL', 'chat')
    os.environ.setdefault('HARNESS_EMBED_MODEL', 'embed')
    sys.path.insert(0, harness_dir)
    record = {'workspace': workspace, 'harness_dir': harness_dir, 'wall_sec': WALL_SEC, 'sessions': SESSIONS,
              'phase': 'import_submission', 'partial': True, 'trust': 'claim; the gateway journal is authoritative'}
    dump(record)
    try:
        import agent as submitted
    except Exception as error:  # noqa: BLE001
        record.update(fatal=f'could not import agent.py: {type(error).__name__}: {error}', partial=False)
        dump(record)
        print(f"[eval] FATAL {record['fatal']}")
        return 0
    function = getattr(submitted, 'run', None)
    if not callable(function):
        record.update(fatal='agent.py has no callable run()', partial=False)
        dump(record)
        print('[eval] FATAL no run()')
        return 0
    task = open(task_file).read()
    model = LiveModel()
    record['phase'] = 'run'
    done = threading.Event()

    def snapshots() -> None:
        while not done.wait(SNAPSHOT_SEC):
            dump({**record, 'calls': model.calls, 'elapsed_sec': round(time.time() - model.started, 1),
                  'upstream_errors': model.errors[:20], 'turns': model.turns[:400], 'partial': True})

    thread = threading.Thread(target=snapshots, daemon=True)
    thread.start()
    raised = []
    for _ in range(SESSIONS):
        try:
            function(task, workspace, model, LiveBudget(model))
            raised.append(None)
        except BaseException as error:  # noqa: BLE001 - spec: run must not raise
            raised.append(f'{type(error).__name__}: {error}'[:400])
        if LiveBudget(model).exhausted():
            break
    done.set()
    thread.join()
    record.update(phase='complete', partial=False, raised=raised[0] if SESSIONS == 1 else raised,
                  sessions_run=len(raised), calls=model.calls, elapsed_sec=round(time.time() - model.started, 1),
                  upstream_errors=model.errors[:20], turns=model.turns[:400])
    dump(record)
    print(f"[eval] calls={model.calls} raised={record['raised']} elapsed={record['elapsed_sec']}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
