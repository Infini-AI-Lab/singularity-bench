"""Worker-side runner for ONE validity scenario (executed only as UID 1101 via run_worker.py).

It imports the submitted harness and hands ``run()`` a model proxy and a time budget.
There is no scenario book, expected file content, check, scripted reply logic or score
here: the scripted model lives in the trusted root parent, which records every call it
answers. Status lines sent from this process are claims made inside a process the
submission controls; the parent's own call log and its post-run workspace checks are
the trusted observations.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

MAX_LINE = 64 * 1024**2
UNBOUNDED = 10 ** 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--task-file', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    # The original stdout pipe becomes the private reply channel; everything the
    # harness prints (including raw writes to fd 1) is redirected to stderr.
    channel = os.fdopen(os.dup(1), 'wb', buffering=0)
    os.dup2(2, 1)
    replies = os.fdopen(os.dup(0), 'rb')  # buffered: one read per reply line
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    def send(value):
        channel.write(json.dumps(value, default=str).encode() + b'\n')

    def receive():
        line = replies.readline(MAX_LINE + 1)
        if not line.endswith(b'\n') or len(line) > MAX_LINE:
            raise ConnectionError('evaluation model channel closed')
        return json.loads(line)

    class ModelProxy:
        """The ModelClient surface; replies come from the trusted scripted model."""

        def complete(self, messages, tools=None, **kw):
            send({'op': 'complete', 'messages': messages, 'tools': tools, 'kw': kw})
            reply = receive()
            if reply.get('ok') is True:
                return reply['response']
            kind = ConnectionError if reply.get('error_type') == 'ConnectionError' else RuntimeError
            raise kind(str(reply.get('message', 'model call failed')))

    class Budget:
        """Time, and only time -- the same surface a graded run hands the harness."""

        def __init__(self, seconds):
            self.deadline = time.time() + seconds

        def seconds_left(self):
            return max(0.0, self.deadline - time.time())

        def exhausted(self):
            return self.seconds_left() <= 0

        def calls_left(self):
            return UNBOUNDED

        def tokens_left(self):
            return UNBOUNDED

    task = open(args.task_file).read()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.module)))
    spec = importlib.util.spec_from_file_location('submitted_harness', args.module)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as error:  # noqa: BLE001 - recorded, never raised
        send({'op': 'status', 'imported': False, 'has_run': False,
              'fatal': f'import failed: {type(error).__name__}: {error}'[:300]})
        return 0
    function = getattr(module, 'run', None)
    if not callable(function):
        send({'op': 'status', 'imported': True, 'has_run': False, 'fatal': 'agent.py defines no callable run()'})
        return 0
    send({'op': 'status', 'imported': True, 'has_run': True})
    raised = None
    try:
        function(task, args.workspace, ModelProxy(), Budget(args.seconds))
    except BaseException as error:  # noqa: BLE001 - the spec says run must not raise; record it
        raised = f'{type(error).__name__}: {error}'[:300]
    send({'op': 'done', 'raised': raised})
    return 0


if __name__ == '__main__':
    sys.exit(main())
