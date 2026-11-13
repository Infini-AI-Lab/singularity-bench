#!/usr/bin/env python3
"""Optional development tasks; no default per-task time or call limit."""
import argparse
import grp
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
CASES = {
    'triton-bias-relu': 'Implement a fused bias + ReLU Triton kernel.',
    'cuda-reduction': 'Repair a CUDA extension build and incorrect tail handling.',
    'gpu-batch-service': 'Fix response routing in a GPU batching service.',
}


class DevelopmentBudget:
    def __init__(self, seconds=None):
        self.deadline = None if seconds is None else time.monotonic() + seconds

    def seconds_left(self):
        # A finite compatibility sentinel avoids breaking harnesses that call int().
        return 10**9 if self.deadline is None else max(0.0, self.deadline - time.monotonic())

    def exhausted(self):
        return self.seconds_left() <= 0

    def calls_left(self):
        return 10**9

    def tokens_left(self):
        return 10**9


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclasses and other ordinary module introspection.
    spec.loader.exec_module(module)
    return module


def prepare(case, destination):
    shutil.copytree(HERE / case, destination)
    # Native author file tools and worker commands share the workspace group.
    for path in [destination, *destination.rglob('*')]:
        share(path)
    return destination.resolve()


def share(path):
    try:
        os.chown(path, -1, grp.getgrnam('workspace').gr_gid)
    except KeyError:  # Also allow running the helper outside the task container.
        pass
    path.chmod(path.stat().st_mode | (0o2070 if path.is_dir() else 0o060))


def drive(case, workspace, harness, model_name, seconds):
    live = load_module('example_live_model', HERE.parent / 'smoke/live.py')
    trace = workspace / f'model-calls-{uuid.uuid4().hex}.jsonl'
    trace_lock = threading.Lock()

    class RecordingModel(live.LiveModel):
        def complete(self, messages, tools=None, **kwargs):
            # Snapshot before calling: a harness may mutate its history afterwards.
            request = json.loads(json.dumps(dict(messages=messages, tools=tools, **kwargs)))
            row = {'model': self.model, 'request': request, 'started': time.time()}
            try:
                response = super().complete(messages, tools=tools, **kwargs)
                row['response'] = response
                return response
            except BaseException as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                row['finished'] = time.time()
                with trace_lock, trace.open('a') as output:
                    output.write(json.dumps(row) + '\n')

    print(f'workspace={workspace}\nmodel={model_name}\ntrace={trace}', flush=True)
    sys.path.insert(0, str(harness.parent))
    module = load_module('submitted_harness', harness)
    model = RecordingModel(model_name, max_calls=None)
    module.run((workspace / 'TASK.md').read_text(), str(workspace), model, DevelopmentBudget(seconds))
    print(f'Harness returned. {model.calls} model calls, {model.tokens} reported tokens. '
          'Workspace and trace retained; run check when you want feedback.', flush=True)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('seconds must be finite and positive')
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    commands.add_parser('diagnose')
    init = commands.add_parser('prepare')
    init.add_argument('case', choices=CASES)
    init.add_argument('destination', type=Path)
    run = commands.add_parser('run')
    run.add_argument('case', choices=CASES)
    run.add_argument('--workspace', type=Path)
    run.add_argument('--harness', type=Path, default=HERE.parent / 'harness/agent.py')
    run.add_argument('--model', choices=('chat', 'chat-b'), default='chat')
    run.add_argument('--seconds', type=positive_seconds, help='an optional limit chosen by you')
    check = commands.add_parser('check')
    check.add_argument('case', choices=CASES)
    check.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'list':
        for name, description in CASES.items():
            print(f'{name}: {description}')
    elif args.command == 'diagnose':
        return subprocess.call([sys.executable, str(HERE / 'checks/diagnose.py')])
    elif args.command == 'prepare':
        print(prepare(args.case, args.destination))
    elif args.command == 'check':
        return subprocess.call([sys.executable, str(HERE / 'checks' / (args.case + '.py')),
                                str(args.workspace.resolve())])
    else:
        workspace = args.workspace
        if workspace is None:
            parent = Path(tempfile.mkdtemp(prefix=f'harness-{args.case}-'))
            share(parent)
            workspace = parent / 'workspace'
        if not workspace.exists():
            prepare(args.case, workspace)
        if not (workspace / 'TASK.md').is_file():
            parser.error('workspace must contain TASK.md; use prepare or a new directory')
        drive(args.case, workspace.resolve(), args.harness.resolve(), args.model, args.seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
