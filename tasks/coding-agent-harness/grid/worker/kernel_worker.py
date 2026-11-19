"""Worker-side host for ONE submitted kernel (UID 1101 via run_worker.py; no answers here).

The trusted parent sends JSON commands on stdin and reads JSON replies from the original
stdout (everything the kernel prints goes to stderr). Tensors cross through data-only
tensor trees (no pickle): inputs in a root-written request directory, outputs in a
worker-written response directory. The oracle libraries are refused at import, as in the
task verifier; attempts are reported to the parent. The parent's clock -- never this
process's -- times each benchmark batch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


class OracleSeal:
    def __init__(self, names):
        self.names, self.seen = set(names), set()

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in self.names:
            self.seen.add(root)
            raise ModuleNotFoundError(repr(root) + " is this grader's own oracle and is not available to a "
                                      'submission: the task is to write the kernel yourself.', name=fullname)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--transport', required=True)
    parser.add_argument('--oracle-libs', default='[]')
    args = parser.parse_args()
    channel = os.fdopen(os.dup(1), 'wb', buffering=0)
    os.dup2(2, 1)
    commands = os.fdopen(os.dup(0), 'rb')  # buffered: one read per command line
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import tensor_files
    seal = OracleSeal(json.loads(args.oracle_libs))
    sys.meta_path.insert(0, seal)
    transport = Path(args.transport)
    cache, prepared = {}, None
    function = None

    def reply(value):
        channel.write(json.dumps({**value, 'oracle_libraries_refused': sorted(seal.seen)}).encode() + b'\n')

    while line := commands.readline():
        command = json.loads(line)
        operation = command.get('op')
        try:
            if operation == 'load':
                sys.path.insert(0, os.path.dirname(args.module))
                spec = importlib.util.spec_from_file_location('submitted_kernel', args.module)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                function = getattr(module, args.entry)
                if not callable(function):
                    raise TypeError(f'{args.entry} is not callable')
                reply({'ok': True})
            elif operation == 'call':
                arguments, keywords, _ = tensor_files.load(transport / 'request' / command['request'], device='cuda',
                                                          cache=cache)
                output = function(*arguments, **keywords)
                torch.cuda.synchronize()
                target = transport / 'response' / command['request']
                tensor_files.dump(output, target)
                reply({'ok': True, 'response': command['request']})
            elif operation == 'bench_prepare':
                arguments, keywords, options = tensor_files.load(transport / 'request' / command['request'],
                                                                device='cuda', cache=cache)
                prepared = (arguments, keywords)
                started, count = time.perf_counter(), 0
                while count < 5 or (time.perf_counter() - started) < float(options.get('warm_s', .2)):
                    function(*arguments, **keywords)
                    torch.cuda.synchronize()
                    count += 1
                reply({'ok': True, 'per_call_estimate': max((time.perf_counter() - started) / count, 1e-6)})
            elif operation == 'bench_batch':
                arguments, keywords = prepared
                for _ in range(int(command['count'])):
                    function(*arguments, **keywords)
                torch.cuda.synchronize()
                reply({'ok': True})
            elif operation == 'drain':
                torch.cuda.synchronize()
                reply({'ok': True})
            elif operation == 'exit':
                reply({'ok': True})
                return 0
            else:
                reply({'ok': False, 'error': 'unknown operation'})
        except BaseException as error:  # noqa: BLE001 - reported to the trusted parent
            reply({'ok': False, 'error': f'{type(error).__name__}: {error}'[:2000]})
    return 0


if __name__ == '__main__':
    sys.exit(main())
