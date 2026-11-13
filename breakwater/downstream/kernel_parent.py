"""Trusted root half of a kernel cell: the task's own sealed verifier, with a remote candidate.

The task verifier (``/private/tests/verify_env.py``) runs here unchanged except for three
interception points, so its references, oracle portfolio, oracle-agreement checks,
correctness tolerances, recomputation challenge and reward formula are the task's own:

1. Loading ``MODULE_PATH`` returns a module whose kernel attribute is a ``RemoteKernel``.
   The submitted code never enters this process.
2. ``_bench`` on a lambda that closes over the remote kernel becomes a parent-clocked
   batch protocol: the worker runs N back-to-back calls and synchronizes; this process's
   clock measures each batch (one pipe round trip per batch, never the worker's clock).
   Reference/oracle ``_bench`` calls run in this process exactly as before.
3. ``_relerr``/``_must_recompute`` are recorded so the final verifier can recompute.

The verifier's own reward.json is retained as the provisional result; the data-only
scorer recomputes it from the raw measurements and requires agreement.
"""
import ast
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
import tensor_files
from registry import KERNELS, require_task


class RemoteFailure(RuntimeError):
    pass


class RemoteKernel:
    """Callable proxy whose every invocation executes in the worker process."""

    def __init__(self, task, transport, oracle_libs, journal):
        entry = KERNELS[require_task(task)]
        self.task, self.transport, self.journal = task, Path(transport), journal
        self.serial = 0
        self.identities = {}
        self.capturing = None
        self.seal_hits = set()
        self.seal_sink = None  # the task verifier's own _SEAL.seen, so its zeroing rule applies
        (self.transport / 'request').mkdir(parents=True, mode=0o755)
        response = self.transport / 'response'
        response.mkdir(mode=0o755)
        os.chown(response, 1101, 1200)
        self.process = subprocess.Popen(
            ['python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', 'python3', '-I', '-B',
             '/opt/breakwater-worker/kernel_worker.py', '--module', entry['module'], '--entry', entry['entry'],
             '--transport', str(self.transport), '--oracle-libs', json.dumps(sorted(oracle_libs))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(journal / 'kernel-worker.stderr', 'ab'),
            start_new_session=True)
        self.loaded = False

    def ensure_loaded(self):
        # Deferred so an unimportable submission fails inside the task verifier's own
        # try/except (reward 0), exactly like an in-process import failure.
        if not self.loaded:
            self.command({'op': 'load'})
            self.loaded = True

    def command(self, value, timeout=1800):
        import select
        self.process.stdin.write(json.dumps(value).encode() + b'\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise RemoteFailure('kernel worker did not answer before its deadline')
        line = self.process.stdout.readline(1 << 20)
        if not line.endswith(b'\n'):
            raise RemoteFailure('kernel worker closed its channel')
        reply = json.loads(line)
        self.seal_hits.update(reply.get('oracle_libraries_refused') or [])
        if self.seal_sink is not None:
            self.seal_sink.update(self.seal_hits)
        if reply.get('ok') is not True:
            raise RemoteFailure(str(reply.get('error', 'kernel call failed')))
        return reply

    def identity(self, tensor):
        import weakref
        key = id(tensor)
        cached = self.identities.get(key)
        if cached is None or cached[0]() is not tensor:
            self.serial += 1
            cached = self.identities[key] = (weakref.ref(tensor), self.serial)
        return cached[1]

    def publish(self, value):
        self.serial += 1
        name = f'{self.serial:06d}'
        target = self.transport / 'request' / name
        tensor_files.dump(value, target, self.identity)
        for path in (target, *target.iterdir()):
            path.chmod(0o755 if path.is_dir() else 0o644)
        return name

    def __call__(self, *args, **kwargs):
        if self.capturing is not None:
            # Inside _bench: record the operands; the parent-clocked protocol runs them.
            self.capturing.append((args, kwargs))
            return None
        self.ensure_loaded()
        name = self.publish((args, kwargs, {}))
        reply = self.command({'op': 'call', 'request': name})
        return tensor_files.load(self.transport / 'response' / reply['response'], device='cuda')

    def benchmark(self, args, kwargs, target_s=0.05, batches=6, warm_s=0.2):
        self.ensure_loaded()
        name = self.publish((args, kwargs, {'warm_s': warm_s}))
        estimate = float(self.command({'op': 'bench_prepare', 'request': name})['per_call_estimate'])
        batch = max(5, min(200, int(target_s / max(estimate, 1e-6))))
        samples = []
        for _ in range(batches):
            started = time.perf_counter()
            self.command({'op': 'bench_batch', 'count': batch})
            samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        self.command({'op': 'drain'})
        drain = time.perf_counter() - started
        best = min(samples) / batch
        return best, {'method': 'parent-clocked-batch-v1', 'batch_size': batch, 'batch_seconds': samples,
                      'drain_seconds': drain, 'deferred_completion_suspect': drain > 2 * max(samples),
                      'ipc_round_trips_per_batch': 1}

    def close(self):
        if self.process.poll() is None:
            try:
                self.command({'op': 'exit'}, timeout=30)
            except Exception:
                pass
            self.process.kill()
        self.process.wait(timeout=30)
        # Grandchildren (compilers, helper processes) leave the session; stop them by UID.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from workers import kill_workers
        kill_workers()


def strip_entrypoint(source):
    """Keep the verifier's definitions; drop only its terminal ``try: main() ...`` invocation."""
    tree = ast.parse(source.read_text(), filename=str(source))
    last = tree.body[-1]
    calls_main = isinstance(last, ast.Try) and any(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'main' for node in last.body)
    plain = isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and \
        isinstance(last.value.func, ast.Name) and last.value.func.id == 'main'
    if not (calls_main or plain):
        raise ValueError('Unexpected kernel verifier layout; refusing to guess its entry point')
    tree.body = tree.body[:-1]
    return tree


def closes_over(function, remote):
    return any(getattr(cell, 'cell_contents', None) is remote for cell in (getattr(function, '__closure__', None) or ()))


def collect(task, tests, raw, transport):
    """Run the instrumented task verifier. -> measurements dict (also written to raw)."""
    entry = KERNELS[require_task(task)]
    tests, raw = Path(tests), Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    source = tests / 'verify_env.py'
    tree = strip_entrypoint(source)
    module = types.ModuleType('downstream_private_verifier')
    module.__file__ = str(source)
    sys.path.insert(0, str(tests))

    class ClientKernel:
        """``kernel_client.RemoteKernel`` as the task verifiers use it (``fn = RemoteKernel(path, entry, task)``;
        candidate timing through ``fn.benchmark(*args)`` and ``fn.last_benchmark``), backed by the worker-hosted
        RemoteKernel below and recorded as a parent-clocked candidate timing."""

        def __init__(self, path, entry_name, task_name):
            if (task_name != task or entry_name != entry['entry']
                    or os.path.realpath(path) != os.path.realpath(entry['module'])):
                raise ValueError('Kernel client request differs from the trusted registry')
            self.last_benchmark = {}

        def __call__(self, *args, **kwargs):
            return remote(*args, **kwargs)

        def benchmark(self, *args, target_s=0.05, batches=6, warm_s=0.2, **kwargs):
            seconds, detail = remote.benchmark(args, kwargs, target_s=target_s, batches=batches, warm_s=warm_s)
            timings.append({'role': 'candidate', 'seconds': seconds, **detail})
            self.last_benchmark = {**detail, 'seconds': seconds}
            return seconds

        def close(self):
            pass

    client = types.ModuleType('kernel_client')
    client.RemoteKernel = ClientKernel
    previous_client = sys.modules.get('kernel_client')
    sys.modules['kernel_client'] = client
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    if os.path.realpath(module.MODULE_PATH) != os.path.realpath(entry['module']):
        raise ValueError('Kernel verifier module path differs from the trusted registry')
    oracle_libs = tuple(getattr(module, 'ORACLE_LIBS', ()))
    remote = RemoteKernel(task, transport, oracle_libs, raw)
    if hasattr(module, '_SEAL'):
        remote.seal_sink = module._SEAL.seen
    timings, comparisons, recompute = [], [], []

    class Loader(importlib.machinery.SourceFileLoader):
        def exec_module(self, target):
            setattr(target, entry['entry'], remote)

    original_spec = importlib.util.spec_from_file_location

    def spec_from_file_location(name, location=None, *args, **kwargs):
        if location is not None and os.path.realpath(str(location)) == os.path.realpath(entry['module']):
            return importlib.util.spec_from_loader(name, Loader(name, str(location)))
        return original_spec(name, location, *args, **kwargs)

    original_bench, original_relerr = module._bench, module._relerr
    original_recompute = module._must_recompute

    def bench(function, target_s=0.05, batches=6, warm_s=0.2):
        if closes_over(function, remote):
            remote.capturing = []
            try:
                function()
            finally:
                captured, remote.capturing = remote.capturing, None
            if len(captured) != 1:
                raise RuntimeError('Candidate benchmark lambda must invoke the kernel exactly once')
            seconds, detail = remote.benchmark(*captured[0], target_s=target_s, batches=batches, warm_s=warm_s)
            timings.append({'role': 'candidate', 'seconds': seconds, **detail})
            return seconds
        seconds = original_bench(function, target_s=target_s, batches=batches, warm_s=warm_s)
        timings.append({'role': 'trusted', 'seconds': seconds})
        return seconds

    def relerr(a, b):
        value = original_relerr(a, b)
        comparisons.append(value if math.isfinite(value) else None)
        return value

    def must_recompute(function, call_args, flip):
        value = original_recompute(function, call_args, flip)
        # +inf is the verifier's "the submission raised here" (not cached); NaN means cached.
        recompute.append(value if math.isfinite(value) else ('inf' if value == math.inf else None))
        return value

    module._bench, module._relerr, module._must_recompute = bench, relerr, must_recompute
    module.REWARD_DIR = str(raw / 'provisional')
    voided = None
    importlib.util.spec_from_file_location = spec_from_file_location
    try:
        module.main()
    except BaseException as error:  # GraderBroken and other grader-side faults void the cell
        voided = f'{type(error).__name__}: {error}'[:2000]
    finally:
        importlib.util.spec_from_file_location = original_spec
        if previous_client is None:
            sys.modules.pop('kernel_client', None)
        else:
            sys.modules['kernel_client'] = previous_client
        remote.close()
    provisional = raw / 'provisional'
    measurements = {'task': task, 'protocol': 'breakwater-kernel-cell-v1', 'timings': timings,
                    'comparisons': comparisons, 'recompute': recompute,
                    'oracle_libraries_refused': sorted(remote.seal_hits), 'voided': voided,
                    'perf_tolerance': getattr(module, 'PERF_TOL', getattr(module, 'REL_TOL', None)), 'recompute_min': module.RECOMPUTE_MIN,
                    'verifier_version': getattr(module, 'VERIFIER_VERSION', None),
                    'correct_shapes': len(module.CORRECT_SHAPES), 'grader_shapes': len(module.GRADER_SHAPES)}
    if voided is None:
        measurements['reward'] = json.loads((provisional / 'reward.json').read_text())
        measurements['details'] = json.loads((provisional / 'details.json').read_text())
    (raw / 'measurements.json').write_text(json.dumps(measurements, sort_keys=True, default=str))
    return measurements
