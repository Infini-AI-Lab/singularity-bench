from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import torch

from common import gpu_call, load, require_cuda


def post(url, body):
    request = urllib.request.Request(url + '/infer', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def host(workspace, ready, evidence):
    require_cuda()
    module = load(workspace, 'service.py')
    original = module.compute_batch

    def observed(items):
        result = gpu_call(lambda: original(items))
        with Path(evidence).open('a') as output:
            output.write('GPU batch completed\n')
        return result

    module.compute_batch = observed
    module.serve(port=0, ready_path=ready)


def check(workspace):
    with tempfile.TemporaryDirectory(prefix='gpu-service-check-') as scratch:
        ready, evidence = Path(scratch) / 'ready', Path(scratch) / 'gpu-evidence'
        process = subprocess.Popen([sys.executable, __file__, '--serve', str(workspace),
                                    str(ready), str(evidence)], start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                assert process.poll() is None, 'Server exited before readiness; inspect its output.'
                assert time.monotonic() < deadline, 'Server did not become ready within 60s.'
                time.sleep(0.05)
            url = ready.read_text()
            with urllib.request.urlopen(url + '/health', timeout=10) as response:
                assert json.load(response)['ready']
            # Independent CPU reference, never imports the candidate's weights().
            w = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
            b = torch.arange(4) / 10
            for round_id in range(3):
                barrier = threading.Barrier(8)

                def request(index):
                    rows = (1, 7, 2, 5, 3, 8, 4, 6)[index]
                    x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / 100 + index
                    body = {'id': f'{round_id}-{index}', 'x': x.tolist()}
                    barrier.wait(timeout=10)
                    out = post(url, body)
                    assert out['id'] == body['id'], 'Response has the wrong request ID'
                    torch.testing.assert_close(torch.tensor(out['y']), torch.relu(x @ w + b))

                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(request, range(8)))
            regression = workspace / 'test_regression.py'
            assert regression.is_file(), 'Add test_regression.py as requested in TASK.md.'
            subprocess.run([sys.executable, str(regression)], cwd=workspace,
                           env=dict(os.environ, SERVICE_URL=url), check=True, timeout=120)
            deadline = time.monotonic() + 10
            while not evidence.exists():
                assert process.poll() is None, 'Server exited while profiling a batch.'
                assert time.monotonic() < deadline, 'No completed GPU batch observed.'
                time.sleep(0.05)
            print('PASS concurrent requests, regression test, and observed GPU computation.', flush=True)
        finally:
            # Kill only the process group created by this checker, including child processes.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    if sys.argv[1] == '--serve':
        host(*sys.argv[2:])
    else:
        check(Path(sys.argv[1]).resolve())
