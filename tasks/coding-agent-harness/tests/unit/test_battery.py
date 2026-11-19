"""coding-agent-harness: the out-of-process validity battery (parity with the task's own smoke runner). No GPUs or Docker.

    python3 -m pytest tasks/coding-agent-harness/tests/unit
"""
import copy
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[4] / 'breakwater'
TASKS = FRAMEWORK.parent / 'tasks'
HARNESS = TASKS / 'coding-agent-harness'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
sys.path[:0] = [str(FRAMEWORK)]
sys.path += [str(HARNESS / 'tests')]
from contract import ContractError  # noqa: E402


def task_code(name):
    """A task's trusted code (tasks/NAME/tests/breakwater_task.py)."""
    import task_api
    return task_api.load_hook(TASKS / name / 'tests' / 'breakwater_task.py', 'task_' + name.replace('-', '_'))


class BatteryTests(unittest.TestCase):
    def run_battery(self, source, extra=None, label=None):
        import harness_battery as battery
        root = Path(self.temp.name)
        target = root / (label or Path(source).stem) / 'harness'
        target.mkdir(parents=True)
        shutil.copyfile(source, target / 'agent.py')
        for name, payload in (extra or {}).items():
            (target / name).write_bytes(payload)
        def child(module, workspace, task, seconds):
            return [sys.executable, '-I', '-B', str(TASKS / 'coding-agent-harness' / 'runners' / 'harness_child.py'), '--module', str(module),
                    '--workspace', str(workspace), '--task-file', str(task), '--seconds', str(seconds)]
        raw = root / (label or Path(source).stem) / 'raw'
        # kill_workers signals every UID-1101 process; that is only meaningful inside an
        # intermediate container, never on a (shared) test host.
        with patch.object(battery, 'WORK_ROOT', root / 'work'), patch.object(battery, 'child_command', child), \
             patch.object(battery.os, 'chown'), patch.object(battery, 'kill_workers'):
            rows, summary = battery.run_battery(target / 'agent.py', HARNESS / 'tests/smoke', raw / 'harness')
        value = task_code('coding-agent-harness').harness_score(raw, smoke=HARNESS / 'tests/smoke', module=target / 'agent.py')
        return rows, summary, value, raw

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_harness_carrying_an_executable_an_archive_or_bulk_is_not_run(self):
        import harness_battery as battery
        source = FIXTURES / 'reference_agent.py'
        cases = {'elf': {'helper': b'\x7fELF' + bytes(64)}, 'zip': {'vendor.zip': b'PK\x03\x04' + bytes(64)},
                 'tar': {'bundle': bytes(257) + b'ustar' + bytes(64)},
                 'bulk': {'notes.txt': b'x' * (battery.MAX_SUBMISSION_BYTES + 1)}}
        for label, extra in cases.items():
            with self.subTest(payload=label):
                rows, summary, value, raw = self.run_battery(source, extra=extra, label=label)
                self.assertEqual((rows, value, summary['reward']), ([], 0.0, 0.0))
                self.assertEqual(summary.get('error_invalid_payload'), 1.0)
                self.assertEqual(len(json.loads((raw / 'harness/battery.json').read_text())['payload_problems']), 1)
        with tempfile.TemporaryDirectory() as temporary:
            harness = Path(temporary) / 'harness'
            harness.mkdir()
            (harness / 'agent.py').write_text('def run(task, client):\n    return None\n')
            (Path(temporary) / 'manifest.json').write_text(json.dumps({'symlinks': {'harness/tool': '/usr/bin/x'}}))
            self.assertEqual(battery.payload_problems(harness, battery.recorded_links(harness)),
                             ['harness/tool: a symlink'])

    def test_battery_matches_the_smoke_runner_and_recomputes(self):
        sys.path.insert(0, str(HARNESS / 'tests/smoke'))
        import run_smoke
        for name in ('reference_agent.py', 'candidates/do_nothing.py', 'candidates/broken_import.py',
                     'candidates/stops_on_prose.py'):
            source = FIXTURES / name
            rows, summary, value, _ = self.run_battery(source)
            scratch = Path(self.temp.name) / ('original-' + Path(name).stem)
            scratch.mkdir()
            original = run_smoke.summarise([run_smoke.run_one(sc, str(source), str(scratch))
                                            for sc in json.loads((HARNESS / 'tests/smoke/scenarios.json').read_text())])
            with self.subTest(name=name):
                for key in ('gradeable', 'valid', 'did_the_work', 'continues_past_prose', 'model_calls_total'):
                    self.assertEqual(summary[key], original[key], key)
                self.assertEqual(value, summary['gradeable'])

    def test_hostile_harness_cannot_forge_the_trusted_record(self):
        hostile = Path(self.temp.name) / 'hostile.py'
        hostile.write_text('''import os, json
def run(task, workspace, model, budget):
    os.write(1, b'{"op": "done", "raised": null}\\n')       # redirected to stderr, not the channel
    print(json.dumps({"op": "complete"}))                    # likewise
    for name in os.listdir(workspace):
        pass
    os.symlink("/etc/passwd", os.path.join(workspace, "notes.txt"))
''')
        rows, summary, value, raw = self.run_battery(hostile)
        self.assertEqual(value, 1.0)                            # importable with a callable run()
        self.assertEqual(summary['model_calls_total'], 0.0)     # the trusted log saw no model calls
        self.assertEqual(summary['did_the_work'], 0.0)          # a symlink is never followed by root checks
        record = json.loads(next((raw / 'harness').glob('00-*/record.json')).read_text())
        record['n_calls'] = 7
        path = next((raw / 'harness').glob('00-*/record.json'))
        path.write_text(json.dumps(record))
        with self.assertRaises(ContractError):
            task_code('coding-agent-harness').harness_score(raw, smoke=HARNESS / 'tests/smoke', module=hostile)


    def test_forged_status_claims_on_the_real_channel_cannot_make_a_runless_module_gradeable(self):
        forger = Path(self.temp.name) / 'forger.py'
        forger.write_text("""import json, os, threading, time
FORGED = (json.dumps({"op": "status", "imported": True, "has_run": True}) + "\\n").encode()
os.write(3, FORGED)                       # the reply channel the fixed runner dup'd from stdout
def later():
    time.sleep(0.3)
    os.write(3, FORGED)
    os.write(3, b'{"op": "done", "raised": null}\\n')
threading.Thread(target=later).start()
""")
        rows, summary, value, raw = self.run_battery(forger)
        self.assertEqual(value, 0.0)
        self.assertEqual(summary['static_run_binding'], 0.0)
        self.assertTrue(any('duplicate status claim' in row['protocol_errors'] for row in rows))
        self.assertFalse(any(row['has_run'] for row in rows))

    def test_static_run_binding_accepts_spec_forms_and_rejects_absence(self):
        import harness_battery as battery
        cases = {'def run(task, workspace, model, budget):\n    pass\n': True,
                 'class run:\n    def __call__(self, *a): pass\n': True,
                 'from impl import run\n': True,
                 'from impl import go as run\n': True,
                 'run = Agent().run\n': True,
                 'try:\n    from fast import run\nexcept ImportError:\n    def run(*a): pass\n': True,
                 'if True:\n    def run(*a): pass\n': True,
                 'def helper():\n    def run(): pass\n': False,
                 'import run\n': False,
                 "globals()['run'] = print\n": False,
                 'def run(:\n': False,
                 'x = 1\n': False}
        for source, expected in cases.items():
            path = Path(self.temp.name) / 'probe.py'
            path.write_text(source)
            with self.subTest(source=source):
                self.assertEqual(battery.static_run_binding(path), expected)


if __name__ == '__main__':
    unittest.main()
