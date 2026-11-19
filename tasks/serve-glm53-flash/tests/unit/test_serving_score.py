"""serve-glm53-flash: the serving score, recomputed from the traffic capture. No GPUs or Docker.

    python3 -m pytest tasks/serve-glm53-flash/tests/unit
"""
import copy
import importlib.util
import io
import resource
import tarfile
from types import SimpleNamespace
import zipfile
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
sys.path[:0] = [str(FRAMEWORK)]
from contract import ContractError  # noqa: E402


def task_code(name):
    """A task's trusted code (tasks/NAME/tests/breakwater_task.py)."""
    import task_api
    return task_api.load_hook(TASKS / name / 'tests' / 'breakwater_task.py', 'task_' + name.replace('-', '_'))


class ServingScoreTests(unittest.TestCase):
    """The verifier recomputes serving income from the complete capture and the accuracy generations."""

    def test_reward_recomputes_from_the_capture_and_rejects_a_changed_reward(self):
        tests = TASKS / 'serve-glm53-flash/tests'
        sys.path.insert(0, str(tests))
        try:
            def load(name):
                spec = importlib.util.spec_from_file_location('serving_' + name, tests / f'{name}.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            simulator, scorer = load('simulator'), load('score')
        finally:
            sys.path.remove(str(tests))
        items = json.loads((tests / 'hidden_eval.json').read_text())
        generations = [{'id': item['id'], 'benchmark': item['benchmark'], 'generation': '\\boxed{' + item['gold'] + '}'}
                       for item in items]
        record = {'primer': False, 'ok': True, 'error': None, 'input_tokens': 1_000_000, 'cached_tokens': 200_000,
                  'reported_output_tokens': 10_000, 'max_tokens': 10_000, 'text': 'x' * 10_000,
                  't_send': 0, 't_first': .1, 't_last': 1, 't_done': 2, 'position': 0}
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / 'traffic-pass-0.json').write_text(json.dumps({'records': [record], 't_start': 0.0, 'window_s': 900}))
            (raw / 'accuracy.generations.json').write_text(json.dumps(generations))
            (raw / 'diagnostics.json').write_text(json.dumps({'reward': 0.093}))
            self.assertEqual(task_code('serve-glm53-flash').serving_score(raw, simulator, scorer.score_candidate), 0.093)
            (raw / 'diagnostics.json').write_text(json.dumps({'reward': 0.2}))
            with self.assertRaises(ContractError):
                task_code('serve-glm53-flash').serving_score(raw, simulator, scorer.score_candidate)
            (raw / 'traffic-pass-0.json').write_text(json.dumps({'records': [record], 't_start': 0.0, 'window_s': 300}))
            with self.assertRaises(ContractError):
                task_code('serve-glm53-flash').serving_score(raw, simulator, scorer.score_candidate)


if __name__ == '__main__':
    unittest.main()
