"""What a task's trusted code gets from Breakwater (SPEC.md, "Task code").

A task's trusted code is ``tests/breakwater_task.py``. Its ``evaluate(run)`` runs in the evaluation stage's trusted
coordinator, with the task's private inputs; ``run`` is an :class:`Evaluation`. Submitted code never runs in the coordinator: ``run_submission`` and
``start_submission`` run it apart from the private inputs (as the worker account under Docker, in the separate
submission sandbox under Apptainer), with the task's declared worker environment.

Its ``score(scoring)`` runs in the verifier on sealed outputs only (no submitted code); ``scoring`` is a
:class:`Scoring`. It returns ``{'reward': float, 'details': {...}, 'validated': [...]}``: the
deterministic reward, what the judge is shown about it, and the checks the scorer applied to every record (a failed
check raises, so a reward exists only if all held). A task scored after the integrity review instead returns
``{'reward': None, 'deferred': {'scored_by': ..., 'rule': ...}}``. An optional ``mechanical(scoring)`` adds task facts
to the mechanical report. A scorer that cannot measure at all (an unavailable oracle, say) raises :class:`Void`: the
attempt then has no score, never a zero.

An optional ``review_views(scoring)`` returns ``[(pattern, view)]``: how the judge sees the task's own evidence
(``external/task/...``) and evaluation outputs (``evaluation/...``) in the indexed review profile. ``view(path, scoring)``
returns text, which the framework shows under a header with the file's digest, or None to leave the file to the
generic presentation; a view that fails falls back to the file as is, and the failure is listed in coverage.json.
"""
import functools
import types
import os
import importlib.util
import json
from pathlib import Path
import subprocess

from contract import atomic_json, read_json, require

CONTRACT = Path('/trusted-task/contract.json')
TASK_CODE = 'breakwater_task.py'  # in the task's tests/ directory
VIEW_ROOTS = ('external/task/', 'evaluation/')  # the only review references a task's views may present


class Void(Exception):
    """The task could not measure the attempt (an unavailable oracle, say): no score, never a zero. The verifier
    writes reports/void.json and keeps the failure reward."""


def load_hook(path, name):
    """A task module by path, with its own directory first on the import path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evaluation:
    """The evaluation stage as a task's ``evaluate(run)`` sees it."""

    def __init__(self, contract=None):
        from workers import worker_prefix
        self.contract = contract if contract is not None else read_json(CONTRACT)
        self.raw = Path('/evaluation/raw')  # the outputs the scorer reads; sealed when the stage ends
        self.inputs = Path('/evaluation/inputs')  # trusted inputs the coordinator prepares; sealed too
        self.public = Path('/eval-inputs')  # answer-free files the submitted code may read
        self.submission = Path(self.contract['submission_mount'] or self.contract['workdir'])
        self.tests = Path('/tests')
        # A smoke run (the launcher's --smoke): the task may use reduced settings; its reward is not a score.
        self.smoke = os.environ.get('BREAKWATER_SMOKE') == '1'
        self._prefix = worker_prefix()

    def asset(self, name):
        """The path at which a declared asset is visible in this stage."""
        for row in self.contract['assets']:
            if row['name'] == name and 'intermediate' in row['stages']:
                return Path(row['target'])
        raise KeyError('No asset named ' + name + ' is visible to the evaluation stage')

    def input_roots(self):
        """Read-only task inputs a submitted symbolic link may point into (never private evaluation data)."""
        return [Path(row['target']) for row in self.contract['assets']
                if 'intermediate' in row['stages'] and not row['target'].startswith('/private/')]

    def worker_command(self, argv):
        """``argv`` as submitted code: run apart from the private inputs, with the worker environment."""
        return [*self._prefix, *argv]

    def run_submission(self, argv, **options):
        """subprocess.run of submitted code (see worker_command)."""
        return subprocess.run(self.worker_command(argv), **options)

    def start_submission(self, argv, **options):
        """subprocess.Popen of submitted code (see worker_command)."""
        return subprocess.Popen(self.worker_command(argv), **options)

    def publish(self, name, data):
        """An answer-free file the submitted code may read, at ``/eval-inputs/NAME``."""
        from workers import apptainer
        require('/' not in name and name not in ('', '.', '..'), 'Invalid public input name')
        self.public.mkdir(mode=0o755, exist_ok=apptainer())
        self.public.chmod(0o755)
        path = self.public / name
        path.write_bytes(data)
        path.chmod(0o444)
        return path


class Scoring:
    """The verifier's view for a task's ``score(scoring)``: sealed outputs and inputs, never submitted code."""

    def __init__(self, contract, reports, smoke=False, metadata=None):
        from spec import HOME_MOUNT
        self.contract, self.smoke = contract, smoke  # smoke: sealed in the authority by the launcher
        # What the task's host hooks recorded before authoring (launch['task_metadata']), sealed by the launcher.
        self.metadata = types.MappingProxyType(json.loads(json.dumps(metadata or {})))
        self.raw = Path('/evaluation/raw')
        self.inputs = Path('/evaluation/inputs')
        self.evidence = Path('/external-evidence')
        self.tests = Path('/tests')
        self.submission = Path(contract.get('submission_root') or contract['workdir'])
        self.home = Path(HOME_MOUNT) if contract.get('snapshot_worker_home') else None
        self.records = Path(reports) / 'deterministic'  # where the scorer may keep its own records
        self.records.mkdir(exist_ok=True)

    def asset(self, name):
        for row in self.contract['assets']:
            if row['name'] == name and 'verifier' in row['stages']:
                return Path(row['target'])
        raise KeyError('No asset named ' + name + ' is visible to the verifier')


def write_json(path, value):
    atomic_json(path, value)


def jsonl(path):
    """JSON Lines split on '\\n' only: U+2028/U+2029 inside a JSON string never split a row."""
    return [json.loads(line) for line in Path(path).read_bytes().split(b'\n') if line.strip()]


def evaluate(contract=None):
    """The evaluation stage's task code: evaluate(run)."""
    run = Evaluation(contract)
    module = load_hook(run.tests / TASK_CODE, 'breakwater_task')
    module.evaluate(run)


@functools.cache
def load(tests):
    """The task's trusted code in the verifier (tests/breakwater_task.py), loaded once for its scorer and its views."""
    return load_hook(Path(tests) / TASK_CODE, 'breakwater_task')


def review_views(module, scoring):
    """The task's review views as [(pattern, view(path) -> str)], each pattern under external/task/ or evaluation/."""
    declare = getattr(module, 'review_views', None)
    if declare is None:
        return []
    views = declare(scoring)
    require(isinstance(views, list) and all(
        isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str) and row[0].startswith(VIEW_ROOTS)
        and '..' not in row[0].split('/') and callable(row[1]) for row in views),
            'review_views returns [(pattern, view)], each pattern under external/task/ or evaluation/')
    return [(pattern, functools.partial(view, scoring=scoring)) for pattern, view in views]


def facts(scoring, module):
    """The task's mechanical facts (its optional mechanical(scoring)), taken before its scorer runs: they are recorded
    even when the scorer fails or the attempt is void."""
    require(read_json('/evaluation/complete.json').get('complete') is True, 'Intermediate evaluation did not complete')
    mechanical = getattr(module, 'mechanical', None)
    result = mechanical(scoring) if mechanical else None
    require(result is None or isinstance(result, dict), 'Invalid task mechanical facts')
    return result or {}


def score(scoring, module):
    """The verifier's deterministic score: the task code's score(scoring), checked for shape. Raises Void when the task
    could not measure the attempt."""
    require(read_json('/evaluation/complete.json').get('complete') is True, 'Intermediate evaluation did not complete')
    result = module.score(scoring)
    require(isinstance(result, dict) and set(result) <= {'reward', 'details', 'validated', 'deferred'},
            'The task scorer returned an unexpected result')
    reward = result.get('reward')
    if result.get('deferred') is not None:
        deferred = result['deferred']
        require(reward is None and isinstance(deferred, dict) and isinstance(deferred.get('scored_by'), str)
                and isinstance(deferred.get('rule'), str), 'A deferred score names its scorer and rule only')
    else:
        import math
        require(type(reward) in (int, float) and math.isfinite(reward), 'Invalid deterministic score')
    validated = result.get('validated') or []
    require(isinstance(validated, list) and all(isinstance(item, str) and item for item in validated),
            'Invalid list of validated checks')
    require(result.get('details') is None or isinstance(result['details'], dict), 'Invalid score details')
    return result


def worker_environment(contract):
    """The environment submitted code runs with in the evaluation stage (task-declared, else a minimal one). It always
    names the workdir (BREAKWATER_WORKDIR), where Docker's worker launcher starts submitted code, unless the task sets
    it."""
    declared = contract.get('worker_env')
    if declared:
        return {'BREAKWATER_WORKDIR': contract['workdir'], **declared}
    return {**GENERIC_ENV, 'BREAKWATER_WORKDIR': contract['workdir']}


# For tasks whose evaluation runs no submitted code, or declares no worker environment.
GENERIC_ENV = {'PATH': '/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp/worker',
               'TMPDIR': '/tmp/worker', 'PYTHONPATH': '/opt/breakwater-drivers', 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
               'PYTHONDONTWRITEBYTECODE': '1'}
