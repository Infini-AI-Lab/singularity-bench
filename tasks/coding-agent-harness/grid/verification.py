"""A grid cell's verification task, as Breakwater's verifier runs it (verify/tests/breakwater_task.py)."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# What Breakwater's digest of a task's tests leaves out (contract.tests_digest); a Docker verifier image copies the same
# directories, so both digests agree.
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc')


def materialize(run, source):
    """run/verification-task: the downstream task's instruction, and as tests the cell's task code and rubric
    (verify/tests) with the data-only scorer and its adapters (downstream/)."""
    task = Path(run) / 'verification-task'
    task.mkdir(mode=0o700)
    shutil.copyfile(Path(source) / 'instruction.md', task / 'instruction.md')
    shutil.copytree(HERE / 'verify' / 'tests', task / 'tests', ignore=IGNORE)
    shutil.copytree(HERE / 'downstream', task / 'tests' / 'downstream', ignore=IGNORE)
    return task


# What a cell records: its agent (the submitted harness, started by a trusted launcher) reaches its model through the
# driver gateway only, and its native output is the protected stdout and stderr.
RECORDING = {'author_gateway': False, 'service_gateways': {'driver-gateway': 'driver-gateway'}, 'native': 'streams'}
