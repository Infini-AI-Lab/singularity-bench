"""A grid cell's verification task: the task code Breakwater's verifier runs in a cell's verifier stage.

A cell's launcher (cell.py, apptainer_cell.py) materializes it as the run's verification task (verification.py): the
downstream task's instruction, and as tests this module, the cell rubric (submission_judge.md) and the data-only scorer
with its per-task adapters (downstream/). The launcher seals the cell record as the run's task metadata.

- score: the cell's reward, recomputed from the trusted evaluation's raw measurements (downstream/cell_score.py). A
  grader that could not measure (an unavailable, wrong or no-faster oracle) leaves the cell void, never zero.
- mechanical: the overlay that crossed to evaluation is exactly the recorded workspace change set, byte for byte, and the
  recorded cell matches the sealed one.
- review_views: the trusted workspace inventories and the kernel transport's tensor payloads, as summaries.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'downstream'))
from contract import digest, read_json, regular, require  # noqa: E402  (Breakwater, beside the verifier)
from task_api import Void, write_json  # noqa: E402


def score(scoring):
    """The recomputed reward; the recomputation is shown to the judge and kept as reports/deterministic/cell-score.json."""
    from cell_score import CellVoid, score as cell_score
    try:
        measured = cell_score(scoring.metadata['task'], scoring.raw)
    except CellVoid as void:
        raise Void(str(void)) from void
    write_json(scoring.records / 'cell-score.json', measured)
    return {'reward': measured['reward'], 'details': {'cell_score': measured}}


def mechanical(scoring):
    """The overlay is exactly the recorded change set; the recorded cell is the sealed one."""
    cell = dict(scoring.metadata)
    evidence, submission = scoring.evidence / 'task', Path(scoring.submission)
    changes = read_json(regular(submission, 'changes.json'))
    final = read_json(regular(evidence, 'cell-workspace-final.json'))
    baseline = read_json(regular(evidence, 'cell-workspace-baseline.json'))
    require(changes.get('workdir') == cell['workdir'], 'Overlay workspace differs from the cell record')
    carried = sorted(changes['added'] + changes['modified'])
    present = sorted(path.relative_to(submission / 'files').as_posix()
                     for path in (submission / 'files').rglob('*') if path.is_file())
    require(present == carried, 'Overlay files differ from the recorded change set')
    for name in carried:
        require(digest(regular(submission / 'files', name)) == final['files'][name]['sha256'], 'Overlay file hash mismatch')
    for name in changes['modified']:
        require(name in baseline['files'], 'A modified file was not in the pristine workspace')
    for name in changes['deleted']:
        require(name in baseline['files'] and name not in final['files'], 'Inconsistent deletion record')
    recorded = read_json(regular(evidence, 'cell.json'))
    require({key: recorded[key] for key in recorded if key != 'harness_files'} == cell, 'Cell record differs from the sealed one')
    return {'overlay': {key: len(changes[key]) for key in ('added', 'modified', 'deleted', 'executable')},
            'ignored_cache_changes': len(final.get('ignored_cache_changes', [])),
            'symlink_changes': {'added_or_changed': final.get('symlinks_added_or_changed', {}),
                                'removed': final.get('symlinks_removed', [])},
            'harness_inventory_sha256': cell['harness_inventory_sha256'], 'driver': cell['driver'],
            'driver_probe_status': cell.get('driver_probe_status')}


def review_views(scoring):
    return [('external/task/cell-workspace-*.json', inventory_view), ('evaluation/*.bin', tensor_view)]


def inventory_view(path, scoring):
    """A trusted workspace inventory the launcher recorded: its counts and records, not every file's digest."""
    value = read_json(path)
    files = value.get('files') if isinstance(value.get('files'), dict) else {}
    return json.dumps({'kind': 'trusted_workspace_inventory', 'file': path.name, 'files': len(files),
                       **{key: item for key, item in value.items() if key != 'files'}}, sort_keys=True, default=str)


def tensor_view(path, scoring):
    """A tensor payload of the kernel transport (worker/tensor_files.py): its layout is in the tree.json beside it."""
    if not (path.parent / 'tree.json').is_file():
        return None  # not a tensor payload: shown as it is
    return json.dumps({'kind': 'tensor_payload', 'tree': 'tree.json',
                       'note': 'tensor bytes are not expanded; tree.json beside it is reviewed as it is'})
