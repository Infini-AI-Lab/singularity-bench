"""Workload and fleet identity bindings, and the absolute three-cell reward.

`aggregate` turns per-cell summaries into the final reward. `hardware_identity` checks the fleet: four
trays of four B300-named GPUs with 16 distinct UUIDs. A suite marked `target-qualified` must also come
with a qualification record binding the workload, the fleet's hardware profile and the driver runtime
(`validate_qualification`), and each graded measurement then records a measurement binding (`binding`). The
released suite is development-provisional and is graded under the task contract's `allow_unqualified`
declaration, so those two paths are not exercised.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import score

QUALIFICATION_SCHEMA = 'serverless-qualification-v5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hardware_identity(nodes):
    if len(nodes) != 4:
        raise ValueError('the graded fleet requires four B300 trays')
    result, ids = [], set()
    for node in nodes:
        devices = node['hardware']['devices']
        gpus = node['gpus']
        if len(devices) != 4 or len(gpus) != 4 or any('B300' not in d['name'] for d in devices):
            raise ValueError('the graded fleet requires four B300 GPUs on every tray')
        if any(g['uuid'] in ids for g in gpus):
            raise ValueError('duplicate physical GPU in the graded fleet')
        ids.update(g['uuid'] for g in gpus)
        result.append({'node': node['node'], 'hardware': node['hardware'],
                       'gpus': [{k: g[k] for k in ('index', 'uuid', 'total_mib')} for g in gpus],
                       'host_ram_gib': node['host_ram']['total_gib'],
                       'host_ram_source': node['host_ram']['source']})
    if len(ids) != 16:
        raise ValueError('the graded fleet requires 16 distinct GPUs')
    return sorted(result, key=lambda n: n['node'])


def driver_runtime():
    return {'python': sys.version, 'architecture': platform.machine(),
            'packages': {p: version(p) for p in ('aiohttp', 'uvloop', 'numpy')}}


def workload_binding(tests_root):
    root = Path(tests_root)
    suite = json.loads((root/'suite-graded.json').read_text())
    return {'score_version': score.SCORE_VERSION,
            'suite_sha256': sha(root/'suite-graded.json'),
            'catalogue_sha256': sha(root/'public/catalogue.json'),
            'build_sha256': sha(root/'build-fingerprint.json'),
            'traces': {c['name']: sha(root/f"trace-graded-{c['name']}.jsonl") for c in suite['cases']}}


def hardware_profile(nodes):
    """Equivalent trays share a profile; hostnames and physical UUIDs are evidence only."""
    profiles = []
    for node in hardware_identity(nodes):
        hardware = node['hardware']
        if not hardware.get('serving_code_sha256'):
            raise ValueError('fleet serving-code identity is missing; re-stage the current fleet image')
        profiles.append({'hardware': {**hardware, 'devices': sorted(
            [{k: v for k, v in d.items() if k != 'uuid'} for d in hardware['devices']], key=digest)},
            'gpu_memory_mib': sorted(g['total_mib'] for g in node['gpus']),
            'host_ram_gib': node['host_ram_gib'], 'host_ram_source': node['host_ram_source']})
    return sorted(profiles, key=digest)


def fleet_image_ids(fleet):
    images = fleet.get('launch', {}).get('images', {})
    if not images or any('sha256:' not in s for s in images.values()):
        raise ValueError('immutable fleet image identities are required')
    return sorted(set(images.values()))


def validate_qualification(tests_root, *, fleet=None, nodes=None, runtime=None):
    root = Path(tests_root)
    artifact = json.loads((root/'qualification.json').read_text())
    if artifact.get('schema') != QUALIFICATION_SCHEMA or artifact.get('workload') != workload_binding(root):
        raise ValueError('qualification record does not match the graded workload or scoring version')
    suite = json.loads((root/'suite-graded.json').read_text())
    if set(suite['qualification'].values()) != {'target-qualified'}:
        raise ValueError('a target-qualified workload is required')
    if not artifact.get('baseline_image') or not artifact.get('driver_image'):
        raise ValueError('qualification record lacks its runtime image identities')
    if fleet is not None:
        # Independently built images may differ in layer metadata. Their serving
        # code and installed runtime are compared through the hardware profile.
        fleet_image_ids(fleet)
    if nodes is not None and hardware_profile(nodes) != artifact['hardware_profile']:
        raise ValueError('fleet configuration differs from the qualified four-tray B300 profile')
    if runtime is not None and runtime != artifact['driver_runtime']:
        raise ValueError('driver runtime differs from the qualification record')
    return artifact


def binding(tests_root, fleet, nodes, *, setup_s=900., retries=2, time_scale=1., runtime=None):
    root = Path(tests_root)
    if time_scale != 1 or setup_s <= 0 or retries != 2:
        raise ValueError('measurement timing configuration differs from the task')
    suite = json.loads((root/'suite-graded.json').read_text())
    if setup_s != suite['setup_s'] or retries != suite['session_retries']:
        raise ValueError('measurement setup/retry budget mismatch')
    images = fleet.get('launch', {}).get('images', {})
    if not images or any('sha256:' not in s for s in images.values()):
        raise ValueError('immutable fleet image identities are required')
    return {'schema': 'serverless-measurement-binding-v5', 'score_version': score.SCORE_VERSION,
            'suite_sha256': sha(root/'suite-graded.json'),
            'catalogue_sha256': sha(root/'public/catalogue.json'),
            'build_sha256': sha(root/'build-fingerprint.json'),
            'traces': {c['name']: sha(root/f"trace-graded-{c['name']}.jsonl") for c in suite['cases']},
            'hardware': hardware_identity(nodes), 'fleet_images': images,
            'driver_runtime': driver_runtime() if runtime is None else runtime,
            'setup_s': setup_s, 'session_retries': retries, 'time_scale': time_scale,
            'policy_resources': {'cpus': 8, 'memory_gib': 32, 'pids': 512},
            'driver_resources': {'cpus': 8, 'memory_gib': 8}}


def aggregate(summaries, labels, expected_models, *, development_only=False):
    if len(summaries) != len(labels) or not summaries:
        raise ValueError('summaries and labels must match')
    groups = {name: [] for name in score.CELLS}
    seen = set()
    rewards = []
    for s, label in zip(summaries, labels):
        name, repeat = label.rsplit('-', 1)
        if name not in groups or not repeat.isdigit() or label in seen:
            raise ValueError('invalid or duplicate cell/repeat label')
        seen.add(label)
        if s.get('score_version') != score.SCORE_VERSION or set(s['model_stats']) != set(expected_models):
            raise ValueError('scoring version or model coverage mismatch')
        if not score.finite(s['utility']) or s['utility'] > 1:
            raise ValueError('utility must be in [0, 1]')
        groups[name].append(s['utility'] if score.gates_pass(s) else 0.)
        rewards.append({'name': label, **score.reward(s), 'utility': s['utility']})
    repeat_sets = [{label.rsplit('-', 1)[1] for label in labels if label.rsplit('-', 1)[0] == n}
                   for n in score.CELLS]
    if not repeat_sets[0] or any(v != repeat_sets[0] for v in repeat_sets):
        raise ValueError('every load cell must have the same complete repeat set')
    cells = {}
    for name, values in groups.items():
        u = math.fsum(values)/len(values)
        cells[name] = {'utility': u, 'reward': u, 'repeats': len(values)}
    tampered = any(s['gates']['tampered'] for s in summaries)
    final = {'reward': 0. if tampered else math.fsum(c['reward'] for c in cells.values())/len(cells),
             'utility': math.fsum(c['utility'] for c in cells.values())/len(cells),
             'served_fraction': math.fsum(s['strict_attainment'] for s in summaries)/len(summaries),
             'completion': math.fsum(s['gates']['completion'] for s in summaries)/len(summaries),
             'gates_pass': not tampered, 'tampered': tampered,
             'cells_passed': sum(score.gates_pass(s) for s in summaries), 'n_traces': len(summaries),
             'score_version': score.SCORE_VERSION,
             'development_only': development_only, 'load_cells': cells}
    worst = min(range(len(summaries)), key=lambda i: summaries[i]['utility'])
    return final, rewards, worst
