"""The two held-out suites: pinned task sources and images, the shared deadline, trial accounting and cleanup."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import tomllib

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
SUITES = json.loads((HERE / 'suites.lock.json').read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


class Deadline:
    """One shared wall-clock budget, with time reserved for bounded cleanup."""
    def __init__(self, seconds, cleanup_seconds=60):
        if not math.isfinite(seconds) or seconds <= cleanup_seconds:
            raise ValueError('Evaluation budget must exceed the cleanup reserve')
        self.start = time.monotonic()
        self.end = self.start + seconds
        self.work_end = self.end - cleanup_seconds

    def remaining(self, cap=None, cleanup=False):
        remaining = (self.end if cleanup else self.work_end) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Combined evaluation deadline reached')
        return min(remaining, cap) if cap is not None else remaining

    def run(self, command, *, cap=60, cleanup=False, **kwargs):
        return subprocess.run(command, timeout=self.remaining(cap, cleanup), **kwargs)


def file_hashes(path):
    return {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(path.rglob('*')) if f.is_file()}


def validate_dev_split(directory):
    """The manifest of the prepared 655/73 split, checked by the task's dataset_contract.py."""
    spec = importlib.util.spec_from_file_location('rl_swe_agent_dataset_contract', TASK / 'dataset_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(directory)


def pin_image(content, identity):
    """Change only the image field, including Dockerfile-only task configs."""
    before = tomllib.loads(content)
    section = re.search(r'^\[environment\]\s*$', content, flags=re.MULTILINE)
    if section is None:
        replacement = content.rstrip() + '\n\n[environment]\ndocker_image = ' + json.dumps(identity) + '\n'
        before['environment'] = {'docker_image': identity}
        if tomllib.loads(replacement) != before:
            raise ValueError('Image pinning changed other task settings')
        return replacement
    end = re.search(r'^\[', content[section.end():], flags=re.MULTILINE)
    stop = section.end() + end.start() if end else len(content)
    environment = content[section.end():stop]
    if 'docker_image' in before['environment']:
        environment, count = re.subn(r'^docker_image\s*=.*$',
            'docker_image = ' + json.dumps(identity), environment, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected exactly one environment docker_image')
    else:
        environment = '\ndocker_image = ' + json.dumps(identity) + '\n' + environment
    replacement = content[:section.end()] + environment + content[stop:]
    before['environment']['docker_image'] = identity
    if tomllib.loads(replacement) != before:
        raise ValueError('Image pinning changed other task settings')
    return replacement


def prepare_suites(paths, output, deadline, sandbox_env, *, image_manifest=None, image_workers=8,
                   task_platform=None, exclude=None):
    """Copy the exact pinned tasks and resolve each task's image once.

    exclude maps a suite to task names left out of it (the SWE-bench Verified exclusions, and the tasks
    outside the fixed subset). The full pinned suite is still verified, the excluded names must exist in
    it, and the selection is the denominator of that suite's score. image_manifest is the image-cache.json
    written by arm64_tasks.py build: its native image IDs are used as they are, never rebuilt or replaced.
    """
    exclude = {suite: set(names) for suite, names in (exclude or {}).items()}
    if set(exclude) - set(paths):
        raise ValueError('Exclusions name a suite that is not evaluated')
    pinned = json.loads(image_manifest.read_text()) if image_manifest else None
    if task_platform not in (None, 'linux/arm64', 'linux/amd64'):
        raise ValueError('Unsupported sandbox image platform')
    selected = {}
    sources = {}
    work = []
    for suite, source in paths.items():
        deadline.remaining()
        spec = SUITES[suite]
        source = source.resolve()
        if spec.get('kind') == 'dev_split':
            manifest = validate_dev_split(source)
            revision = manifest['revision']
            if revision != spec['revision'] or manifest['dataset'] != spec['repository']:
                raise ValueError('Dev dataset differs from the pinned release')
            available = sorted(manifest['dev_ids'])
            source = source / 'tasks'
        else:
            revision = deadline.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                    check=True, capture_output=True, text=True).stdout.strip()
            if revision != spec['commit']:
                raise ValueError(f'{suite}: source revision differs from suites.lock.json')
            deadline.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--', '.'], check=True)
            untracked = deadline.run(['git', '-C', str(source), 'ls-files', '--others', '--exclude-standard', '.'],
                                     check=True, capture_output=True, text=True).stdout
            if untracked.strip():
                raise ValueError(f'{suite}: untracked files in benchmark source')
            available = sorted(p.parent.name for p in source.glob('*/task.toml'))
        if len(available) != spec['task_count']:
            raise ValueError(f'{suite}: expected {spec["task_count"]} pinned tasks, got {len(available)}')
        excluded = exclude.get(suite, set())
        if not excluded <= set(available):
            raise ValueError(f'{suite}: excluded tasks are not in the pinned suite')
        tasks = [name for name in available if name not in excluded]
        target_root = output / 'tasks' / suite
        target_root.mkdir(parents=True)
        selected[suite] = {'path': str(target_root.resolve()), 'task_names': tasks, 'result_names': {}}
        sources[suite] = {'revision': revision, 'tasks': {}}
        for name in tasks:
            deadline.remaining()
            hashes = file_hashes(source / name)
            sources[suite]['tasks'][name] = hashes
            target = target_root / name
            shutil.copytree(source / name, target)
            task_config = tomllib.loads((target / 'task.toml').read_text())
            result_name = task_config.get('task', {}).get('name', name)
            if result_name in selected[suite]['result_names']:
                raise ValueError(f'{suite}: duplicate declared task name {result_name}')
            selected[suite]['result_names'][result_name] = name
            work.append((suite, name, target, hashes))
    keys = {f'{suite}/{name}' for suite, name, _, _ in work}
    if pinned is not None and set(pinned) != keys:
        raise ValueError('The image manifest must cover exactly the selected tasks of both suites')
    write(output / 'benchmark-source.json', sources)

    def prepare(item):
        suite, name, target, hashes = item
        key = f'{suite}/{name}'
        content = (target / 'task.toml').read_text()
        tag = tomllib.loads(content).get('environment', {}).get('docker_image')
        previous = pinned[key] if pinned is not None else None
        if task_platform == 'linux/arm64' and suite == 'swebench-verified' and previous is None:
            raise ValueError('ARM64 SWE-bench requires the native image-cache.json from '
                             'arm64_tasks.py build; refusing the x86 upstream Dockerfiles')
        if previous and (previous['source_hashes'] != hashes or previous['original_tag'] != tag):
            raise ValueError(f'{key}: task source changed since the image manifest was written')
        identity = previous['image_id'] if previous else tag
        inspected = None
        if identity:
            inspected = deadline.run(['docker', 'image', 'inspect', identity, '--format', '{{.Id}}'],
                                     capture_output=True, text=True, env=sandbox_env)
        if inspected is None or inspected.returncode:
            if previous:
                raise RuntimeError(f'{key}: pinned image missing; refusing substitution')
            with (output / f'image-{suite}-{name}.log').open('w') as log:
                if tag:
                    command = ['docker', 'pull', *(['--platform', task_platform] if task_platform else []), tag]
                    identity = tag
                else:
                    context = target / 'environment'
                    if not (context / 'Dockerfile').is_file():
                        raise ValueError(f'{key}: no Dockerfile or prebuilt image')
                    # Build the task's own context unchanged; image-cache.json records the resulting ID.
                    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
                    identity = f'rl-swe-agent-eval-task:{digest}'
                    command = ['docker', 'build', *(['--platform', task_platform] if task_platform else []), '-t', identity, str(context)]
                deadline.run(command, cap=1800, stdout=log, stderr=subprocess.STDOUT,
                             check=True, env=sandbox_env)
            inspected = deadline.run(['docker', 'image', 'inspect', identity, '--format', '{{.Id}}'],
                                     check=True, capture_output=True, text=True, env=sandbox_env)
        identity = inspected.stdout.strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
            raise ValueError(f'{key}: invalid Docker image identity')
        if task_platform:
            actual_platform = deadline.run(['docker', 'image', 'inspect', identity,
                '--format', '{{.Os}}/{{.Architecture}}'], check=True, capture_output=True,
                text=True, env=sandbox_env).stdout.strip()
            if actual_platform != task_platform:
                raise ValueError(f'{key}: expected {task_platform}, found {actual_platform}; '
                                 'prepare native images with arm64_tasks.py')
        (target / 'task.toml').write_text(pin_image(content, identity))
        entry = {'original_tag': tag, 'image_id': identity, 'source_hashes': hashes,
                 'prepared_config_sha256': hashlib.sha256((target / 'task.toml').read_bytes()).hexdigest()}
        if task_platform:
            entry['platform'] = task_platform
        return key, entry

    with ThreadPoolExecutor(max_workers=image_workers) as pool:
        images = dict(pool.map(prepare, work))
    write(output / 'image-cache.json', images)
    return selected


def summarize_trials(job_dir, task_names, returncode, result_names=None):
    """Never turn missing/error trials into a misleading completed-only score."""
    expected = set(task_names)
    result_names = result_names or {name: name for name in task_names}
    results = {}
    malformed = []
    for path in sorted(job_dir.glob('*/result.json')):
        try:
            result = json.loads(path.read_text())
            name = result_names[result['task_name']]
            if name not in expected or name in results:
                raise ValueError('Unexpected or duplicate task result')
            results[name] = result
        except (ValueError, KeyError) as error:
            malformed.append(f'{path}: {error}')
    finished = {name: row for name, row in results.items() if row.get('finished_at')}
    resolved = []
    errors = {}
    graded = []
    for name, row in finished.items():
        reward = ((row.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        if row.get('exception_info'):
            errors[name] = row['exception_info']['exception_type']
        if isinstance(reward, (int, float)) and math.isfinite(reward) and reward in (0, 1):
            graded.append(name)
            if reward == 1:
                resolved.append(name)
        elif name not in errors:
            errors[name] = 'MissingOrInvalidReward'
    complete = (returncode == 0 and len(finished) == len(expected) and not malformed)
    # Finished agent-error trials may still have a valid verifier reward.
    score_valid = complete and len(graded) == len(expected)
    return {'selected': len(expected), 'finished': len(finished), 'graded': len(graded),
            'resolved': len(resolved), 'errors': errors,
            'unfinished_tasks': sorted(expected - finished.keys()), 'malformed_results': malformed,
            'complete': complete, 'score_valid': score_valid,
            'resolved_rate': len(resolved) / len(expected) if score_valid else None,
            'resolved_fraction_of_selected': len(resolved) / len(expected) if expected else None,
            'harbor_returncode': returncode}


def stop_processes(processes, deadline):
    active = [proc for proc in processes if proc.poll() is None]
    for proc in active:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    grace = min(time.monotonic() + 20, deadline.end)
    while any(proc.poll() is None for proc in active) and time.monotonic() < grace:
        time.sleep(.1)
    for proc in active:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=deadline.remaining(2, cleanup=True))


def cleanup_sandboxes(job_dirs, deadline, sandbox_env):
    # Harbor session IDs are trial_name + '__env'. Match exact Compose labels,
    # never prune the shared daemon or stop unrelated trials.
    def project_name(name):
        name = name.lower()
        if not re.match(r'^[a-z0-9]', name):
            name = '0' + name
        return re.sub(r'[^a-z0-9_-]', '-', name)
    projects = {project_name(trial.name + '__env')
                for job_dir in job_dirs for trial in job_dir.iterdir()
                if trial.is_dir() and (trial / 'config.json').is_file()}
    if not projects:
        return
    for list_args, remove_args in [
        (['ps', '-a'], ['rm', '-f']),
        (['network', 'ls'], ['network', 'rm']),
    ]:
        result = deadline.run(['docker', *list_args, '--format', '{{json .}}'], cap=10,
                              cleanup=True, capture_output=True, text=True, check=True, env=sandbox_env)
        ids = []
        for line in result.stdout.splitlines():
            row = json.loads(line)
            labels = dict(pair.split('=', 1) for pair in row.get('Labels', '').split(',') if '=' in pair)
            if labels.get('com.docker.compose.project') in projects:
                ids.append(row['ID'])
        if ids:
            deadline.run(['docker', *remove_args, *ids], cap=15, cleanup=True,
                         capture_output=True, check=True, env=sandbox_env)
