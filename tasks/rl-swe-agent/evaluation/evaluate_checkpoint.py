#!/usr/bin/env python3
"""Trusted held-out evaluation of a submitted rl-swe-agent checkpoint.

Audits the export against the initialization weights (mini_slime.model.checkpoint from this repository, in a
container without network), serves it with SGLang on the four GPUs, and runs mini-swe-agent 2.4.6 through
Harbor, one attempt per task, on the 73 development tasks and the fixed 100-task SWE-bench Verified subset,
under one four-hour budget that includes setup and cleanup. The result is OUTPUT/summary.json, which score.py
turns into the reward. The submitted training code is never executed, and no oracle answer enters a policy
request.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from suites import Deadline, SUITES, prepare_suites, summarize_trials, stop_processes, cleanup_sandboxes, write

TASK = HERE.parent
LOCK = json.loads((TASK / 'sources.lock.json').read_text())
CONTEXT = 65536  # tokens per request, prompt and output together
MAX_TOKENS = 8192  # generated tokens per model call
EVAL_HOURS = 4  # both suites together, including setup and cleanup
CONCURRENCY = {'rl-swe-agent-dev': 8, 'swebench-verified': 24}
SANDBOX_PLATFORM = 'linux/arm64'  # GB300 hosts; every task image is checked before use
# FlashInfer's SM100 prefill kernel rejects GLM-4.7-Flash's 256-dim heads on GB300; TRT-LLM MLA serves it.
ATTENTION_BACKEND = 'trtllm_mla'
AGENT_VERSION = '2.4.6'


def run_logged(command, log, timeout, env=None, deadline=None):
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=deadline.remaining(timeout) if deadline else timeout, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='The sealed export: RUN_DIR/submission/out/final_checkpoint of the Breakwater run')
    p.add_argument('--base-model', type=Path, default=TASK / 'artifacts/GLM-4.7-Flash',
                   help='The GLM-4.7-Flash initialization weights that the export audit compares against')
    p.add_argument('--dev-data', type=Path, default=TASK / 'data',
                   help='The prepared 655/73 dataset (prepare_inputs.py); only its 73 development tasks are evaluated')
    p.add_argument('--swe-benchmark', type=Path, default=TASK / '.deps/harbor-datasets/datasets/swebench-verified',
                   help='The pinned SWE-bench Verified task directory of the harbor-datasets checkout')
    p.add_argument('--image-manifest', type=Path, required=True,
                   help='image-cache.json from arm64_tasks.py build: the native images of both suites')
    p.add_argument('--harbor', type=Path, default=TASK / '.deps/harbor',
                   help='The pinned Harbor checkout whose harbor CLI runs the trials')
    p.add_argument('--uv', default='uv', help='The uv executable that runs Harbor from its lockfile')
    p.add_argument('--eval-image', default='rl-swe-agent-eval:local',
                   help='The evaluation runtime image (evaluation/Dockerfile), used by the audit and the policy server')
    p.add_argument('--gpus', default='0,1,2,3', help='Four distinct physical GPU indices')
    p.add_argument('--api-host', required=True,
                   help='Address at which the task sandboxes reach the policy server on this host')
    p.add_argument('--port', type=int, default=32200)
    p.add_argument('--sandbox-docker-host', help='Optional separate Docker endpoint for the Harbor sandboxes only')
    p.add_argument('--output', type=Path, required=True, help='A new directory for the evaluation record')
    a = p.parse_args()
    if not re.fullmatch(r'[0-7](,[0-7]){3}', a.gpus) or len(set(a.gpus.split(','))) != 4:
        p.error('--gpus needs exactly four distinct physical GPU indices')
    sandbox_env = dict(os.environ)
    if a.sandbox_docker_host:
        sandbox_env.pop('DOCKER_CONTEXT', None)
        sandbox_env['DOCKER_HOST'] = a.sandbox_docker_host
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    summary = {'context_limit': CONTEXT, 'evaluation_budget_seconds': EVAL_HOURS * 3600,
               'started_utc': datetime.now(timezone.utc).isoformat()}
    serving = 'rl-swe-agent-eval-' + hashlib.sha256(str(a.output).encode()).hexdigest()[:16]
    summary['serving_container'] = serving
    server_proc = None
    audit_containers = []
    jobs = {}
    selected = {}
    budget = None
    def interrupted(signum, frame):
        raise InterruptedError(f'Evaluation interrupted by signal {signum}')
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        budget = Deadline(EVAL_HOURS * 3600)
        checkpoint = a.checkpoint.resolve()
        if not (checkpoint / '.complete').is_file():
            raise ValueError('The checkpoint lacks the .complete marker of a validated mini_slime export')
        summary['checkpoint'] = str(checkpoint)
        runtime_image = budget.run(['docker', 'image', 'inspect', a.eval_image, '--format', '{{.Id}}'],
                                   capture_output=True, text=True, check=True).stdout.strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', runtime_image):
            raise ValueError('Invalid evaluation runtime image ID')
        uv = shutil.which(a.uv)
        if uv is None:
            raise FileNotFoundError(f'uv executable not found: {a.uv}')
        harbor_root = a.harbor.resolve()
        harbor_commit = budget.run(['git', '-C', str(harbor_root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        if harbor_commit != LOCK['harbor']['commit']:
            raise ValueError('Harbor checkout does not match the pinned commit in sources.lock.json')
        harbor_patch = budget.run(['git', '-C', str(harbor_root), 'diff', 'HEAD'],
                                 capture_output=True, check=True).stdout
        swe_tasks = sorted(path.parent.name for path in a.swe_benchmark.glob('*/task.toml'))
        exclusions_file = (HERE / 'swe-exclusions.json').read_bytes()
        exclusions = json.loads(exclusions_file)['tasks']
        if set(exclusions) - set(swe_tasks):
            raise ValueError(f'Excluded tasks not in the benchmark: {sorted(set(exclusions) - set(swe_tasks))}')
        subset_file = (HERE / 'swe-subset.json').read_bytes()
        subset = json.loads(subset_file)
        if set(subset['tasks']) - (set(swe_tasks) - set(exclusions)):
            raise ValueError('SWE subset names excluded or unknown tasks')
        if subset['swe_exclusions_sha256'] != hashlib.sha256(exclusions_file).hexdigest():
            raise ValueError('SWE subset was drawn against different exclusions')
        settings = {'runtime_image_id': runtime_image, 'context_limit': CONTEXT,
                    'harbor_commit': harbor_commit, 'harbor_patch_sha256': hashlib.sha256(harbor_patch).hexdigest(),
                    'max_tokens': MAX_TOKENS, 'agent': 'mini-swe-agent', 'agent_version': AGENT_VERSION,
                    'n_attempts': 1, 'concurrency': CONCURRENCY,
                    'agent_timeouts': {name: spec['agent_timeout_sec'] for name, spec in SUITES.items()},
                    'suites': SUITES, 'sandbox_platform': SANDBOX_PLATFORM, 'attention_backend': ATTENTION_BACKEND,
                    'excluded_tasks': {'swebench-verified': sorted(exclusions)},
                    'excluded_reason': {'swebench-verified': exclusions},
                    'exclusions_sha256': hashlib.sha256(exclusions_file).hexdigest(),
                    'swe_subset': {k: v for k, v in subset.items() if k != 'strata'},
                    'swe_subset_sha256': hashlib.sha256(subset_file).hexdigest()}
        write(a.output / 'eval-settings.json', settings)
        summary['settings'] = settings
        # The export audit runs this repository's checkpoint checker, never the submission's copy.
        audit_name = serving + '-audit'
        audit_containers.append(audit_name)
        run_logged(['docker', 'run', '--rm', '--init', '--name', audit_name,
            '--network', 'none', '--memory', '100g', '--cpus', '4',
            '-v', f'{TASK / "mini_slime"}:/trusted/mini_slime:ro',
            '-v', f'{a.base_model.resolve()}:/base:ro', '-v', f'{checkpoint}:/candidate:ro',
            '-v', f'{a.output}:/report', '-w', '/trusted', '--entrypoint', 'python', runtime_image,
            '-m', 'mini_slime.model.checkpoint', '--base', '/base', '--updated', '/candidate',
            '--report', '/report/export-audit.json'], a.output / 'export-audit.log', 1200, deadline=budget)
        summary['export_audit_pass'] = True
        # SWE-bench Verified tasks outside the fixed subset leave the suite and its denominator, like the exclusions.
        excluded = sorted(set(swe_tasks) - set(subset['tasks']))
        keep = {f'swebench-verified/{name}' for name in subset['tasks']}
        manifest = a.output / 'image-manifest-subset.json'
        write(manifest, {k: v for k, v in json.loads(a.image_manifest.read_text()).items()
                         if not k.startswith('swebench-verified/') or k in keep})
        selected = prepare_suites({'rl-swe-agent-dev': a.dev_data, 'swebench-verified': a.swe_benchmark},
                                  a.output, budget, sandbox_env, image_manifest=manifest,
                                  task_platform=SANDBOX_PLATFORM, exclude={'swebench-verified': excluded})
        command = ['docker', 'run', '--rm', '--init', '--name', serving, '--gpus', f'"device={a.gpus}"',
            '--network', 'host', '--ipc', 'host', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-e', f'EVAL_CONTEXT_LIMIT={CONTEXT}', '-v', f'{checkpoint}:/eval-model:ro',
            '--entrypoint', 'python', runtime_image, '-m', 'sglang.launch_server',
            '--model-path', '/eval-model', '--served-model-name', 'model', '--tp', '1', '--dp-size', '4',
            '--context-length', str(CONTEXT), '--mem-fraction-static', '0.7', '--tool-call-parser', 'glm47',
            '--reasoning-parser', 'glm45', '--cuda-graph-max-bs-decode', '8',
            '--cuda-graph-backend-prefill', 'disabled', '--host', '0.0.0.0', '--port', str(a.port),
            '--attention-backend', ATTENTION_BACKEND]
        with (a.output / 'serve.log').open('w') as stream:
            server_proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        health_deadline = time.monotonic() + 900
        while True:
            budget.remaining()
            if server_proc.poll() is not None:
                raise RuntimeError('Checkpoint server failed to initialize')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2).close()
                break
            except OSError:
                if time.monotonic() >= health_deadline:
                    raise TimeoutError('Checkpoint server health timeout')
                time.sleep(2)
        for suite, dataset in selected.items():
            suite_output = a.output / suite
            suite_output.mkdir()
            config = {'job_name': suite, 'jobs_dir': str(suite_output / 'trials'),
                'n_concurrent_trials': settings['concurrency'][suite], 'n_attempts': 1,
                'datasets': [{'path': dataset['path'], 'task_names': dataset['task_names']}],
                'agents': [{'name': 'mini-swe-agent', 'model_name': 'openai/model',
                    'override_timeout_sec': settings['agent_timeouts'][suite],
                    'kwargs': {'version': AGENT_VERSION, 'max_seq_len': CONTEXT,
                        'max_tokens': settings['max_tokens'],
                        'model_info': {'max_input_tokens': CONTEXT,
                                      'max_output_tokens': settings['max_tokens'],
                                      'input_cost_per_token': 0., 'output_cost_per_token': 0.}},
                    'env': {'OPENAI_API_BASE': f'http://{a.api_host}:{a.port}/v1',
                            'OPENAI_API_KEY': 'dummy', 'MSWEA_COST_TRACKING': 'ignore_errors'}}],
                'environment': {'type': 'docker', 'delete': True, 'force_build': False}}
            config_file = suite_output / 'job-config.json'
            write(config_file, config)
            budget.remaining()
            with (suite_output / 'harbor.log').open('w') as stream:
                jobs[suite] = subprocess.Popen([uv, 'run', '--directory', str(harbor_root),
                    '--frozen', '--python', '3.12', 'harbor', 'run', '--config', str(config_file)],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=sandbox_env)
        while any(job.poll() is None for job in jobs.values()):
            budget.remaining()
            if server_proc.poll() is not None:
                raise RuntimeError('The shared policy server exited during evaluation')
            time.sleep(min(1, budget.remaining()))
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        summary.update(error=f'{type(error).__name__}: {error}',
                       deadline_reached=budget is not None and time.monotonic() >= budget.work_end)
    except Exception as error:
        summary.update(error=f'{type(error).__name__}: {error}')
    finally:
        # Up to 60 seconds inside the four-hour budget are reserved for both jobs and Docker cleanup.
        cleanup_errors = []
        if budget is not None:
            try:
                stop_processes(list(jobs.values()), budget)
            except Exception as error:
                cleanup_errors.append(str(error))
            if server_proc is not None:
                try:
                    budget.run(['docker', 'rm', '-f', serving], cap=10, cleanup=True,
                               capture_output=True, check=True)
                    server_proc.wait(timeout=budget.remaining(2, cleanup=True))
                except Exception as error:
                    cleanup_errors.append(str(error))
            for name in audit_containers:
                try:
                    removed = budget.run(['docker', 'rm', '-f', name], cap=5, cleanup=True,
                                         capture_output=True, text=True)
                    if removed.returncode and 'No such container' not in removed.stderr:
                        cleanup_errors.append(removed.stderr)
                except Exception as error:
                    cleanup_errors.append(str(error))
            job_dirs = [a.output / suite / 'trials' / suite for suite in jobs]
            try:
                cleanup_sandboxes([path for path in job_dirs if path.is_dir()], budget, sandbox_env)
            except Exception as error:
                cleanup_errors.append(str(error))
            summary['evaluation_elapsed_seconds'] = time.monotonic() - budget.start
        summary['benchmarks'] = {}
        for suite, dataset in selected.items():
            job_dir = a.output / suite / 'trials' / suite
            job = jobs.get(suite)
            result = summarize_trials(job_dir, dataset['task_names'], job.returncode if job else None,
                                      dataset['result_names'])
            summary['benchmarks'][suite] = result
            result_path = job_dir / 'result.json'
            if result_path.is_file():
                try:
                    result['harbor_result'] = json.loads(result_path.read_text())
                except ValueError:
                    result['result_manifest_error'] = 'Invalid Harbor job result JSON'
        if cleanup_errors:
            summary['cleanup_errors'] = cleanup_errors
        summary['complete'] = (set(summary['benchmarks']) == set(SUITES)
                               and all(row['complete'] for row in summary['benchmarks'].values())
                               and not summary.get('error') and not cleanup_errors)
        rates = [row['resolved_rate'] for row in summary['benchmarks'].values()]
        summary['macro_resolved_rate'] = (sum(rates) / len(rates)
            if summary['complete'] and all(rate is not None for rate in rates) else None)
        summary['pass'] = summary['complete'] and all(
            not row['errors'] and row['score_valid'] for row in summary['benchmarks'].values())
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(a.output / 'summary.json', summary)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(summary), flush=True)
    return 124 if summary.get('deadline_reached') else (0 if summary['pass'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
