"""Shared Slime RL: CHECKPOINT OUTPUT, fixed operator recipe and private data."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from completion import validate_completion

ROOT = Path(__file__).resolve().parent
SLIME = ROOT / 'vendor/slime'
ROLLOUT_SAFETY_CEILING = 1_000_000  # A guard only; the production stopping rule is elapsed time.
GPUS = 4  # The GB300 node's four GPUs; by default the actor and the rollout engines share all of them.


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    for key in ('batch_size', 'responses_per_prompt', 'max_input_tokens', 'max_generation_tokens'):
        if type(recipe.get(key)) is not int or recipe[key] < 1:
            raise ValueError(f'{key} must be a positive integer in the shared recipe')
    if 'time_budget_seconds' in recipe:
        if 'steps' in recipe:
            raise ValueError('A recipe sets either a wall-clock budget or a fixed number of steps, not both')
        for key in ('time_budget_seconds', 'checkpoint_interval_seconds', 'final_save_reserve_seconds'):
            if type(recipe.get(key)) not in (int, float) or not math.isfinite(recipe[key]) or recipe[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if recipe['final_save_reserve_seconds'] >= recipe['time_budget_seconds']:
            raise ValueError('Final save reserve must be shorter than the training budget')
        if recipe['lr_schedule'] != 'linear':
            raise ValueError('The timed recipe implements linear wall-clock LR decay')
    elif type(recipe.get('steps')) is not int or recipe['steps'] < 1:
        raise ValueError('A recipe without a wall-clock budget requires a positive number of steps')
    if recipe['batch_unit'] != 'prompts':
        raise ValueError('batch_size means prompts; every prompt has responses_per_prompt trajectories')
    if recipe.get('reward') != 'binary_correctness' or recipe.get('require_think_format'):
        raise ValueError('This reward hook implements binary correctness without a thinking-format gate')
    if recipe['reference_policy'] != 'frozen_submission':
        raise ValueError('this implementation uses a frozen copy of the submitted policy')
    return recipe


def train_args(recipe, checkpoint, data, output, gpus=GPUS, placement='colocate'):
    timed = 'time_budget_seconds' in recipe
    rollout_limit = ROLLOUT_SAFETY_CEILING if timed else recipe['steps']
    prompts, samples = recipe['batch_size'], recipe['responses_per_prompt']
    length = recipe['max_input_tokens'] + recipe['max_generation_tokens']
    total_gpus = gpus
    actor_gpus = rollout_gpus = total_gpus if placement == 'colocate' else total_gpus // 2
    # Architecture of the pinned Llama-3.2-1B Base; embeddings stay tied by default.
    model_args = [
        '--swiglu', '--num-layers', '16', '--hidden-size', '2048',
        '--ffn-hidden-size', '8192', '--num-attention-heads', '32',
        '--group-query-attention', '--num-query-groups', '8',
        '--max-position-embeddings', '131072', '--use-rotary-position-embeddings',
        '--disable-bias-linear', '--normalization', 'RMSNorm', '--norm-epsilon', '1e-5',
        '--rotary-base', '500000', '--vocab-size', '128256', '--kv-channels', '64',
        '--use-rope-scaling', '--rotary-scaling-factor', '32.0',
    ]
    return [
        '--actor-num-nodes', '1', '--actor-num-gpus-per-node', str(actor_gpus),
        '--rollout-num-gpus', str(rollout_gpus), '--rollout-num-gpus-per-engine', '1',
        '--num-gpus-per-node', str(total_gpus),
        *(['--colocate'] if placement == 'colocate' else []),
        *model_args,
        '--hf-checkpoint', str(checkpoint), '--load', str(checkpoint), '--ref-load', str(checkpoint),
        '--save', str(output / 'megatron'), '--save-interval', str(rollout_limit),
        '--save-hf', str(output / 'hf_step_{rollout_id}'),
        '--no-save-optim', '--no-load-optim', '--no-load-rng', '--finetune',
        '--prompt-data', str(data), '--input-key', 'prompt', '--label-key', 'label',
        '--metadata-key', 'metadata', '--rollout-shuffle',
        '--custom-rm-path', 'rewards.reward',
        '--num-rollout', str(rollout_limit), '--rollout-batch-size', str(prompts),
        '--n-samples-per-prompt', str(samples), '--global-batch-size', str(prompts * samples),
        '--micro-batch-size', '1', '--seed', str(recipe['seed']), '--rollout-seed', str(recipe['seed']),
        '--rollout-max-prompt-len', str(recipe['max_input_tokens']),
        '--rollout-max-response-len', str(recipe['max_generation_tokens']),
        '--rollout-max-context-len', str(length), '--sglang-context-length', str(length),
        '--rollout-temperature', str(recipe['temperature']), '--rollout-top-p', str(recipe['top_p']),
        '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1', '--seq-length', str(length),
        '--bf16', '--attention-backend', 'fused', '--use-distributed-optimizer',
        '--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1',
        '--attention-dropout', '0', '--hidden-dropout', '0',
        '--advantage-estimator', recipe['algorithm'], '--use-kl-loss',
        '--kl-loss-type', 'low_var_kl', '--kl-loss-coef', str(recipe['kl_loss_coefficient']),
        '--eps-clip', str(recipe['clip_low']), '--eps-clip-high', str(recipe['clip_high']),
        '--entropy-coef', str(recipe['entropy_coefficient']),
        '--optimizer', 'adam', '--lr', str(recipe['learning_rate']), '--min-lr', '0',
        '--lr-decay-style', 'constant' if timed else recipe['lr_schedule'], '--lr-warmup-iters', '0',
        '--lr-decay-iters', str(rollout_limit),
        '--adam-beta1', str(recipe['adam_beta1']), '--adam-beta2', str(recipe['adam_beta2']),
        '--weight-decay', str(recipe['weight_decay']), '--clip-grad', '1.0',
        '--sglang-mem-fraction-static', '0.6' if placement == 'colocate' else '0.7',
        *(['--custom-megatron-before-train-step-hook-path', 'wall_clock.wall_clock_lr'] if timed else []),
    ]


def select_checkpoint(output, recipe, returncode, elapsed):
    """A hard stop may recover a previous complete save, never a partial later one."""
    timed = 'time_budget_seconds' in recipe
    if returncode not in ((0, 124) if timed else (0,)):
        raise RuntimeError(f'RL process failed with exit code {returncode}')
    receipt = output / ('latest_checkpoint.json' if returncode == 124 else 'training_complete.json')
    complete = json.loads(receipt.read_text())
    if returncode == 124:
        complete.update(stop_reason='hard_deadline', elapsed_seconds=elapsed, supervisor_exit_code=124)
    validate_completion(complete, recipe)
    final = Path(complete['checkpoint_path'])
    if (final.is_symlink() or not final.is_dir() or final.parent.resolve() != output.resolve() or
            not (final / 'SAVE_COMPLETE').is_file()):
        raise RuntimeError('RL did not publish a complete checkpoint in this run')
    marker = (final / 'SAVE_COMPLETE').read_text()
    if marker != f"optimizer_steps={complete['optimizer_steps']}\n":
        raise RuntimeError('Checkpoint completion marker disagrees with the run receipt')
    return final, complete


def validate_data(data, recipe, base):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    manifest = json.loads(data.with_name('manifest.json').read_text())
    checksum = hashlib.sha256(data.read_bytes()).hexdigest()
    if checksum != manifest['sha256'] or manifest['counts'] != recipe['counts']:
        raise ValueError('RL data differs from its fixed mixture manifest')
    counts, ids = Counter(), set()
    with data.open() as stream:
        for line in stream:
            row = json.loads(line)
            domain = row['metadata']['domain']
            if domain not in recipe['counts'] or row['id'] in ids:
                raise ValueError('unknown domain or duplicate RL prompt')
            ids.add(row['id'])
            counts[domain] += 1
            if len(tokenizer.encode(row['prompt'], add_special_tokens=False)) > recipe['max_input_tokens']:
                raise ValueError('RL mixture contains an input longer than 2K tokens')
    if counts != recipe['counts']:
        raise ValueError(f'incorrect RL counts: {dict(counts)}')
    return checksum


def canonical_checkpoint(submission, base, destination):
    # Validation code is verifier-owned, never imported from the agent workspace.
    from checkpoint_check import validate
    report = validate(submission, base)
    if not report['valid']:
        raise ValueError(f'invalid submitted checkpoint: {report["errors"]}')
    destination.mkdir()
    for path in base.iterdir():
        if path.is_file() and path.suffix in ('.json', '.model', '.txt') and 'safetensors' not in path.name:
            shutil.copy2(path, destination / path.name)
    for path in submission.iterdir():
        if path.suffix == '.safetensors' or path.name == 'model.safetensors.index.json':
            shutil.copy2(path, destination / path.name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base', type=Path, default=Path('/opt/base-model'))
    parser.add_argument('--data', type=Path, default=Path('/opt/midtrain-rl-data/train.jsonl'))
    parser.add_argument('--recipe', type=Path, default=ROOT / 'recipe.json')
    parser.add_argument('--placement', choices=('colocate', 'separate'), default='colocate',
                        help='share all GPUs between rollout and training with offloading (default)')
    parser.add_argument('--dry-run', action='store_true', help='print the shared command without GPU imports or writes')
    parser.add_argument('--eval-data', type=Path, help='Run the three Slime benchmarks immediately after training')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    args.output = args.output.resolve()
    canonical = args.output / 'input_checkpoint'
    command = train_args(recipe, canonical, args.data.resolve(), args.output, placement=args.placement)
    if args.dry_run:
        workers = GPUS if args.placement == 'colocate' else GPUS // 2
        layout = f'actor DP{workers}/TP1 + {workers} rollout TP1 engines; {args.placement} on {GPUS} physical GPUs'
        print(json.dumps({'recipe': recipe, 'hardware': 'gb300', 'placement': args.placement, 'layout': layout,
                          'optimizer_steps': recipe.get('steps'), 'time_budget_seconds':recipe.get('time_budget_seconds'),
                          'trajectories_per_step': recipe['batch_size'] * recipe['responses_per_prompt']}))
        print(shlex.join([sys.executable, str(ROOT / 'train_driver.py'), *command]))
        return 0
    from verify_vendor import verify
    verify()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; submissions must not resume another run')
    if not os.environ.get('MIDTRAIN_CODE_SANDBOX_URL'):
        raise RuntimeError('Start the isolated code sandbox and set MIDTRAIN_CODE_SANDBOX_URL')
    request = urllib.request.Request(os.environ['MIDTRAIN_CODE_SANDBOX_URL'].rstrip('/') + '/health')
    if os.environ.get('MIDTRAIN_CODE_SANDBOX_TOKEN'):
        request.add_header('Authorization', 'Bearer ' + os.environ['MIDTRAIN_CODE_SANDBOX_TOKEN'])
    with urllib.request.urlopen(request, timeout=30) as response:
        if json.load(response).get('status') != 'ok':
            raise RuntimeError('code sandbox is not ready')
    data_hash = validate_data(args.data, recipe, args.base)
    args.output.mkdir(parents=True)
    report = canonical_checkpoint(args.checkpoint.resolve(), args.base.resolve(), canonical)
    env = {**os.environ,
           'PYTHONPATH': f'{ROOT}:{SLIME}:/opt/Megatron-LM',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'OMP_NUM_THREADS': '4',
           'NVTE_FLASH_ATTN': '0', 'NVTE_FUSED_ATTN': '1', 'NVTE_UNFUSED_ATTN': '0',
           'WANDB_MODE': 'disabled'}
    for key in ('SLIME_JOB_START_MONOTONIC', 'SLIME_JOB_DEADLINE_MONOTONIC', 'MIDTRAIN_RL_TIME_BUDGET_SECONDS'):
        env.pop(key, None)
    if 'time_budget_seconds' in recipe:
        env.update(MIDTRAIN_RL_TIME_BUDGET_SECONDS=str(recipe['time_budget_seconds']),
                   SLIME_CHECKPOINT_SECONDS=str(recipe['checkpoint_interval_seconds']),
                   SLIME_FINAL_SAVE_RESERVE_SECONDS=str(recipe['final_save_reserve_seconds']))
    # Respect CUDA_VISIBLE_DEVICES; use only this run's allocated GPUs.
    subprocess.run([sys.executable, str(ROOT / 'check_runtime.py')], check=True, env=env)
    (args.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (args.output / 'run.json').write_text(json.dumps({'recipe': recipe, 'hardware': 'gb300', 'placement': args.placement, 'data_sha256': data_hash,
        'checkpoint': str(args.checkpoint.resolve()), 'structural_check': report}, indent=2) + '\n')
    with (args.output / 'train.log').open('w') as log:
        training_started = time.monotonic()
        driver = [sys.executable, str(ROOT / 'train_driver.py'), *command]
        if 'time_budget_seconds' in recipe:
            sys.path.insert(0, str(SLIME))
            from wall_clock import supervise
            returncode = supervise(driver, recipe['time_budget_seconds'], env=env, stdout=log, stderr=subprocess.STDOUT)
        else:
            returncode = subprocess.run(driver, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic() - training_started
    final, complete = select_checkpoint(args.output, recipe, returncode, elapsed)
    (args.output / 'rl_checkpoint.json').write_text(json.dumps(dict(complete, data_sha256=data_hash), indent=2) + '\n')
    print(f'RL_CHECKPOINT_PATH={final}')
    if args.eval_data:
        subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--checkpoint', str(final),
                        '--data', str(args.eval_data.resolve()), '--output', str(args.output / 'evaluation')],
                       check=True, env=env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
