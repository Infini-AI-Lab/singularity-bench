"""Slime evaluation only: AMC-83 avg@4, HumanEvalPlus once, GPQA Diamond unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXPECTED = {'amc': (83, 4), 'humanevalplus': (164, 1), 'gpqa_diamond': (198, 1)}
# recipe.json keys that only the timed training run uses.
TIMED_KEYS = ('time_budget_seconds', 'checkpoint_interval_seconds', 'final_save_reserve_seconds')


def jsonl_rows(path):
    # Unicode separators can occur inside JSON strings; only newline delimits rows.
    with Path(path).open() as stream:
        return [line.rstrip('\n') for line in stream if line.strip()]


def summarize(data, expected=EXPECTED):
    if set(data) != set(expected):
        raise ValueError('Evaluation must finish every configured suite')
    scores = {}
    for name, (items, repeats) in expected.items():
        rewards = data[name]['rewards']
        if len(rewards) != items * repeats or any(v not in (0.0, 1.0) for v in rewards):
            raise ValueError(f'{name}: incomplete evaluation or invalid rewards')
        scores[name] = {'items': items, 'samples_per_item': repeats,
                        'responses': len(rewards), 'score': sum(rewards) / len(rewards),
                        'metric': 'avg@4' if name == 'amc' else 'accuracy@1'}
    return {'suites': scores, 'final_reward': sum(v['score'] for v in scores.values()) / len(scores),
            'formula': '(amc_avg_at_4 + humanevalplus_accuracy_at_1 + gpqa_diamond_accuracy_at_1) / 3',
            'score_scale': '0 to 1', 'backend': 'Slime native evaluation rollouts + SGLang'}


def evaluation_command(checkpoint, data, output):
    """Slime arguments for an evaluation-only launch on two GPUs with recipe.json's settings.

    The shared launcher receives a fixed rollout count in place of the wall-clock budget, and
    --num-rollout is then set to 0, so slime runs only the evaluation rollouts.
    """
    import launch
    recipe = {key: value for key, value in launch.load_recipe(ROOT / 'recipe.json').items() if key not in TIMED_KEYS}
    recipe['steps'] = 10
    command = launch.train_args(recipe, checkpoint, data, output, gpus=2)
    command[command.index('--num-rollout') + 1] = '0'
    # GPQA includes prompts longer than the training mixture's 2K input cap.
    # Allocate 4K input + 16K response without dropping benchmark questions.
    for option in ('--rollout-max-context-len', '--sglang-context-length'):
        command[command.index(option) + 1] = '20480'
    return command


def generate_and_save(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout
    if not evaluation:
        raise ValueError('This hook is evaluation-only')
    result = generate_rollout(args, rollout_id, data_source, evaluation=True)
    expected = json.loads(os.environ['MIDTRAIN_EVAL_EXPECTED'])
    output = Path(os.environ['MIDTRAIN_EVAL_OUTPUT'])
    report = summarize(result.data, expected)
    for name, suite in result.data.items():
        with (output / f'{name}.generations.jsonl').open('w') as stream:
            for sample in suite['samples']:
                stream.write(json.dumps({'index': sample.index, 'metadata': sample.metadata,
                    'response': sample.response, 'reward': sample.reward,
                    'response_length': sample.response_length}, ensure_ascii=False) + '\n')
    report['checkpoint'] = args.hf_checkpoint
    pending = output / 'summary.json.tmp'
    pending.write_text(json.dumps(report, indent=2) + '\n')
    pending.replace(output / 'summary.json')
    print(json.dumps(report), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-limit', type=int, help='Integration probe only; never produces a final benchmark reward')
    cli = parser.parse_args()
    if cli.smoke_limit is not None and cli.smoke_limit < 1:
        parser.error('--smoke-limit must be positive')
    manifest = json.loads((cli.data / 'manifest.json').read_text())
    cli.output.mkdir(parents=True, exist_ok=False)
    expected, datasets = {}, []
    for name, (items, repeats) in EXPECTED.items():
        source = cli.data / f'{name}.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['datasets'][name]['sha256']:
            raise ValueError(f'{name}: dataset hash mismatch')
        rows = jsonl_rows(source)
        if len(rows) != items:
            raise ValueError(f'{name}: wrong question count')
        if cli.smoke_limit:
            rows = rows[:cli.smoke_limit]
            source = cli.output / f'{name}.probe.jsonl'
            source.write_text('\n'.join(rows) + '\n')
        expected[name] = (len(rows), repeats)
        datasets.append({'name': name, 'path': str(source.resolve()),
                         'n_samples_per_eval_prompt': repeats})
    config = {'eval': {'defaults': {'input_key': 'prompt', 'label_key': 'label',
        'metadata_key': 'metadata', 'custom_rm_path': 'rewards.reward',
        'temperature': 0.6, 'top_p': 0.95, 'top_k': -1, 'max_response_len': 16384,
        'apply_chat_template': False}, 'datasets': datasets}}
    config_path = cli.output / 'eval-config.json'
    config_path.write_text(json.dumps(config, indent=2))
    os.environ['MIDTRAIN_EVAL_OUTPUT'] = str(cli.output.resolve())
    os.environ['MIDTRAIN_EVAL_EXPECTED'] = json.dumps(expected)
    command = evaluation_command(cli.checkpoint.resolve(), cli.data / 'amc.jsonl', cli.output.resolve())
    from slime.utils.arguments import parse_args
    sys.argv = [sys.argv[0], *command, '--debug-rollout-only', '--eval-interval', '1',
                '--eval-config', str(config_path.resolve()), '--eval-function-path', 'evaluate.generate_and_save',
                '--eval-max-prompt-len', '4096', '--eval-max-context-len', '20480']
    args = parse_args()
    import ray
    from slime.observability.logging_utils import configure_logger
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager
    ray.init(address='local', num_cpus=min(32, len(os.sched_getaffinity(0))), num_gpus=2,
             object_store_memory=4 * 1024**3, include_dashboard=False,
             _temp_dir=tempfile.mkdtemp(prefix='midtrain-eval-', dir='/tmp'))
    manager = None
    try:
        configure_logger()
        groups = create_placement_groups(args)
        manager, _ = create_rollout_manager(args, groups['rollout'])
        ray.get(manager.eval.remote(0))
    finally:
        try:
            if manager is not None:
                ray.get(manager.dispose.remote())
        finally:
            ray.shutdown()
    summary = cli.output / 'summary.json'
    report = json.loads(summary.read_text())
    report['dataset_manifest'] = manifest
    report['integration_probe'] = bool(cli.smoke_limit)
    if cli.smoke_limit:
        report['probe_mean'] = report.pop('final_reward')
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
