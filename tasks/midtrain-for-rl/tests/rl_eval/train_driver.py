"""Shared Slime GRPO driver: the time-budgeted RL loop, or a fixed number of rollouts when no budget is set."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

# In graded runs this directory and the vendored source come from the trusted tests/ tree.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor/slime'))
from wall_clock import TrainingClock


def allocated_gpu_count(args):
    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    return max(actor_gpus, args.rollout_num_gpus) if args.colocate else actor_gpus + args.rollout_num_gpus


def atomic_json(path, value):
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def run_training_loop(args, actor, manager, wait, *, clock=None, now=None):
    now = now or time.monotonic
    if clock is None and os.environ.get('MIDTRAIN_RL_TIME_BUDGET_SECONDS'):
        clock = TrainingClock.from_environment()
    if clock is not None and not math.isfinite(clock.deadline):
        raise ValueError('Time-budgeted RL requires a finite supervised deadline')
    output = Path(args.save_hf).parent
    last_step = last_saved = -1
    latest = None
    reason = 'step_limit'

    def update_rollout():
        if args.offload_rollout:
            wait(manager.onload_weights.remote())
        actor.update_weights()
        if args.offload_rollout:
            wait(manager.onload_kv.remote())

    def save(step):
        nonlocal last_saved, latest
        actor.save_model(step, force_sync=True)
        checkpoint = Path(args.save_hf.format(rollout_id=step)).resolve()
        elapsed = now() - clock.start if clock else None
        # A save finishing after the hard budget is never made eligible.
        if clock and elapsed > clock.deadline - clock.start:
            raise TimeoutError('Checkpoint save exceeded the RL time budget')
        (checkpoint / 'SAVE_COMPLETE').write_text(f'optimizer_steps={step + 1}\n')
        latest = {'checkpoint_path':str(checkpoint), 'optimizer_steps':step + 1,
                  'mode':'wall_clock' if clock else 'steps',
                  'time_budget_seconds':clock.deadline-clock.start if clock else None,
                  'checkpoint_elapsed_seconds':elapsed}
        atomic_json(output / 'latest_checkpoint.json', latest)
        last_saved = step
        if clock:
            clock.saved(now())

    update_rollout()
    for rollout_id in range(args.num_rollout):
        if clock and clock.stopping(now()):
            if last_step < 0:
                raise RuntimeError('RL budget exhausted before any complete optimizer update')
            if last_saved != last_step:
                if args.offload_rollout:
                    wait(manager.offload.remote())
                save(last_step)
            reason = 'wall_clock_reserve'
            break
        data_ref = wait(manager.generate.remote(rollout_id))
        if args.offload_rollout:
            wait(manager.offload.remote())
        wait(actor.async_train(rollout_id, data_ref))
        last_step = rollout_id
        ending = rollout_id == args.num_rollout - 1
        stopping = clock is not None and clock.stopping(now())
        if (clock and clock.due(now(), final=ending)) or (clock is None and ending):
            save(rollout_id)
        wait(manager.cleanup_rollout_data.remote(rollout_id))
        if stopping or (clock is not None and clock.stopping(now())):
            if last_saved != last_step:
                save(last_step)
            reason = 'wall_clock_reserve'
            break
        if ending:
            break
        if not args.offload_train:
            actor.clear_memory()
        update_rollout()
    if latest is None:
        raise RuntimeError('No complete RL checkpoint was saved')
    complete = dict(latest, stop_reason=reason,
                    elapsed_seconds=now()-clock.start if clock else None)
    atomic_json(output / 'training_complete.json', complete)
    return complete


def main():
    import ray
    from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from slime.utils.arguments import parse_args

    args = parse_args()
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        raise ValueError('One complete rollout batch must equal one optimizer batch')
    if args.start_rollout_id != 0 or args.use_critic or args.release_train:
        raise ValueError('Expected fresh GRPO without critic or release-train')
    if args.colocate and not (args.offload_train and args.offload_rollout):
        raise ValueError('Colocation requires both actor and rollout offloading')
    total_gpus = allocated_gpu_count(args)
    ray.init(address='local', num_cpus=min(32, len(os.sched_getaffinity(0))), num_gpus=total_gpus,
             object_store_memory=4 * 1024**3, include_dashboard=False,
             _temp_dir=tempfile.mkdtemp(prefix='midtrain-rl-', dir='/tmp'))
    manager = None
    try:
        configure_logger()
        init_tracking(args)
        groups = create_placement_groups(args)
        manager, _ = create_rollout_manager(args, groups['rollout'])
        actor, _ = create_training_models(args, groups, manager)
        run_training_loop(args, actor, manager, ray.get)
        finish_tracking(args)
    finally:
        try:
            if manager is not None:
                ray.get(manager.dispose.remote())
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()
