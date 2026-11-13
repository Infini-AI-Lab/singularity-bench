import ray
import time
from checkpoint_clock import CheckpointClock, mark_completed_checkpoint, publish_completed_checkpoint

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    release_train = args.release_train
    checkpoint_clock = CheckpointClock.from_environment()

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        saved_actor = False
        stop_for_deadline = checkpoint_clock.stopping(time.monotonic())
        if release_train or checkpoint_clock.due(time.monotonic(), final=rollout_id == args.num_rollout - 1):
            force_sync = True
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            if actor_trains and args.save_hf:
                mark_completed_checkpoint(args, rollout_id)
                saved_actor = True

        ray.get(rollout_manager.cleanup_rollout_data.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if saved_actor:
            # Save and engine update both contain this rollout's post-training weights.
            # Mark the recovery checkpoint before evaluating so a deadline interruption
            # preserves recovery while leaving the previous selected best untouched.
            publish_completed_checkpoint(args, rollout_id,
                validate=lambda: ray.get(rollout_manager.validate_checkpoint.remote(rollout_id)))
            checkpoint_clock.saved(time.monotonic())

        if stop_for_deadline or checkpoint_clock.stopping(time.monotonic()):
            break

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
