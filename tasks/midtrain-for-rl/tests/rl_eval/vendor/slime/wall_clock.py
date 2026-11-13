"""CPU-only wall-clock checkpoints, LR scheduling and Linux job supervision."""
import ctypes
import math
import os
import signal
import subprocess
import time


class TrainingClock:
    def __init__(self, start, deadline, interval=3600, reserve=900):
        if (not math.isfinite(start) or math.isnan(deadline) or deadline <= start or
                not math.isfinite(interval) or interval <= 0 or
                not math.isfinite(reserve) or reserve < 0 or reserve >= deadline - start):
            raise ValueError('Invalid wall-clock checkpoint/deadline configuration')
        self.start = start
        self.next_save = start + interval
        self.deadline = deadline
        self.interval = interval
        self.reserve = reserve

    @classmethod
    def from_environment(cls):
        return cls(float(os.environ.get('SLIME_JOB_START_MONOTONIC', time.monotonic())),
                   float(os.environ.get('SLIME_JOB_DEADLINE_MONOTONIC', 'inf')),
                   float(os.environ.get('SLIME_CHECKPOINT_SECONDS', '3600')),
                   float(os.environ.get('SLIME_FINAL_SAVE_RESERVE_SECONDS', '900')))

    def stopping(self, now):
        return now >= self.deadline - self.reserve

    def due(self, now, final=False):
        return final or self.stopping(now) or now >= self.next_save

    def saved(self, now):
        self.next_save = now + self.interval


def synchronized_fraction(value):
    """All actor ranks must use one LR, even when their hooks start at different times."""
    import torch
    import torch.distributed as dist
    if dist.is_initialized():
        device = 'cuda' if dist.get_backend() == 'nccl' else 'cpu'
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.broadcast(tensor, src=0)
        return tensor.item()
    return value


def wall_clock_lr(args, rollout_id, step_id, model, optimizer, scheduler):
    """Apply linear LR decay over elapsed time rather than over a step count.

    Megatron's underlying scheduler must use constant decay. Its per-group maxima
    are refreshed before each update; its normal step() then preserves those LRs.
    """
    clock = TrainingClock.from_environment()
    if not math.isfinite(clock.deadline):
        raise ValueError('Wall-clock LR requires an explicit training deadline')
    fraction = max(0.0, min(1.0, (clock.deadline - time.monotonic()) / (clock.deadline - clock.start)))
    fraction = synchronized_fraction(fraction)
    if not hasattr(scheduler, '_wall_clock_base_lrs'):
        scheduler._wall_clock_base_lrs = [group.get('max_lr', scheduler.max_lr) for group in optimizer.param_groups]
    scheduler.max_lr = args.lr * fraction
    for group, base in zip(optimizer.param_groups, scheduler._wall_clock_base_lrs, strict=True):
        group['max_lr'] = base * fraction
        group['min_lr'] = 0.0
        rate = group['max_lr'] * group.get('lr_mult', 1.0)
        if hasattr(group.get('lr'), 'fill_'):
            group['lr'].fill_(rate)
        else:
            group['lr'] = rate


def kill_job_children():
    import psutil
    parent = psutil.Process()
    stopped = {}
    while True:
        children = [p for p in parent.children(recursive=True) if p.pid not in stopped]
        if not children:
            break
        for child in children:
            stopped[child.pid] = child
            try:
                child.send_signal(signal.SIGSTOP)
            except psutil.NoSuchProcess:
                pass
    for child in reversed(list(stopped.values())):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def supervise(command, seconds, *, env=None, stdout=None, stderr=None):
    """Bound this child and its detached Ray descendants; timeout returns 124."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Training time budget must be positive and finite')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot enable job descendant tracking')
    start = time.monotonic()
    child_env = dict(os.environ if env is None else env)
    child_env.update(SLIME_JOB_START_MONOTONIC=str(start), SLIME_JOB_DEADLINE_MONOTONIC=str(start + seconds))
    child_env.setdefault('SLIME_FINAL_SAVE_RESERVE_SECONDS', str(min(900, seconds / 2)))
    process = subprocess.Popen(command, env=child_env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        return process.wait(timeout=max(0, start + seconds - time.monotonic()))
    except subprocess.TimeoutExpired:
        print(f'{seconds:g}-second hard deadline reached: killing this job and its workers. '
              'Only a completed checkpoint may be used.', flush=True)
        return 124
    finally:
        kill_job_children()
