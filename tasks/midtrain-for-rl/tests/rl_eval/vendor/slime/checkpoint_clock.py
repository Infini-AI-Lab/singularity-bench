"""Wall-clock saves at completed rollout boundaries; no catch-up save bursts."""
import json
import os
from pathlib import Path
import time
import subprocess
import sys


from wall_clock import TrainingClock as CheckpointClock


def mark_completed_checkpoint(args, rollout_id):
    checkpoint = Path(args.save_hf.format(rollout_id=rollout_id)).resolve()
    # Called only after all actor ranks finish the synchronous HF + optimizer save.
    marker = checkpoint / 'SAVE_COMPLETE'
    marker.write_text(f'rollout_id={rollout_id}\n')
    pointer = checkpoint.parent / 'latest_checkpoint.json'
    pending = pointer.with_suffix('.json.tmp')
    pending.write_text(json.dumps({'checkpoint_path': str(checkpoint),
                                    'rollout_id': rollout_id}, indent=2) + '\n')
    os.replace(pending, pointer)
    return checkpoint


def publish_completed_checkpoint(args, rollout_id, validate=None):
    checkpoint = mark_completed_checkpoint(args, rollout_id)
    if validate is not None:
        try:
            validate()
        except Exception as exc:
            print(f'Holdout evaluation failed for {checkpoint}: {exc}; keeping prior best.', flush=True)
            return
    handoff_script = os.environ.get('SLIME_HANDOFF_SCRIPT')
    if handoff_script:
        # Synchronous, approximately hourly. The external budget supervisor
        # also bounds copying/checking/loading; failures preserve prior handoff.
        result = subprocess.run([sys.executable, handoff_script,
            '--output', str(checkpoint.parent), '--candidate', str(checkpoint),
            '--base', args.hf_checkpoint,
            '--policy', 'best'])
        if result.returncode:
            print(f'Handoff validation/publication failed for {checkpoint}; '
                  'inspect the handoff pointer; training recovery checkpoint is still available.', flush=True)
