"""GRPO group advantages (one group per task and policy version) and the clipped policy loss, masked before exp."""
from collections import defaultdict
import torch

def compute_group_advantages(samples, epsilon=1e-6):
    groups=defaultdict(list)
    for s in samples:
        s.validate()
        groups[s.group_id].append(s)
    for group in groups.values():
        if len(group)<2:
            raise ValueError('GRPO needs at least two samples per group')
        if len({(s.prompt,s.policy_version,s.metadata.get('instance_id')) for s in group}) != 1:
            raise ValueError('Mixed tasks or policy versions within one group')
        rewards=torch.tensor([s.reward for s in group],dtype=torch.float64)
        std=rewards.std(unbiased=False)
        advantages=(rewards-rewards.mean())/(std+epsilon) if std>epsilon else torch.zeros_like(rewards)
        for s,a in zip(group,advantages.tolist()):
            s.advantage=a
    return samples

def clipped_policy_loss(new_logprobs,old_logprobs,advantages,loss_mask,*,clip_low=0.2,clip_high=0.28,reduction='mean'):
    if not (new_logprobs.shape==old_logprobs.shape==advantages.shape==loss_mask.shape):
        raise ValueError('Loss inputs must have identical shapes')
    active=loss_mask.bool()
    if not active.any():
        raise ValueError('No action tokens')
    new,old,adv=new_logprobs[active],old_logprobs[active],advantages[active]
    if not all(torch.isfinite(t).all() for t in [new,old,adv]):
        raise ValueError('Nonfinite action loss inputs')
    ratio=(new-old).exp()
    clipped=ratio.clamp(1-clip_low,1+clip_high)
    values=-torch.minimum(ratio*adv,clipped*adv)
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite policy loss')
    if reduction=='sum':
        return values.sum()
    if reduction!='mean':
        raise ValueError('Unknown reduction')
    return values.mean()
