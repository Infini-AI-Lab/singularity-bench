import torch
from .grpo import clipped_policy_loss

def policy_loss_from_logits(logits,batch,**kwargs):
    if logits.shape[:2] != batch.target_ids.shape:
        raise ValueError('Logits do not align with shifted targets')
    # Only materialize FP32 distributions for actual action positions.
    active=batch.loss_mask.bool()
    chosen=logits[active].float().log_softmax(-1).gather(-1,batch.target_ids[active].unsqueeze(-1)).squeeze(-1)
    return clipped_policy_loss(chosen,batch.old_logprobs[active],batch.advantages[active],
                               torch.ones_like(chosen),**kwargs)
