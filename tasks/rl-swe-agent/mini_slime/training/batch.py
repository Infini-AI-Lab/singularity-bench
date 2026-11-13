"""Shift token provenance exactly once; never reconstruct a contiguous response mask."""
from dataclasses import dataclass
import torch
from mini_slime.rollout.sample import Sample

@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    loss_mask: torch.Tensor

    def to(self, device):
        return type(self)(**{k:v.to(device) for k,v in vars(self).items()})

def build_training_batch(samples: list[Sample], pad_token_id: int):
    if not samples:
        raise ValueError('Empty batch')
    for s in samples:
        s.validate()
        if s.advantage is None:
            raise ValueError('Compute group advantages first')
    width = max(len(s.prompt_ids)+len(s.response_ids) for s in samples)-1
    rows = {k:[] for k in ('input_ids','target_ids','old_logprobs','advantages','loss_mask')}
    for s in samples:
        seq=s.prompt_ids+s.response_ids
        padded=seq+[pad_token_id]*(width+1-len(seq))
        start=len(s.prompt_ids)-1
        mask=[0]*width; old=[0.0]*width; adv=[0.0]*width
        for j,active in enumerate(s.action_mask):
            mask[start+j]=active
            if active:
                old[start+j]=s.rollout_logprobs[j]
                adv[start+j]=s.advantage
        for key,value in zip(rows, [padded[:-1],padded[1:],old,adv,mask]):
            rows[key].append(value)
    return TrainingBatch(**{k:torch.tensor(v,dtype=torch.long if k.endswith('_ids') else torch.float32) for k,v in rows.items()})
