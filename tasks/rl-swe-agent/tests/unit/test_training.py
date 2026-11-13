import copy
import pytest
import torch
from mini_slime.rollout.sample import Sample
from mini_slime.training.batch import build_training_batch
from mini_slime.training.grpo import compute_group_advantages,clipped_policy_loss
from mini_slime.training.loss import policy_loss_from_logits

def sample(reward=1.0):
    return Sample('task',[1,2],[3,4,5,6],[-2.0,0.0,0.0,-3.0],[1,0,0,1],reward,'g','0',{'instance_id':'task'})

def test_multiturn_shift_and_padding():
    a=sample();b=sample(0.0);b.prompt_ids=[1]
    compute_group_advantages([a,b]);batch=build_training_batch([a,b],0)
    assert batch.loss_mask.tolist()==[[0,1,0,0,1],[1,0,0,1,0]]
    assert batch.target_ids[0].tolist()==[2,3,4,5,6]
    assert batch.old_logprobs[0].tolist()==[0,-2,0,0,-3]
    assert a.action_mask==[1,0,0,1]

def test_loss_only_has_action_position_gradients():
    a=sample();b=sample(0.0);compute_group_advantages([a,b])
    batch=build_training_batch([a,b],0)
    logits=torch.randn(2,5,9,requires_grad=True)
    loss=policy_loss_from_logits(logits,batch);loss.backward()
    assert torch.equal(logits.grad[~batch.loss_mask.bool()],torch.zeros_like(logits.grad[~batch.loss_mask.bool()]))
    assert logits.grad[batch.loss_mask.bool()].abs().sum()>0

def test_mask_before_exponentiation():
    new=torch.tensor([0.0,10000.0],requires_grad=True)
    loss=clipped_policy_loss(new,torch.zeros(2),torch.ones(2),torch.tensor([1,0]))
    assert torch.isfinite(loss)
    loss.backward();assert new.grad.tolist()==[-1.0,0.0]

def test_uniform_rewards_zero_advantage():
    items=[sample(),sample()];compute_group_advantages(items)
    assert [s.advantage for s in items]==[0,0]

def test_reject_mixed_version():
    a=sample();b=sample(0);b.policy_version='1'
    with pytest.raises(ValueError,match='Mixed'):compute_group_advantages([a,b])

@pytest.mark.parametrize('change',[{'rollout_logprobs':[-1]}, {'action_mask':[0,0,0,0]}, {'reward':float('nan')},{'response_ids':[-1,4,5,6]}])
def test_reject_malformed(change):
    a=sample()
    for k,v in change.items():setattr(a,k,v)
    with pytest.raises(ValueError):a.validate()

def test_real_cpu_optimizer_and_zero_advantage_control():
    # Synthetic test tensors validate math only, never represented as dataset/RL evidence.
    initial=torch.tensor([[-1.0,-2.0],[-2.0,-1.0]])
    def update(advantages):
        weights=torch.nn.Parameter(initial.clone());opt=torch.optim.SGD([weights],lr=.1)
        old=initial.log_softmax(-1)[:,0]
        loss=clipped_policy_loss(weights.log_softmax(-1)[:,0],old,advantages,torch.ones(2))
        loss.backward();norm=weights.grad.norm().item();opt.step()
        return weights.detach(),norm
    actual,norm=update(torch.tensor([1.0,-1.0]))
    control,zero=update(torch.zeros(2))
    assert norm>0 and not torch.equal(actual,initial)
    assert zero==0 and torch.equal(control,initial)
