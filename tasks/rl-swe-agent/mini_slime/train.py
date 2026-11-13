"""One synchronous GRPO update from one complete collected group.

Experimental L40/H200 backend: HF GLM + four-rank PyTorch FSDP2 CPU offload.
This is mini_slime training, not a wrapper around Miles train.py. Stop inference
before launching; export/reload/resample before every subsequent update.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

from mini_slime.rollout.sample import Sample
from mini_slime.training.batch import build_training_batch
from mini_slime.training.grpo import compute_group_advantages
from mini_slime.training.loss import policy_loss_from_logits
from mini_slime.model.checkpoint import audit,fingerprint
from mini_slime.model.fsdp_init import shard_cpu_model
from mini_slime.config import PROFILES, validate_gpu_names


def read_samples(path,version,require_variation=True):
    samples=[Sample(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not samples:raise ValueError('Empty training samples')
    for s in samples:
        s.validate()
        if s.policy_version!=str(version):raise ValueError('Wrong sample policy version')
        if s.metadata.get('split')!='train':raise ValueError('Only training-split trajectories may be optimized')
    if len({s.group_id for s in samples})!=1:
        raise ValueError('Initial trainer accepts exactly one complete prompt group')
    compute_group_advantages(samples)
    if require_variation and not any(abs(s.advantage)>0 for s in samples):
        raise ValueError('Uniform reward group: no reward-driven update; choose another training task/seed')
    return samples


def train(a):
    import torch
    import torch.distributed as dist
    from mini_slime.model.checkpoint import inventory
    from torch.distributed.fsdp import CPUOffloadPolicy,MixedPrecisionPolicy,fully_shard
    from torch.distributed.checkpoint.state_dict import StateDictOptions,get_model_state_dict
    from transformers import AutoModelForCausalLM,AutoTokenizer
    rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=4 or torch.cuda.device_count()!=4:
        raise RuntimeError('Use torchrun with exactly four allocated GPUs')
    torch.cuda.set_device(local_rank)
    validate_gpu_names(a.profile, [torch.cuda.get_device_name(i) for i in range(4)])
    profile=PROFILES[a.profile]
    samples=read_samples(a.samples,a.policy_version,not a.roundtrip_only)
    if any(len(s.prompt_ids)+len(s.response_ids)>profile.context for s in samples):
        raise ValueError(f'Trajectory exceeds {a.profile} context; recollect, do not truncate training data')
    receipt=json.loads((a.samples.parent/'policy_receipt.json').read_text())
    if receipt['version']!=str(a.policy_version):raise ValueError('Receipt version mismatch')
    if len(samples)%world:raise ValueError('Sample count must be divisible by four')
    if len({s.metadata.get('group_size') for s in samples})!=1 or samples[0].metadata.get('group_size')!=len(samples):
        raise ValueError('Incomplete group relative to collection manifest')
    for s in samples:
        if s.metadata.get('dataset_revision')!='39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f':
            raise ValueError('Unexpected dataset revision')
    torch.manual_seed(a.seed)
    dist.init_process_group('nccl')
    started=time.monotonic()
    try:
        if rank==0:
            if fingerprint(a.model)!=receipt['checkpoint_sha256']:
                raise ValueError('Training initialization differs from rollout checkpoint')
            a.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        if rank==0:
            from mini_slime.preflight import host_memory
            (a.output/'host-memory.json').write_text(json.dumps(host_memory(),indent=2)+'\n')
        tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=False)
        # Retain FP32 masters on CPU. Never call model.cuda(): the full model
        # exceeds an L40's VRAM even before optimizer/activation allocations.
        model,loading=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,
            attn_implementation='sdpa',trust_remote_code=False,output_loading_info=True,
            device_map=None,low_cpu_mem_usage=True)
        if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
            raise RuntimeError(f'HF GLM mapping needs attention before training: {loading}')
        if model.config.model_type!='glm4_moe_lite':raise ValueError('Wrong model family')
        model.config.use_cache=False
        # We consume logits only, never model.loss or labels. MTP loss is disabled
        # in configs too, and needs a runtime gradient audit on the cluster.
        model.config.mtp_loss_scaling_factor=0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,output_dtype=torch.bfloat16)
        # Page-locking hundreds of GiB can exhaust a smaller workstation host.
        offload=CPUOffloadPolicy(pin_memory=a.profile!='l40-debug')
        torch.cuda.reset_peak_memory_stats()
        units=shard_cpu_model(model,fully_shard=fully_shard,mp_policy=policy,
            offload_policy=offload,max_unit_bytes=int(torch.cuda.get_device_properties(local_rank).total_memory*.2))
        if rank==0:
            (a.output/'initialization.json').write_text(json.dumps({'profile':a.profile,
                'method':'CPU load, layer-wise FSDP2, CPU offload; no whole-model CUDA move',
                'units':units,'rank0_peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        action_count=sum(sum(s.action_mask) for s in samples)
        total_loss=torch.zeros((),device='cuda')
        if not a.roundtrip_only:
            model.train()
            for s in samples[rank::world]:
                pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                batch=build_training_batch([s],pad).to('cuda')
                logits=model(input_ids=batch.input_ids,use_cache=False).logits
                loss=policy_loss_from_logits(logits,batch,reduction='sum')*world/action_count
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();total_loss+=loss.detach()/world
                del logits,batch,loss
            norm_sq=0.0
            for param in model.parameters():
                grad=param.grad
                if grad is not None:
                    grad=grad.to_local() if hasattr(grad,'to_local') else grad
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    norm_sq+=float(grad.float().square().sum())
            norm=torch.tensor(norm_sq,device='cuda',dtype=torch.float64)
            dist.all_reduce(norm);dist.all_reduce(total_loss)
            grad_norm=math.sqrt(float(norm))
            if not math.isfinite(grad_norm) or grad_norm==0:raise ValueError('Zero/nonfinite policy gradient')
            optimizer.step()
        else:
            grad_norm=0.0
        # Full-state CPU offload gathers the export using the installed FSDP API.
        state=get_model_state_dict(model,options=StateDictOptions(full_state_dict=True,cpu_offload=True))
        if rank==0:
            export=a.output/'checkpoint'
            model.save_pretrained(export,state_dict=state,safe_serialization=True,max_shard_size='2GB')
            tokenizer.save_pretrained(export)
            # Restore checkpoint dtype to BF16 for parity with initialization; FP32
            # masters are training state, not a changed architecture/dtype contract.
            del state
            from safetensors import safe_open
            from safetensors.torch import save_file
            base_config,base_map=inventory(a.model)
            dtypes={}
            for shard in set(base_map.values()):
                with safe_open(shard,framework='pt',device='cpu') as f:
                    for key in f.keys():dtypes[key]=f.get_tensor(key).dtype
            for shard in export.glob('*.safetensors'):
                with safe_open(shard,framework='pt',device='cpu') as f:
                    values={k:f.get_tensor(k).to(dtypes[k]) for k in f.keys()}
                save_file(values,str(shard),metadata={'format':'pt'})
            config=json.loads((export/'config.json').read_text())
            # FSDP2 changes the Python class name at runtime; HF save_pretrained
            # may serialize that wrapper name. The export uses the native class.
            config['architectures']=base_config.get('architectures')
            for key in ['dtype','torch_dtype']:
                config.pop(key,None)
                if key in base_config:config[key]=base_config[key]
            (export/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            # Recompute index total_size after export dtype conversion.
            index=export/'model.safetensors.index.json'
            if index.exists():
                data=json.loads(index.read_text());size=0
                for shard in export.glob('*.safetensors'):
                    with safe_open(shard,framework='pt',device='cpu') as f:
                        size+=sum(f.get_tensor(k).numel()*f.get_tensor(k).element_size() for k in f.keys())
                data.setdefault('metadata',{})['total_size']=size
                index.write_text(json.dumps(data,indent=2)+'\n')
            report=audit(a.model,export,require_changed=not a.roundtrip_only)
            report.update(profile=a.profile,rank0_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                optimizer_steps=0 if a.roundtrip_only else 1,gradient_norm=grad_norm,
                policy_loss=float(total_loss),raw_rewards=[s.reward for s in samples],
                advantages=[s.advantage for s in samples],mtp_coefficient=0.0,weight_decay=0.0,kl_coefficient=0.0,
                elapsed_seconds=time.monotonic()-started,source_policy_version=str(a.policy_version),
                optimizer_resume='not implemented in this initial one-step backend')
            (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            (export/'.complete').write_text('validated by mini_slime.model.checkpoint\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--policy-version',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--roundtrip-only',action='store_true')
    p.add_argument('--profile',choices=['h200-debug','l40-debug'],default='h200-debug')
    a=p.parse_args()
    if not 0<a.lr<=1e-3:p.error('Learning rate must be in (0, 1e-3]')
    train(a)

if __name__=='__main__':main()
