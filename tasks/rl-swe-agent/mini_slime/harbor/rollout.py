"""Collect one complete prompt group via a separately running Miles session server."""
import argparse
import asyncio
from dataclasses import asdict
import json
import signal
import uuid
from pathlib import Path
import httpx
from mini_slime.config import PROFILES
from .client import TrialResult,request_payload
from .bridge import convert_sample

async def collect_one(client,record,*,session_url,external_session_url,instance_id,worker_url,model,profile,version,group_id,temperature,decode):
    base=None
    outcome={'instance_id':record['metadata']['instance_id']}
    try:
        created=await client.post(session_url+'/sessions',json={'evaluation':False,'temperature':float(temperature)},timeout=30)
        created.raise_for_status()
        sid=created.json()['session_id']
        base=session_url+'/sessions/'+sid
        external=external_session_url+'/sessions/'+sid
        outcome['session_id']=sid
        response=await client.post(worker_url+'/run',json=request_payload(record,external,instance_id,
            model=model,context=profile.context,response=profile.response,temperature=temperature),timeout=profile.episode_seconds+180)
        response.raise_for_status()
        result=TrialResult.parse(response.json())
        outcome.update(status=result.status,worker_result=result.raw)
        if not result.eligible:
            return None,outcome
        wire=await client.post(base+'/samples',json={'max_seq_len':profile.context},timeout=120)
        wire.raise_for_status()
        reply=decode(wire.content,record)
        if len(reply.samples)!=1:
            raise ValueError(f'Expected one v1 sample; got {len(reply.samples)} ({reply.empty_reason})')
        sample=convert_sample(reply.samples[0],result,record,group_id,version,sid)
        sample.metadata['session_metadata']=reply.session_metadata
        outcome['action_tokens']=sum(sample.action_mask)
        outcome['total_tokens']=len(sample.prompt_ids)+len(sample.response_ids)
        return sample,outcome
    except Exception as exc:
        # Keep protocol/codec failures in the outcome ledger. Cancellation is a
        # BaseException and still propagates to the run-scoped flush handler.
        outcome.update(status='collection_error',error=f'{type(exc).__name__}: {exc}')
        return None,outcome
    finally:
        # DELETE removes this session only. Worker cleanup is separately heartbeat/flush scoped.
        if base is not None:
            try:
                response=await client.delete(base,timeout=30)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                outcome['session_cleanup_error']=str(exc)

async def run(a):
    from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
    from miles.utils.types import Sample as NativeSample
    profile=PROFILES[a.profile]
    record=json.loads(a.data.read_text().splitlines()[a.row])
    if record.get('metadata',{}).get('dataset_revision')!='39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f':
        raise ValueError('Unexpected dataset provenance')
    count=a.samples or profile.samples
    if count<2 or count>profile.concurrency:
        raise ValueError('Samples must be between 2 and profile concurrency')
    receipt=json.loads(a.policy_receipt.read_text())
    if record.get('metadata',{}).get('split')!='train':
        raise ValueError('This optimization collector accepts only the training split')
    if receipt['version']!=str(a.policy_version):
        raise ValueError('Receipt and requested version differ')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'policy_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def decode(blob,rec):
        return decode_samples_and_merge_input_sample(blob,NativeSample(prompt=rec['prompt'],metadata=dict(rec['metadata'])))
    # All trials use one dedicated server instance, so scoped flush cannot affect other runs.
    async with httpx.AsyncClient() as client:
        for url in [a.worker_url,a.session_url]:
            response=await client.get(url+'/health',timeout=15); response.raise_for_status()
        health=await client.get(a.session_url+'/health',timeout=15)
        if health.json().get('session_server_instance_id')!=a.instance_id:
            raise ValueError('Session instance ID mismatch; refusing scoped flush')
        backend=await client.get(receipt['backend_url']+'/get_model_info',timeout=30)
        backend.raise_for_status()
        if backend.json().get('model_path')!=receipt['checkpoint']:
            raise ValueError('SGLang model path differs from published receipt')
        group=str(uuid.uuid4())
        tasks=[collect_one(client,record,session_url=a.session_url.rstrip('/'),external_session_url=a.external_session_url.rstrip('/'),
               instance_id=a.instance_id,worker_url=a.worker_url.rstrip('/'),model=a.model,profile=profile,
               version=a.policy_version,group_id=group,temperature=a.temperature,decode=decode) for _ in range(count)]
        try:
            results=await asyncio.gather(*tasks)
        finally:
            # Never flush all trials. This UUID must belong to this run's private session server.
            headers={}
            import os
            if os.getenv('HARBOR_ADMIN_SECRET'):
                headers['Authorization']='Bearer '+os.environ['HARBOR_ADMIN_SECRET']
            try:
                response=await client.post(a.worker_url.rstrip('/')+'/flush',json={'session_server_instance_id':a.instance_id},headers=headers,timeout=30)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                (a.output/'cleanup_error.txt').write_text(str(exc)+'\n')
        (a.output/'outcomes.json').write_text(json.dumps([r for _,r in results],indent=2)+'\n')
        # Reject the entire group if any member failed. Never condition group inclusion on reward.
        if any(s is None for s,_ in results):
            raise RuntimeError('Incomplete group; outcomes saved, no training trajectories published')
        for sample,_ in results:
            sample.metadata['group_size']=count
        (a.output/'samples.jsonl').write_text(''.join(json.dumps(asdict(s))+'\n' for s,_ in results))
        print(json.dumps({'output':str(a.output),'samples':count,'rewards':[s.reward for s,_ in results]}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--row',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,default='h200-debug')
    p.add_argument('--worker-url',required=True);p.add_argument('--session-url',required=True)
    p.add_argument('--external-session-url',required=True);p.add_argument('--instance-id',required=True)
    p.add_argument('--policy-receipt',type=Path,required=True)
    p.add_argument('--policy-version',required=True);p.add_argument('--model',default='model')
    p.add_argument('--samples',type=int);p.add_argument('--temperature',type=float,default=0.8)
    a=p.parse_args()
    if a.row<0 or not 0<a.temperature<=2:
        p.error('Invalid row/temperature')
    uuid.UUID(a.instance_id)
    async def entry():
        task=asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,task.cancel)
        await run(a)
    asyncio.run(entry())

if __name__=='__main__':
    main()
