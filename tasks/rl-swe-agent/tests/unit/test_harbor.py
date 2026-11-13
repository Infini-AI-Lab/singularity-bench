import asyncio
from types import SimpleNamespace as NS
import httpx
import pytest
from mini_slime.config import PROFILES,generation_budget
from mini_slime.harbor.client import TrialResult,request_payload
from mini_slime.harbor.bridge import convert_sample
from mini_slime.harbor.rollout import collect_one

RECORD={'prompt':'fix','metadata':{'instance_id':'task_1','dataset_revision':'rev'}}

def native(version='0'):
    return NS(status='completed',response_length=3,tokens=[1,2,3,4,5],loss_mask=[1,0,1],
       rollout_log_probs=[-1,0,-2],metadata={},weight_versions=[NS(spans=[
       NS(version=version,abs_start=2,abs_end=3),NS(version=version,abs_start=4,abs_end=5)])])

def result(reward=1):
    return TrialResult.parse({'exit_status':'Submitted','reward':reward,'eval_report':{'reward':reward}})

@pytest.mark.parametrize('status',['Error: NoResult','AgentError','TimeLimitExceeded','Flushed','SequenceLengthLimitExceeded','Unknown'])
def test_errors_never_become_zero_rewards(status):
    r=TrialResult.parse({'exit_status':status,'reward':0,'eval_report':{}})
    assert not r.eligible and r.reward is None

def test_legitimate_zero_is_eligible():
    assert result(0).eligible
    assert result(0).reward==0

@pytest.mark.parametrize('payload',[
    {'exit_status':'Submitted','reward':0,'eval_report':{}},
    {'exit_status':'Submitted','reward':1,'eval_report':{'reward':0}},
    {'exit_status':'Submitted','reward':float('nan'),'eval_report':{'reward':float('nan')}}])
def test_malformed_rewards_rejected(payload):
    with pytest.raises(ValueError):TrialResult.parse(payload)

def test_native_conversion_preserves_ids_masks_and_logprobs():
    s=convert_sample(native(),result(),RECORD,'group','0','session')
    assert s.prompt_ids==[1,2] and s.response_ids==[3,4,5]
    assert s.action_mask==[1,0,1] and s.rollout_logprobs==[-1,0,-2]

@pytest.mark.parametrize('mode',['stale','missing','overlap','truncated'])
def test_version_and_completion_fail_closed(mode):
    n=native('1' if mode=='stale' else '0')
    if mode=='missing':n.weight_versions=[]
    if mode=='overlap':n.weight_versions*=2
    if mode=='truncated':n.status='truncated'
    with pytest.raises(ValueError):convert_sample(n,result(),RECORD,'g','0','s')

def test_request_uses_real_wire_contract():
    p=request_payload(RECORD,'http://trainer:30001/sessions/abc','uuid',model='model',context=8192,response=2048,temperature=.8)
    assert p['base_url']=='http://trainer:30001/sessions/abc/v1'
    assert p['session_server_id']=='trainer:30001'
    assert p['model']=='openai/model'
    assert p['sampling_params']['max_tokens']==2048

def test_context_budget():
    assert generation_budget(8192,8000,2048)==191
    assert generation_budget(8192,8192,2048)==0

@pytest.mark.parametrize('failed',[False,True])
def test_http_collection_and_session_cleanup(failed):
    seen=[]
    def handler(req):
        seen.append((req.method,req.url.path))
        if req.url.path=='/sessions':return httpx.Response(200,json={'session_id':'abc'})
        if req.url.path=='/run':
            raw={'exit_status':'AgentError','reward':0,'eval_report':{}} if failed else result().raw
            return httpx.Response(200,json=raw)
        if req.url.path.endswith('/samples'):return httpx.Response(200,content=b'native-safetensors')
        return httpx.Response(200,json={})
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collect_one(client,RECORD,session_url='http://s',external_session_url='http://external',
                instance_id='uuid',worker_url='http://w',model='model',profile=PROFILES['h200-debug'],
                version='0',group_id='g',temperature=.8,decode=lambda b,r:NS(samples=[native()],session_metadata={},empty_reason=None))
    sample,outcome=asyncio.run(exercise())
    assert ('DELETE','/sessions/abc') in seen
    assert (sample is None)==failed
    assert (('POST','/sessions/abc/samples') in seen)!=failed

def test_session_creation_failure_is_recorded_without_trial():
    seen=[]
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(503,text='unavailable')
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collect_one(client,RECORD,session_url='http://s',external_session_url='http://external',
                instance_id='uuid',worker_url='http://w',model='model',profile=PROFILES['h200-debug'],
                version='0',group_id='g',temperature=.8,decode=lambda *a:None)
    sample,outcome=asyncio.run(exercise())
    assert sample is None and outcome['status']=='collection_error'
    assert seen==['/sessions']
