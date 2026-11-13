"""Pinned harbor-miles /run protocol. No automatic trial retries or fake rewards."""
from dataclasses import dataclass
import math
from urllib.parse import urlsplit

@dataclass(frozen=True)
class TrialResult:
    status: str
    reward: float | None
    raw: dict

    @property
    def eligible(self):
        return self.status == 'completed'

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw,dict):
            raise ValueError('Worker response must be an object')
        status=raw.get('exit_status')
        if status != 'Submitted':
            category={'SequenceLengthLimitExceeded':'token_limit','TimeLimitExceeded':'timeout','Flushed':'cancelled'}.get(status,'infrastructure_error')
            return cls(category,None,raw)
        report=raw.get('eval_report')
        if not isinstance(report,dict) or 'reward' not in report or 'reward' not in raw:
            raise ValueError('Submitted trial lacks explicit verifier reward')
        reward=raw['reward']
        if type(reward) not in (int,float) or not math.isfinite(reward) or reward not in (0,1) or report['reward']!=reward:
            raise ValueError('Invalid or inconsistent binary verifier reward')
        return cls('completed',float(reward),raw)

def request_payload(record, session_url, server_instance_id, *, model, context, response, temperature):
    meta=record['metadata']
    return {'instance_id':meta['instance_id'],'agent_name':'mini-swe-agent',
            'base_url':session_url.rstrip('/')+'/v1','model':'openai/'+model,
            'sampling_params':{'temperature':float(temperature),'max_tokens':response},
            'max_seq_len':context,'session_server_id':urlsplit(session_url).netloc,
            'session_server_instance_id':server_instance_id,'api_key':'dummy'}
