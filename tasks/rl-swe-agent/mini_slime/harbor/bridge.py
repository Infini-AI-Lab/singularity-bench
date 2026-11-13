"""Convert native Miles v1 samples to mini_slime, retaining exact IDs/masks."""
from mini_slime.rollout.sample import Sample

def convert_sample(native, result, record, group_id, expected_version, session_id):
    if not result.eligible:
        raise ValueError('Infrastructure/aborted trial is not optimization data')
    if getattr(native.status,'value',native.status) != 'completed':
        raise ValueError('Only completed native trajectories accepted initially')
    n=native.response_length
    if type(n) is not int or n<=0 or n>=len(native.tokens):
        raise ValueError('Invalid native response boundary')
    if native.loss_mask is None or native.rollout_log_probs is None:
        raise ValueError('Native sample lacks masks/logprobs')
    sample=Sample(prompt=record['prompt'],prompt_ids=list(native.tokens[:-n]),
        response_ids=list(native.tokens[-n:]),rollout_logprobs=list(native.rollout_log_probs),
        action_mask=list(native.loss_mask),reward=result.reward,group_id=group_id,
        policy_version=str(expected_version),metadata={**record['metadata'],'session_id':session_id,
        'worker_result':result.raw,'native_metadata':native.metadata})
    sample.validate()
    # The version claim comes from backend spans, never solely a client label.
    covered=set()
    spans=[]
    for call in native.weight_versions:
        for span in call.spans:
            if str(span.version)!=str(expected_version):
                raise ValueError(f'Stale serving weights: {span.version}, expected {expected_version}')
            if not 0<=span.abs_start<span.abs_end<=len(native.tokens):
                raise ValueError('Invalid backend weight-version span')
            positions=set(range(span.abs_start,span.abs_end))
            if covered & positions:
                raise ValueError('Overlapping weight-version spans')
            covered.update(positions)
            spans.append(dict(version=str(span.version),start=span.abs_start,end=span.abs_end))
    actions={len(sample.prompt_ids)+i for i,m in enumerate(sample.action_mask) if m}
    if not actions <= covered:
        raise ValueError('Missing backend version evidence for action tokens')
    sample.metadata['weight_version_spans']=spans
    return sample
