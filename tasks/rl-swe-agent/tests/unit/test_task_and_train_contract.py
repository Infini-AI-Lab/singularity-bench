import json
from pathlib import Path
import tomllib
from dataclasses import asdict
import pytest
from mini_slime.rollout.sample import Sample
from mini_slime.train import read_samples

ROOT=Path(__file__).resolve().parents[2]

def test_task_identity_and_dataset_lock():
    task=tomllib.loads((ROOT/'task.toml').read_text())
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    assert task['agent']['timeout_sec']==43200
    assert task['environment']['gpus']==4
    assert lock['harbor']['commit']=='45ce769ae580ccd2a1f9adf174ee66b872a4052d'
    assert lock['dataset']['revision']=='39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f'
    assert 'Qwen' not in (ROOT/'instruction.md').read_text()

def samples_file(tmp_path,rewards=(1,0),split='train'):
    path=tmp_path/'samples.jsonl'
    samples=[Sample('task',[1],[2],[-1],[1],r,'group','0',{'split':split,'instance_id':'t'}) for r in rewards]
    path.write_text(''.join(json.dumps(asdict(s))+'\n' for s in samples))
    return path

def test_training_rejects_dev_split_and_uniform_rewards(tmp_path):
    with pytest.raises(ValueError,match='training-split'):read_samples(samples_file(tmp_path,split='dev'),'0')
    with pytest.raises(ValueError,match='Uniform'):read_samples(samples_file(tmp_path,rewards=(1,1)),'0')
    assert len(read_samples(samples_file(tmp_path,rewards=(1,1)),'0',require_variation=False))==2

def test_training_rejects_wrong_policy(tmp_path):
    with pytest.raises(ValueError,match='policy version'):read_samples(samples_file(tmp_path),'2')
