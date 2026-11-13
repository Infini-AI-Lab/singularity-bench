import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from mini_slime.model.checkpoint import audit,fingerprint
from mini_slime.sync.policy import PolicyBarrier

def checkpoint(p,value=1.0):
    p.mkdir()
    (p/'config.json').write_text(json.dumps({'model_type':'glm4_moe_lite','hidden_size':2}))
    (p/'tokenizer.json').write_text('{}');(p/'tokenizer_config.json').write_text('{}')
    save_file({'weight':torch.full((2,2),value)},str(p/'model.safetensors'))
    return p

def test_audit_changed_and_finite(tmp_path):
    base=checkpoint(tmp_path/'base');updated=checkpoint(tmp_path/'updated',2.0)
    assert audit(base,updated)['changed_tensors']==1
    assert fingerprint(base)!=fingerprint(updated)
    with pytest.raises(ValueError,match='No changed'):audit(base,base)
    save_file({'weight':torch.full((2,2),float('nan'))},str(updated/'model.safetensors'))
    with pytest.raises(ValueError,match='Invalid exported'):audit(base,updated)

def test_partial_or_symlink_rejected(tmp_path):
    base=checkpoint(tmp_path/'base');updated=checkpoint(tmp_path/'updated')
    save_file({'other':torch.ones(2)},str(updated/'model.safetensors'))
    with pytest.raises(ValueError,match='inventory'):audit(base,updated)
    (updated/'tokenizer.json').unlink();(updated/'tokenizer.json').symlink_to(base/'tokenizer.json')
    with pytest.raises(ValueError,match='Symlink'):fingerprint(updated)

def test_sync_failure_stays_dirty(monkeypatch):
    import mini_slime.sync.policy as module
    barrier=PolicyBarrier()
    with pytest.raises(RuntimeError):barrier.require_ready()
    monkeypatch.setattr(module,'update_sglang_from_disk',lambda *a:{'success':True})
    barrier.sync('url','path',0);assert barrier.require_ready()=='0'
    barrier.mark_updated()
    def fail(*args):raise RuntimeError('failed sync')
    monkeypatch.setattr(module,'update_sglang_from_disk',fail)
    with pytest.raises(RuntimeError):barrier.sync('url','path',1)
    with pytest.raises(RuntimeError):barrier.require_ready()
    with pytest.raises(ValueError):barrier.sync('url','path',0)
