"""Strict safetensors identity/inventory check; never load executable checkpoint code."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors import safe_open

STRUCTURE=('model_type','architectures','vocab_size','hidden_size','intermediate_size',
           'num_hidden_layers','num_attention_heads','num_key_value_heads','n_routed_experts',
           'n_shared_experts','num_experts_per_tok','moe_intermediate_size','q_lora_rank',
           'kv_lora_rank','qk_rope_head_dim','qk_nope_head_dim','v_head_dim','tie_word_embeddings',
           'rope_theta','rope_scaling','max_position_embeddings','num_nextn_predict_layers')

def inventory(path):
    path=Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Checkpoint must be a real directory')
    files=list(path.rglob('*'))
    if any(p.is_symlink() for p in files):
        raise ValueError('Symlink in checkpoint')
    if any(p.suffix in {'.py','.bin','.pt','.pkl','.pickle','.so','.sh'} for p in files if p.is_file()):
        raise ValueError('Executable/pickle checkpoint assets are not accepted')
    config=json.loads((path/'config.json').read_text())
    if config.get('model_type')!='glm4_moe_lite' or config.get('auto_map'):
        raise ValueError('Expected native GLM-4.7-Flash config')
    if not (path/'tokenizer_config.json').is_file() or not (path/'tokenizer.json').is_file():
        raise ValueError('Missing complete tokenizer files')
    mapping={}
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key in mapping:
                    raise ValueError(f'Duplicate tensor: {key}')
                mapping[key]=shard
    if not mapping:
        raise ValueError('No safetensors')
    index=path/'model.safetensors.index.json'
    if index.exists():
        expected=json.loads(index.read_text())['weight_map']
        if expected!={k:v.name for k,v in mapping.items()}:
            raise ValueError('Shard index does not match actual inventory')
    elif len(set(mapping.values()))!=1:
        raise ValueError('Multiple shards need an index')
    return config,mapping

def audit(base,updated,require_changed=True):
    import torch
    bc,bm=inventory(base);uc,um=inventory(updated)
    if any(bc.get(k)!=uc.get(k) for k in STRUCTURE):
        raise ValueError('Model architecture changed')
    if bm.keys()!=um.keys():
        raise ValueError(f'Tensor inventory mismatch: missing={sorted(bm.keys()-um.keys())[:8]}, extra={sorted(um.keys()-bm.keys())[:8]}')
    changed=0;max_delta=0.0
    for key in bm:
        with safe_open(bm[key],framework='pt',device='cpu') as f, safe_open(um[key],framework='pt',device='cpu') as g:
            x=f.get_tensor(key);y=g.get_tensor(key)
        if x.shape!=y.shape or x.dtype!=y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f'Invalid exported tensor: {key}')
        if not torch.equal(x,y):
            changed+=1
            max_delta=max(max_delta,float((x.float()-y.float()).abs().max()))
    if require_changed and not changed:
        raise ValueError('No changed weights')
    return {'tensors':len(bm),'changed_tensors':changed,'max_abs_change':max_delta,'finite':True}

def fingerprint(path):
    """Hash full checkpoint files, used to tie a collected group to its initialization."""
    _,mapping=inventory(path)
    hashes={}
    for p in sorted(set(mapping.values())|{Path(path)/'config.json',Path(path)/'tokenizer.json',Path(path)/'tokenizer_config.json'}):
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(8*1024*1024): h.update(chunk)
        hashes[p.name]=h.hexdigest()
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--updated',type=Path,required=True)
    p.add_argument('--allow-unchanged',action='store_true');p.add_argument('--report',type=Path)
    a=p.parse_args();result=audit(a.base,a.updated,not a.allow_unchanged)
    payload=json.dumps(result,indent=2)+'\n'
    if a.report:a.report.write_text(payload)
    print(payload,end='')

if __name__=='__main__':main()
