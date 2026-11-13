"""Explicitly load a complete checkpoint into SGLang and record the source hash."""
import argparse
import json
from pathlib import Path
from .policy import PolicyBarrier
from mini_slime.model.checkpoint import fingerprint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--version',type=int,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if a.receipt.exists():raise FileExistsError(a.receipt)
    source_hash=fingerprint(a.checkpoint)
    result=PolicyBarrier().sync(a.url,a.checkpoint,a.version)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps({'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':source_hash,
        'version':str(a.version),'backend_url':a.url.rstrip('/'),'response':result},indent=2)+'\n')

if __name__=='__main__':main()
