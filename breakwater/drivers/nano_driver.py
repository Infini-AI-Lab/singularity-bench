"""Immutable label-free submitted-code runner, executed ONLY as worker in agent."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['generation', 'ppl_logits'], required=True)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--transport-fd', type=int)
args = parser.parse_args()
root = Path('/app')
checkpoint = root / 'out/final_checkpoint'
if checkpoint.is_symlink() or not (checkpoint / 'config.json').is_file():
    raise ValueError('Required final checkpoint missing')
# The model adapter sees only example IDs, prompt IDs, and generation parameters.
# It has no answers, correctness logic, scoring code, judge key or output-store path.
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('generation_model', root / 'generation_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_model(str(checkpoint))
if args.stage == 'ppl_logits':
    import socket
    import torch
    from nano_logits_wire import send_logits
    with socket.socket(fileno=args.transport_fd) as transport, torch.inference_mode():
        while raw := transport.recv(65536):
            request = json.loads(raw)
            if set(request) != {'input_ids'} or not 1 <= len(request['input_ids']) <= 2048:
                raise ValueError('Invalid forward request')
            with contextlib.redirect_stdout(sys.stderr), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = module.ppl_logits(model, request['input_ids'])
            send_logits(transport, logits, len(request['input_ids']))
    raise SystemExit(0)
for line in args.input.open():
    row = json.loads(line)
    if set(row) != {'example_id', 'prompt_ids'}:
        raise ValueError('Invalid label-free prompt')
    with contextlib.redirect_stdout(sys.stderr):
        tokens = module.generate(model, row['prompt_ids'], 16)
    print(json.dumps({'example_id': row['example_id'], 'generated_token_ids': tokens}), flush=True)
