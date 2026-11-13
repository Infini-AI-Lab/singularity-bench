"""Build an inspectable, reduced-memory SGLang launch; --execute starts it."""
import argparse
import json
import os
from pathlib import Path
import sys
from mini_slime.config import PROFILES, validate_gpu_names


def command(model, profile, *, host='0.0.0.0', port=32000):
    if profile not in ('l40-debug', 'h200-debug'):
        raise ValueError('This launcher starts one TP4 debug engine, not the GB300 fleet')
    if not 0 < port < 65536:
        raise ValueError('Invalid port')
    c = PROFILES[profile]
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(model),
           '--served-model-name', 'model', '--tp', str(c.engine_tp),
           '--context-length', str(c.context), '--dtype', 'bfloat16',
           '--tool-call-parser', 'glm47', '--reasoning-parser', 'glm45',
           '--host', host, '--port', str(port)]
    if profile == 'l40-debug':
        cmd += ['--mem-fraction-static', '0.60', '--attention-backend', 'triton',
                '--moe-runner-backend', 'triton', '--disable-cuda-graph',
                '--disable-custom-all-reduce', '--chunked-prefill-size', '512',
                '--max-running-requests', '2']
    else:
        cmd += ['--mem-fraction-static', '0.7', '--cuda-graph-max-bs-decode', '8',
                '--cuda-graph-backend-prefill', 'disabled']
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--profile', choices=['l40-debug', 'h200-debug'], default='l40-debug')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=32000)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    cmd = command(a.model, a.profile, host=a.host, port=a.port)
    print(json.dumps({'profile': a.profile, 'argv': cmd, 'execute': a.execute}, indent=2), flush=True)
    if a.execute:
        import torch
        validate_gpu_names(a.profile, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        if not a.model.is_dir():
            raise FileNotFoundError(a.model)
        os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
