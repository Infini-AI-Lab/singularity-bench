"""Fixed teacher-forcing, vocabulary normalization and streaming Parquet writer.

The root supervisor never imports the submitted architecture. A separate worker
supplies fully sealed numeric tensors; provenance/causality remain audit gates.
"""
from pathlib import Path

VOCAB = 151669
BLOCK = 2048


def target_logprobs(logits, targets):
    import torch
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 2 or logits.shape[0] != len(targets) or logits.shape[1] < VOCAB:
        raise ValueError('ppl_logits must return [input length, vocabulary >=151669] tensor')
    scores = logits[:, :VOCAB].float()
    if not torch.isfinite(scores).all():
        raise ValueError('Scored vocabulary logits must all be finite')
    labels = torch.as_tensor(targets, dtype=torch.long, device=scores.device)
    if torch.any(labels < 0) or torch.any(labels >= VOCAB):
        raise ValueError('Invalid target token ID')
    return -torch.nn.functional.cross_entropy(scores, labels, reduction='none')


def evaluate(source, output):
    import json
    import socket
    import subprocess
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nano_logits_wire import receive_logits
    if not torch.cuda.is_available():
        raise ValueError('Protected PPL evaluation requires allocated CUDA GPUs')
    size = Path(source).stat().st_size
    if size < 8 or size % 4:
        raise ValueError('Expected uint32 token stream')
    tokens = np.memmap(source, dtype='<u4', mode='r')
    schema = pa.schema([('token_index', pa.int64()), ('token_id', pa.int64()), ('logprob', pa.float32())])
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    process = subprocess.Popen(['python3', '-I', '/opt/breakwater-runtime/run_worker.py',
        'python3', '/opt/breakwater-drivers/nano_driver.py', '--stage', 'ppl_logits',
        '--input', str(source), '--output', str(output), '--transport-fd', str(child.fileno())],
        pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    child.close()
    try:
        with parent, pq.ParquetWriter(output, schema) as writer, torch.inference_mode():
            parent.settimeout(1800)
            for start in range(0, len(tokens)-1, BLOCK):
                stop = min(start + BLOCK, len(tokens)-1)
                inputs = tokens[start:stop].astype(np.int64).tolist()
                parent.sendall(json.dumps({'input_ids': inputs}).encode())
                logits, digest = receive_logits(parent, stop-start)
                values = target_logprobs(logits, tokens[start+1:stop+1].astype(np.int64)).cpu().numpy()
                writer.write_table(pa.Table.from_arrays([
                    pa.array(np.arange(start+1, stop+1), type=pa.int64()),
                    pa.array(tokens[start+1:stop+1].astype(np.int64), type=pa.int64()),
                    pa.array(values, type=pa.float32())], schema=schema))
                print(json.dumps({'stage': 'trusted_ppl', 'start': start, 'stop': stop,
                    'logits_sha256': digest, 'rows': stop-start}), flush=True)
        if process.wait(timeout=30):
            raise ValueError('Forward worker failed')
    finally:
        parent.close()
        if process.poll() is None:
            process.kill()
        process.wait()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('Trusted PPL supervisor requires root')
    evaluate(args.input, args.output)
