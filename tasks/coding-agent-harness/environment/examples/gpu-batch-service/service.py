import argparse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time

import torch


def weights():
    return torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32, torch.arange(4) / 10


def compute_batch(items):
    # Sort by request size before packing, then send each request its own output.
    packed = sorted(items, key=lambda item: len(item[0]['x']), reverse=True)
    x = torch.tensor([row for body, _ in packed for row in body['x']], device='cuda')
    w, b = weights()
    y = torch.relu(x @ w.cuda() + b.cuda()).cpu().tolist()
    cursor = 0
    for body, future in items:
        count = len(body['x'])
        future.set_result({'id': body['id'], 'y': y[cursor:cursor + count]})
        cursor += count


def serve(port=8080, ready_path=None):
    torch.zeros(1, device='cuda')
    pending = queue.Queue()

    def batches():
        while True:
            items = [pending.get()]
            deadline = time.monotonic() + 0.025
            while len(items) < 16:
                try:
                    items.append(pending.get(timeout=max(0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            try:
                compute_batch(items)
            except Exception as exc:
                for _, future in items:
                    if not future.done():
                        future.set_exception(exc)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {'ready': True, 'device': torch.cuda.get_device_name()})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                future = Future()
                pending.put((body, future))
                self.reply(200, future.result(timeout=60))
            except Exception as exc:
                self.reply(500, {'error': str(exc)})

    threading.Thread(target=batches, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    address = f'http://127.0.0.1:{server.server_port}'
    if ready_path:
        Path(ready_path).write_text(address)
    print(f'READY {address}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    serve(args.port)
