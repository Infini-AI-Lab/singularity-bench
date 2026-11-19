"""Bounded data-only tensor trees for private transport and retained raw outputs.

There is no pickle, numpy object array, imported class, or arbitrary file path.
Tensor identities/strides preserve in-place recomputation challenges.
"""
import hashlib
import json
import math
from pathlib import Path

DTYPES = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
          'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8,
          'float8_e4m3fn': 1, 'float8_e5m2': 1, 'float8_e4m3fnuz': 1, 'float8_e5m2fnuz': 1}
MAX_BYTES = 4 * 1024**3
MAX_TOTAL = 16 * 1024**3
MAX_TENSORS = 64


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate tensor metadata key')
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
        raise ValueError('Invalid tensor manifest')
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def dump(value, directory, identity=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    count, total = 0, 0
    def visit(item, depth=0):
        nonlocal count, total
        if depth > 32:
            raise ValueError('Tensor tree too deep')
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            dtype = str(tensor.dtype).removeprefix('torch.')
            if dtype not in DTYPES:
                raise ValueError('Unsupported tensor dtype')
            size = tensor.numel() * tensor.element_size()
            total += size
            if size > MAX_BYTES or total > MAX_TOTAL or count >= MAX_TENSORS:
                raise ValueError('Tensor inventory exceeds bound')
            filename = f'{count:04d}.bin'
            count += 1
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            (directory / filename).write_bytes(raw)
            return {'tensor': filename, 'shape': list(item.shape), 'stride': list(item.stride()),
                    'dtype': dtype, 'bytes': size, 'sha256': hashlib.sha256(raw).hexdigest(),
                    'identity': identity(item) if identity else count}
        if item is None or type(item) in (bool, int, str):
            return {'scalar': item}
        if type(item) is float and math.isfinite(item):
            return {'scalar': item}
        if isinstance(item, (list, tuple)):
            return {'tuple' if isinstance(item, tuple) else 'list': [visit(v, depth+1) for v in item]}
        if isinstance(item, dict) and all(type(k) is str for k in item):
            return {'dict': {k: visit(v, depth+1) for k, v in item.items()}}
        raise ValueError('Unsupported tensor tree value')
    tree = visit(value)
    (directory / 'tree.json').write_text(json.dumps(tree, allow_nan=False))
    return tree


def load(directory, device='cpu', cache=None):
    import torch
    directory = Path(directory)
    seen, active, total = set(), set(), 0
    def visit(item, depth=0):
        nonlocal total
        if depth > 32 or not isinstance(item, dict):
            raise ValueError('Invalid tensor tree')
        if set(item) == {'tensor', 'shape', 'stride', 'dtype', 'bytes', 'sha256', 'identity'}:
            name, shape, stride, dtype = item['tensor'], item['shape'], item['stride'], item['dtype']
            if name != f'{len(seen):04d}.bin' or name in seen or len(seen) >= MAX_TENSORS:
                raise ValueError('Invalid tensor file inventory')
            seen.add(name)
            if dtype not in DTYPES or not isinstance(shape, list) or len(shape) > 16 or \
                    any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('Invalid tensor shape/dtype')
            if not isinstance(stride, list) or len(stride) != len(shape) or \
                    any(type(n) is not int or n < 0 for n in stride):
                raise ValueError('Invalid tensor stride')
            size = math.prod(shape) * DTYPES[dtype]
            total += size
            storage = 0 if not math.prod(shape) else 1 + sum((n-1)*s for n,s in zip(shape,stride))
            if size > MAX_BYTES or total > MAX_TOTAL or storage * DTYPES[dtype] > MAX_BYTES or \
                    type(item['bytes']) is not int or item['bytes'] != size:
                raise ValueError('Oversized tensor storage')
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                raise ValueError('Invalid tensor payload')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Corrupt tensor payload')
            value = torch.frombuffer(bytearray(raw), dtype=getattr(torch, dtype)).reshape(shape) if size else \
                torch.empty(shape, dtype=getattr(torch, dtype))
            identity = item['identity']
            if type(identity) is not int or identity <= 0:
                raise ValueError('Invalid tensor identity')
            active.add(identity)
            old = cache.get(identity) if cache is not None else None
            if old is None or tuple(old.shape) != tuple(shape) or old.dtype != value.dtype or tuple(old.stride()) != tuple(stride):
                old = torch.empty_strided(shape, stride, dtype=value.dtype, device=device)
                if cache is not None:
                    cache[identity] = old
            old.copy_(value)
            return old
        if set(item) == {'scalar'}:
            value = item['scalar']
            if value is None or type(value) in (bool, int, str) or type(value) is float and math.isfinite(value):
                return value
        for kind in ('list', 'tuple', 'dict'):
            if set(item) == {kind}:
                if kind == 'dict' and isinstance(item[kind], dict):
                    return {k: visit(v, depth+1) for k,v in item[kind].items()}
                if kind != 'dict' and isinstance(item[kind], list):
                    values = [visit(v, depth+1) for v in item[kind]]
                    return tuple(values) if kind == 'tuple' else values
        raise ValueError('Invalid tensor node')
    value = visit(read_json(directory / 'tree.json'))
    if {p.name for p in directory.iterdir()} != seen | {'tree.json'}:
        raise ValueError('Undeclared tensor payload')
    if cache is not None:
        for identity in set(cache) - active:
            del cache[identity]
    return value
