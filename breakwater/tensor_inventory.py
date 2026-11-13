"""Non-executable safetensors inventory for judge coverage and checkpoint lint."""
import json
from pathlib import Path
import struct

from bundle_storage import SafetyError, strict_json

DTYPES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2,
          'I8': 1, 'U64': 8, 'U32': 4, 'U16': 2, 'U8': 1, 'BOOL': 1, 'F8_E4M3': 1,
          'F8_E4M3FN': 1, 'F8_E5M2': 1}


def inventory(path: Path):
    with path.open('rb') as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise SafetyError('Invalid safetensors header')
        size = struct.unpack('<Q', first)[0]
        if not 2 <= size <= 100_000_000:
            raise SafetyError('Invalid safetensors header size')
        raw = stream.read(size)
        if len(raw) != size:
            raise SafetyError('Truncated safetensors header')
        header = strict_json(raw)
        if not isinstance(header, dict):
            raise SafetyError('Tensor header must be an object')
    intervals = []
    for name, value in header.items():
        if name == '__metadata__':
            if not isinstance(value, dict) or any(type(v) is not str for v in value.values()):
                raise SafetyError('Invalid safetensors metadata')
            continue
        if not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise SafetyError('Invalid tensor descriptor')
        shape, offsets = value['shape'], value['data_offsets']
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise SafetyError('Invalid tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise SafetyError('Invalid tensor offsets')
        count = 1
        for dimension in shape:
            count *= dimension
        dtype = value['dtype']
        if dtype not in DTYPES or offsets[1] - offsets[0] != count * DTYPES[dtype]:
            raise SafetyError('Tensor byte accounting mismatch')
        intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise SafetyError('Overlapping or undeclared checkpoint bytes')
        cursor = end
    if cursor + 8 + size != path.stat().st_size:
        raise SafetyError('Undeclared checkpoint payload')
    return header


def finite_checkpoint(directory: Path):
    """Check every safetensors payload with bounded, non-executable integer reads."""
    import numpy as np
    paths = sorted(directory.rglob('*.safetensors'))
    if not paths:
        raise SafetyError('Checkpoint has no safetensors weights')
    descriptors = {}
    masks = {'F64': ('<u8', 0x7ff0000000000000), 'F32': ('<u4', 0x7f800000),
             'F16': ('<u2', 0x7c00), 'BF16': ('<u2', 0x7f80),
             'F8_E5M2': ('u1', 0x7c)}
    for path in paths:
        if path.is_symlink():
            raise SafetyError('Checkpoint shard must not be a symlink')
        header = inventory(path)
        with path.open('rb') as stream:
            offset = 8 + struct.unpack('<Q', stream.read(8))[0]
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                start, end = info['data_offsets']
                stream.seek(offset + start)
                remaining = end - start
                dtype = info['dtype']
                while remaining:
                    raw = stream.read(min(remaining, 4 * 1024 * 1024))
                    if not raw:
                        raise SafetyError('Truncated tensor payload')
                    remaining -= len(raw)
                    if dtype in masks:
                        integer_dtype, mask = masks[dtype]
                        bits = np.frombuffer(raw, dtype=integer_dtype)
                        invalid = np.any((bits & mask) == mask)
                    elif dtype in ('F8_E4M3', 'F8_E4M3FN'):
                        invalid = np.any((np.frombuffer(raw, dtype='u1') & 0x7f) == 0x7f)
                    else:
                        invalid = False
                    if invalid:
                        raise SafetyError(f'Nonfinite checkpoint tensor: {path.name}:{name}')
        descriptors[path.relative_to(directory).as_posix()] = header
    return descriptors
