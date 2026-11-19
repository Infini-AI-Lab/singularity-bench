"""Sealed float32 tensor transport; no pickle, imports or arbitrary host paths."""
import array
import fcntl
import hashlib
import json
import mmap
import os
import socket
import stat

VOCAB = 151669
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


def send_logits(sock, logits, length):
    import torch
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda or not logits.is_floating_point() or logits.ndim != 2 or logits.shape[0] != length or logits.shape[1] < VOCAB:
        raise ValueError('Expected CUDA floating logits [input length, vocabulary >=151669]')
    values = logits[:, :VOCAB].detach().float().contiguous().cpu().numpy()
    fd = os.memfd_create('nano-logits', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, values.nbytes)
        # Anonymous tensor IPC does not dump multi-TB forward tensors into the
        # syscall journal. Seal before transfer; parent verifies every operand.
        with mmap.mmap(fd, values.nbytes) as memory:
            memory[:] = memoryview(values).cast('B')
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        metadata = json.dumps({'shape': [length, VOCAB], 'dtype': 'float32'}).encode()
        if sock.sendmsg([metadata], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]) != len(metadata):
            raise ValueError('Partial tensor transport')
    finally:
        os.close(fd)


def receive_logits(sock, length, device='cuda'):
    import numpy as np
    import torch
    raw, ancillary, flags, _ = sock.recvmsg(4096, socket.CMSG_SPACE(16), socket.MSG_CMSG_CLOEXEC)
    fds = []
    try:
        for level, kind, data in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                raise ValueError('Unexpected tensor transport control message')
            descriptors = array.array('i'); descriptors.frombytes(data[:len(data)-len(data)%4]); fds.extend(descriptors)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(fds) != 1 or json.loads(raw) != {'shape': [length, VOCAB], 'dtype': 'float32'} or not 1 <= length <= 2048:
            raise ValueError('Invalid bounded logits response')
        fd = fds[0]; info = os.fstat(fd); size = length * VOCAB * 4
        if not stat.S_ISREG(info.st_mode) or info.st_size != size or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS != SEALS:
            raise ValueError('Logits must be a fully sealed float32 tensor')
        # No shared writes are needed. A private read-only mapping also works on
        # kernels that reject MAP_SHARED on a descriptor with F_SEAL_WRITE.
        with mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ) as memory:
            digest = hashlib.sha256(memory).hexdigest()
            # Copy out of the sealed mapping; no alias remains after fd close.
            values = np.frombuffer(memory, dtype='<f4').copy().reshape(length, VOCAB)
        return torch.from_numpy(values).to(device), digest
    finally:
        for fd in fds: os.close(fd)
