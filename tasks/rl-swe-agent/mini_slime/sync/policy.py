"""Synchronous policy-version barrier: rollouts are ready only after the latest checkpoint is loaded into SGLang."""
from .disk_sync import update_sglang_from_disk

class PolicyBarrier:
    def __init__(self):
        self.version=None
        self.dirty=True

    def mark_updated(self):
        self.dirty=True

    def sync(self,url,path,version):
        if type(version) is not int or version<0 or (self.version is not None and version<=self.version):
            raise ValueError('Policy versions must increase')
        self.dirty=True
        result=update_sglang_from_disk(url,str(path),version)
        self.version=version
        self.dirty=False
        return result

    def require_ready(self):
        if self.dirty or self.version is None:
            raise RuntimeError('Policy not synchronized')
        return str(self.version)
