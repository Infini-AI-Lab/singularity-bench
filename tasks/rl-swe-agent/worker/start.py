#!/usr/bin/env python3
"""Start the supplied Harbor sandbox worker on the GB300 host, outside the agent container.

Runs Harbor's agent server from the pinned checkout (sources.lock.json) after checking that both patches in
worker/patches are applied, and serves only the staged training tasks with native ARM64 images
(arm64_tasks.py stage). Settings come from mini_slime.config PROFILES['gb300']: 65,536-token context,
8,192 generated tokens per turn, 32 concurrent trials and a 5,400-second episode limit.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
LOCK=json.loads((ROOT/'sources.lock.json').read_text())
PATCHES=[ROOT/'worker/patches/harbor-agent-version.patch',ROOT/'worker/patches/harbor-cancel-cleanup.patch']

async def idle_prune_loop():
    # Leave normal per-trial teardown enabled; never run a shared-host global prune.
    await asyncio.Event().wait()

def resolve_docker_host_alias():
    # Session servers register as host.docker.internal:PORT, the name task containers use for this host, and the
    # worker's heartbeat polls that address from the host, where the name does not resolve. In this process it is
    # the loopback, so heartbeats reach the session servers and in-flight trials are not flushed.
    import socket
    resolve=socket.getaddrinfo
    def getaddrinfo(host,*args,**kwargs):   # anyio (httpx) passes the name IDNA-encoded as bytes
        return resolve('127.0.0.1' if host in ('host.docker.internal',b'host.docker.internal') else host,*args,**kwargs)
    socket.getaddrinfo=getaddrinfo

def validate_arm64_tasks(tasks):
    """GB300 workers accept only prepared native task images on an ARM64 daemon."""
    info = json.loads(subprocess.check_output(['docker','info','--format','{{json .}}'],text=True))
    if info.get('OSType') != 'linux' or info.get('Architecture') not in ('aarch64','arm64'):
        raise ValueError('GB300 task sandboxes require a native Linux ARM64 Docker daemon')
    images = set()
    configs = sorted(tasks.glob('*/task.toml'))
    if not configs:
        raise ValueError('No task definitions found')
    for path in configs:
        image = tomllib.loads(path.read_text()).get('environment',{}).get('docker_image')
        if not image or not image.startswith('sha256:'):
            raise ValueError(f'{path}: use the pinned ARM64 task tree from arm64_tasks.py stage')
        images.add(image)
    for image in sorted(images):
        platform = subprocess.check_output(['docker','image','inspect',image,
            '--format','{{.Os}}/{{.Architecture}}'],text=True).strip()
        if platform != 'linux/arm64':
            raise ValueError(f'{image}: expected linux/arm64, found {platform}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harbor',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--trials',type=Path,required=True)
    p.add_argument('--port',type=int,default=11000)
    a=p.parse_args();harbor=a.harbor.resolve()
    a.trials=a.trials.resolve();a.tasks=a.tasks.resolve()
    commit=subprocess.check_output(['git','-C',str(harbor),'rev-parse','HEAD'],text=True).strip()
    if commit!=LOCK['harbor']['commit']:raise RuntimeError('Wrong Harbor commit')
    for patch in PATCHES:
        if not patch.is_file() or not patch.stat().st_size:raise ValueError(f'Missing patch: {patch}')
        subprocess.run(['git','-C',str(harbor),'apply','--reverse','--check',str(patch)],check=True)
    subprocess.run(['docker','info'],check=True,stdout=subprocess.DEVNULL)
    if not a.tasks.is_dir():raise ValueError('Missing worker task directory')
    validate_arm64_tasks(a.tasks)
    sys.path.insert(0,str(ROOT))
    from mini_slime.config import PROFILES
    c=PROFILES['gb300']
    os.environ.update(HARBOR_TASKS_DIR=str(a.tasks.resolve()),HARBOR_ENV_TYPE='docker',
        HARBOR_DELETE_CONTAINERS='true',HARBOR_KEEP_SANDBOX='false',HARBOR_TIMEOUT_MULTIPLIER='1.0',
        HARBOR_MAX_SEQ_LEN=str(c.context),AGENT_MAX_INPUT_TOKENS=str(c.context),
        AGENT_MAX_OUTPUT_TOKENS=str(c.response),HARBOR_RESPONSE_LENGTH_POLICY='abort',OPENAI_API_KEY='dummy')
    a.trials.mkdir(parents=True,exist_ok=False)
    diff=subprocess.check_output(['git','-C',str(harbor),'diff','HEAD'])
    (a.trials/'worker-provenance.json').write_text(json.dumps({'commit':commit,'profile':'gb300',
        'diff_sha256':hashlib.sha256(diff).hexdigest(),'patch_sha256':{patch.name:hashlib.sha256(patch.read_bytes()).hexdigest() for patch in PATCHES},
        'global_prune':'disabled','tasks':str(a.tasks.resolve()),'host_docker_internal':'127.0.0.1 in the worker process'},indent=2)+'\n')
    (a.trials/'source.patch').write_bytes(diff)
    os.chdir(harbor);sys.path.insert(0,str(harbor))
    resolve_docker_host_alias()
    import miles_agent_server
    miles_agent_server._docker_prune_loop=idle_prune_loop
    sys.argv=[str(harbor/'miles_agent_server.py'),'--host','0.0.0.0','--port',str(a.port),
        '--dashboard-port','0','--dashboard-log-path',str(a.trials.resolve()/'requests.jsonl'),
        '--max-concurrent',str(c.concurrency),'--agent-timeout',str(c.episode_seconds),
        '--trials-dir',str(a.trials.resolve())]
    print('Harbor worker patches checked; global periodic prune disabled.',flush=True)
    miles_agent_server.main()

if __name__=='__main__':main()
