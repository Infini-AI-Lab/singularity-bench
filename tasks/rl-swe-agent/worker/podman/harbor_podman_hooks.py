"""Rootless-podman adaptations of the pinned Harbor agent server and Harbor itself, applied in every process.

environment() gives a Python process the variables under which every Python process it starts imports this module at
startup (a sitecustomize shim in a temporary directory, on PYTHONPATH); worker/start.py --engine podman and the
evaluator's --sandboxes podman use it, and the worker also imports it itself. With HARBOR_PODMAN_SANDBOXES=1 it
patches, as each module is first
imported (the server runs every trial in a fresh ``python -m agent_server.trial_runner`` subprocess, so an in-process
patch would not reach trials, and that module runs as __main__, so the patch sits in modules it imports):
  harbor.environments.factory  the ``docker`` environment type resolves to podman_env.PodmanEnvironment
  agent_server.docker_ops      compose-project listing and teardown act on those containers; the Docker Hub login
                               check and the global prune are no-ops (every task image is pinned and already in the
                               local store)
The pinned Harbor checkout and its patches are not modified.
"""
import importlib.abc
import importlib.machinery
import os
import sys

TARGETS = {'agent_server.docker_ops', 'harbor.environments.factory'}
PREFIX = 'harbor-'


def patch_docker_ops(module):
    import asyncio
    podman = os.environ.get('HARBOR_PODMAN', 'podman')

    async def run(*cmd):
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return proc.returncode or 0, out.decode(errors='replace'), err.decode(errors='replace')

    async def list_projects():
        rc, out, err = await run(podman, 'ps', '-a', '--format', '{{.Names}}')
        if rc:
            module.logger.warning('podman ps failed: %s', err.strip())
            return []
        return [name[len(PREFIX):] for name in out.split() if name.startswith(PREFIX)]

    async def run_docker(*cmd):
        args = list(cmd[1:])
        if args[:1] == ['compose'] and 'down' in args and '-p' in args:
            return await run(podman, 'rm', '--force', '--ignore', PREFIX + args[args.index('-p') + 1])
        if args[:2] == ['ps', '-aq'] and '--filter' in args:
            project = args[args.index('--filter') + 1].split('=', 2)[-1]
            return await run(podman, 'ps', '-aq', '--filter', f'name=^{PREFIX}{project}$')
        if args[:1] in (['network'], ['volume']):
            return 0, '', ''  # podman_env creates no per-trial networks or volumes
        if args[:1] in (['rm'],):
            return await run(podman, *args)
        return 1, '', f'unsupported with podman sandboxes: {" ".join(cmd)}'

    async def no_prune():
        return {'skipped': 'rootless podman store; containers are removed per trial'}

    module._list_compose_projects = list_projects
    module._run_docker = run_docker
    module._prune_docker = no_prune
    module._check_docker_login = lambda: None


def patch_environment_factory(module):
    from harbor.models.environment_type import EnvironmentType
    original = module._load_environment_class

    def load(env_type):
        if env_type == EnvironmentType.DOCKER:
            from podman_env import PodmanEnvironment
            return PodmanEnvironment
        return original(env_type)

    module._load_environment_class = load


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            (patch_docker_ops if name == 'agent_server.docker_ops' else patch_environment_factory)(module)
        loader.exec_module = exec_module
        return spec


def environment(base):
    """``base`` plus what makes a Python process and its subprocesses run Harbor's docker environment as rootless
    podman: HARBOR_PODMAN_SANDBOXES, and a PYTHONPATH whose sitecustomize imports this module."""
    import tempfile
    from pathlib import Path
    shim = Path(tempfile.mkdtemp(prefix='harbor-podman-'))
    (shim / 'sitecustomize.py').write_text('import harbor_podman_hooks  # noqa: F401\n')
    here = str(Path(__file__).resolve().parent)
    return {**base, 'HARBOR_PODMAN_SANDBOXES': '1',
            'PYTHONPATH': os.pathsep.join([str(shim), here, *filter(None, [base.get('PYTHONPATH')])])}


def install():
    """Patch the two modules in this process as they are first imported."""
    if not any(isinstance(item, Finder) for item in sys.meta_path):
        sys.meta_path.insert(0, Finder())


if os.environ.get('HARBOR_PODMAN_SANDBOXES') == '1':
    install()
