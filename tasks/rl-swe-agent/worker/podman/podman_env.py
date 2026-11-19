"""Harbor's Docker environment, run with rootless podman: the task sandboxes of the sandbox worker and the evaluator on
hosts without Docker (an Apptainer cluster). Selected by harbor_podman_hooks.py in this directory, which maps Harbor's
``docker`` environment type to ``PodmanEnvironment`` when HARBOR_PODMAN_SANDBOXES=1.

Everything above the container runtime is Harbor's own DockerEnvironment: mounts, task and persistent environment,
working directory, users, uploads and downloads, output streaming and timeouts. The few ``docker compose`` subcommands
it issues for a prebuilt single-service task are translated here to ``podman``.

Differences from Docker, all because the runtime is rootless, without subordinate UID ranges or cgroup delegation:
  * container root is the calling user (no other UIDs exist inside); apt is told, by a read-only bind of
    apt-root-mapped.conf, not to switch to its _apt user, and tar does not restore file owners
    (TAR_OPTIONS=--no-same-owner);
  * task CPU, memory and storage limits are recorded but not enforced;
  * networking is slirp4netns with public egress (the tasks declare public networking); a sandbox reaches services
    on the host's loopback as host.docker.internal (10.0.2.2);
  * tasks must use a prebuilt image ([environment] docker_image) and no compose sidecars.
"""
import asyncio
import os
import shutil
import subprocess
from pathlib import Path

from harbor.environments.base import ExecResult, OutputCallback
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

PODMAN = os.environ.get('HARBOR_PODMAN', 'podman')
HOST_ALIAS = 'host.docker.internal:10.0.2.2'
# apt drops privileges to _apt, which a root-mapped container cannot do (agent installers run apt-get).
APT_CONFIG = Path(__file__).resolve().parent / 'apt-root-mapped.conf'


class PodmanEnvironment(DockerEnvironment):
    @classmethod
    def preflight(cls) -> None:
        if not shutil.which(PODMAN):
            raise SystemExit('podman is not on PATH')
        subprocess.run([PODMAN, 'info'], capture_output=True, timeout=60, check=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._enable_egress_control:
            raise ValueError('The podman environment supports public task networking only')
        if not self.task_env_config.docker_image:
            raise ValueError('The podman environment requires a prebuilt task image')
        self._container = 'harbor-' + _sanitize_docker_compose_project_name(self.session_id)

    def _validate_daemon_mode(self) -> None:
        if self._is_windows_container:
            raise RuntimeError('Windows task containers are not supported by podman')

    async def _validate_image_os(self, image_name: str) -> None:
        result = await self._podman(['image', 'inspect', '--format', '{{.Os}}', image_name], check=True)
        if (result.stdout or '').strip() != 'linux':
            raise RuntimeError(f'{image_name} is not a Linux image')

    async def prepare_logs_for_host(self) -> None:
        """Container root is the submitting user; bind-mounted logs are already host-owned."""

    async def _remove_project_network(self) -> None:
        pass

    async def _remove_base_image(self) -> None:
        pass

    async def _cleanup_build_cache(self) -> None:
        pass

    async def _podman(self, args, *, check=False, timeout_sec=None, stdin_data=None,
                      on_output: OutputCallback | None = None) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            PODMAN, *args,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        if on_output is not None:
            result = await self._collect_streamed_output(process, timeout_sec=timeout_sec, stdin_data=stdin_data,
                                                         on_output=on_output)
        else:
            result = await self._collect_buffered_output(process, timeout_sec=timeout_sec, stdin_data=stdin_data)
        if check and result.return_code != 0:
            raise RuntimeError(f'podman command failed for environment {self.environment_name}. '
                               f'Command: podman {" ".join(args)}. Return code: {result.return_code}. '
                               f'Output: {result.stdout}')
        return result

    def _run_args(self) -> list[str]:
        # A kernel may refuse a fresh sysfs mount in a rootless network namespace: bind the host's read-only. A
        # site's containers.conf may set pidns="host", under which podman cannot stop a sandbox and its processes
        # outlive removal: each sandbox gets its own PID namespace, as with Docker.
        args = ['run', '--detach', '--name', self._container, '--pid', 'private',
                '--network', 'slirp4netns:allow_host_loopback=true',
                '--add-host', HOST_ALIAS, '--dns', '8.8.8.8', '--volume', '/sys:/sys:ro',
                '--volume', f'{APT_CONFIG}:/etc/apt/apt.conf.d/99-root-mapped-sandbox:ro',
                # Root's tar restores archived owners, which do not exist here (agent installers unpack as root).
                '--env', 'TAR_OPTIONS=--no-same-owner']
        for mount in self._mounts:
            if mount.get('type', 'bind') != 'bind' or not mount.get('source') or not mount.get('target'):
                raise ValueError(f'Unsupported mount for podman: {mount}')
            Path(mount['source']).mkdir(parents=True, exist_ok=True)
            args += ['--volume', f"{mount['source']}:{mount['target']}" + (':ro' if mount.get('read_only') else '')]
        for key, value in self._startup_env().items():
            args += ['--env', f'{key}={value}']
        if self.task_env_config.workdir:
            args += ['--workdir', self.task_env_config.workdir]
        return args + [self.task_env_config.docker_image, 'sh', '-c', 'sleep infinity']

    async def _run_docker_compose_command(self, command, check=True, timeout_sec=None, stdin_data=None,
                                          on_output: OutputCallback | None = None) -> ExecResult:
        verb, rest = command[0], command[1:]
        if verb == 'build':
            raise RuntimeError('The podman environment uses prebuilt task images only')
        if verb == 'down':
            return await self._podman(['rm', '--force', '--ignore', self._container], timeout_sec=timeout_sec)
        if verb == 'stop':
            return await self._podman(['stop', '--ignore', '--time', '10', self._container], check=check,
                                      timeout_sec=timeout_sec)
        if verb == 'up':
            return await self._podman(self._run_args(), check=check, timeout_sec=timeout_sec)
        if verb == 'cp':
            paths = [self._container + arg[len('main'):] if arg.startswith('main:') else arg for arg in rest]
            return await self._podman(['cp', *paths], check=check, timeout_sec=timeout_sec)
        if verb == 'exec':
            args, index = ['exec'] + (['--interactive'] if stdin_data is not None else []), 0
            while index < len(rest) and rest[index].startswith('-'):
                flag = rest[index]
                if flag == '-T':
                    index += 1
                    continue
                if flag not in ('-w', '-e', '-u'):
                    raise ValueError(f'Unsupported exec flag {flag}')
                args += [flag, rest[index + 1]]
                index += 2
            if rest[index] != 'main':
                raise ValueError('The podman environment has no compose sidecars')
            return await self._podman(args + [self._container, *rest[index + 1:]], check=check,
                                      timeout_sec=timeout_sec, stdin_data=stdin_data, on_output=on_output)
        raise ValueError(f'Unsupported compose command for podman: {command}')
