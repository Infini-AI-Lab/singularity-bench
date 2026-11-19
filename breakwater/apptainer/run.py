#!/usr/bin/env python3
"""Breakwater launcher for the unprivileged Apptainer backend.

    python3 breakwater/apptainer/run.py --task tasks/<name> --image TASK.sif --cli-dir AUTHOR_CLI_DIR \
      --judge-dir JUDGE_DIR --run-dir RUN_DIR --asset-root ASSET_ROOT [--bindings bindings.json] \
      [--author-cli claude-code|codex] --author-model MODEL --author-upstream https://PROVIDER \
      --author-key-env KEY_VAR [--judge-key-env KEY_VAR]

It reads the same task specification (task.toml, breakwater/SPEC.md) as the Docker launcher and writes the same run
directory layout: external/ (evidence), agent-local/, submission/ (the sealed workspace), evaluation/ (the sealed
raw evaluation outputs), authority/ (the sealed launch record) and reports/ (reward and verifier reports). The three
stages run in fresh sandboxes from the task's SIF, one after another: authoring, evaluation and the verifier
(breakwater/verifier.py, with the judge runner from install_judge.py and the judge key). Keys never reach a
sandbox's environment except the judge key in the verifier's.

The agent CLI is Claude Code or Codex (install_cli.py --agent), with the same settings as in the Docker author image.
Claude Code runs each command through its shell prefix, the broker's client. Codex sends every command and file
operation to its executor, ``codex exec-server``, which runs beside it in the author sandbox and starts each command
as the broker's client (codex_forward.py). Either way, each command the agent runs is a command of the broker in the
command sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def _docker_launcher():
    """The Docker launcher's contract, binding and seeding logic (this file shares its name, hence by path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('breakwater_launcher', HERE.parent / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract_tools = _docker_launcher()
from agent_stage import AuthoringStage, image_environment  # noqa: E402
from evaluation_stage import FRAMEWORK, EvaluationStage  # noqa: E402
from contract import (CREDENTIAL_ENV, FRAMEWORK_REVISION, ContractError, RulesViolation, atomic_json, digest,  # noqa: E402
                      require, seal, tests_digest, write_canary)
import host_hooks  # noqa: E402
import plan as plan_tools  # noqa: E402
import spec  # noqa: E402
from environment_guard import shadow_policy, shadows  # noqa: E402
from sandbox import SCRATCH, Sandbox  # noqa: E402


# Forwarded to the verifier, which reaches the judge's provider over the host network.
PROXY_ENV = ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy')

# Claude Code inside the author sandbox: the gateway is the relay on 127.0.0.1:8001 and every command goes through
# the shell-prefix client to the command sandbox.
CLAUDE_SETTINGS = {
    'env': {'CLAUDE_CODE_SHELL_PREFIX': '/opt/breakwater-runtime/breakwater-exec', 'CLAUDE_CODE_SHELL': '/bin/bash',
            'SHELL': '/bin/bash', 'DISABLE_AUTOUPDATER': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL': '1', 'CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS': '1'},
    'disableAllHooks': True, 'allowManagedHooksOnly': True, 'allowManagedMcpServersOnly': True,
    'allowedMcpServers': [], 'strictKnownMarketplaces': [], 'disableSideloadFlags': True,
    'allowManagedPermissionRulesOnly': True, 'permissions': {'deny': ['Read(//opt/author-cli/**)', 'Read(//proc/**)']},
}


# Codex's execution environment: its own executor beside it in the author sandbox, every command forwarded to the
# broker (codex_forward.py).
CODEX_FORWARD = '/opt/breakwater-runtime/codex-forward.py'


def codex_command(model, instruction, workdir):
    """Codex in the author sandbox, as the Docker author image launches it (author/launch.py)."""
    argv = ['/opt/author-cli/codex', 'exec', '--strict-config', '--skip-git-repo-check', '--json', '--cd', workdir,
            '--', instruction]
    env = {'HOME': '/home/author', 'USER': 'author', 'LOGNAME': 'author', 'LANG': 'C.UTF-8', 'SHELL': '/bin/bash',
           'PATH': '/usr/local/bin:/usr/bin:/bin', 'CODEX_HOME': '/home/author/.codex',
           'BREAKWATER_AUTHOR_KEY': 'gateway-only-not-a-provider-key', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null'}
    return argv, env


def codex_files(model, workdir, folder):
    """Codex's settings (author/codex_config.py, as in the Docker author image) and its execution environment as
    read-only files for the author sandbox: the model is the gateway's relay, and the one execution environment is
    codex_forward.py, which runs Codex's executor there and forwards every command to the broker. Returns (host file,
    target) pairs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('codex_config', HERE.parent / 'author' / 'codex_config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    files = config.texts(model, workdir, base_url='http://127.0.0.1:8001/v1',
                         executor=('/usr/bin/env', ['python3', '-I', CODEX_FORWARD]))
    targets = {'config.toml': '/home/author/.codex/config.toml',
               'environments.toml': '/home/author/.codex/environments.toml',
               'requirements.toml': '/etc/codex/requirements.toml'}
    pairs = []
    for name, target in targets.items():
        (folder / name).write_text(files[name])
        (folder / name).chmod(0o444)
        pairs.append((folder / name, target))
    return pairs + [(HERE / 'codex_forward.py', CODEX_FORWARD)]


def agent_cli(args, task, workdir, folder):
    """The agent CLI: its argv and environment, read-only files for the author sandbox and facts for the launch
    record."""
    cli = {'author_files': [], 'record': {}}
    if args.cli_command:
        cli['argv'], cli['env'] = json.loads(args.cli_command)
        return cli
    installed = args.cli_dir / 'harness'
    installed = installed.read_text().strip() if installed.is_file() else 'claude-code'
    require(installed == args.author_cli, f'--cli-dir holds {installed}, not {args.author_cli}: install it with '
            f'install_cli.py --agent {args.author_cli}')
    version = args.cli_dir / 'version'
    cli['record'] = {'agent_cli_version': version.read_text().strip() if version.is_file() else None}
    instruction = (task / 'instruction.md').read_text()
    folder.mkdir(mode=0o700)
    if args.author_cli == 'codex':
        cli['argv'], cli['env'] = codex_command(args.author_model, instruction, workdir)
        cli['author_files'] = codex_files(args.author_model, workdir, folder)
        return cli
    cli['argv'], cli['env'] = claude_command(args.author_model, instruction)
    settings = folder / 'managed-settings.json'
    settings.write_text(json.dumps(CLAUDE_SETTINGS, indent=1))
    settings.chmod(0o444)
    cli['author_files'] = [(settings, '/etc/claude-code/managed-settings.json')]
    return cli


def command_environment(config, image_env):
    """The agent's commands' environment: the image's, plus the task's authoring environment
    ([breakwater.authoring.env]), as the Docker author image gives them."""
    return {**image_env, **config['authoring_env']}


def model_access_service(config, key_env, held, external):
    """The development gateway of [breakwater.model_access] as a service for the agent's commands, at the task's host
    and port (gateway:8000 by default), with the same settings as Docker's sidecar; only this host process holds its
    key."""
    access = config['development_gateway']
    if not access:
        return []
    require(held.get(key_env), f'Set {key_env} to the development gateway key (model_access)')
    folder = external / 'dev-gateway'
    return [{'name': 'dev-gateway', 'port': access['port'], 'host': access['host'],
             'argv': [sys.executable, '-I', str(HERE.parent / 'dev_gateway.py')],
             'env': {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'UPSTREAM_API_KEY': held[key_env],
                     **spec.model_access_env(access),
                     'GATEWAY_LOG': str(folder / 'requests.jsonl'), 'GATEWAY_STATE': str(folder / 'budget.json'),
                     **{name: os.environ[name] for name in PROXY_ENV if name in os.environ}}}]


def host_service_ports(forwarded):
    """The ports of the host services named by loopback URLs among the forwarded evaluation variables. The coordinator
    has no network of its own, so a URL of any other host is refused."""
    import urllib.parse
    ports = []
    for name, value in forwarded.items():
        url = urllib.parse.urlsplit(value)
        if url.scheme not in ('http', 'https') or not url.netloc:
            continue
        require(url.hostname in ('127.0.0.1', 'localhost'), f'{name} must name a service on this host '
                f'(http://127.0.0.1:PORT): the evaluation reaches host services only through relays')
        ports.append(url.port or (443 if url.scheme == 'https' else 80))
    return sorted(set(ports))


def claude_command(model, instruction):
    argv = ['/opt/author-cli/claude', '--verbose', '--output-format=stream-json', '--permission-mode=bypassPermissions',
            '--setting-sources', '', '--tools', 'Bash,Read,Write,Edit', '--model', model, '--print', '--', instruction]
    env = {'HOME': '/home/author', 'USER': 'author', 'LANG': 'C.UTF-8', 'SHELL': '/bin/bash',
           'PATH': '/usr/local/bin:/usr/bin:/bin', 'ANTHROPIC_API_KEY': 'gateway-only-not-a-provider-key',
           'ANTHROPIC_BASE_URL': 'http://127.0.0.1:8001', 'ANTHROPIC_MODEL': model,
           'CLAUDE_CONFIG_DIR': '/logs/agent/sessions', 'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1',
           'DISABLE_ERROR_REPORTING': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
           **{key: model for key in ('ANTHROPIC_DEFAULT_SONNET_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL',
                                     'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL')}}
    return argv, env


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def tree_inventory(root, excludes=()):
    """Regular files (sha256, bytes) and symbolic links (link text) of a host directory, relative paths."""
    files, links = {}, {}
    root = Path(root)
    for directory, dirs, names in os.walk(root):
        for name in [*dirs, *names]:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if any(relative == item or relative.startswith(item.rstrip('/') + '/') for item in excludes):
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                links[relative] = os.readlink(path)
            elif stat.S_ISREG(info.st_mode):
                files[relative] = {'sha256': file_sha256(path), 'bytes': info.st_size}
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
    return files, links


def snapshot_tree(source, destination, excludes, links):
    """Copy a stopped stage's workspace into new host-owned files. Symbolic links are recorded (relative path ->
    link text) and skipped; hard links and special files fail the snapshot, as in the Docker launcher."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(mode=0o700)
    for directory, dirs, names in os.walk(source):
        for name in sorted([*dirs, *names]):
            path = Path(directory) / name
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if any(str(relative) == item or str(relative).startswith(item.rstrip('/') + '/') for item in excludes):
                if name in dirs:
                    dirs.remove(name)
                continue
            info = path.lstat()
            target = destination / str(relative)
            if stat.S_ISLNK(info.st_mode):
                links[str(relative)] = os.readlink(path)
                if name in dirs:
                    dirs.remove(name)
            elif stat.S_ISDIR(info.st_mode):
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(info.st_mode):
                require(info.st_nlink == 1, 'Hard-linked file in the submitted workspace: ' + str(relative))
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as reader, target.open('xb') as writer:
                    shutil.copyfileobj(reader, writer, length=1 << 20)
                target.chmod(0o755 if info.st_mode & 0o111 else 0o644)
            else:
                raise RulesViolation('Special file in the submitted workspace: ' + str(relative))


def worker_environment(config):
    """The submitted code's environment (the task's evaluation.worker_env), as the Docker coordinator gives its worker."""
    import task_api
    return task_api.worker_environment(config)


def seed(config, task, image, destination, scratch=None):
    """The agent's starting workspace: the task tree, or the image's own workdir."""
    if config['workspace_source'] == 'task':
        contract_tools.TASK = task
        contract_tools.seed_workspace(config, None, destination)
        return
    destination.mkdir(mode=0o700)
    box = Sandbox(image, binds=[(destination, '/breakwater-seed', 'rw')], scratch=scratch)
    try:
        code, _, err = box.run(['sh', '-c', f'cp -a {config["workdir"]}/. /breakwater-seed/'], timeout=1800)
        require(code == 0, 'Could not seed the workspace from the image: ' + err[-500:])
    finally:
        box.stop()


# The prefetched caches the Docker author image moves from /root into the worker's home (author/install_task.py).
WORKER_CACHES = ('.cache/huggingface', '.cache/torch', '.cache/torch_extensions', '.triton')


def worker_home_seed(image, destination, scratch=None):
    """The worker's starting home, as the Docker author image makes it: the image's /home/worker and its prefetched
    caches from /root, so offline kernels and models are found. Each stage's fresh worker home starts as a copy."""
    destination.mkdir(mode=0o700)
    script = 'set -e; if [ -d /home/worker ]; then cp -a /home/worker/. /breakwater-seed/; fi; ' + '; '.join(
        f'if [ -e /root/{cache} ]; then mkdir -p /breakwater-seed/{os.path.dirname(cache) or "."} && '
        f'cp -a /root/{cache} /breakwater-seed/{cache}; fi' for cache in WORKER_CACHES)
    box = Sandbox(image, binds=[(destination, '/breakwater-seed', 'rw')], scratch=scratch)
    try:
        code, _, err = box.run(['sh', '-c', script], timeout=1800)
        require(code == 0, 'Could not seed the worker home from the image: ' + err[-500:])
    finally:
        box.stop()


def read_only(root):
    """Directories 0555 and files 0444, as the Docker host leaves evidence before the verifier reads it."""
    for directory, _, names in os.walk(root):
        Path(directory).chmod(0o555)
        for name in names:
            (Path(directory) / name).chmod(0o444)


def verify(*, run, task, config, contract, image, image_env, bindings, judge_dir, judge_key, scratch, identity, stages,
           seals, authority, reports, logs, extra_roots=(), extra=None, home=None, judge_concurrency=None):
    """The verifier stage: a fresh sandbox from the run's image (the Docker verifier image is the task image plus the
    framework and the judge) with every input read-only, its own reports directory, the host network for the judge
    and the judge key. The host records the sandbox and seals the authority with it, as Docker inspects the verifier
    before sealing, then runs breakwater/verifier.py there. Returns the verifier's exit code and the authority digest.

    ``task`` is the verification task's directory: its instruction.md, its tests/ (the task code the verifier runs and
    the rubric, tests/submission_judge.md). The submission is bound at the contract's submission_root, else its workdir.
    A custom launcher (one that runs a submitted agent, say) passes a directory of its own and seals ``recording`` and
    ``task_metadata``
    through ``extra``."""
    external, local, workspace, evaluation = (run / name for name in ('external', 'agent-local', 'submission', 'evaluation'))
    rows = [row for row in bindings if 'verifier' in row['stages']]
    verifier = Sandbox(image, network=True, role='verifier', scratch=scratch, binds=[
        (external, '/external-evidence', 'ro'), (authority, '/authority', 'ro'), (local, '/agent-local', 'ro'),
        (workspace, config.get('submission_root') or config['workdir'], 'ro'), (evaluation, '/evaluation', 'ro'),
        (reports, '/reports', 'rw'), (FRAMEWORK, '/opt/breakwater', 'ro'), (contract, '/trusted-task/contract.json', 'ro'),
        (task / 'instruction.md', '/trusted-task/instruction.md', 'ro'),
        (task / 'tests' / 'submission_judge.md', '/trusted-task/tests/submission_judge.md', 'ro'),
        (task / 'tests', '/tests', 'ro'),
        (judge_dir, '/opt/breakwater-judge', 'ro'), *[(row['source'], row['target'], 'ro') for row in rows],
        *([(home, spec.HOME_MOUNT, 'ro')] if home else [])])
    try:
        stages = {**stages, 'verifier': {'started_ns': verifier.started_ns, 'started_wall_ns': time.time_ns(),
                                         'sandboxes': [verifier.describe()]}}
        authority_sha = seal(authority, identity, backend='apptainer', stages=stages, **seals,
                             external_host_root=str(external),
                             protected_host_roots=[str(external), str(authority), str(reports), *map(str, extra_roots)],
                             pipeline={key: config[key] for key in spec.SEALED_PIPELINE},
                             contract_sha256=digest(contract), framework_revision=FRAMEWORK_REVISION,
                             tests_sha256=tests_digest(task / 'tests'), task_directory=task.name, **(extra or {}))
        read_only(authority)
        environment = {**image_env, 'BREAKWATER_STAGE': 'verifier', 'BREAKWATER_BACKEND': 'apptainer',
                       'BREAKWATER_CODEX': '/opt/breakwater-judge/codex', 'FIREWORKS_API_KEY': judge_key,
                       'HOME': '/tmp/verifier-home', 'XDG_CACHE_HOME': '/tmp/verifier-home/cache',
                       'PYTHONDONTWRITEBYTECODE': '1', **{name: os.environ[name] for name in PROXY_ENV if name in os.environ},
                       **({'BREAKWATER_JUDGE_CONCURRENCY': str(judge_concurrency)} if judge_concurrency else {})}
        # This is the only stage that can read external evidence or call the judge model.
        code, out, err = verifier.run(['python3', '-I', '-B', '/opt/breakwater/verifier.py', '--authority',
                                       authority_sha], cwd='/tmp', timeout=config['judge_timeout_sec'] + 600,
                                      env=environment)
        Path(f'{logs}.stdout').write_text(out)
        Path(f'{logs}.stderr').write_text(err)
    finally:
        verifier.stop_proven()
    return code, authority_sha


def filesystem_type(path):
    """The type of the filesystem that holds path, from this process's mount table."""
    path, best, kind = os.path.realpath(path), '', None
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        point = fields[4].replace('\\040', ' ')
        if (path == point or path.startswith(point.rstrip('/') + '/')) and len(point) >= len(best):
            best, kind = point, fields[fields.index('-') + 1]
    return kind


def say(text, stream):
    """Best effort: the failure being reported may be a full disk under stdout or stderr too."""
    try:
        print(text, file=stream, flush=True)
    except OSError:
        pass


def check_no_keys(held, *environments):
    """No held key's value may appear in an environment given to a sandboxed process."""
    keys = [value for value in held.values() if len(value) >= 8]
    require(not any(key in str(value) for environment in environments for value in environment.values() for key in keys),
            'A provider or judge key would reach a sandbox')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True, help='the task runtime SIF')
    parser.add_argument('--cli-dir', type=Path, required=True, help='host directory holding the agent CLI')
    parser.add_argument('--judge-dir', type=Path, required=True, help='host directory holding the judge runner')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, help="moved asset sources, relative to --asset-root (default: the task's)")
    parser.add_argument('--author-cli', choices=('claude-code', 'codex'), default='claude-code',
                        help='the agent CLI in --cli-dir (install_cli.py --agent)')
    parser.add_argument('--author-model', required=True)
    parser.add_argument('--author-upstream', required=True)
    parser.add_argument('--author-key-env', required=True)
    parser.add_argument('--author-auth', choices=('bearer', 'anthropic'), default='bearer')
    parser.add_argument('--judge-key-env', default='FIREWORKS_API_KEY', help='variable holding the judge key')
    parser.add_argument('--judge-concurrency', type=int, choices=range(1, 65), metavar='1-64',
                        help='judge calls in flight (default 8): raise it, within the provider\'s rate limit, so a long '
                             "run's review fits the task's judge budget")
    parser.add_argument('--dev-key-env',
                        help="variable holding the development gateway's key (tasks with [breakwater.model_access])")
    parser.add_argument('--scratch', type=Path, help="host directory for each sandbox's private /tmp, on local disk "
                                                     '(default: $BREAKWATER_SCRATCH or the system temporary directory)')
    parser.add_argument('--endpoint', action='append', default=[], metavar='NAME=HOST:PORT',
                        help="where one of the task's [[breakwater.endpoints]] really is (by default the service the task "
                             'declares); the sandboxes still use its declared name and port')
    parser.add_argument('--input', action='append', default=[], metavar='NAME=PATH',
                        help="a file the task's [breakwater.operator] inputs declare (its host hooks read it)")
    parser.add_argument('--value-env', action='append', default=[], metavar='NAME=VAR',
                        help="the variable holding a declared operator value (default: the task's)")
    parser.add_argument('--secret-env', action='append', default=[], metavar='NAME=VAR',
                        help="the variable holding a declared operator secret (default: the task's); held, never in a "
                             'sandbox')
    parser.add_argument('--smoke', action='store_true',
                        help="a smoke run: every stage runs, but the task's evaluation uses its reduced smoke settings "
                             '(breakwater/SPEC.md); the run is sealed as smoke and its reward is not a benchmark score')
    parser.add_argument('--cli-command', help='JSON [argv, env] replacing the agent CLI (tests and other CLIs)')
    parser.add_argument('--coordinator-command', help='JSON argv replacing intermediate.py (tests)')
    args = parser.parse_args()
    task = args.task.resolve()
    config = spec.load(task)
    run = Path(os.path.realpath(args.run_dir))
    # The task's host side: the operator's inputs, values and secrets (secrets leave this environment now); its
    # preflight hook runs below, before the run directory exists.
    task_host = host_hooks.TaskHost(task, config, 'apptainer', args.smoke, args.input, args.value_env, args.secret_env,
                                    avoid=(task, run))
    # Keys leave this process's environment, so no spawner, Slurm step or sandbox inherits them; the gateways and the
    # verifier are given theirs explicitly. They include the task's own credentials (its model-access key, say).
    dev_key_env = args.dev_key_env or (config['development_gateway'] or {}).get('key_env')
    held = {name: os.environ.pop(name) for name in {*CREDENTIAL_ENV, *config['credential_env'], args.author_key_env,
                                                    args.judge_key_env,
                                                    *([dev_key_env] if dev_key_env else [])} if name in os.environ}
    require(held.get(args.author_key_env), 'Missing the author provider key')
    require(held.get(args.judge_key_env), 'Missing the judge key')
    require((args.judge_dir / 'codex').is_file(), 'Missing the judge runner: install it with install_judge.py --dest DIR')
    # A task that needs a setting this backend does not support is refused, not run half-isolated.
    unported = spec.unsupported(config, 'apptainer')
    require(not unported, 'The Apptainer backend does not run tasks that use these: ' + ', '.join(unported))
    require(args.author_upstream.startswith(('https://', 'http://127.0.0.1')), 'Author upstream must use HTTPS')
    # Codex speaks the Responses API, which the gateway relays with bearer authentication only.
    protocol = 'responses' if args.author_cli == 'codex' else 'messages'
    require(protocol == 'messages' or args.author_auth == 'bearer', 'Codex needs --author-auth bearer')
    task_host.preflight(reserved=spec.reserved_names(config))
    # Its credential variables, as the verifier checks them, include any the operator renamed.
    config = {**config, 'credential_env': sorted({*config['credential_env'], *task_host.operator.credential_names})}
    # The task's static endpoints with the operator's overrides, healthy before authoring starts, and its hooks'.
    static = plan_tools.static_endpoints(config, args.endpoint)
    plan_tools.check_health(static)
    roles = plan_tools.by_role([*static, *task_host.endpoints()])
    from endpoints import addresses
    for role, rows in roles.items():
        try:
            addresses(rows)
        except ValueError as error:
            raise ContractError(f"The {role}'s endpoints: {error}") from error
    low, high = spec.port_range(config['inbound_ports'], 'inbound_ports') if config['inbound_ports'] else (1, 0)
    inbound = list(range(low, high + 1))  # servers the agent's commands run, reachable from the host's loopback
    scratch = Path(os.path.realpath(args.scratch or SCRATCH))
    require(scratch.is_dir(), 'The scratch directory does not exist: ' + str(scratch))
    if filesystem_type(scratch) in ('tmpfs', 'ramfs'):
        say(f"breakwater: each sandbox's /tmp goes under {scratch}, which is held in memory; pass --scratch on "
            'local disk', sys.stderr)
    if os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURMD_NODENAME'):
        say('breakwater: in a Slurm job but not on one of its nodes (an salloc shell on a login node?): sandboxes run '
            'on this host, without per-stage memory limits; start the launcher on a node of the job (srun or sbatch)',
            sys.stderr)
    require(not run.exists() and not contract_tools.overlaps(run, task), 'Use a fresh run directory outside the task')
    bindings = spec.bindings(config, args.asset_root, run, args.bindings)
    workdir = config['workdir']
    excludes = [row['target'][len(workdir) + 1:] for row in bindings
                if 'agent' in row['stages'] and row['target'].startswith(workdir + '/')] + config['snapshot_excludes']
    run.mkdir(parents=True, mode=0o700)
    task_host.adopt(run)
    external, authority, local, workspace, evaluation, reports = (
        run / name for name in ('external', 'authority', 'agent-local', 'submission', 'evaluation', 'reports'))
    for folder in (external, authority, local, evaluation, reports):
        folder.mkdir(mode=0o700)
    # The resolved contract the coordinator and the verifier read (sealed into the authority by its digest).
    contract = run / 'contract.json'
    atomic_json(contract, config)
    contract.chmod(0o444)
    task_plan_sha = task_host.record(external, static)
    write_canary(external)
    identity = {'task': config['task'], 'run_id': secrets.token_hex(16)}
    stages = {}
    try:
        live = run / 'workspace-live'
        seed(config, task, args.image, live, scratch)
        task_host.write_workspace_files(live)
        policy = shadow_policy(config['shadow_policy'])
        baseline_files, baseline_links = tree_inventory(live, excludes)
        image_sha = file_sha256(args.image)
        baseline = {'version': 'breakwater-apptainer-v1', 'identity': identity, 'image_sha256': image_sha,
                    'specification': {'shadow': policy}, 'workspace_shadows': shadows(live, policy, excludes)}
        atomic_json(external / 'environment-baseline.json', baseline)
        # The instructions promise the agent a read-only copy of the pre-attempt inventory.
        baseline_dir = run / 'environment-baseline'
        baseline_dir.mkdir(mode=0o700)
        shutil.copyfile(external / 'environment-baseline.json', baseline_dir / 'environment.json')
        (baseline_dir / 'environment.json').chmod(0o444)
        baseline_bind = (baseline_dir, '/opt/environment-baseline', 'ro')
        image_env = image_environment(args.image)
        home_seed = run / 'worker-home-seed'
        worker_home_seed(args.image, home_seed, scratch)
        atomic_json(external / 'workspace-baseline.json', {'files': baseline_files, 'symlinks': baseline_links})
        cli = agent_cli(args, task, workdir, run / 'agent-config')
        services = model_access_service(config, dev_key_env, held, external)
        # In the command sandbox the inbound host is its own loopback, where the servers the agent's commands run
        # listen; the host relays those ports to them (each replacement sandbox gets its own relays).
        atomic_json(external / 'authoring-plan.json', spec.authoring_plan(
            config, endpoints=[{'name': row['name'], 'host': row['host'], 'ports': plan_tools.ranges(row['ports'])}
                               for row in roles['agent']],
            hosts=[[config['inbound_host'], '127.0.0.1']] if inbound else [],
            reply_ports=config['inbound_ports'].replace('-', ':') if inbound else None,
            env={**config['authoring_env'], **task_host.agent_env()}))
        task_private = task_host.write_private_files(run)
        home_live = run / 'home-live' if config['snapshot_worker_home'] else None
        command_env = {**command_environment(config, image_env), **task_host.agent_env()}
        check_no_keys({**held, **{f'operator {name}': value for name, value in task_host.operator.secrets.items()}},
                      cli['env'], image_env, command_env)
        task_host.before_authoring(external)
        agent_binds = [(row['source'], row['target'], 'ro') for row in bindings if 'agent' in row['stages']] + [baseline_bind]
        stage = AuthoringStage(image=args.image, workspace=live, workdir=workdir, external=external, local=local,
                               cli_dir=args.cli_dir, cli_argv=cli['argv'], cli_env=cli['env'],
                               author_files=cli['author_files'],
                               gateway={'model': args.author_model, 'key': held[args.author_key_env],
                                        'upstream': args.author_upstream, 'auth': args.author_auth,
                                        'protocol': protocol},
                               budget=config['agent_timeout_sec'], agent_binds=agent_binds, gpus=config['gpus'] > 0,
                               command_env=command_env,
                               scratch=scratch, services=services, worker_home_seed=home_seed,
                               endpoints=roles['agent'],
                               hosts={config['inbound_host']: '127.0.0.1'} if inbound else None, inbound_ports=inbound,
                               home_snapshot=home_live,
                               memory_mb=(config.get('agent_resources') or {}).get('memory_mb')
                               or config['resources'].get('memory_mb'),
                               cpus=(config.get('agent_resources') or {}).get('cpus') or config['resources'].get('cpus'))
        record = stages['agent'] = {**stage.run(), 'agent_cli': args.author_cli, **cli['record']}
        if not (config['agent_deadline'] == 'evaluate' or (record['exit_code'] == 0 and not record['deadline_reached'])):
            raise ContractError('The agent failed; evidence is retained without a reward')
        # The environment check: the image cannot have changed, and the workspace may not shadow a protected package.
        links = {}
        snapshot_tree(live, workspace, excludes, links)
        report = {'baseline_sha256': digest(external / 'environment-baseline.json'), 'image_sha256_before': image_sha,
                  'image_sha256_after': file_sha256(args.image),
                  'workspace_shadow_check': 'PASS' if shadows(workspace, policy, links=links) == baseline['workspace_shadows']
                  else 'FAIL'}
        protected = contract_tools.protected_paths_check(baseline_files, workspace, config['protected_workspace_paths'])
        atomic_json(external / 'protected-workspace.json', protected)
        home = home_sha = None
        home_links = {}
        if home_live is not None:
            # The worker's home as the agent left it, sealed with the submission (workspace.snapshot_home).
            require(home_live.is_dir(), 'The worker home was not snapshotted')
            home = run / 'submission-home'
            snapshot_tree(home_live, home, [], home_links)
            shutil.rmtree(home_live, ignore_errors=True)
        atomic_json(external / 'workspace-links.json', {'workdir': workdir, 'symlinks': links,
                    **({'home': {'path': '/home/worker', 'symlinks': home_links}} if home else {})})
        report['status'] = 'PASS' if (report['image_sha256_after'] == image_sha and report['workspace_shadow_check'] == 'PASS'
                                      and record['process_wall'] == {'author': True, 'command': True}) else 'FAIL'
        atomic_json(external / 'runtime-integrity.json', report)
        if report['status'] != 'PASS':
            raise RulesViolation('Environment binary/package integrity, import-shadow or process-wall check failed')
        submission_sha = seal(workspace, identity, symlinks=links)
        home_sha = seal(home, identity, symlinks=home_links) if home else None
        local_sha = seal(local, identity)
        shutil.rmtree(live, ignore_errors=True)
        # The evaluation stage: a fresh coordinator sandbox with the private inputs, submitted code apart from them.
        rows = [row for row in bindings if 'intermediate' in row['stages']]
        worker_env = worker_environment(config)
        # As the Docker coordinator passes its LD_LIBRARY_PATH to the worker: under --nv the driver libraries are bound
        # at /.singularity.d/libs, which only the image's environment puts on the search path. Without it, submitted
        # code entered with nsenter finds no CUDA driver.
        if image_env.get('LD_LIBRARY_PATH'):
            worker_env['LD_LIBRARY_PATH'] = image_env['LD_LIBRARY_PATH']
        # Operator variables the task forwards to its coordinator (evaluation.env); a loopback URL among them names a
        # service on this host, which the coordinator reaches at the same URL through a relay.
        missing = [name for name in config['intermediate_env'] if not os.environ.get(name)]
        require(not missing, 'Set the variables the task forwards to its evaluation: ' + ', '.join(missing))
        forwarded = {name: os.environ[name] for name in config['intermediate_env']}
        coordinator_env = {**image_env, 'HOME': '/tmp', **forwarded, **({'BREAKWATER_SMOKE': '1'} if args.smoke else {})}
        check_no_keys({**held, **{f'operator {name}': value for name, value in task_host.operator.secrets.items()}},
                      worker_env, coordinator_env)
        evaluation_record = stages['intermediate'] = EvaluationStage(
            image=args.image, task=task, contract=contract, submission=workspace, workdir=workdir, evaluation=evaluation,
            external=external,
            worker_env=worker_env, coordinator_env=coordinator_env, timeout=config['evaluation_timeout_sec'],
            private_binds=[(row['source'], row['target'], 'ro') for row in rows if row['target'].startswith('/private/')]
            + ([(task_private, '/private/operator', 'ro')] if task_private else []),
            public_binds=[(row['source'], row['target'], 'ro') for row in rows if not row['target'].startswith('/private/')]
            + [baseline_bind], gpus=config['gpus'] > 0, memory_mb=config['resources'].get('memory_mb'),
            cpus=config['resources'].get('cpus'),
            submission_mount=config['submission_mount'], scratch=scratch, ports=config['submission_ports'],
            host_ports=host_service_ports(forwarded), shared=config['evaluation_shared'], worker_home_seed=home_seed,
            coordinator_endpoints=roles['coordinator'], submission_endpoints=roles['submission'], home=home,
            runners=task / config['runners'] if config['runners'] else None,
            coordinator_argv=json.loads(args.coordinator_command) if args.coordinator_command else None).run()
        shutil.rmtree(home_seed, ignore_errors=True)  # copied into each stage's worker home
        # Both execution stages are proven empty: the task's host side may now collect its evidence.
        task_host.after_evaluation(external, {stage: [stages[stage]['started_wall_ns'], stages[stage]['ended_wall_ns']]
                                              for stage in ('agent', 'intermediate')})
        evaluation_record['image_sha256_after'] = file_sha256(args.image)
        require(evaluation_record['image_sha256_after'] == image_sha, 'The image changed during the evaluation stage')
        require(evaluation_record['exit_code'] == 0 and not evaluation_record['timed_out'], 'Intermediate evaluation failed')
        require(contract_tools.read_json(evaluation / 'complete.json').get('complete') is True,
                'Missing intermediate completion')
        # The coordinator publishes /evaluation read-only (0555/0444) when it completes. Docker's host seals as root;
        # here the launcher is the same UID, so it reopens only the top directory to add the manifest.
        evaluation.chmod(0o700)
        evaluation_sha = seal(evaluation, identity)
        external_sha = seal(external, identity, finalized=True, stage_order=['agent', 'intermediate'])
        for root in (external, local, evaluation):
            read_only(root)
        code, authority_sha = verify(
            run=run, task=task, config=config, contract=contract, image=args.image, image_env=image_env, bindings=bindings,
            judge_dir=args.judge_dir, judge_key=held[args.judge_key_env], scratch=scratch, identity=identity,
            stages=stages, authority=authority, reports=reports, logs=run / 'verifier',
            seals={'external_sha256': external_sha, 'evaluation_sha256': evaluation_sha,
                   'submission_sha256': submission_sha, 'local_log_sha256': local_sha, 'image_sha256': image_sha},
            home=home, extra={'recording': spec.recording(config), **({'smoke': True} if args.smoke else {}),
                              **({'home_sha256': home_sha} if home_sha else {}), **task_host.sealed(task_plan_sha)},
            judge_concurrency=args.judge_concurrency)
        print(json.dumps({'task': config['task'], 'run_id': identity['run_id'], 'agent_cli': args.author_cli,
                          **({'smoke': True} if args.smoke else {}),
                          'submission_sha256': submission_sha,
                          'exit_code': record['exit_code'], 'deadline_reached': record['deadline_reached'],
                          'command_sandbox_restarts': record['command_sandbox_restarts'],
                          'evaluation_exit_code': evaluation_record['exit_code'],
                          'evaluation_timed_out': evaluation_record['timed_out'], 'evaluation_sha256': evaluation_sha,
                          'external_sha256': external_sha, 'authority_sha256': authority_sha,
                          'verifier_exit_code': code, 'reports': str(reports)}))
        return code
    except BaseException as error:
        task_host.on_failure()
        # Failed attempts keep their evidence and never turn into a pass. Every write here is best effort, since the
        # failure may be a full disk; the traceback and a JSON line on stdout are always attempted.
        say(traceback.format_exc().rstrip(), sys.stderr)
        violation = isinstance(error, RulesViolation)
        status, message = 'RULE_VIOLATION' if violation else 'INCONCLUSIVE', f'{type(error).__name__}: {error}'
        records = []
        if violation:
            records += [('reward.json', {'reward': config['rule_violation_reward']}),
                        ('rule-violation.json', {'reward': config['rule_violation_reward'], 'reason': 'violation of rules',
                                                 'detail': str(error), 'judge_skipped': True})]
        elif not (reports / 'reward.json').exists():
            records.append(('reward.json', {'reward': config['failure_reward']}))
        records.append(('pipeline-error.json', {'status': status, 'error': message, 'completed_stages': list(stages),
                                                'stages': stages}))
        written = []
        for name, value in records:
            try:
                atomic_json(reports / name, value)
                written.append(name)
            except OSError as failure:
                say(f'breakwater: could not write {reports / name}: {failure}', sys.stderr)
        say(json.dumps({'task': config['task'], 'run_id': identity['run_id'], 'status': status, 'error': message,
                        'completed_stages': list(stages), 'reports': str(reports), 'written': written}), sys.stdout)
        return 2


if __name__ == '__main__':
    sys.exit(main())
