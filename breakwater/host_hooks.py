"""Host hooks: a task's trusted code on the launcher's host (SPEC.md, [breakwater.host] and [breakwater.operator]).

A task that declares ``[breakwater.host] module = "tests/breakwater_host.py"`` defines there::

    from host_hooks import HostPlan, Endpoint, FromInput, FromSecret, Text

    def preflight(host) -> HostPlan          # required: before the run directory exists; no infrastructure side effects
    def before_authoring(host)               # optional: after seeding, before the first sandbox starts
    def after_evaluation(host, windows)      # optional: after both execution stages are proven empty, before sealing
    def on_failure(host)                     # optional: best effort on the failure path; its errors are ignored

``host`` has ``task`` (the task directory), ``contract``, ``backend`` ('docker' or 'apptainer'), ``smoke``,
``input(name)`` (the path of an operator input), ``value(name)`` and ``secret(name)`` (operator values and secrets),
``state`` (a private directory that persists across the phases and is never sealed) and, in before_authoring and
after_evaluation, ``evidence``: ``evidence.json(name, value)`` and ``evidence.path(name)`` write under the run's
external/task/, which is sealed and reviewed. ``windows`` gives each execution stage's wall-clock span:
{'agent': [start_ns, end_ns], 'intermediate': [start_ns, end_ns]}.

preflight returns a HostPlan: endpoints (each with the roles that reach it: 'agent', 'coordinator', 'submission'),
an environment for the agent's commands, files for the workspace (from an operator input, or text) and private files
for the coordinator only (from an input, a secret, or text; bound read-only at /private/operator), and metadata that
the launcher seals as the run's task metadata (the verifier's ``scoring.metadata``).

The launcher runs each phase in its own process (``python3 -I -B host_hooks.py PHASE``), in the hook state directory,
with only PATH, LANG and the task's declared operator values and secrets in its environment, within [breakwater.host]
timeout_sec. The phase's result comes back as a file in that directory, never on stdout; the launcher validates the
plan and applies it, writes secret files itself, and seals the phases' times, which the verifier checks lie outside the
execution stages. A hook is trusted task code, like the task's evaluation: it is not a security boundary, only a
contained, recorded place for host work.
"""
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PHASES = ('preflight', 'before_authoring', 'after_evaluation', 'on_failure')
REQUEST, RESULT = 'hook-request.json', 'hook-result.json'
REACH = ('agent', 'coordinator', 'submission')
HOST_NAME = re.compile(r'(?=.*[A-Za-z])[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?')
FILE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
METADATA_BYTES = 1024 ** 2


# ---- What a task's host module builds -------------------------------------------------------------------------------

class Text:
    """A file's contents, given as text."""

    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {'text': self.value}


class FromInput:
    """A file's contents: the operator input of this name (--input NAME=PATH)."""

    def __init__(self, name):
        self.name = name

    def to_json(self):
        return {'input': self.name}


class FromSecret:
    """A private file's contents: the operator secret of this name, written by the launcher, never by the hook."""

    def __init__(self, name):
        self.name = name

    def to_json(self):
        return {'secret': self.name}


class Endpoint:
    """A host and ports the sandboxes of the given roles may reach, as ``name`` (the host name they use)."""

    def __init__(self, name, host, ports, reach=('agent',)):
        self.name, self.host, self.ports, self.reach = name, host, list(ports), list(reach)

    def to_json(self):
        return {'name': self.name, 'host': self.host, 'ports': self.ports, 'reach': self.reach}


class HostPlan:
    def __init__(self, endpoints=(), agent_env=None, workspace_files=None, private_files=None, metadata=None):
        self.endpoints, self.agent_env = list(endpoints), dict(agent_env or {})
        self.workspace_files, self.private_files = dict(workspace_files or {}), dict(private_files or {})
        self.metadata = dict(metadata or {})

    def to_json(self):
        def source(value):
            if not hasattr(value, 'to_json'):
                raise TypeError('A file is Text, FromInput or FromSecret')
            return value.to_json()
        return {'endpoints': [endpoint.to_json() for endpoint in self.endpoints], 'agent_env': self.agent_env,
                'workspace_files': {path: source(value) for path, value in self.workspace_files.items()},
                'private_files': {name: source(value) for name, value in self.private_files.items()},
                'metadata': self.metadata}


# ---- What a hook sees (the hook's process) --------------------------------------------------------------------------

class Evidence:
    """The task's evidence: files under the run's external/task/, sealed and reviewed."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(part in ('', '.', '..') for part in relative.parts):
            raise ValueError(f'Evidence paths are relative paths under external/task/: {name!r}')
        target = self.root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def json(self, name, value):
        target = self.path(name)
        pending = target.with_name(target.name + '.tmp')
        pending.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        pending.replace(target)
        return target


class HostContext:
    def __init__(self, request):
        self.task, self.contract = Path(request['task']), request['contract']
        self.backend, self.smoke, self.phase = request['backend'], request['smoke'], request['phase']
        self.state = Path(request['state'])
        self.evidence = Evidence(request['evidence']) if request.get('evidence') else None
        self._inputs, self._values, self._secrets = request['inputs'], request['values'], request['secrets']

    def input(self, name):
        return Path(self._inputs[name])

    def value(self, name):
        return os.environ[self._values[name]]

    def secret(self, name):
        return os.environ[self._secrets[name]]


def child(phase):
    """One phase in the hook's own process: the request and the result are files in the working directory."""
    request = json.loads(Path(REQUEST).read_text())
    try:
        sys.path[:0] = [str(Path(request['module']).parent), str(HERE)]
        location = importlib.util.spec_from_file_location('breakwater_host', request['module'])
        module = importlib.util.module_from_spec(location)
        location.loader.exec_module(module)
        host = HostContext(request)
        result = {}
        if phase == 'preflight':
            plan = module.preflight(host)
            if not isinstance(plan, HostPlan):
                raise TypeError('preflight returns a HostPlan')
            result['plan'] = plan.to_json()
        else:
            function = getattr(module, phase, None)
            if function is not None:
                function(host, request['windows']) if phase == 'after_evaluation' else function(host)
        outcome = {'ok': True, **result}
    except BaseException:  # noqa: BLE001 - every failure is reported to the launcher
        outcome = {'ok': False, 'error': traceback.format_exc()[-4000:]}
    pending = Path(RESULT + '.tmp')
    pending.write_text(json.dumps(outcome, sort_keys=True, allow_nan=False))
    pending.replace(RESULT)
    return 0 if outcome['ok'] else 1


# ---- The launcher's side --------------------------------------------------------------------------------------------

def require(condition, message):
    from contract import require as contract_require
    contract_require(condition, message)


def pairs(values, label):
    """NAME=VALUE options as {name: value}, each name once."""
    result = {}
    for value in values or ():
        name, _, item = str(value).partition('=')
        require(name and item and name not in result, f'{label} takes NAME=VALUE, once per name')
        result[name] = item
    return result


class Operator:
    """The operator's inputs, values and secrets for the task's [breakwater.operator] declarations (--input NAME=PATH,
    --value-env NAME=VAR, --secret-env NAME=VAR). The secrets leave this process's environment here."""

    def __init__(self, config, inputs=(), value_env=(), secret_env=(), environ=None, avoid=()):
        environ = os.environ if environ is None else environ
        given, values, secrets = pairs(inputs, '--input'), pairs(value_env, '--value-env'), pairs(secret_env, '--secret-env')
        require(set(given) <= set(config['operator_inputs']), 'Undeclared operator input: '
                + ', '.join(sorted(set(given) - set(config['operator_inputs']))))
        require(set(values) <= set(config['operator_values']), 'Undeclared operator value: '
                + ', '.join(sorted(set(values) - set(config['operator_values']))))
        require(set(secrets) <= set(config['operator_secrets']), 'Undeclared operator secret: '
                + ', '.join(sorted(set(secrets) - set(config['operator_secrets']))))
        missing = [name for name in config['operator_inputs'] if name not in given]
        require(not missing, 'Missing operator inputs (--input NAME=PATH): ' + ', '.join(missing))
        from contract import overlaps
        self.inputs = {}
        for name, value in given.items():
            path = Path(value).resolve()
            require(path.is_file() and stat.S_ISREG(path.stat().st_mode), f'The operator input {name} is not a file')
            require(not any(overlaps(path, root) for root in avoid),
                    f'The operator input {name} lies inside the task or the run directory')
            self.inputs[name] = path
        self.value_vars = {name: values.get(name, variable) for name, variable in config['operator_values'].items()}
        self.secret_vars = {name: secrets.get(name, variable) for name, variable in config['operator_secrets'].items()}
        unset = [variable for variable in [*self.value_vars.values(), *self.secret_vars.values()] if not environ.get(variable)]
        require(not unset, 'Set the operator variables: ' + ', '.join(sorted(set(unset))))
        self.values = {name: environ[variable] for name, variable in self.value_vars.items()}
        self.secrets = {name: environ.pop(variable) for name, variable in self.secret_vars.items()}

    @property
    def credential_names(self):
        return sorted(set(self.secret_vars.values()))

    def environment(self):
        """The hook process's environment: PATH, LANG, and the declared values and secrets under their variables."""
        return {'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'), 'LANG': os.environ.get('LANG', 'C.UTF-8'),
                **{self.value_vars[name]: value for name, value in self.values.items()},
                **{self.secret_vars[name]: value for name, value in self.secrets.items()}}

    def digests(self):
        return {name: file_sha256(path) for name, path in sorted(self.inputs.items())}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(chunk)
    return value.hexdigest()


def contains(data, secrets):
    """Whether bytes contain any secret value."""
    return any(secret and secret.encode() in data for secret in secrets)


class TaskHost:
    """The task's host side for one run: its operator inputs, values and secrets, its hooks and their plan. A task
    without a host module or operator declarations gets an inert one."""

    def __init__(self, task, config, backend, smoke=False, inputs=(), value_env=(), secret_env=(), avoid=()):
        self.task, self.config, self.backend, self.smoke = Path(task), config, backend, smoke
        self.operator = Operator(config, inputs, value_env, secret_env, avoid=avoid)
        self.module = self.task / config['host_module'] if config['host_module'] else None
        require(self.module is None or self.module.is_file(), 'The declared host module does not exist')
        self.state = Path(tempfile.mkdtemp(prefix='breakwater-host-')) if self.module else None
        if self.state:
            self.state.chmod(0o700)
        self.times, self.plan = {}, {'endpoints': [], 'agent_env': {}, 'workspace_files': {}, 'private_files': {},
                                     'metadata': {}}
        self.private = None

    # -- running a phase

    def run(self, phase, evidence=None, windows=None):
        if self.module is None:
            return {}
        request = {'phase': phase, 'task': str(self.task), 'module': str(self.module), 'contract': self.config,
                   'backend': self.backend, 'smoke': self.smoke, 'state': str(self.state),
                   'evidence': str(evidence) if evidence else None, 'windows': windows,
                   'inputs': {name: str(path) for name, path in self.operator.inputs.items()},
                   'values': self.operator.value_vars, 'secrets': self.operator.secret_vars}
        (self.state / RESULT).unlink(missing_ok=True)
        (self.state / REQUEST).write_text(json.dumps(request, sort_keys=True))
        started = time.time_ns()
        with (self.state / f'hook-{phase}.log').open('ab') as log:
            try:
                code = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'host_hooks.py'), phase], cwd=self.state,
                                      env=self.operator.environment(), stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=log, timeout=self.config['host_timeout_sec']).returncode
            except subprocess.TimeoutExpired:
                code = None
        self.times[phase] = [started, time.time_ns()]
        require(code is not None, f"The task's {phase} hook ran out of time ({self.config['host_timeout_sec']} s)")
        result = json.loads((self.state / RESULT).read_text()) if (self.state / RESULT).is_file() else {}
        require(code == 0 and result.get('ok') is True,
                f"The task's {phase} hook failed: " + str(result.get('error') or f'exit status {code}')[-2000:])
        return result

    def preflight(self, reserved=()):
        """The task's plan, validated. ``reserved``: names no endpoint may take (the framework's own hosts)."""
        if self.module is None:
            return self.plan
        self.plan = validate(self.run('preflight')['plan'], self, reserved)
        return self.plan

    def adopt(self, run):
        """The hook state moves into the run directory, beside the run's other private files."""
        if self.state is not None:
            target = Path(run) / 'host-state'
            shutil.move(str(self.state), target)
            self.state = target

    def before_authoring(self, external):
        self.run('before_authoring', evidence=self.evidence(external))

    def after_evaluation(self, external, windows):
        self.run('after_evaluation', evidence=self.evidence(external), windows=windows)

    def on_failure(self):
        try:
            self.run('on_failure')
        except Exception:  # noqa: BLE001 - best effort on the failure path
            pass

    def evidence(self, external):
        folder = Path(external) / 'task'
        folder.mkdir(mode=0o700, exist_ok=True)
        return folder

    # -- applying the plan

    def endpoints(self):
        """The hook's endpoints as plan.py rows (no port map, no health path)."""
        return [{'name': row['name'], 'host': row['host'], 'ports': list(row['ports']), 'port_map': None,
                 'reach': list(row['reach']), 'health': None} for row in self.plan['endpoints']]

    def agent_env(self):
        return dict(self.plan['agent_env'])

    def content(self, source):
        if 'text' in source:
            return source['text'].encode()
        if 'input' in source:
            return self.operator.inputs[source['input']].read_bytes()
        return self.operator.secrets[source['secret']].encode()

    def write_workspace_files(self, workspace):
        """The plan's workspace files, before the workspace inventory: regular files only, never a secret."""
        for relative, source in self.plan['workspace_files'].items():
            data = self.content(source)
            require(not contains(data, self.operator.secrets.values()),
                    f'The workspace file {relative} contains an operator secret')
            target = Path(workspace).joinpath(*PurePosixPath(relative).parts)
            require(not any(parent.is_symlink() for parent in [target, *target.parents]
                            if Path(workspace) in [parent, *parent.parents]), f'The workspace path {relative} is a link')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.write_bytes(data)

    def write_private_files(self, run):
        """The plan's private files, 0400 in run/task-private, bound read-only at /private/operator in the coordinator
        only. Returns the directory, or None."""
        if not self.plan['private_files']:
            return None
        self.private = Path(run) / 'task-private'
        self.private.mkdir(mode=0o700)
        for name, source in self.plan['private_files'].items():
            target = self.private / name
            target.write_bytes(self.content(source))
            target.chmod(0o400)
        self.private.chmod(0o500)
        return self.private

    def check_no_secrets(self, *texts):
        """No operator secret appears in these strings (environments, plans)."""
        require(not any(contains(text.encode() if isinstance(text, str) else text, self.operator.secrets.values())
                        for text in texts), 'An operator secret would reach a sandbox')

    def record(self, external, static=()):
        """external/task-plan.json: what the task's host side and static endpoints contributed, with digests for file
        contents and secret-derived values. Written only when there is something to record. Returns its digest."""
        if not (self.module or static or self.operator.inputs or self.operator.values or self.operator.secrets):
            return None
        def described(source):
            if 'secret' in source:
                return {'secret': source['secret'], 'sha256': hashlib.sha256(self.content(source)).hexdigest()}
            return {**({'input': source['input']} if 'input' in source else {'text': True}),
                    'sha256': hashlib.sha256(self.content(source)).hexdigest()}
        value = {'host_module': self.config['host_module'],
                 'endpoints': [{key: row[key] for key in ('name', 'host', 'ports', 'reach')}
                               for row in [*static, *self.endpoints()]],
                 'agent_env': self.agent_env(),
                 'workspace_files': {path: described(source) for path, source in self.plan['workspace_files'].items()},
                 'private_files': {name: described(source) for name, source in self.plan['private_files'].items()},
                 'metadata_sha256': hashlib.sha256(json.dumps(self.plan['metadata'], sort_keys=True).encode()).hexdigest(),
                 'operator': {'inputs': self.operator.digests(), 'values': sorted(self.operator.values),
                              'secrets': sorted(self.operator.secrets)}}
        path = Path(external) / 'task-plan.json'
        path.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n')
        return file_sha256(path)

    def sealed(self, plan_sha256):
        """What the authority records of the task's host side."""
        return {**({'task_plan_sha256': plan_sha256} if plan_sha256 else {}),
                **({'host_hooks': dict(self.times)} if self.times else {}),
                **({'task_metadata': self.plan['metadata']} if self.plan['metadata'] else {})}


def validate(plan, host, reserved=()):
    """A preflight plan, checked: endpoint names, hosts, ports and roles; the agent environment; file paths and sources;
    metadata. Raises ContractError."""
    from spec import CREDENTIAL_ENV, RESERVED_ENV, VARIABLE, WORKER_FIXED_ENV
    require(isinstance(plan, dict) and set(plan) == {'endpoints', 'agent_env', 'workspace_files', 'private_files',
                                                     'metadata'}, 'Invalid host plan')
    config, operator = host.config, host.operator
    names = set()
    for row in plan['endpoints']:
        require(isinstance(row, dict) and set(row) == {'name', 'host', 'ports', 'reach'}, 'Invalid endpoint in the host plan')
        name = row['name']
        try:
            # An address as the name: on Apptainer a loopback address, where the relay listens in the sandbox; on Docker
            # any other address, which the container reaches as itself.
            address = ipaddress.IPv4Address(name)
            literal = address.is_loopback == (host.backend == 'apptainer')
        except (ipaddress.AddressValueError, ValueError):
            literal = False
        require(isinstance(name, str) and (HOST_NAME.fullmatch(name) or literal) and name not in reserved,
                f'Invalid or reserved endpoint name: {name!r}')
        require(isinstance(row['host'], str) and row['host'], f'The endpoint {name} has no host')
        require(isinstance(row['ports'], list) and row['ports'] and len(row['ports']) <= 4096
                and all(type(port) is int and 1 <= port <= 65535 for port in row['ports']),
                f'The endpoint {name} has invalid ports')
        require(isinstance(row['reach'], list) and row['reach'] and set(row['reach']) <= set(REACH),
                f'The endpoint {name} names an unknown role')
        names.add(name)
    require(len(names) <= 250, 'At most 250 endpoint names')
    env = plan['agent_env']
    require(isinstance(env, dict) and all(
        isinstance(name, str) and VARIABLE.fullmatch(name) and name not in CREDENTIAL_ENV
        and name not in operator.credential_names and name not in config['credential_env'] and name not in RESERVED_ENV
        and name not in WORKER_FIXED_ENV and not name.startswith(('LD_', 'PYTHON')) and isinstance(value, str)
        and '\0' not in value and len(value) <= 65536 for name, value in env.items()),
            'Invalid agent environment in the host plan: names in capitals, string values, never a credential or a '
            'variable the framework sets')
    require(not set(env) & set(config['authoring_env']), 'The host plan sets a variable the task already declares')
    host.check_no_secrets(*env.values())
    for relative, source in plan['workspace_files'].items():
        path = PurePosixPath(relative)
        require(isinstance(relative, str) and not path.is_absolute() and path.parts
                and all(part not in ('', '.', '..') for part in path.parts), f'Invalid workspace file path {relative!r}')
        check_source(source, operator, secret=False, label='workspace file ' + relative)
    for name, source in plan['private_files'].items():
        require(isinstance(name, str) and FILE_NAME.fullmatch(name), f'Invalid private file name {name!r}')
        check_source(source, operator, secret=True, label='private file ' + name)
    metadata = plan['metadata']
    require(isinstance(metadata, dict) and len(json.dumps(metadata)) <= METADATA_BYTES, 'Invalid or oversized metadata')
    host.check_no_secrets(json.dumps(metadata))
    return plan


def check_source(source, operator, secret, label):
    require(isinstance(source, dict) and len(source) == 1, f'Invalid source for the {label}')
    kind, value = next(iter(source.items()))
    if kind == 'text':
        require(isinstance(value, str), f'Invalid text for the {label}')
    elif kind == 'input':
        require(value in operator.inputs, f'The {label} names an undeclared input')
    else:
        require(kind == 'secret' and secret and value in operator.secrets,
                f'The {label} has an invalid source (secrets go to private files only)')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in PHASES:
        raise SystemExit('host_hooks.py PHASE')
    sys.path.insert(0, str(HERE))
    import host_hooks  # the module the task's code imports, so that its HostPlan is this one
    raise SystemExit(host_hooks.child(sys.argv[1]))
