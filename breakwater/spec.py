"""The Breakwater task specification: a Harbor task whose task.toml carries [breakwater] tables (SPEC.md).

``load(task_dir)`` reads and validates a task and returns its resolved contract: every setting with its default, as
the launchers, the evaluation coordinator and the verifier use it. The launcher hands sandboxes this contract as JSON
(``contract.json``), so code inside a task image never needs a TOML parser. ``assets(contract)`` lists the task's
declared inputs; ``bindings(...)`` resolves them against an operator's asset root.
"""
import json
import os
import math
from pathlib import Path
import re

from contract import CREDENTIAL_ENV, overlaps, read_json, require

# Contract keys sealed into the authority; the verifier compares them with its own copy of the contract.
SEALED_PIPELINE = ('task', 'gpus', 'review_profile', 'workdir', 'agent_deadline', 'rule_violation_reward',
                   'snapshot_excludes', 'snapshot_worker_home', 'review_metadata_paths', 'recomputed', 'agent_exit',
                   'options')
DEFAULTS = {
    'gpus': 4,
    'gpu_name_patterns': ['GB300', 'B300'],
    'gpu_type': 'GB300',
    'workdir': '/app',
    'workspace_source': 'task',
    'workspace_excludes': [],
    'submission_mount': None,
    'agent_extra_networks': [],
    'intermediate_network': 'none',
    'intermediate_cap_add': [],
    'intermediate_env': [],
    'intermediate_add_hosts': [],
    'resources': {'cpus': None, 'memory_mb': None, 'nofile': None, 'shm_size': '16g', 'memlock': None,
                  'stack': None, 'pids': None, 'cpuset': None},
    'agent_resources': {},
    'shadow_policy': None,
    'protected_workspace_paths': [],
    'snapshot_excludes': [],
    'snapshot_worker_home': False,
    'dataset_manifest': None,
    'dataset_roots': [],
    'review_profile': 'exhaustive',
    'review_metadata_paths': [],
    'agent_deadline': 'fail',
    'failure_reward': 0.0,
    'rule_violation_reward': 0.0,
    'development_gateway': None,
    'worker_env': None,
    'runners': None,
    'submission_ports': [],
    'evaluation_shared': [],
    'recomputed': [],
    'assets': [],
    'agent_exit': None,  # resolved: 'any' when the task grades at the deadline, else 'zero'
    'authoring_env': {},
    'inbound_ports': None,
    'inbound_host': None,
    'host_module': None,
    'host_timeout_sec': 1800,
    'operator_inputs': [],
    'operator_values': {},
    'operator_secrets': {},
    'options': {},
    'endpoints': [],
    'credential_env': [],  # computed: the declared secrets' variables and the model-access key's
}
RESOURCE_FIELDS = ('cpus', 'memory_mb', 'nofile', 'shm_size', 'memlock', 'stack', 'pids', 'cpuset')
# Variables the controller sets itself; with credentials and loader/interpreter variables, they are never forwarded
# from the host to the evaluation stage.
RESERVED_ENV = ('BREAKWATER_STAGE', 'PATH', 'HOME', 'PYTHONDONTWRITEBYTECODE')
EVALUATION_HOSTS = ('host.docker.internal:host-gateway',)
# Where a snapshotted worker home is mounted for the verifier (workspace.snapshot_home).
HOME_MOUNT = '/submission-home'
STAGES = {'agent': 'agent', 'evaluation': 'intermediate', 'verifier': 'verifier'}
ASSET_ROOTS = ('private', 'models', 'assets', 'app', 'workspace')
OPERATOR_NAME = re.compile(r'[a-z][a-z0-9-]*')
VARIABLE = re.compile(r'[A-Z][A-Z0-9_]*')
HOST_NAME = re.compile(r'(?=.*[A-Za-z])[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?')
REACH = ('agent', 'coordinator', 'submission')
# The worker environment the framework sets for the agent's commands, which a task's authoring env may not replace.
WORKER_FIXED_ENV = ('HOME', 'USER', 'LOGNAME', 'LANG', 'SHELL', 'PATH', 'BREAKWATER_WORKDIR', 'PYTHONUNBUFFERED',
                    'HF_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR')
MODEL_ACCESS_KEYS = ('upstream', 'chat_models', 'embed_model', 'max_chat_tokens', 'max_embed_tokens', 'host', 'port',
                     'key_env')
MODEL_ACCESS_DEFAULTS = {'host': 'gateway', 'port': 8000, 'key_env': 'FIREWORKS_API_KEY'}

# task.toml keys -> resolved contract keys, per table. Anything else in a [breakwater] table is an error.
TABLES = {
    'breakwater': {'name': 'task', 'failure_reward': 'failure_reward', 'rule_violation_reward': 'rule_violation_reward',
                   'agent_deadline': 'agent_deadline', 'agent_exit': 'agent_exit', 'review_profile': 'review_profile',
                   'review_metadata_paths': 'review_metadata_paths'},
    'judge': {'timeout_sec': 'judge_timeout_sec'},
    'workspace': {'source': 'workspace_source', 'excludes': 'workspace_excludes',
                  'protected_paths': 'protected_workspace_paths', 'snapshot_excludes': 'snapshot_excludes',
                  'snapshot_home': 'snapshot_worker_home', 'submission_mount': 'submission_mount',
                  'dataset_manifest': 'dataset_manifest', 'dataset_roots': 'dataset_roots',
                  'shadow_policy': 'shadow_policy'},
    'authoring': {'extra_networks': 'agent_extra_networks', 'env': 'authoring_env', 'inbound_ports': 'inbound_ports',
                  'inbound_host': 'inbound_host'},
    'host': {'module': 'host_module', 'timeout_sec': 'host_timeout_sec'},
    'operator': {'inputs': 'operator_inputs', 'values': 'operator_values', 'secrets': 'operator_secrets'},
    'evaluation': {'timeout_sec': 'evaluation_timeout_sec', 'network': 'intermediate_network',
                   'cap_add': 'intermediate_cap_add', 'env': 'intermediate_env', 'add_hosts': 'intermediate_add_hosts',
                   'worker_env': 'worker_env', 'runners': 'runners', 'ports': 'submission_ports',
                   'shared': 'evaluation_shared'},
    'scoring': {'recomputed': 'recomputed'},
}
WHOLE = {'resources': 'resources', 'agent_resources': 'agent_resources', 'model_access': 'development_gateway',
         'options': 'options'}


# Settings a backend does not support, with their task.toml names; the launcher refuses a task that uses one.
PENDING = {'apptainer': {}}


def unsupported(config, backend):
    """The task.toml settings of a resolved contract that ``backend`` cannot run."""
    pending = PENDING.get(backend, {})
    # On Apptainer, evaluation.network "bridge" (with its cap_add and add_hosts, for a firewall and a host name the
    # coordinator no longer needs) means the host services named by loopback URLs in evaluation.env, each relayed, and
    # authoring.extra_networks (a Docker network) means endpoints on other hosts, so it needs endpoints the agent
    # reaches: declared ones, or a host module's.
    names = [name for key, name in pending.items() if config.get(key) not in (None, False, [], 'none')]
    if (backend == 'apptainer' and config.get('agent_extra_networks') and not config.get('host_module')
            and not any('agent' in row['reach'] for row in config.get('endpoints') or [])):
        names.append('authoring.extra_networks')
    return names


def relative_paths(values, label, *, anchored=False):
    """Workspace-relative paths without empty, '.' or '..' components; a leading '/' only if anchored."""
    require(isinstance(values, list), 'Invalid ' + label)
    for value in values:
        require(isinstance(value, str) and value and '\0' not in value, 'Invalid ' + label)
        body = value[1:] if anchored and value.startswith('/') else value
        require(all(part not in ('', '.', '..') for part in body.split('/')), 'Invalid ' + label)
    return values


def cpu_list(cpuset):
    cpus = set()
    for item in cpuset.split(','):
        first, _, last = item.partition('-')
        require(int(first) <= int(last or first) < 65536, 'Invalid cpuset range')
        cpus.update(range(int(first), int(last or first) + 1))
    return sorted(cpus)


def check_resources(values, label):
    require(isinstance(values, dict) and set(values) <= set(RESOURCE_FIELDS), 'Unknown ' + label + ' field')
    for key, value in values.items():
        if value is None:
            continue
        if key == 'cpus':
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        elif key == 'shm_size':
            valid = isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*[bkmgBKMG]?', value) is not None
        elif key == 'cpuset':
            valid = (isinstance(value, str) and re.fullmatch(r'[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*', value) is not None
                     and bool(cpu_list(value)))
        elif key in ('memlock', 'stack'):
            valid = type(value) is int and (value == -1 or value > 0)
        else:
            valid = type(value) is int and value > 0
        require(valid, f'Invalid {label} value: {key}')
    if values.get('cpuset') and values.get('cpus'):
        require(len(cpu_list(values['cpuset'])) >= values['cpus'], 'The cpuset has fewer CPUs than the CPU quota')
    return values


def read(task_dir):
    """task.toml as the settings it declares, in resolved-contract keys (no defaults applied)."""
    import tomllib  # host side only: sandboxes read the resolved contract as JSON
    task_dir = Path(task_dir)
    path = task_dir / 'task.toml'
    require(path.is_file(), 'Not a task: ' + str(task_dir) + ' has no task.toml')
    document = tomllib.loads(path.read_text())
    section = document.get('breakwater')
    require(isinstance(section, dict), 'task.toml has no [breakwater] table')
    agent, environment = document.get('agent') or {}, document.get('environment') or {}
    require(type(agent.get('timeout_sec')) in (int, float), 'task.toml needs [agent] timeout_sec (the authoring budget)')
    # The task's identity in run records: [breakwater] name, else the last part of Harbor's [task] name, else the
    # directory name (so a copy of the task under another name keeps its identity only with an explicit name).
    harbor_name = str((document.get('task') or {}).get('name') or '').rpartition('/')[2]
    settings = {'task': harbor_name or task_dir.name, 'agent_timeout_sec': float(agent['timeout_sec'])}
    if 'gpus' in environment:
        settings['gpus'] = environment['gpus']
    if 'workdir' in environment:
        settings['workdir'] = environment['workdir']
    if 'gpu_types' in environment:
        types = environment['gpu_types']
        require(isinstance(types, list) and all(isinstance(item, str) and item for item in types), 'Invalid gpu_types')
        settings['gpu_name_patterns'] = list(types)
        settings['gpu_type'] = types[0] if types else 'none'
    for name, value in section.items():
        if name in TABLES['breakwater']:
            settings[TABLES['breakwater'][name]] = value
        elif name in WHOLE:
            require(isinstance(value, dict), f'[breakwater.{name}] must be a table')
            settings[WHOLE[name]] = value
        elif name in TABLES:
            require(isinstance(value, dict), f'[breakwater.{name}] must be a table')
            unknown = set(value) - set(TABLES[name])
            require(not unknown, f'Unknown key in [breakwater.{name}]: ' + ', '.join(sorted(unknown)))
            settings.update({TABLES[name][key]: item for key, item in value.items()})
        elif name in ('assets', 'endpoints'):
            settings[name] = value
        else:
            raise ValueError('Unknown [breakwater] key: ' + name)
    require('judge_timeout_sec' in settings, 'task.toml needs [breakwater.judge] timeout_sec (the judge budget)')
    require('evaluation_timeout_sec' in settings, 'task.toml needs [breakwater.evaluation] timeout_sec')
    if settings.get('gpus') == 0 and 'gpu_name_patterns' not in settings:
        settings.update(gpu_name_patterns=[], gpu_type='none')
    return settings


def load(task_dir):
    """The resolved, validated contract of a task directory."""
    config = resolve(read(task_dir))
    require((Path(task_dir) / 'tests' / 'breakwater_task.py').is_file(),
            'A task needs tests/breakwater_task.py (its evaluate and score entry points)')
    require(config['runners'] is None or (Path(task_dir) / config['runners']).is_dir(),
            'The declared runners directory does not exist')
    return config


def resolve(config):
    """Defaults applied to declared settings, then every rule checked."""
    require(isinstance(config.get('resources', {}), dict) and isinstance(config.get('agent_resources', {}), dict),
            'Resources must be objects')
    merged = {**DEFAULTS, **config}
    merged['resources'] = {**DEFAULTS['resources'], **(config.get('resources') or {})}
    merged['agent_resources'] = dict(config.get('agent_resources') or {})
    require(isinstance(merged.get('task'), str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]*', merged['task']),
            'Invalid task name')
    for key in ('agent_timeout_sec', 'evaluation_timeout_sec', 'judge_timeout_sec'):
        require(type(merged.get(key)) in (int, float) and math.isfinite(merged[key]) and merged[key] > 0,
                'Invalid ' + key)
    require(type(merged['gpus']) is int and 0 <= merged['gpus'] <= 4, 'Declared GPU count must be between 0 and 4')
    require(isinstance(merged['workdir'], str) and merged['workdir'].startswith('/') and merged['workdir'] != '/',
            'Invalid workdir')
    require(merged['workspace_source'] in ('task', 'image'), 'Unknown workspace source')
    excludes = relative_paths(merged['workspace_excludes'], 'workspace exclude', anchored=True)
    if merged['workspace_source'] == 'task':
        # A plain entry is a name excluded anywhere in the task tree; a leading '/' anchors a
        # path at the task root (so /README.md does not hide nanoGPT/README.md).
        require(all(item.startswith('/') or '/' not in item for item in excludes),
                'A plain workspace exclude is one name; anchor a longer path at the task root with a leading /')
    require(merged['intermediate_network'] in ('none', 'bridge'), 'Unknown intermediate network')
    require(set(merged['intermediate_cap_add']) <= {'NET_ADMIN', 'NET_RAW'}, 'Only network capabilities may be added to the coordinator')
    names = merged['intermediate_env']
    require(isinstance(names, list)
            and all(isinstance(name, str) and re.fullmatch(r'[A-Z][A-Z0-9_]*', name) and name not in CREDENTIAL_ENV
                    and name not in RESERVED_ENV and not name.startswith(('LD_', 'PYTHON')) for name in names)
            and len(set(names)) == len(names),
            'Invalid evaluation-stage variable: credentials and loader/interpreter variables are never forwarded')
    hosts = merged['intermediate_add_hosts']
    require(isinstance(hosts, list) and all(host in EVALUATION_HOSTS for host in hosts if isinstance(host, str))
            and all(isinstance(host, str) for host in hosts) and len(set(hosts)) == len(hosts),
            'Unexpected evaluation host entry')
    require(not hosts or merged['intermediate_network'] == 'bridge', 'Evaluation host entries need the bridge network')
    check_resources(merged['resources'], 'resource')
    check_resources(merged['agent_resources'], 'agent resource')
    check_resources({**merged['resources'], **merged['agent_resources']}, 'agent resource')
    require(merged['review_profile'] in ('exhaustive', 'indexed'), 'Unknown review profile')
    protected = relative_paths(merged['protected_workspace_paths'], 'protected workspace path')
    snapshot_excludes = relative_paths(merged['snapshot_excludes'], 'snapshot exclude')
    relative_paths(merged['review_metadata_paths'], 'review metadata path')
    def covers(prefix, value):
        return value == prefix or value.startswith(prefix + '/')
    require(not any(covers(item, path) or covers(path, item) for item in snapshot_excludes for path in protected),
            'Protected workspace paths may not be excluded from the sealed snapshot')
    require(type(merged['snapshot_worker_home']) is bool, 'snapshot_worker_home must be true or false')
    # The agent-declared training-dataset manifest the observer verifies and archives after authoring.
    inside = merged['workdir'].rstrip('/') + '/'
    manifest, dataset_roots = merged['dataset_manifest'], merged['dataset_roots']
    require(manifest is None or (isinstance(manifest, str) and manifest.startswith(inside)),
            'The dataset manifest must be a path inside the workspace')
    require(isinstance(dataset_roots, list) and all(isinstance(root, str) and root.startswith(inside) for root in dataset_roots)
            and (manifest is not None or not dataset_roots), 'Dataset roots must lie inside the workspace and need a manifest')
    require(merged['agent_deadline'] in ('fail', 'evaluate'), 'Unknown agent deadline policy')
    for key in ('failure_reward', 'rule_violation_reward'):
        require(type(merged[key]) in (int, float) and math.isfinite(merged[key]), 'Invalid ' + key.replace('_', ' '))
    worker_env = merged['worker_env']
    require(worker_env is None or (isinstance(worker_env, dict) and all(
        isinstance(name, str) and re.fullmatch(r'[A-Z][A-Z0-9_]*', name) and name not in CREDENTIAL_ENV
        and isinstance(value, str) for name, value in worker_env.items())),
            'Invalid evaluation worker environment: names in capitals, string values, never a credential')
    ports = merged['submission_ports']
    require(isinstance(ports, list) and len(set(ports)) == len(ports)
            and all(type(port) is int and 0 < port < 65536 for port in ports), 'Invalid evaluation ports')
    shared = merged['evaluation_shared']
    require(isinstance(shared, list) and len(set(shared)) == len(shared) and all(
        isinstance(path, str) and path.startswith('/') and os.path.normpath(path) == path and path != merged['workdir']
        and path.split('/')[1] not in ('', 'private', 'evaluation', 'eval-inputs', 'opt', 'proc', 'sys', 'dev', 'tests',
                                       'trusted-task', 'run', 'tmp') for path in shared), 'Invalid evaluation shared paths')
    runners = merged['runners']
    require(runners is None or (isinstance(runners, str) and relative_paths([runners], 'runners directory')),
            'Invalid runners directory')
    recomputed = merged['recomputed']
    require(isinstance(recomputed, list) and all(isinstance(item, str) and item.startswith(('raw/', 'inputs/'))
                                                 and '..' not in item.split('/') for item in recomputed),
            'Recomputed outputs are raw/ or inputs/ paths (glob patterns allowed)')
    if merged['submission_mount'] is not None:
        require(str(merged['submission_mount']).startswith('/') and merged['submission_mount'] != merged['workdir'],
                'Invalid submission mount')
    merged['assets'] = assets(merged)
    resolve_extensions(merged)
    return merged


PLAN_VERSION = 1


def authoring_plan(config, endpoints=(), hosts=(), rejects=(), reply_ports=None, env=None):
    """What the agent's commands get beyond the workspace and the author gateway, as each backend applies it: the
    framework services they reach (the model-access gateway), the task's endpoints, ports refused before any accept,
    host names, the source ports whose replies inbound connections need, and the task's authoring environment. The
    Docker launcher hands it to the author image's bootstrap (BREAKWATER_AUTHORING_PLAN), which applies it and returns
    its digest; both backends record it as external/authoring-plan.json."""
    access = config['development_gateway']
    return {'version': PLAN_VERSION,
            'services': [{'host': access['host'], 'port': access['port']}] if access else [],
            'endpoints': [dict(item) for item in endpoints], 'rejects': [dict(item) for item in rejects],
            'hosts': [list(item) for item in hosts], 'reply_ports': reply_ports,
            'env': dict(config['authoring_env'] if env is None else env)}


def model_access_env(access):
    """The development gateway's settings (dev_gateway.py) for [breakwater.model_access], on either backend."""
    return {'GATEWAY_ROLE': 'development', 'GATEWAY_SOURCE': 'development-gateway', 'UPSTREAM_BASE_URL': access['upstream'],
            'CHAT_MODELS': json.dumps(access['chat_models'], sort_keys=True),
            **({'EMBED_MODEL': access['embed_model']} if access.get('embed_model') else {}),
            **{name.upper(): str(int(access[name])) for name in ('max_chat_tokens', 'max_embed_tokens') if name in access}}


def plan_text(plan):
    """The plan as the bytes a launcher sends, records and digests."""
    return json.dumps(plan, sort_keys=True, separators=(',', ':'))


def reserved_names(config):
    """Host names no endpoint may take: the agent's gateway, localhost, the model-access host and the inbound host."""
    access = config['development_gateway']
    return {'author-gateway', 'localhost', *([access['host']] if access else []),
            *([config['inbound_host']] if config['inbound_host'] else [])}


def recording(config):
    """What a run of this contract records, as its launcher seals it for the verifier: the author gateway's journal,
    the development gateway's when the task declares model access, and the agent CLI's native JSONL log."""
    return {'author_gateway': True,
            'service_gateways': {'dev-gateway': 'development-gateway'} if config['development_gateway'] else {},
            'native': 'jsonl'}


def port_range(value, label, limit=64):
    """A "LOW-HIGH" range of at most ``limit`` ports, all at least 1024: (low, high)."""
    require(isinstance(value, str) and re.fullmatch(r'\d{4,5}-\d{4,5}', value) is not None,
            f'{label} is a range such as "30000-30015"')
    low, high = map(int, value.split('-'))
    require(1024 <= low <= high <= 65535 and high - low < limit, 'Invalid ' + label)
    return low, high


def resolve_extensions(merged):
    """The extension points a task may declare (SPEC.md): agent_exit, [breakwater.operator], [breakwater.model_access],
    [breakwater.authoring] env and inbound ports, [breakwater.host], [breakwater.options] and [[breakwater.endpoints]].
    Validated and given their defaults here; trusted framework code applies them."""
    if merged['agent_exit'] is None:
        merged['agent_exit'] = 'any' if merged['agent_deadline'] == 'evaluate' else 'zero'
    require(merged['agent_exit'] in ('any', 'zero'), 'Unknown agent exit policy')
    require(not (merged['agent_deadline'] == 'fail' and merged['agent_exit'] == 'any'),
            'agent_exit "any" needs agent_deadline "evaluate"')
    inputs, values, secrets = merged['operator_inputs'], merged['operator_values'], merged['operator_secrets']
    require(isinstance(inputs, list) and len(set(inputs)) == len(inputs)
            and all(isinstance(name, str) and OPERATOR_NAME.fullmatch(name) for name in inputs), 'Invalid operator inputs')
    for label, table in (('values', values), ('secrets', secrets)):
        require(isinstance(table, dict) and all(
            isinstance(name, str) and OPERATOR_NAME.fullmatch(name) and isinstance(variable, str)
            and VARIABLE.fullmatch(variable) and variable not in RESERVED_ENV and not variable.startswith(('LD_', 'PYTHON'))
            for name, variable in table.items()), f'Invalid operator {label}: names in lower case, variables in capitals')
    names = [*inputs, *values, *secrets]
    require(len(set(names)) == len(names), 'Operator input, value and secret names must be distinct')
    require(not set(values.values()) & set(secrets.values()), 'An operator variable is a value or a secret, not both')
    access = merged['development_gateway']
    if access is not None:
        require(isinstance(access, dict) and set(access) <= set(MODEL_ACCESS_KEYS),
                'Unknown [breakwater.model_access] key: ' + ', '.join(sorted(set(access) - set(MODEL_ACCESS_KEYS))))
        access = {**MODEL_ACCESS_DEFAULTS, **access}
        require(isinstance(access.get('upstream'), str) and access['upstream'].startswith(('https://', 'http://')),
                '[breakwater.model_access] needs its upstream URL')
        models = access.get('chat_models')
        require(isinstance(models, dict) and models and all(
            isinstance(alias, str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]*', alias) and isinstance(model, str) and model
            for alias, model in models.items()), '[breakwater.model_access] needs chat_models: alias = "model"')
        require(access.get('embed_model') is None or (isinstance(access['embed_model'], str) and access['embed_model']),
                'Invalid model_access embed_model')
        for key in ('max_chat_tokens', 'max_embed_tokens'):
            require(key not in access or (type(access[key]) is int and access[key] > 0), f'Invalid model_access {key}')
        require(isinstance(access['host'], str) and HOST_NAME.fullmatch(access['host']) is not None,
                'Invalid model_access host')
        require(type(access['port']) is int and 1 <= access['port'] <= 65535, 'Invalid model_access port')
        require(isinstance(access['key_env'], str) and VARIABLE.fullmatch(access['key_env']) is not None,
                'Invalid model_access key_env')
        merged['development_gateway'] = access
    merged['credential_env'] = sorted(set(secrets.values()) | ({access['key_env']} if access else set()))
    env = merged['authoring_env']
    require(isinstance(env, dict) and all(
        isinstance(name, str) and VARIABLE.fullmatch(name) and name not in CREDENTIAL_ENV
        and name not in merged['credential_env'] and name not in RESERVED_ENV and name not in WORKER_FIXED_ENV
        and not name.startswith(('LD_', 'PYTHON')) and isinstance(value, str) and '\0' not in value
        and len(value) <= 65536 for name, value in env.items()),
            'Invalid authoring environment: names in capitals, string values, never a credential, a loader or '
            'interpreter variable, or a variable the framework sets')
    if merged['inbound_ports'] is not None:
        port_range(merged['inbound_ports'], 'inbound_ports')
        merged['inbound_host'] = merged['inbound_host'] or 'host.docker.internal'
    require(merged['inbound_host'] is None or (isinstance(merged['inbound_host'], str)
                                               and HOST_NAME.fullmatch(merged['inbound_host']) is not None),
            'Invalid inbound_host')
    module = merged['host_module']
    require(module is None or (isinstance(module, str) and relative_paths([module], 'host module')
                               and module.startswith('tests/') and module.endswith('.py')),
            'The host module is a Python file under tests/')
    timeout = merged['host_timeout_sec']
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, 'Invalid host timeout_sec')
    require(isinstance(merged['options'], dict), '[breakwater.options] must be a table')
    endpoints = merged['endpoints']
    require(isinstance(endpoints, list) and len(endpoints) <= 250, 'Endpoints are a list of at most 250 tables')
    reserved = reserved_names(merged)
    rows, seen = [], set()
    for item in endpoints:
        require(isinstance(item, dict) and {'name', 'port'} <= set(item) <= {'name', 'port', 'service', 'reach', 'health'},
                'An endpoint has a name and a port, and optionally service, reach and health')
        name, port = item['name'], item['port']
        require(isinstance(name, str) and HOST_NAME.fullmatch(name) is not None and name not in seen
                and name not in reserved, f'Invalid, duplicate or reserved endpoint name: {name!r}')
        require(type(port) is int and 1 <= port <= 65535, 'Invalid endpoint port')
        service = item.get('service', f'127.0.0.1:{port}')
        host, _, service_port = service.rpartition(':') if isinstance(service, str) else ('', '', '')
        require(bool(host) and service_port.isdigit() and 1 <= int(service_port) <= 65535,
                'An endpoint service is HOST:PORT')
        reach = item.get('reach', ['agent'])
        require(isinstance(reach, list) and reach and set(reach) <= set(REACH) and len(set(reach)) == len(reach),
                'An endpoint reach lists agent, coordinator or submission')
        health = item.get('health')
        require(health is None or (isinstance(health, str) and health.startswith('/')), 'An endpoint health check is a path')
        seen.add(name)
        rows.append({'name': name, 'port': port, 'service': service, 'reach': reach, 'health': health})
    merged['endpoints'] = rows


def assets(config):
    """The declared inputs as {name, source, target, stages} rows; stages use the internal stage names."""
    rows, names, seen = [], set(), set()
    for item in config.get('assets') or []:
        require(isinstance(item, dict) and set(item) == {'name', 'source', 'target', 'stages'},
                'An asset needs exactly name, source, target and stages')
        name, source, target, stages = item['name'], item['source'], Path(str(item['target'])), item['stages']
        require(isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]*', name) and name not in names,
                'Asset names must be unique lower-case identifiers')
        names.add(name)
        require(isinstance(source, str) and source and not source.startswith('/') and '..' not in Path(source).parts,
                'An asset source is a path relative to the asset root')
        require(target.is_absolute() and '..' not in target.parts and len(target.parts) >= 3, 'Invalid asset target')
        require(target.parts[1] in ASSET_ROOTS, 'Unexpected asset destination')
        stages = [STAGES.get(stage, stage) for stage in stages] if isinstance(stages, list) else None
        require(stages and set(stages) <= {'agent', 'intermediate', 'verifier'} and len(set(stages)) == len(stages),
                'Invalid asset stages')
        require(not ('agent' in stages and target.parts[1] == 'private'), 'Private evaluation input exposed to agent')
        for stage in stages:
            require((stage, str(target)) not in seen, 'Duplicate asset destination')
            seen.add((stage, str(target)))
        rows.append({'name': name, 'source': source, 'target': str(target), 'stages': stages})
    return rows


def bindings(config, asset_root, run_root, path=None):
    """The declared assets resolved under ``asset_root``: [{source (absolute), target, stages}].

    An operator's bindings file (``path``: [{source, target, stages}], sources relative to the asset root) may move
    sources, but must declare exactly the task's targets and stage visibility."""
    asset_root = Path(asset_root)
    declared = config['assets']
    if path is None:
        rows = [{key: row[key] for key in ('source', 'target', 'stages')} for row in declared]
    else:
        rows = read_json(path)
        require(isinstance(rows, list) and all(isinstance(row, dict) and set(row) == {'source', 'target', 'stages'}
                                               for row in rows), 'Unknown input binding field')
        rows = [{**row, 'stages': [STAGES.get(stage, stage) for stage in row['stages']]
                 if isinstance(row['stages'], list) else row['stages']} for row in rows]
        require({(row['target'], tuple(sorted(row['stages']))) for row in rows}
                == {(row['target'], tuple(sorted(row['stages']))) for row in declared},
                'Binding destinations/stage visibility must match the task contract')
    resolved, seen = [], set()
    for row in rows:
        source = (asset_root / row['source']).resolve()
        target = Path(row['target'])
        require(source.exists() and source.is_relative_to(asset_root.resolve()) and not overlaps(source, run_root),
                'Input must be an existing asset under the dedicated asset root')
        require(target.is_absolute() and '..' not in target.parts and len(target.parts) >= 3, 'Invalid asset target')
        require(target.parts[1] in ASSET_ROOTS, 'Unexpected asset destination')
        require(isinstance(row['stages'], list) and row['stages']
                and set(row['stages']) <= {'agent', 'intermediate', 'verifier'}, 'Invalid binding stages')
        require(not ('agent' in row['stages'] and target.parts[1] == 'private'), 'Private evaluation input exposed to agent')
        for stage in row['stages']:
            require((stage, str(target)) not in seen, 'Duplicate asset destination')
            seen.add((stage, str(target)))
        resolved.append({'source': str(source), 'target': str(target), 'stages': list(row['stages'])})
    return resolved
