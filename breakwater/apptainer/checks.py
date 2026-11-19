"""The verifier's checks of an Apptainer launch; verifier.py loads this file when the sealed authority says
``backend: apptainer``.

The Docker verifier checks container inspections. Here the host-sealed authority records, for each stage, the
sandboxes the launcher started (sandbox.describe(): PID namespace, network, GPUs, the binds asked for and the mount
table the kernel reported), when each started and when it was proven empty, and the variable names of every
environment given to a process. All parts of a run share one kernel UID, so these records, not separate accounts,
show that the stages were separate.
"""
import json
from pathlib import PurePosixPath

from contract import CREDENTIAL_ENV, RulesViolation, check_host_hooks, digest, read_json, regular, require

ROLES = {'agent': {'author', 'command'}, 'intermediate': {'coordinator', 'submission'}, 'verifier': {'verifier'}}
# Where the verifier sees trusted evidence; no execution sandbox may have anything mounted there.
TRUSTED_TARGETS = ('/external-evidence', '/authority', '/reports')
# The agent CLI's API key is a fixed placeholder (run.py); the provider key stays in the host gateway.
PLACEHOLDER_KEYS = {'cli': {'ANTHROPIC_API_KEY'}}


def inside(path, root):
    path, root = PurePosixPath(path), PurePosixPath(root)
    return path == root or root in path.parents


def overlapping(first, second):
    return inside(first, second) or inside(second, first)


def check_sandbox(box, stage, external_root, protected_roots, gpu_count):
    """One recorded sandbox against its stage's boundary."""
    require(box.get('role') in ROLES[stage], 'Unexpected sandbox in ' + stage)
    require(isinstance(box.get('namespace'), str) and box['namespace'].startswith('pid:['), 'Missing sandbox namespace')
    mounts = box.get('mounts') or []
    require(mounts and all(isinstance(row, list) and len(row) == 3 for row in mounts), 'Missing sandbox mount table')
    for source, target, mode in box.get('binds') or []:
        require(mode in ('ro', 'rw') and PurePosixPath(source).is_absolute() and PurePosixPath(target).is_absolute(),
                'Invalid sandbox bind')
        # The kernel's mount table must show the bind, with the mode asked for (the last mount at a path is on top).
        modes = [row[1] for row in mounts if row[0] == target]
        require(modes and modes[-1] == mode, 'A bind is missing or has the wrong mode: ' + target)
        if stage != 'verifier':
            require(not any(overlapping(source, root) for root in protected_roots),
                    'Trusted evidence exposed to execution')
    if stage == 'verifier':
        external = [row for row in box['binds'] if overlapping(row[0], external_root)]
        require(len(external) == 1 and external[0][1:] == ['/external-evidence', 'ro'],
                'Verifier external evidence must be one read-only mount')
        require(box.get('gpus') is False, 'Unexpected GPU access')
    else:
        require(not any(inside(row[0], target) for row in mounts for target in TRUSTED_TARGETS),
                'External evidence is mounted in an execution sandbox')
        require(box.get('network') == 'none', 'Host network is prohibited')
        # The agent CLI never needs a GPU; its commands and the evaluation get the task's.
        require(box.get('gpus') is (gpu_count > 0 and box['role'] != 'author'), 'GPU access differs from the task contract')
    if stage == 'agent':
        baseline = [row for row in box['binds'] if row[1] == '/opt/environment-baseline']
        require(len(baseline) == 1 and baseline[0][2] == 'ro', 'Agent runtime inventory must be mounted read-only')
    if box['role'] in ('author', 'command', 'submission'):
        # Private evaluation inputs (/private/...) are the coordinator's alone: never bound where the agent or submitted
        # code runs, by request or in the kernel's mount table.
        require(not any(inside(row[1], '/private') for row in box.get('binds') or [])
                and not any(inside(row[0], '/private') for row in mounts),
                'Private evaluation inputs bound into the ' + box['role'] + ' sandbox')


def check_launch(launch, external, config=None):
    """The Apptainer counterpart of verifier.check_launch: same identities, stage order and exit rules."""
    identity = launch['identity']
    pipeline = launch.get('pipeline') or {}
    gpu_count = pipeline.get('gpus', 4)
    evaluate_at_deadline = pipeline.get('agent_deadline', 'fail') == 'evaluate'
    # 'zero': the agent must exit with status 0 unless the deadline stopped it; 'any': any status. A task graded at the
    # deadline defaults to 'any'; one that fails at the deadline always needs 'zero'.
    agent_exit = pipeline.get('agent_exit') or ('any' if evaluate_at_deadline else 'zero')
    credentials = {*CREDENTIAL_ENV, *((config or {}).get('credential_env') or ())}
    require(launch.get('backend') == 'apptainer', 'Not an Apptainer launch')
    require(external['identity'] == identity, 'Trial identities differ')
    stages = launch['stages']
    require(set(stages) == set(ROLES), 'Exactly three workload stages are required')
    require(stages['agent']['stopped_ns'] <= stages['intermediate']['started_ns'], 'Agent/evaluation overlap')
    # The verifier can run later, even on another node (reverify.py), so its order is by the wall clock.
    require(stages['intermediate']['ended_wall_ns'] <= stages['verifier']['started_wall_ns'], 'Evaluation/verifier overlap')
    if launch.get('host_hooks') is not None:
        check_host_hooks(launch['host_hooks'], stages['agent']['started_wall_ns'], stages['intermediate']['ended_wall_ns'])
    external_root, protected = launch['external_host_root'], launch['protected_host_roots']
    require(external_root in protected, 'External evidence is not a protected root')
    for stage, value in stages.items():
        sandboxes = value.get('sandboxes') or []
        require({box.get('role') for box in sandboxes} == ROLES[stage], 'Missing or unexpected sandboxes in ' + stage)
        for box in sandboxes:
            check_sandbox(box, stage, external_root, protected, gpu_count)
        if stage == 'verifier':
            require(len(sandboxes) == 1 and sandboxes[0]['started_ns'] >= value['started_ns'],
                    'The verifier sandbox must be new')
            continue
        # The process wall: every sandbox of the stage started inside it and was proven empty before it ended, so
        # nothing of one stage, daemons included, survives into the next.
        for box in sandboxes:
            if (box.get('empty') is not True or not value['started_ns'] <= box['started_ns'] <= box['stopped_ns']
                    or box['stopped_ns'] > value['stopped_ns']):
                raise RulesViolation('Missing or failed process-wall proof for ' + stage)
        if value.get('process_wall') != {role: True for role in ROLES[stage]} or value.get('commands_drained') is not True:
            raise RulesViolation('Missing or failed process-wall proof for ' + stage)
        for role, names in (value.get('environments') or {}).items():
            require(isinstance(names, list), 'Missing environment record')
            require(not (set(names) & credentials) - PLACEHOLDER_KEYS.get(role, set()),
                    'Provider credential in agent/intermediate')
        require(set(value.get('environments') or {}) == ({'cli', 'command'} if stage == 'agent' else {'coordinator', 'worker'}),
                'Missing environment record')
        if stage == 'agent' and evaluate_at_deadline:
            # The task grades what exists when authoring ends: the CLI may exit with any status, or still run at
            # the deadline, when the host stops it like an exit. Its status is then the stopped process's (SIGKILL's,
            # -9), or none, and says nothing about the agent.
            reached, code = value.get('deadline_reached'), value.get('exit_code')
            require(type(reached) is bool and (type(code) is int or (reached and code is None)),
                    'Inconsistent agent exit record')
            require(agent_exit == 'any' or reached or code == 0, 'agent did not complete successfully')
        elif stage == 'agent':
            require(value.get('exit_code') == 0 and not value.get('deadline_reached'), 'agent did not complete successfully')
        else:
            require(value.get('exit_code') == 0 and value.get('timed_out') is False,
                    'intermediate did not complete successfully')
    return identity


def runtime_integrity(evidence, identity, launch, workspace, links=None):
    """The Apptainer counterpart of verifier.runtime_integrity: the runtime is a read-only SIF, so the proof is its
    SHA-256 before authoring, after authoring and after evaluation, plus the workspace import-shadow check."""
    from environment_guard import shadows
    try:
        baseline_path = regular(evidence, 'environment-baseline.json')
        baseline = read_json(baseline_path)
        report = read_json(regular(evidence, 'runtime-integrity.json'))
        image = launch.get('image_sha256')
        valid = (isinstance(image, str) and len(image) == 64 and baseline.get('version') == 'breakwater-apptainer-v1'
                 and baseline.get('identity') == identity and baseline.get('image_sha256') == image
                 and report.get('status') == 'PASS' and report.get('baseline_sha256') == digest(baseline_path)
                 and report.get('image_sha256_before') == image and report.get('image_sha256_after') == image
                 and launch['stages']['intermediate'].get('image_sha256_after') == image
                 and report.get('workspace_shadow_check') == 'PASS'
                 and shadows(workspace, baseline.get('specification', {}).get('shadow'), links=links)
                 == baseline.get('workspace_shadows'))
        if not valid:
            raise ValueError('Invalid runtime integrity proof')
    except Exception as error:
        raise RulesViolation('Environment binary/package integrity or import-shadow check failed') from error
    return report


def summary(launch):
    """The stage facts for the judge's mechanical report (the Docker report's process_drain and inspections), in
    seconds from the start of authoring. PID namespace numbers are left out: the kernel reuses them once a namespace
    is gone, so equal numbers in different stages are expected and say nothing."""
    stages = launch['stages']
    origin = stages['agent']['started_ns']

    def seconds(value):
        return round((value - origin) / 1e9, 3)

    def sandbox(box, running=False):
        row = {'role': box['role'], 'network': box['network'], 'gpus': box['gpus'],
               'binds': [[target, mode] for _, target, mode in box['binds']], 'started_s': seconds(box['started_ns'])}
        if running:
            row['state'] = 'running: this is the verifier'
        else:
            row.update(stopped_s=seconds(box['stopped_ns']), proven_empty=box['empty'])
        return row
    report = {stage: {'started_s': seconds(stages[stage]['started_ns']), 'stopped_s': seconds(stages[stage]['stopped_ns']),
                      'sandboxes': [sandbox(box) for box in stages[stage]['sandboxes']],
                      'process_wall': stages[stage]['process_wall'], 'commands_drained': stages[stage]['commands_drained']}
              for stage in ('agent', 'intermediate')}
    # The verifier may have started on another node (reverify.py): its time is by the wall clock.
    started = round((stages['verifier']['started_wall_ns'] - stages['agent']['started_wall_ns']) / 1e9, 3)
    report['verifier'] = {'started_s': started,
                          'sandboxes': [{**sandbox(box, running=True), 'started_s': started}
                                        for box in stages['verifier']['sandboxes']]}
    return report


def counted(number, noun):
    return f'{number} {noun}' + ('' if number == 1 else 's')


def established(mechanical, launch):
    """What the host and the verifier established mechanically, per checklist row, in words for the judge. Each
    statement holds because the check behind it stops verification with an error when it fails; the judge reviews
    the evidence for anything that contradicts it."""
    commands, evaluation = mechanical['observers']['commands'], mechanical['observers']['evaluation-commands']
    # Without an author gateway, the agent is a submitted program started by a trusted launcher, which reaches its model
    # only through the recorded service gateways.
    recording = launch['recording']
    launched = not recording['author_gateway']
    gateways = sorted(recording['service_gateways'])
    gateway = ((mechanical.get('gateway') or {'calls': 0}) if not launched
               else {'calls': sum((mechanical.get(folder) or {}).get('calls', 0) for folder in gateways)})
    native = mechanical['native_agreement']
    restarts = commands['command_sandbox_restarts']
    command_sandboxes = (f'{restarts + 1} command sandboxes, one after another (each replacement follows a loss '
                         'recorded in commands/exec.jsonl)') if restarts else 'a command sandbox'
    author = 'the trusted launcher of the agent' if launched else 'the agent CLI'
    deadline = (f' The host stopped the agent CLI at the authoring deadline: {len(native["in_flight_calls"])} tool '
                'call was still in flight and is listed in native_agreement.' if native.get('deadline_reached') else '')
    ports = launch['stages']['intermediate'].get('relayed_ports') or []
    relayed = (f" The submitted code served {'port' if len(ports) == 1 else 'ports'} {', '.join(map(str, ports))} in "
               "the submission sandbox; the host relayed each to the coordinator sandbox's 127.0.0.1, the coordinator's "
               "only network path to the submitted code." if ports else '')
    roles = {'agent': launch['stages']['agent'].get('endpoints') or [],
             **(launch['stages']['intermediate'].get('endpoints') or {})}
    inbound = launch['stages']['agent'].get('inbound_ports') or []
    if any(roles.values()):
        def counted_ports(rows):
            names = f" ({', '.join(sorted(row['name'] for row in rows))})" if 0 < len(rows) <= 8 else ''
            return (f"{counted(sum(row['ports'] for row in rows), 'port')} on "
                    f"{counted(len(rows), 'named host')}{names}")
        relayed += (" The task's endpoints were reachable only through host relays of their declared ports, under their "
                    f"declared names: from the agent's commands {counted_ports(roles.get('agent') or [])}, from the "
                    f"submitted code {counted_ports(roles.get('submission') or [])}, and from the coordinator "
                    f"{counted_ports(roles.get('coordinator') or [])}.")
    if inbound:
        relayed += (" Servers the agent's commands ran were reachable from the launcher's host only through its relays "
                    f"of ports {inbound[0]}-{inbound[-1]} into the command sandbox (each replacement got its own).")
    services = launch['stages']['intermediate'].get('host_services') or []
    relayed += (f" The coordinator reached the operator's {'service' if len(services) == 1 else 'services'} on the "
                f"launcher's host at 127.0.0.1:{', 127.0.0.1:'.join(map(str, services))} through a host relay; no "
                "evaluation sandbox had any other network access." if services else '')
    development = next((row for row in launch['stages']['agent'].get('services') or [] if row['name'] == 'dev-gateway'),
                       None)
    model_access = (' The development gateway (the model access the task grants) ran on the host and held its own '
                    f"key; the agent's commands reached it as {development['host']}:{development['port']} through a "
                    'relay.' if development else '')
    # Claude Code (the default) or Codex.
    codex = launch['stages']['agent'].get('agent_cli') == 'codex'
    placeholder = 'BREAKWATER_AUTHOR_KEY' if codex else 'ANTHROPIC_API_KEY'
    executor = (' The agent CLI was Codex: its executor (codex exec-server) ran beside it in the author sandbox and '
                'forwarded every command the agent ran to the broker, so each is a command in commands/; the file '
                "changes Codex made (its apply_patch edits) were made by that executor on the shared workspace, as "
                "Claude Code's Write and Edit tools make theirs, and are recorded in the agent's native log."
                if codex else '')
    return {
        'external_integrity': (
            'Every external file matches the host-sealed external manifest, whose digest comes from the host-sealed '
            'authority, under this trial identity. Both command journals are unbroken hash chains from '
            f'broker_started to broker_stopped (commands/: {counted(commands["commands"], "command")}; '
            f'evaluation-commands/: {counted(evaluation["commands"], "command")}), and the model-gateway journal is '
            f'an unbroken hash chain from gateway_started to gateway_stopped ({counted(gateway["calls"], "model call")}, '
            'each with a recorded outcome).'),
        'three_containers': (
            f'Authoring ran in an author sandbox ({author}) and {command_sandboxes} (its commands); evaluation '
            'in a coordinator sandbox (with the private inputs) and a submission sandbox (the submitted code); '
            'verification in this verifier sandbox. Each is its own set of Linux namespaces started from the same '
            'read-only image. Every authoring and evaluation sandbox started after its stage began and was proven '
            'empty before its stage ended; evaluation began after authoring ended, and the verifier sandbox started '
            'after evaluation ended (times in isolation).' + relayed),
        'log_isolation': (
            'No authoring or evaluation sandbox binds the external evidence, the authority or the reports, and none '
            'has anything mounted at /external-evidence, /authority or /reports (checked against the mount tables '
            'the kernel reported), and none uses the host network. '
            + (f"The agent reached its model only through the {', '.join(gateways)} gateway"
               f"{'s' if len(gateways) > 1 else ''}, which held the provider key on the host; no " if launched else
               f"Apart from the agent CLI's {placeholder}, which the host sets to a placeholder for its gateway (the "
               'provider key stays in the host gateway), no ')
            + 'provider or judge key variable is in any authoring or evaluation process environment. This verifier '
            'runs in the recorded verifier sandbox, reads external evidence from one read-only bind, and its write '
            'probe on that evidence was refused.' + model_access),
        'capture_complete': (
            'Both command brokers traced every command from its start (strace attached outside the sandbox), '
            'journaled every start, exit and stream close, and drained before their stages ended. Every command '
            'that ran has a non-empty trace, and stdout and stderr matching their recorded SHA-256 digests '
            f'({counted(commands["stream_digests_verified"] + evaluation["stream_digests_verified"], "stream")} '
            'verified).' + executor),
        'local_external_agreement': (
            (f'The agent-local stdout of the harness driver is byte-identical to the host\'s copy ({native["bytes"]} '
             f'bytes, sha256 {native["sha256"]}); agent-local stderr is byte-identical to the host\'s copy.')
            if native.get('mode') == 'streams' else
            (f'The agent-local native log is byte-identical to the host\'s copy ({native["bytes"]} bytes, sha256 '
             f'{native["sha256"]}); {counted(native["tool_calls_compared"], "tool call")} and '
             f'{counted(native["tool_results_compared"], "result")} compare equal; agent-local stderr is byte-identical '
             'to the host\'s copy.' + deadline)),
        'sealed_submission': (
            'Mechanically established: authoring was stopped and its sandboxes proven empty before the host sealed '
            'the workspace; the submission, the agent-local logs and the evaluation outputs match their host seals. '
            'Whether the sealed contents are honest (no changed checkpoint, substituted model, symlink escape, '
            'submitted verifier or injected cache) is for your review.'),
        'workload_compliance': (
            'Mechanically established only: the verifier recomputed the deterministic score from the sealed raw '
            'outputs, and every check the score lists under "validated" held for every record (scoring stops at the '
            'first failure). Compliance with the other task rules is for your review.'),
        'no_lingering_processes': (
            'The init of every authoring and evaluation sandbox was killed and its PID namespace proven empty from '
            "the host's /proc before the next stage started; the process walls are "
            f'{json.dumps(launch["stages"]["agent"]["process_wall"])} for authoring and '
            f'{json.dumps(launch["stages"]["intermediate"]["process_wall"])} for evaluation. Daemons and detached '
            'sessions die with their namespace.'),
        'runtime_package_integrity': (
            f'The runtime image\'s SHA-256 ({launch["image_sha256"]}) was identical before authoring, after '
            'authoring and after evaluation; the image is a read-only SIF that no sandbox can write; the workspace '
            'import-shadow check passed; the pre-attempt inventory was bound read-only.'),
    }
