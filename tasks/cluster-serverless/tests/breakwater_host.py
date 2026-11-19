"""cluster-serverless on the launcher's host (Breakwater host hooks, breakwater/SPEC.md): the operator's GPU fleet.

The operator starts the fleet (bin/cluster.sh up) and passes its private manifest and its public view to the launcher
(--input fleet=RUN/fleet.json --input public-fleet=RUN/fleet.public.json), with the fleet's grader token (secret) and
the development receipt key (public by design) in the environment.

- preflight: the two manifests agree and have the graded shape; the graded suite is target-qualified, or the task
  declares allow_unqualified; the inference audit directory is readable here. The plan: the controller and every
  node's engine-port window for the agent's commands, the submitted plane and the coordinator, and the node managers
  for the coordinator alone (which holds the grader token); the public fleet view in the workspace; the private
  manifests and the token at /private/operator in the coordinator.
- before_authoring: no engine survives from an earlier session; the journals already present are noted.
- after_evaluation: every engine of this run stops and syncs its journal; this run's journals become evidence
  (external/task/fleet-inference/), checked, with the execution stages' windows.
- on_failure: the fleet is reset.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_fleet as fleet_tools  # noqa: E402
from contract import digest, read_json, require  # noqa: E402
from host_hooks import Endpoint, FromInput, FromSecret, HostPlan  # noqa: E402


def manifests(host):
    return read_json(host.input('fleet')), read_json(host.input('public-fleet'))


def check_options(options):
    """[breakwater.options]: repeats (a positive integer), time_scale (a positive number), allow_unqualified."""
    require(isinstance(options, dict) and set(options) <= {'repeats', 'time_scale', 'allow_unqualified'},
            'Unknown cluster option: ' + ', '.join(sorted(set(options) - {'repeats', 'time_scale', 'allow_unqualified'})))
    scale = options.get('time_scale', 1)
    require(type(options.get('repeats', 1)) is int and options.get('repeats', 1) >= 1
            and type(scale) in (int, float) and math.isfinite(scale) and scale > 0
            and type(options.get('allow_unqualified', False)) is bool, 'Invalid cluster options')


def preflight(host):
    fleet, public = manifests(host)
    fleet_tools.check_public_fleet(public, fleet)
    catalogue = read_json(host.task / 'tests' / 'public' / 'catalogue.json')
    fleet_tools.check_fleet_shape(fleet, catalogue)
    fleet_tools.check_fleet_shape(public, catalogue)
    options = host.contract['options']
    check_options(options)
    suite = read_json(host.task / 'tests' / 'suite-graded.json')
    # Before authoring, not after four hours: a development (unqualified) suite needs an explicit declaration.
    require(set(suite['qualification'].values()) == {'target-qualified'} or options.get('allow_unqualified'),
            'The graded suite is not target-qualified; declare allow_unqualified in [breakwater.options]')
    fleet_tools.audit_directory(fleet)
    managers = fleet_tools.manager_ports(fleet)
    if host.backend == 'apptainer':
        # A sandbox reaches a fleet host by name or by a loopback address, relayed: no other address can be named.
        import ipaddress
        for name in {row['name'] for row in fleet_tools.endpoints(fleet, trusted=True)}:
            try:
                address = ipaddress.ip_address(name)
            except ValueError:
                continue
            require(address.is_loopback, f'{name}: start the fleet with CLUSTER_FLEET_RUNTIME=apptainer, which keeps '
                                         'node names in its manifest')
    shared = [Endpoint(row['name'], row['host'], row['ports'], ('agent', 'submission', 'coordinator'))
              for row in fleet_tools.endpoints(fleet)]
    trusted = [Endpoint(host_name, host_name, [port], ('coordinator',)) for host_name, port in managers]
    workdir = host.contract['workdir']
    return HostPlan(
        endpoints=shared + trusted,
        agent_env={'CLUSTER_RECEIPT_KEY': host.value('receipt-key'), 'CLUSTER_NODES': f'{workdir}/fleet.json',
                   'CLUSTER_CATALOGUE': f'{workdir}/catalogue.json', 'CLUSTER_FLEET_MODE': 'real'},
        workspace_files={'fleet.json': FromInput('public-fleet')},
        private_files={'grader-token': FromSecret('grader-token'), 'fleet.json': FromInput('fleet'),
                       'fleet.public.json': FromInput('public-fleet')},
        metadata={'fleet_run_id': fleet_tools.run_id(fleet)})


def before_authoring(host):
    fleet, public = manifests(host)
    fleet_tools.reset(fleet, host.secret('grader-token'))
    before = fleet_tools.existing_journals(fleet_tools.audit_directory(fleet))
    (host.state / 'journals-before.json').write_text(json.dumps(before, sort_keys=True))
    host.evidence.json('fleet-manifests.json', {
        'private_sha256': digest(host.input('fleet')), 'public_sha256': digest(host.input('public-fleet')),
        'public': public, 'run_id': fleet_tools.run_id(fleet), 'preexisting_journals': before})


def after_evaluation(host, windows):
    # Collected only after both execution stages are proven empty. The reset repeats the coordinator's final one
    # (idempotent), so every worker incarnation of this run has stopped and synced its journal, even when the evaluation
    # ended abnormally.
    fleet, _ = manifests(host)
    fleet_tools.reset(fleet, host.secret('grader-token'))
    folder = host.evidence.path('fleet-inference')  # collect() creates it
    before = json.loads((host.state / 'journals-before.json').read_text())
    copied = fleet_tools.collect(fleet_tools.audit_directory(fleet), folder, before)
    report = fleet_tools.check_journals(folder, fleet_tools.run_id(fleet))
    host.evidence.json('fleet-inference.json', {**report, 'collected': copied, 'windows': windows})


def on_failure(host):
    fleet, _ = manifests(host)
    fleet_tools.reset(fleet, host.secret('grader-token'))
