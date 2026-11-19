"""A stand-in task host module for test_host_hooks.py. What it does follows the operator value 'mode'."""
import os
import time

from host_hooks import Endpoint, FromInput, FromSecret, HostPlan, Text


def note(host, phase):
    with (host.state / 'order').open('a') as stream:
        stream.write(phase + '\n')


def preflight(host):
    mode = host.value('mode')
    if mode == 'fail-preflight':
        raise RuntimeError('preflight refused')
    if mode == 'slow':
        time.sleep(30)
    note(host, 'preflight')
    plans = {
        'credential-env': HostPlan(agent_env={'TASK_TOKEN': 'x'}),  # the secret's own variable
        'secret-env': HostPlan(agent_env={'TASK_URL': 'http://x/?t=' + host.secret('token')}),
        'secret-file': HostPlan(workspace_files={'leak.txt': Text('token=' + host.secret('token'))}),
        'secret-metadata': HostPlan(metadata={'token': host.secret('token')}),
        'ip-name': HostPlan(endpoints=[Endpoint('10.0.0.5', '10.0.0.5', [1])]),
        'loopback-name': HostPlan(endpoints=[Endpoint('127.0.0.5', '127.0.0.5', [1])]),
        'bad-reach': HostPlan(endpoints=[Endpoint('store', 'h', [1], reach=['verifier'])]),
        'bad-port': HostPlan(endpoints=[Endpoint('store', 'h', [0])]),
        'reserved': HostPlan(endpoints=[Endpoint('author-gateway', 'h', [1])]),
        'secret-workspace-source': HostPlan(workspace_files={'x': FromSecret('token')}),
        'bad-path': HostPlan(workspace_files={'../x': Text('x')}),
        'not-a-plan': {'endpoints': []},
    }
    if mode in plans:
        return plans[mode]
    return HostPlan(endpoints=[Endpoint('store', '127.0.0.1', [9000, 9001], reach=['agent', 'coordinator'])],
                    agent_env={'TASK_STORE': 'http://store:9000'},
                    workspace_files={'config/fleet.json': FromInput('fleet'), 'note.txt': Text('hello')},
                    private_files={'token': FromSecret('token'), 'fleet.json': FromInput('fleet')},
                    metadata={'fleet_id': 'f1'})


def before_authoring(host):
    note(host, 'before_authoring')
    host.evidence.json('before.json', {'environment': sorted(os.environ), 'phase': host.phase})
    if host.value('mode') == 'evidence-outside':
        host.evidence.path('../escape.json')


def after_evaluation(host, windows):
    if host.value('mode') == 'fail-after':
        raise RuntimeError('collection failed')
    note(host, 'after_evaluation')
    host.evidence.json('journals/after.json', {'windows': windows, 'order': (host.state / 'order').read_text()})


def on_failure(host):
    note(host, 'on_failure')
    raise RuntimeError('a failure hook error is ignored')
