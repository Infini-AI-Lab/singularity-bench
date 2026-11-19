"""The Docker author image's firewall: the hosts lines and iptables rules author/task_config.firewall_plan derives from an
image policy are exactly the ones the bootstrap applied before (fixtures/bootstrap_rules.json). No Docker or root.
"""
import json
from pathlib import Path
import sys
import unittest

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRAMEWORK / 'author')]
import task_config  # noqa: E402

GOLDEN = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'bootstrap_rules.json').read_text())


def decide(commands, uid, address, port):
    """What iptables does with a new TCP connection from ``uid`` to address:port under these commands: the first matching
    OUTPUT rule's target, else the chain's policy."""
    policy = None
    for argv in commands:
        if argv[0] != 'iptables':
            continue
        if argv[1] == '-P':
            policy = argv[3]
            continue
        arguments, index, target, matched = argv[3:], 0, None, True
        while index < len(arguments):
            key = arguments[index]
            if key == '-m':
                index += 2
                continue
            value = arguments[index + 1]
            if key == '--uid-owner' and int(value) != uid or key == '-d' and value != address:
                matched = False
            if key == '--dport':
                low, _, high = value.partition(':')
                matched = matched and int(low) <= port <= int(high or low)
            if key in ('--sport', '--ctstate'):  # replies only: never a new connection
                matched = False
            if key == '-j':
                target = value
            index += 2
        if matched:
            return target
    return policy


class FirewallPlanTests(unittest.TestCase):
    def test_each_policy_gets_the_recorded_hosts_and_rules(self):
        for name, case in GOLDEN['cases'].items():
            with self.subTest(policy=name):
                hosts, commands = task_config.firewall_plan(
                    case['task'], case['plan'], [tuple(row) for row in case['resolved']],
                    resolve=lambda host: GOLDEN['addresses'][host], hostname=lambda: GOLDEN['hostname'])
                self.assertEqual((hosts, commands), (case['hosts'], case['commands']))

    def test_endpoints_on_other_hosts_allow_what_the_captured_rules_allowed(self):
        # Before endpoints, the author image read a set of remote hosts from its own policy and opened each one's
        # port window and a controller port, refusing a management port on each host. The plan those hosts become as
        # endpoints decides every connection the same way, with the same host names.
        case = GOLDEN['cases']['remote_endpoints']
        addresses = sorted({address for values in GOLDEN['addresses'].values() for address in values} | {'8.8.8.8'})
        for uid in (1100, 1101, 0):
            for address in addresses:
                for port in (22, 8000, 8001, 9000, 9099, 9100, 9101, 9150, 9299, 9300, 19099, 19100, 19101, 30000):
                    with self.subTest(uid=uid, address=address, port=port):
                        self.assertEqual(decide(case['commands'], uid, address, port),
                                         decide(case['captured_commands'], uid, address, port))
        def lines(text):
            return {tuple(line.split()) for line in text.split('\n') if line.strip()}
        self.assertEqual(lines(case['hosts']), lines(case['captured_hosts']))

    def test_an_endpoint_on_a_framework_address_or_off_ipv4_is_refused(self):
        case = GOLDEN['cases']['model_access']
        resolved = [tuple(row) for row in case['resolved']]
        for host in ('172.18.0.2', '172.18.0.3', 'author-gateway'):  # the author gateway, the service, by name
            plan = {**case['plan'], 'endpoints': [{'name': 'store', 'host': host, 'ports': ['9000']}]}
            with self.subTest(host=host), self.assertRaisesRegex(SystemExit, 'framework service address'):
                task_config.firewall_plan(case['task'], plan, resolved, resolve=lambda name: GOLDEN['addresses'][name],
                                          hostname=lambda: 'x')
        plan = {**case['plan'], 'endpoints': [{'name': 'store', 'host': 'store.local', 'ports': ['9000']}]}
        with self.assertRaises((SystemExit, ValueError)):
            task_config.firewall_plan(case['task'], plan, resolved, resolve=lambda name: ['::1'], hostname=lambda: 'x')
        plan['rejects'] = [{'host': '10.9.0.1', 'port': 9100}]
        hosts, commands = task_config.firewall_plan(case['task'], plan, resolved, resolve=lambda name: ['10.9.0.1'],
                                                    hostname=lambda: 'x')
        self.assertIn('\n10.9.0.1 store\n', hosts)
        reject = ['iptables', '-A', 'OUTPUT', '-m', 'owner', '--uid-owner', '1101', '-p', 'tcp', '-d', '10.9.0.1',
                  '--dport', '9100', '-j', 'REJECT']
        accept = ['iptables', '-A', 'OUTPUT', '-m', 'owner', '--uid-owner', '1101', '-d', '10.9.0.1', '-p', 'tcp',
                  '--dport', '9000', '-j', 'ACCEPT']
        self.assertLess(commands.index(reject), commands.index(accept))


PLAN = {'version': 1, 'services': [{'host': 'gateway', 'port': 8000}], 'endpoints': [], 'rejects': [], 'hosts': [],
        'reply_ports': None, 'env': {'TASK_BASE_URL': 'http://gateway:8000/v1'}}


class AuthoringPlanTests(unittest.TestCase):
    def test_the_plan_is_validated_and_digested_exactly(self):
        import hashlib
        text = json.dumps(PLAN)
        plan, digest = task_config.authoring_plan({'BREAKWATER_AUTHORING_PLAN': text})
        self.assertEqual((plan, digest), (PLAN, hashlib.sha256(text.encode()).hexdigest()))
        env = task_config.worker_environment({'workdir': '/app'}, {'PATH': '/usr/bin', 'SECRET': 'x'}, plan)
        self.assertEqual((env['TASK_BASE_URL'], 'SECRET' in env, env['HOME']), ('http://gateway:8000/v1', False, '/home/worker'))
        self.assertNotIn('TASK_BASE_URL', task_config.worker_environment({'workdir': '/app'}, {}))
        for change in ({'version': 2}, {'extra': 1}, {'services': [{'host': 'gateway'}]},
                       {'services': [{'host': 'gateway', 'port': 0}]}, {'services': [{'host': 'bad host', 'port': 1}]},
                       {'endpoints': [{'name': 'w', 'host': 'h', 'ports': ['9:8']}]},
                       {'endpoints': [{'name': 'w', 'host': 'h', 'ports': []}]}, {'hosts': [['w', '::1']]},
                       {'reply_ports': '1:2:3'}, {'env': {'HOME': '/x'}}, {'env': {'LD_PRELOAD': 'x'}},
                       {'env': {'PYTHONPATH': 'x'}}, {'env': {'TASK': 1}}, {'env': {'lower': 'x'}}):
            with self.subTest(change=change), self.assertRaises((SystemExit, ValueError)):
                task_config.authoring_plan({'BREAKWATER_AUTHORING_PLAN': json.dumps({**PLAN, **change})})
        with self.assertRaises(SystemExit):
            task_config.authoring_plan({})


if __name__ == '__main__':
    unittest.main()
