#!/usr/bin/env python3
"""The breakwater command.

    breakwater check TASK_DIR                           validate a task against the specification (SPEC.md)
    breakwater run ARGS...                              run a task on Apptainer (the default backend): authoring,
                                                        evaluation, verification
    breakwater run --backend docker ARGS...             run a task on Docker
    breakwater reverify ARGS...                         run the verifier again on a finished Apptainer run
    breakwater install-cli --dest DIR [--agent codex]   the pinned agent CLI for the Apptainer backend
    breakwater install-judge --dest DIR                 the pinned judge runner for the Apptainer backend

Apptainer runs without root or a daemon (the usual setup of GPU clusters with Slurm); Docker suits a host with a
Docker daemon and the NVIDIA Container Toolkit. ``run``, ``reverify`` and the installers hand their arguments to the
backend's launcher: ``breakwater run --help`` lists Apptainer's, ``breakwater run --backend docker --help`` Docker's.
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    ('run', 'docker'): HERE / 'run.py',
    ('run', 'apptainer'): HERE / 'apptainer' / 'run.py',
    ('reverify', 'apptainer'): HERE / 'apptainer' / 'reverify.py',
    ('install-cli', 'apptainer'): HERE / 'apptainer' / 'install_cli.py',
    ('install-judge', 'apptainer'): HERE / 'apptainer' / 'install_judge.py',
}


def check(task):
    """Validate a task directory and summarize its contract."""
    sys.path.insert(0, str(HERE))
    import spec
    from contract import ContractError
    try:
        config = spec.load(task)
    except (ContractError, OSError, ValueError) as error:
        print(f'{task}: not a valid Breakwater task: {error}', file=sys.stderr)
        return 1
    pending = spec.unsupported(config, 'apptainer')
    print(json.dumps({
        'task': config['task'], 'gpus': config['gpus'], 'workdir': config['workdir'],
        'budgets_sec': {'authoring': config['agent_timeout_sec'], 'evaluation': config['evaluation_timeout_sec'],
                        'judge': config['judge_timeout_sec']},
        'assets': [{'name': row['name'], 'target': row['target'],
                    'stages': [{'intermediate': 'evaluation'}.get(stage, stage) for stage in row['stages']]}
                   for row in config['assets']],
        'runners': config['runners'], 'review_profile': config['review_profile'],
        'agent_exit': config['agent_exit'],
        'model_access': ({key: config['development_gateway'][key] for key in ('host', 'port', 'key_env')}
                         | {'chat_models': sorted(config['development_gateway']['chat_models'])}
                         if config['development_gateway'] else None),
        'authoring_env': sorted(config['authoring_env']),
        'inbound_ports': config['inbound_ports'],
        'endpoints': [{key: row[key] for key in ('name', 'port', 'reach')} for row in config['endpoints']],
        'operator': {'inputs': config['operator_inputs'], 'values': sorted(config['operator_values']),
                     'secrets': sorted(config['operator_secrets'])},
        'host_module': config['host_module'], 'options': sorted(config['options']),
        'backends': {'apptainer': 'supported' if not pending else 'unsupported: ' + ', '.join(pending),
                     'docker': 'supported'},
    }, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='breakwater', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'run', 'reverify', 'install-cli', 'install-judge'))
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command, rest = argv[0], argv[1:]
    parser.parse_args([command])
    if command == 'check':
        if len(rest) != 1:
            parser.error('breakwater check TASK_DIR')
        return check(Path(rest[0]))
    backend = 'apptainer'  # the default: no root or daemon, as GPU clusters with Slurm provide
    if command == 'run' and '--backend' in rest:
        index = rest.index('--backend')
        if index + 1 >= len(rest) or rest[index + 1] not in ('apptainer', 'docker'):
            parser.error('--backend is apptainer (the default) or docker')
        backend = rest[index + 1]
        rest = rest[:index] + rest[index + 2:]
    script = SCRIPTS.get((command, backend))
    if script is None:
        parser.error(f'{command} is not available for the {backend} backend')
    # The launcher replaces this process, so it sees exactly the arguments, environment and signals given here.
    os.execv(sys.executable, [sys.executable, '-I', str(script), *rest])


if __name__ == '__main__':
    raise SystemExit(main())
