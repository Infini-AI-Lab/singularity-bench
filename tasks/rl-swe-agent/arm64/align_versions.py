"""Align the testbed environment with the published x86 image of the same SWE-bench task.

Upstream recipes leave most dependencies unpinned, so a rebuild today resolves newer releases
than the published images, and new deprecation warnings fail tests run with warnings as errors.
x86-site-packages.txt lists the metadata directories found in the published x86 image (pulled
and mounted, never executed), as pythonX.Y/<dir>. This reinstalls those exact versions without dependency
resolution, installs missing distributions, and removes pytest plugins the x86 image lacks.
The editable task repository and pip itself are left alone. Distributions with no installable
aarch64 build are recorded in /opt/x86-alignment.json rather than failing the image. A published
image built for another Python minor (requests-1724 ships 2.7, its recipe asks for 3.9) has no
versions that apply, so the environment is then left as the recipe built it. Only the testbed
interpreter's own lib/pythonX.Y counts: recipes that create an unpinned environment and then
downgrade the interpreter (matplotlib: 3.14 -> 3.11) leave the newer minor's site-packages behind,
unimportable and ignored.

Runs under the base Miniconda Python; the testbed may be as old as Python 2.7.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ENV = Path('/opt/miniconda3/envs/testbed')
SKIP = {'pip'}


def key(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def parse(entry):
    """'.../pytest-7.1.2.dist-info' or '.../foo-1.0-py3.9.egg-info' -> (name, version)."""
    stem = re.sub(r'\.(dist|egg)-info$', '', Path(entry).name)
    parts = stem.split('-')
    return (parts[0], parts[1]) if len(parts) > 1 else (parts[0], None)


def interpreter_dir():
    """'python3.11': the testbed interpreter's lib directory (Python 2.7 compatible probe)."""
    return subprocess.check_output([str(ENV/'bin/python'), '-c',
                                    'import sys; print("python%d.%d" % sys.version_info[:2])']).decode().strip()


def installed():
    found = {}
    for path in sorted(ENV.glob(f'lib/{interpreter_dir()}/site-packages/*.*-info')):
        if not path.name.endswith(('.dist-info', '.egg-info')):
            continue
        name, version = parse(path.name)
        # Only the editable task repository counts as local; conda-forge packages also record a
        # file: URL (their build directory) and must still be aligned.
        direct = path/'direct_url.json'
        editable = direct.exists() and json.loads(direct.read_text()).get('dir_info', {}).get('editable', False)
        found[key(name)] = {'name': name, 'version': version, 'path': path, 'local': editable}
    return found


def is_pytest_plugin(path):
    entry_points = path/'entry_points.txt'
    return entry_points.exists() and '[pytest11]' in entry_points.read_text()


def pip(*args):
    env = dict(os.environ)
    env.pop('PIP_CONSTRAINT', None)  # exact versions take precedence over the build-tool ceilings
    return subprocess.run([str(ENV/'bin/python'), '-m', 'pip', *args, '--no-input', '--disable-pip-version-check'],
                          env=env).returncode


def main(reference):
    lines = Path(reference).read_text().split()
    x86_python = sorted({line.split('/')[0] for line in lines})
    local_python = [interpreter_dir()]
    stale = sorted(p.name for p in ENV.glob('lib/python*')
                   if not p.is_symlink() and p.name != local_python[0] and (p/'site-packages').is_dir())
    if x86_python != local_python:
        report = {'reference': reference, 'skipped': f'x86 image has {x86_python}, testbed interpreter is {local_python}'}
        Path('/opt/x86-alignment.json').write_text(json.dumps(report, indent=2) + '\n')
        print('x86 alignment:', json.dumps(report))
        return
    want = {}
    for line in lines:
        name, version = parse(line)
        if version and '+' not in version and key(name) not in SKIP:
            want[key(name)] = (name, version)
    have = installed()
    install = sorted(f'{name}=={version}' for k, (name, version) in want.items()
                     if not (k in have and (have[k]['local'] or have[k]['version'] == version)))
    remove = sorted(have[k]['name'] for k in set(have) - set(want)
                    if not have[k]['local'] and is_pytest_plugin(have[k]['path']))
    failed = []
    if remove:
        pip('uninstall', '-y', *remove)
    if install and pip('install', '--no-deps', *install):
        failed = [spec for spec in install if pip('install', '--no-deps', spec)]
    have = installed()
    mismatched = sorted(f'{name}=={version}' for k, (name, version) in want.items()
                        if k not in have or not (have[k]['local'] or have[k]['version'] == version))
    report = {'reference': reference, 'installed': install, 'removed_pytest_plugins': remove,
              'failed': failed, 'still_mismatched': mismatched, 'ignored_stale_site_packages': stale}
    Path('/opt/x86-alignment.json').write_text(json.dumps(report, indent=2) + '\n')
    print('x86 alignment:', json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}))
    for spec in mismatched:
        print('  not aligned:', spec)


if __name__ == '__main__':
    main(sys.argv[1])
