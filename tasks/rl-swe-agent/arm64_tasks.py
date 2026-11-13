#!/usr/bin/env python3
"""Audit, plan, check, build and stage native ARM64 Harbor task images; never use x86 emulation.

Covers the 728 OpenThoughts-Agent-v1-RL tasks (655 training, 73 development) and the 500 SWE-bench
Verified tasks. Audit, check, build and stage use only the standard library. Planning additionally
needs the pinned SWE-bench recipe generator in arm64/requirements.txt.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
HARBOR_REV = '86723674f04e4209ac479d0fb75d9d9f44b4377e'
HARNESS_REV = 'c7c22a916c9215e709722bc5ab18df4062dc6248'
PLATFORM = 'linux/arm64'
# Conda packages with no linux-aarch64 build (CDAT is unmaintained), by SWE-bench repository: dropped from
# the recipe's environment file and from the x86 alignment list. Tasks whose graded tests import them
# cannot pass and are excluded at evaluation (evaluation/swe-exclusions.json).
NO_ARM64_BUILD = {'pydata/xarray': ('cdms2', 'cdtime')}
# Compiled packages the x86 image got from conda but today's aarch64 solve leaves out: the alignment would
# pip-install a manylinux wheel that loads the system libstdc++ first (pyarrow), after which conda's
# libeccodes (GLIBCXX_3.4.32) cannot load. Requested from conda at the x86 version instead.
CONDA_FROM_X86 = {'pydata/xarray': ('pyarrow',)}
FROM = re.compile(r'^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+\S+)?\s*$', re.M | re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes(path):
    return {str(f.relative_to(path)): sha(f.read_bytes())
            for f in sorted(path.rglob('*')) if f.is_file()}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def verify_checkout(path, revision):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError(f'Expected {revision} at {path}; found {actual}')
    subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--', '.'], check=True)
    untracked = subprocess.check_output(['git', '-C', str(path), 'ls-files', '--others', '--exclude-standard', '.'])
    if untracked.strip():
        raise ValueError(f'Untracked source files at {path}')


def swe_tasks(path):
    verify_checkout(path, HARBOR_REV)
    tasks = sorted(p.parent for p in path.glob('*/task.toml'))
    if len(tasks) != 500:
        raise ValueError(f'Expected all 500 SWE-bench Verified tasks, found {len(tasks)}')
    return tasks


def arm_candidate(dockerfile):
    matches = FROM.findall(dockerfile)
    if len(matches) != 1 or not matches[0].startswith('swebench/sweb.eval.x86_64.'):
        raise ValueError('Unexpected pinned SWE Dockerfile; review before conversion')
    return matches[0].replace('sweb.eval.x86_64.', 'sweb.eval.arm64.', 1)


def registry_arm64(image):
    """Inspect Docker Hub manifests/configs without pulling layers or using Docker."""
    repository, tag = image.rsplit(':', 1)
    try:
        query = urlencode({'service': 'registry.docker.io', 'scope': f'repository:{repository}:pull'})
        with urlopen('https://auth.docker.io/token?' + query, timeout=30) as response:
            token = json.load(response)['token']
        headers = {'Authorization': 'Bearer ' + token, 'Accept': ', '.join([
            'application/vnd.oci.image.index.v1+json',
            'application/vnd.docker.distribution.manifest.list.v2+json',
            'application/vnd.oci.image.manifest.v1+json',
            'application/vnd.docker.distribution.manifest.v2+json'])}
        def get(kind, reference):
            request = Request(f'https://registry-1.docker.io/v2/{repository}/{kind}/{reference}', headers=headers)
            with urlopen(request, timeout=30) as response:
                payload = response.read()
                return json.loads(payload), response.headers.get('Docker-Content-Digest', 'sha256:' + sha(payload))
        manifest, digest = get('manifests', tag)
        if 'manifests' in manifest:
            candidates = [m for m in manifest['manifests']
                          if m.get('platform', {}).get('os') == 'linux'
                          and m.get('platform', {}).get('architecture') == 'arm64']
            if not candidates:
                return {'image': image, 'status': 'no_arm64_manifest'}
            digest = candidates[0]['digest']
            manifest, _ = get('manifests', digest)
        config, _ = get('blobs', manifest['config']['digest'])
        platform = config.get('os', '') + '/' + config.get('architecture', '')
        return {'image': image, 'status': 'available' if platform == PLATFORM else 'wrong_platform',
                'platform': platform, 'digest': digest}
    except HTTPError as error:
        return {'image': image, 'status': 'missing' if error.code == 404 else 'registry_error',
                'error': f'HTTP {error.code}'}
    except Exception as error:
        return {'image': image, 'status': 'registry_error', 'error': str(error)}


def audit(a):
    tasks = swe_tasks(a.swe_tasks)
    candidates = {task.name: arm_candidate((task/'environment/Dockerfile').read_text()) for task in tasks}
    report = {'harbor_revision': HARBOR_REV, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'swe_tasks': len(tasks), 'x86_swe_dockerfiles': len(candidates)}
    if a.registry:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            report['registry'] = dict(zip(candidates, pool.map(registry_arm64, candidates.values())))
    if a.training_data:
        training = sorted((a.training_data/'tasks').glob('*/environment/Dockerfile'))
        report['training_tasks'] = len(training)
        report['training_from'] = sorted({tuple(FROM.findall(p.read_text())) for p in training})
    save(a.output, report)
    print(f'Audit written to {a.output}')


def legacy_environment(script, minor, pins=()):
    """Use native source Python only where upstream requests unavailable legacy minors.

    pins (name==version from the published x86 image) stand in for conda's binary resolution;
    unconstrained, pip on these interpreters picks sdists of current releases."""
    constrain = ''
    if pins:
        constrain = ' -c /tmp/x86-pins.txt'
    pattern = re.compile(r'^conda create -n testbed python=' + re.escape(minor) + r'\s*(.*?)\s+-y$', re.M)
    def replace(match):
        packages = match[1].strip()
        return ('conda create -n testbed --no-default-packages -y\n'
                f'cp -a /opt/legacy-python/{minor}/. /opt/miniconda3/envs/testbed/\n'
                'conda activate testbed\n'
                + ("cat > /tmp/x86-pins.txt <<'PINS'\n" + '\n'.join(pins) + '\nPINS\n' if pins else '')
                # Isolated sdist builds (numpy) see only PIP_CONSTRAINT, not -c files.
                + "echo 'Cython<3' > /tmp/legacy-build-constraints.txt\n"
                'export PIP_CONSTRAINT="$PIP_CONSTRAINT /tmp/legacy-build-constraints.txt"\n'
                # The source Python ships pip 18, which predates manylinux2014 and PEP 600 tags and so
                # rejects every current aarch64 wheel (bcrypt then needs Rust). Requires-Python picks
                # the newest pip each minor supports.
                "python -m pip install -U 'pip<22'\n"
                # matplotlib sdists (no aarch64 wheel for these minors) otherwise download FreeType 2.6.1
                # from SourceForge at build time; build against the base image's libfreetype6-dev.
                + ("printf '[libs]\\nsystem_freetype = true\\n' > /tmp/mplsetup.cfg\n"
                   'export MPLSETUPCFG=/tmp/mplsetup.cfg\n' if 'matplotlib' in script else '')
                # Conda shipped binary numpy; from source, numpy needs Cython<3 and must precede
                # packages whose setup_requires would otherwise fetch the latest numpy sdist.
                # cppy: setup_requires of kiwisolver sdists, fetched by easy_install (no constraints) otherwise.
                + f"python -m pip install{constrain} 'Cython<3' 'cppy<1.2'\n"
                + (f'python -m pip install{constrain} numpy\n' if re.search(r'(^|\s)numpy(\s|$)', packages) else '')
                + (f'python -m pip install{constrain} ' + packages + '\n' if packages else '')
                + f'python -c "import sys; assert sys.version_info[:2] == ({minor.replace(".", ", ")})"')
    result, count = pattern.subn(replace, script)
    if count != 1:
        raise ValueError(f'Cannot safely convert Python {minor} environment; preserve and inspect the recipe')
    return result


def pin_python(script, minor, version):
    """python=3.9 -> python=3.9.20 in the recipe's conda create/install lines.

    Conda resolves python=X.Y to the newest patch release, and later patches change stdlib behaviour
    the published x86 images predate (3.9.23+ html.parser fails Django's strip_tags tests)."""
    if not version.startswith(minor + '.'):
        raise ValueError(f'x86 interpreter {version} is not Python {minor}')
    result, count = re.subn(r'(?m)^(.*\bconda (?:create|install)\b.*\bpython=)' + re.escape(minor) + r'(?=\s)',
                            lambda m: m[1] + version, script)
    if not count:
        raise ValueError(f'No conda python={minor} request to pin')
    # An environment.yml without a python entry lets `conda env update` re-solve the interpreter too;
    # for xarray 2905-4966 that solve overflows libsolv ("Out of memory allocating 18446744071562067968*4 bytes").
    if 'conda env update' in result and not re.search(r'(?m)^\s*- python\b', result):
        result = re.sub(r'(?m)^dependencies:\n', f'dependencies:\n  - python={version}\n', result, count=1)
    return result


def parse_metadata_dir(entry):
    """'pytest-7.1.2.dist-info' or 'foo-1.0-py3.6.egg-info' -> (name, version or None)."""
    parts = re.sub(r'\.(dist|egg)-info$', '', Path(entry).name).split('-')
    return (parts[0], parts[1]) if len(parts) > 1 else (parts[0], None)


def replace_base(dockerfile, image):
    if len(FROM.findall(dockerfile)) != 1:
        raise ValueError('Expected exactly one Dockerfile FROM')
    return FROM.sub(f'FROM --platform={PLATFORM} {image}', dockerfile, count=1)


def plan(a):
    from dataset_contract import validate
    manifest = validate(a.training_data)
    tasks = swe_tasks(a.swe_tasks)
    verify_checkout(a.harness, HARNESS_REV)
    sys.path.insert(0, str(a.harness.resolve()))
    import swebench
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, USE_X86
    from swebench.harness.test_spec.test_spec import make_test_spec
    if swebench.__version__ != '4.0.4' or not Path(swebench.__file__).resolve().is_relative_to(a.harness.resolve()):
        raise ValueError('Recipe generator must come from the pinned SWE-bench checkout')
    # Upstream recipe generation fetches requirements from immutable repo commits.
    import requests
    original_get = requests.get
    def bounded_get(*args, **kwargs):
        kwargs.setdefault('timeout', 45)
        return original_get(*args, **kwargs)
    requests.get = bounded_get
    a.output.mkdir(parents=True, exist_ok=False)
    nodes, images, errors = {}, {}, {}
    registry = json.loads(a.registry_audit.read_text()).get('registry', {}) if a.registry_audit else {}
    def node(kind, dockerfile, files=None, dependencies=(), executable=()):
        files = dict(files or {})
        files['.dockerignore'] = files.get('.dockerignore', b'') + b'\nbuild.log\n'
        fingerprint = sha(json.dumps([dockerfile, {k: sha(v) for k, v in files.items()}, sorted(executable)], sort_keys=True).encode())[:24]
        image = f'rl-swe-agent-arm64-{kind}:{fingerprint}'
        if image not in nodes:
            context = a.output/'contexts'/f'{kind}-{fingerprint}'
            context.mkdir(parents=True)
            (context/'Dockerfile').write_text(dockerfile)
            for name, data in files.items():
                target = context/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if name in executable:
                    target.chmod(0o755)
            nodes[image] = {'image': image, 'context': str(context.relative_to(a.output)),
                            'dependencies': list(dependencies), 'files': hashes(context),
                            'executable': sorted(executable)}
        return image
    # All 728 environments of the release: the 655 training tasks and the 73 development tasks.
    training_names = [t['instance_id'] for t in manifest['tasks']]
    for name in sorted(training_names):
        task = a.training_data/'tasks'/name
        context = task/'environment'
        dockerfile = (context/'Dockerfile').read_text()
        if FROM.findall(dockerfile) != ['ubuntu:24.04']:
            raise ValueError(f'{name}: review this nonstandard training base before conversion')
        files = {str(p.relative_to(context)): p.read_bytes() for p in context.rglob('*')
                 if p.is_file() and p.name != 'Dockerfile'}
        executable = [name for name in files if (context/name).stat().st_mode & 0o111]
        image = node('training', replace_base(dockerfile, 'ubuntu:24.04'), files, executable=executable)
        images[f'rl-swe-agent-dev/{name}'] = {'image': image, 'source_hashes': hashes(task),
            'original_tag': tomllib.loads((task/'task.toml').read_text()).get('environment', {}).get('docker_image'),
            'evaluation': name in manifest['dev_ids'], 'strategy': 'native Ubuntu build'}
    # Distribution metadata of each published x86 image (see arm64/align_versions.py).
    x86_site_packages = json.loads((ROOT/'arm64/x86-site-packages.json').read_text())
    # Testbed interpreter of each published x86 image, probed once per environment group.
    x86_python = json.loads((ROOT/'arm64/x86-python.json').read_text())
    python_pins = {}
    legacy_counts = {}
    legacy_pins = {}
    forced_x86 = []
    for task in tasks:
        key = 'swebench-verified/' + task.name
        try:
            row = json.loads((task/'tests/config.json').read_text())
            if row['instance_id'] != task.name or not re.fullmatch(r'[0-9a-f]{40}', row['base_commit']):
                raise ValueError('Task ID/base commit mismatch')
            spec = make_test_spec(row, namespace=None)
            spec.arch = 'arm64'  # Deliberately override upstream USE_X86, including all 86 affected tasks.
            if row['instance_id'] in USE_X86:
                forced_x86.append(row['instance_id'])
            minor = MAP_REPO_VERSION_TO_SPECS[row['repo']][row['version']]['python']
            dropped = NO_ARM64_BUILD.get(row['repo'], ())
            x86_entries = [e for e in x86_site_packages[task.name] if parse_metadata_dir(e)[0].lower() not in dropped]
            candidate = registry.get(task.name, {})
            strategy = 'native source build'
            dependencies = []
            if a.prefer_published and candidate.get('status') == 'available' and candidate.get('platform') == PLATFORM:
                base = candidate['image'].split(':')[0] + '@' + candidate['digest']
                strategy = 'published ARM64 digest'
            else:
                basefile = spec.base_dockerfile
                # Headers for sdists without ARM64 wheels (e.g. lxml under pikepdf, pylibmc).
                basefile += '\nRUN apt-get update && apt-get install -y --no-install-recommends gfortran libopenblas-dev liblapack-dev pkg-config libjpeg-dev libpng-dev libfreetype6-dev libxml2-dev libxslt1-dev libmemcached-dev && rm -rf /var/lib/apt/lists/*\n'
                # Environment, instance and verifier installs all resolve build tools the same way.
                basefiles = {'build-constraints.txt': (ROOT/'arm64/build-constraints.txt').read_bytes()}
                basefile += ('COPY build-constraints.txt /opt/swebench-build-constraints.txt\n'
                             'ENV PIP_CONSTRAINT=/opt/swebench-build-constraints.txt\n')
                # Conda ignores PIP_CONSTRAINT (it installs the newest pip, setuptools_scm and sometimes
                # Cython); bring what it installed within the constraints. vcs-versioning comes only with
                # setuptools_scm 9+ and is absent from every published x86 image.
                env_script = spec.setup_env_script + (
                    "\nsource /opt/miniconda3/bin/activate\nconda activate testbed\n"
                    "for req in $(grep -v '^#' /opt/swebench-build-constraints.txt); do\n"
                    "  if python -m pip show \"${req%%[<>=]*}\" >/dev/null 2>&1; then python -m pip install \"$req\"; fi\n"
                    "done\n"
                    "python -m pip uninstall -y vcs-versioning || true\n"
                    "if python -c 'import Cython' 2>/dev/null; then python -m pip install 'Cython<3'; fi\n")
                for package in dropped:
                    env_script = re.sub(rf'(?m)^\s*- {re.escape(package)}\s*\n', '', env_script)
                for package in CONDA_FROM_X86.get(row['repo'], ()):
                    version = dict((n.lower(), v) for n, v in map(parse_metadata_dir, x86_entries)).get(package)
                    if version and 'conda env update' in env_script:
                        env_script = re.sub(r'(?m)^dependencies:\n', f'dependencies:\n  - {package}={version}\n',
                                            env_script, count=1)
                if minor in ('3.5', '3.6'):
                    basefiles['legacy_python.sh'] = (ROOT/'arm64/legacy_python.sh').read_bytes()
                    basefile += f'COPY legacy_python.sh /opt/legacy_python.sh\nRUN bash /opt/legacy_python.sh {minor}\n'
                    # Environments are shared, so take pins from the first task built on this recipe.
                    x86 = [e for e in x86_entries if e.startswith(f'python{minor}/')]
                    pins = sorted({f'{name}=={version}' for name, version in map(parse_metadata_dir, x86)
                                   if version and '+' not in version and name.lower() != 'pip'})
                    env_script = legacy_environment(env_script, minor, legacy_pins.setdefault(spec.setup_env_script, pins))
                    legacy_counts[minor] = legacy_counts.get(minor, 0) + 1
                else:
                    env_script = pin_python(env_script, minor, python_pins.setdefault(spec.setup_env_script, x86_python[task.name]))
                base = node('base', basefile, basefiles)
                envfile = replace_base(spec.env_dockerfile, base)
                environment = node('env', envfile, {'setup_env.sh': env_script.encode()}, [base])
                instancefile = replace_base(spec.instance_dockerfile, environment) + (
                    '\nCOPY align_versions.py x86-site-packages.txt /root/\n'
                    'RUN /opt/miniconda3/bin/python /root/align_versions.py /root/x86-site-packages.txt\n')
                if row['repo'] == 'matplotlib/matplotlib':
                    # The published x86 images carry jammy's release tzdata; 2026 updates changed far-future
                    # Canada/Pacific offsets that test_dates compares (2040-01-01: -07:00 instead of -08:00).
                    instancefile += ('RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends '
                                     'tzdata=2022a-0ubuntu1 && apt-mark hold tzdata && rm -rf /var/lib/apt/lists/*\n')
                # Verifiers of setuptools_scm repositories (pytest, astropy, matplotlib) reinstall the task
                # repository. With the tags removed below it would version as 0.1.dev (pytest then rejects its
                # own minversion), so pin the version the tagged x86 checkout reports.
                for name, version in sorted(map(parse_metadata_dir, x86_entries)):
                    if version and '+g' in version:
                        instancefile += f'ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{re.sub(r"[-_.]+", "_", name).upper()}={version}\n'
                # Retain the base/setup commits required by Harbor's verifier, remove future refs/reflogs.
                repo_script = spec.install_repo_script + '\ngit checkout --detach\ngit for-each-ref --format="delete %(refname)" refs/heads refs/remotes refs/tags | git update-ref --stdin\ngit reflog expire --expire=now --all\ngit gc --prune=now\n'
                base = node('instance', instancefile, {
                    'setup_repo.sh': repo_script.encode(),
                    'align_versions.py': (ROOT/'arm64/align_versions.py').read_bytes(),
                    'x86-site-packages.txt': '\n'.join(x86_entries).encode() + b'\n'}, [environment])
                dependencies = [base]
            wrapper = replace_base((task/'environment/Dockerfile').read_text(), base)
            # Verify the intended checkout and interpreter in both downloaded and source-built images.
            wrapper += ('\nRUN test "$(uname -m)" = aarch64 && '
                        f'git -C /testbed merge-base --is-ancestor {row["base_commit"]} HEAD && '
                        '/opt/miniconda3/envs/testbed/bin/python -c '
                        + shlex.quote(f'import platform,sys; assert platform.machine() == "aarch64"; assert sys.version_info[:2] == ({minor.replace(".", ", ")})') + '\n')
            wrapper += ('RUN cd /testbed && '
                        f'(test "$(git rev-parse HEAD)" = {row["base_commit"]} || '
                        f'test "$(git rev-parse HEAD^)" = {row["base_commit"]}) && '
                        'git checkout --detach && '
                        'git for-each-ref --format="delete %(refname)" refs/heads refs/remotes refs/tags | '
                        'git update-ref --stdin\n'
                        'RUN cd /testbed && git reflog expire --expire=now --all && git gc --prune=now\n')
            files = {str(p.relative_to(task/'environment')): p.read_bytes()
                     for p in (task/'environment').rglob('*') if p.is_file() and p.name != 'Dockerfile'}
            executable = [name for name in files if (task/'environment'/name).stat().st_mode & 0o111]
            image = node('swe', wrapper, files, dependencies, executable=executable)
            images[key] = {'image': image, 'source_hashes': hashes(task), 'original_tag': None,
                           'evaluation': True, 'strategy': strategy, 'python': minor,
                           'base_commit': row['base_commit'], 'dropped_without_arm64_build': list(dropped)}
        except Exception as error:
            errors[key] = f'{type(error).__name__}: {error}'
    requests.get = original_get
    result = {'platform': PLATFORM, 'harbor_revision': HARBOR_REV, 'harness_revision': HARNESS_REV,
              'nodes': nodes, 'images': images, 'errors': errors,
              'legacy_python_tasks': legacy_counts, 'overridden_upstream_x86_tasks': forced_x86,
              'runtime_validated': False}
    save(a.output/'plan.json', result)
    print(f'Planned {len(images)} task images in {len(nodes)} build nodes; {len(errors)} planning errors. See {a.output}/plan.json')
    return 1 if errors else 0


def inspect_arm64(image, env, timeout=60):
    result = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{json .}}'],
                            capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode:
        return None
    data = json.loads(result.stdout)
    if data.get('Os') != 'linux' or data.get('Architecture') != 'arm64':
        raise ValueError(f'{image}: expected linux/arm64, found {data.get("Os")}/{data.get("Architecture")}')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', data.get('Id', '')):
        raise ValueError(f'{image}: invalid image ID')
    return data['Id']


def check_plan(a):
    """Offline verification of every generated context and shell script."""
    path = a.plan.resolve()
    data = json.loads(path.read_text())
    if data['platform'] != PLATFORM or data['errors']:
        raise ValueError('Incomplete/non-ARM64 build plan')
    nodes = data['nodes']
    scripts = 0
    for image, node in nodes.items():
        context = path.parent/node['context']
        if {k: v for k, v in hashes(context).items() if k != 'build.log'} != node['files']:
            raise ValueError(f'{image}: build context changed')
        dockerfile = (context/'Dockerfile').read_text()
        if len(FROM.findall(dockerfile)) != 1 or not re.search(r'^FROM --platform=linux/arm64(?:/v8)? ', dockerfile, re.M):
            raise ValueError(f'{image}: missing explicit ARM64 FROM')
        if 'sweb.eval.x86_64.' in dockerfile:
            raise ValueError(f'{image}: x86 task base remains')
        if not set(node['dependencies']) <= set(nodes):
            raise ValueError(f'{image}: missing dependency')
        # Original task seeds may deliberately contain broken shell code to repair.
        # Syntax-check only the generated build scripts, never the benchmark input.
        for script in [context/name for name in ('setup_env.sh', 'setup_repo.sh', 'legacy_python.sh')
                       if (context/name).is_file()]:
            subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)
            scripts += 1
    for key, task in data['images'].items():
        if task['image'] not in nodes:
            raise ValueError(f'{key}: missing task build node')
    report = {'task_images': len(data['images']), 'build_nodes': len(nodes),
              'shell_scripts_checked': scripts, 'planning_errors': len(data['errors']),
              'legacy_python_tasks': data['legacy_python_tasks'],
              'upstream_x86_overrides': len(data['overridden_upstream_x86_tasks']),
              'platform': PLATFORM, 'static_checks_passed': True, 'runtime_validated': False}
    save(path.parent/'static-checks.json', report)
    print(json.dumps(report, indent=2))


def build(a):
    plan_path = a.plan.resolve()
    root = plan_path.parent
    data = json.loads(plan_path.read_text())
    if data['platform'] != PLATFORM or data['errors']:
        raise ValueError('Fix plan errors before building; no task may be silently dropped')
    env = dict(os.environ)
    if a.docker_host:
        env.pop('DOCKER_CONTEXT', None)
        env['DOCKER_HOST'] = a.docker_host
    info = subprocess.check_output(['docker', 'info', '--format', '{{json .}}'], env=env, text=True, timeout=30)
    info = json.loads(info)
    if info.get('OSType') != 'linux' or info.get('Architecture') not in ('aarch64', 'arm64'):
        raise ValueError('Build on a native Linux ARM64 daemon; x86 emulation is not an ARM64 validation')
    receipt = {'platform': PLATFORM, 'plan_sha256': sha(plan_path.read_bytes()), 'built': {}, 'failed': {}}
    pending = dict(data['nodes'])
    def build_node(item):
        image, node = item
        context = root/node['context']
        try:
            if {k: v for k, v in hashes(context).items() if k != 'build.log'} != node['files']:
                raise ValueError('Build context changed after planning')
            if sorted(str(p.relative_to(context)) for p in context.rglob('*')
                      if p.is_file() and p.name != 'build.log' and p.stat().st_mode & 0o111) != node.get('executable', []):
                raise ValueError('Build context executable permissions changed after planning')
            identity = inspect_arm64(image, env)
            if not identity:
                with (context/'build.log').open('w') as stream:
                    subprocess.run(['docker', 'build', '--platform', PLATFORM, '-t', image, str(context)],
                                   check=True, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=a.timeout)
                identity = inspect_arm64(image, env)
            if not identity:
                raise ValueError('Built image missing')
            return image, identity, None
        except Exception as error:
            return image, None, f'{type(error).__name__}: {error}'
    while pending:
        ready = [(image, node) for image, node in pending.items()
                 if all(dep in receipt['built'] for dep in node['dependencies'])]
        if not ready:
            receipt['failed'].update({image: 'Dependency failed or graph incomplete' for image in pending})
            break
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for image, identity, error in pool.map(build_node, ready):
                pending.pop(image)
                (receipt['failed'] if error else receipt['built'])[image] = error or identity
        save(root/'build-receipt.json', receipt)
    smoke, cache = {}, {}
    for key, task in data['images'].items():
        identity = receipt['built'].get(task['image'])
        if not identity:
            continue
        try:
            command = ['docker', 'run', '--rm', '--platform', PLATFORM, '--network', 'none',
                       '--entrypoint', 'sh', identity, '-ec', 'test "$(uname -m)" = aarch64; '
                       + ('/opt/miniconda3/envs/testbed/bin/python' if key.startswith('swebench-') else 'python3')
                       + ' -c \'import platform; assert platform.machine() == "aarch64"\'']
            subprocess.run(command, env=env, timeout=120, check=True, capture_output=True, text=True)
            smoke[key] = 'passed'
            if task['evaluation']:
                cache[key] = {'original_tag': task['original_tag'], 'image_id': identity,
                              'source_hashes': task['source_hashes'], 'platform': PLATFORM}
        except Exception as error:
            smoke[key] = str(error)
    complete = not receipt['failed'] and len(smoke) == len(data['images']) and all(v == 'passed' for v in smoke.values())
    receipt.update(task_smoke=smoke, complete=complete)
    save(root/'build-receipt.json', receipt)
    # Only publish a reusable manifest when EVERY requested image is native and starts successfully.
    if complete:
        save(root/'image-cache.json', cache)
    else:
        (root/'image-cache.json').unlink(missing_ok=True)
    print(f'ARM64 builds complete={complete}; see {root}/build-receipt.json')
    return 0 if complete else 1


def stage(a):
    """Copy task definitions with pinned native images; never change the original dataset."""
    plan_path = a.plan.resolve()
    data = json.loads(plan_path.read_text())
    receipt = json.loads((plan_path.parent/'build-receipt.json').read_text())
    if not receipt.get('complete') or receipt['plan_sha256'] != sha(plan_path.read_bytes()):
        raise ValueError('A complete build/smoke receipt for this exact plan is required')
    sys.path.insert(0, str(ROOT/'evaluation'))
    from suites import pin_image
    a.output.mkdir(parents=True, exist_ok=False)
    staged = {}
    for key, task in data['images'].items():
        suite, name = key.split('/', 1)
        source = (a.training_data/'tasks' if suite == 'rl-swe-agent-dev' else a.swe_tasks)/name
        if hashes(source) != task['source_hashes']:
            raise ValueError(f'{key}: original task changed after planning')
        if receipt['task_smoke'].get(key) != 'passed':
            raise ValueError(f'{key}: missing native runtime smoke')
        staged_suite = ('rl-swe-agent-train' if suite == 'rl-swe-agent-dev'
                        and not task.get('evaluation', False) else suite)
        target = a.output/'tasks'/staged_suite/name
        shutil.copytree(source, target)
        config = target/'task.toml'
        identity = receipt['built'][task['image']]
        config.write_text(pin_image(config.read_text(), identity))
        staged[key] = {'image_id': identity, 'platform': PLATFORM,
                       'source_hashes': task['source_hashes'], 'prepared_hashes': hashes(target)}
    save(a.output/'staged-images.json', staged)
    print(f'Staged {len(staged)} tasks under {a.output}/tasks; original instructions/tests unchanged')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    audit_p = commands.add_parser('audit')
    audit_p.add_argument('--swe-tasks', type=Path, required=True)
    audit_p.add_argument('--training-data', type=Path)
    audit_p.add_argument('--output', type=Path, required=True)
    audit_p.add_argument('--registry', action='store_true')
    audit_p.add_argument('--workers', type=int, default=4)
    audit_p.set_defaults(func=audit)
    plan_p = commands.add_parser('plan')
    plan_p.add_argument('--swe-tasks', type=Path, required=True)
    plan_p.add_argument('--training-data', type=Path, default=ROOT/'data')
    plan_p.add_argument('--harness', type=Path, required=True)
    plan_p.add_argument('--output', type=Path, required=True)
    plan_p.add_argument('--registry-audit', type=Path)
    plan_p.add_argument('--prefer-published', action='store_true', help='Use audited ARM64 digests where available; default rebuilds from source')
    plan_p.set_defaults(func=plan)
    check_p = commands.add_parser('check')
    check_p.add_argument('--plan', type=Path, required=True)
    check_p.set_defaults(func=check_plan)
    build_p = commands.add_parser('build')
    build_p.add_argument('--plan', type=Path, required=True)
    build_p.add_argument('--docker-host')
    build_p.add_argument('--workers', type=int, default=2)
    build_p.add_argument('--timeout', type=int, default=7200)
    build_p.set_defaults(func=build)
    stage_p = commands.add_parser('stage')
    stage_p.add_argument('--plan', type=Path, required=True)
    stage_p.add_argument('--training-data', type=Path, default=ROOT/'data')
    stage_p.add_argument('--swe-tasks', type=Path, required=True)
    stage_p.add_argument('--output', type=Path, required=True)
    stage_p.set_defaults(func=stage)
    a = p.parse_args()
    if hasattr(a, 'workers') and a.workers < 1:
        p.error('--workers must be positive')
    if hasattr(a, 'timeout') and a.timeout < 1:
        p.error('--timeout must be positive')
    return a.func(a) or 0


if __name__ == '__main__':
    raise SystemExit(main())
