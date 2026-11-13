"""CPU-only regression tests for native image conversion and architecture gates.

Run without installing Torch/pytest:
python3 -m unittest discover -s tasks/rl-swe-agent/tests/unit -p test_arm64_tasks.py -v
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import arm64_tasks as arm

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker = load('arm_worker', ROOT/'worker/start.py')
sys.path.insert(0, str(ROOT/'evaluation'))
import suites as ev
IMAGE = 'sha256:' + 'a'*64


class Arm64Tests(unittest.TestCase):
    def test_candidate_preserves_instance_and_only_changes_arch(self):
        self.assertEqual(arm.arm_candidate('# FROM ignored\nFROM swebench/sweb.eval.x86_64.django_1776_django-123:latest\n'),
                         'swebench/sweb.eval.arm64.django_1776_django-123:latest')
        for dockerfile in ['FROM ubuntu:24.04\n', 'FROM a\nFROM b\n']:
            with self.assertRaises(ValueError):
                arm.arm_candidate(dockerfile)

    def test_wrapper_replaces_exactly_one_base_and_preserves_commands(self):
        before = '# template\nFROM --platform=linux/amd64 old:tag\nWORKDIR /testbed\nRUN echo test\n'
        after = arm.replace_base(before, 'native@sha256:abc')
        self.assertIn('FROM --platform=linux/arm64 native@sha256:abc', after)
        self.assertEqual(before.split('WORKDIR')[1], after.split('WORKDIR')[1])

    def test_legacy_conversion_preserves_python_minor_and_dependencies(self):
        for minor in ('3.5', '3.6'):
            original = f'conda create -n testbed python={minor} numpy scipy -y\nconda activate testbed\npython -m pip install exact==1.2\n'
            result = arm.legacy_environment(original, minor)
            self.assertIn(f'/opt/legacy-python/{minor}/.', result)
            self.assertIn('python -m pip install numpy scipy', result)
            self.assertTrue(result.endswith('python -m pip install exact==1.2\n'))
            self.assertNotIn(f'conda create -n testbed python={minor}', result)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as f:
                f.write(result); f.flush()
                subprocess.run(['bash', '-n', f.name], check=True)
        with self.assertRaises(ValueError):
            arm.legacy_environment('conda env create --file environment.yml', '3.6')

    def test_environment_yml_update_keeps_the_pinned_interpreter(self):
        script = ("cat <<'EOF' > environment.yml\nname: testbed\ndependencies:\n  - numpy\nEOF\n"
                  "conda create -c conda-forge -n testbed python=3.10 -y\nconda env update -f environment.yml\n")
        pinned = arm.pin_python(script, '3.10', '3.10.15')
        self.assertIn('dependencies:\n  - python=3.10.15\n  - numpy\n', pinned)
        self.assertIn('-n testbed python=3.10.15 -y', pinned)
        plain = 'conda create -n testbed python=3.9 -y\n'
        self.assertEqual(arm.pin_python(plain, '3.9', '3.9.20'), 'conda create -n testbed python=3.9.20 -y\n')

    def test_image_metadata_must_be_native_even_if_tag_says_arm64(self):
        for architecture in ('arm64', 'amd64'):
            metadata = {'Os': 'linux', 'Architecture': architecture, 'Id': IMAGE}
            result = SimpleNamespace(returncode=0, stdout=json.dumps(metadata))
            with patch.object(arm.subprocess, 'run', return_value=result):
                if architecture == 'arm64':
                    self.assertEqual(arm.inspect_arm64('looks-arm64:latest', {}), IMAGE)
                else:
                    with self.assertRaisesRegex(ValueError, 'expected linux/arm64'):
                        arm.inspect_arm64('looks-arm64:latest', {})

    def test_worker_rejects_unstaged_and_wrong_arch_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            tasks = Path(temp); task = tasks/'one'; task.mkdir()
            config = task/'task.toml'; config.write_text('[agent]\ntimeout_sec=600\n')
            def answer(command, **kwargs):
                return json.dumps({'OSType': 'linux', 'Architecture': 'aarch64'}) if command[1] == 'info' else 'linux/amd64\n'
            with patch.object(worker.subprocess, 'check_output', side_effect=answer):
                with self.assertRaisesRegex(ValueError, 'pinned ARM64 task tree'):
                    worker.validate_arm64_tasks(tasks)
                config.write_text(f'[environment]\ndocker_image="{IMAGE}"\n')
                with self.assertRaisesRegex(ValueError, 'expected linux/arm64'):
                    worker.validate_arm64_tasks(tasks)

    def test_worker_checks_daemon_and_accepts_only_native_images(self):
        with tempfile.TemporaryDirectory() as temp:
            tasks = Path(temp); (tasks/'one').mkdir()
            (tasks/'one/task.toml').write_text(f'[environment]\ndocker_image="{IMAGE}"\n')
            for arch in ('aarch64', 'x86_64'):
                def answer(command, **kwargs):
                    return json.dumps({'OSType': 'linux', 'Architecture': arch}) if command[1] == 'info' else 'linux/arm64\n'
                with patch.object(worker.subprocess, 'check_output', side_effect=answer):
                    if arch == 'aarch64':worker.validate_arm64_tasks(tasks)
                    else:
                        with self.assertRaisesRegex(ValueError, 'native Linux ARM64'):
                            worker.validate_arm64_tasks(tasks)

    def test_build_refuses_x86_daemon_before_building_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'plan.json'
            path.write_text(json.dumps({'platform': 'linux/arm64', 'errors': {}}))
            info = json.dumps({'OSType': 'linux', 'Architecture': 'x86_64'})
            with patch.object(arm.subprocess, 'check_output', return_value=info), patch.object(arm.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'native Linux ARM64'):
                    arm.build(SimpleNamespace(plan=path, docker_host=None))
                run.assert_not_called()

    def test_failed_build_records_all_failures_and_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); context = root/'context'; context.mkdir()
            (context/'Dockerfile').write_text('FROM ubuntu:24.04\n')
            path = root/'plan.json'
            path.write_text(json.dumps({'platform': 'linux/arm64', 'errors': {}, 'nodes': {
                'one': {'context': 'context', 'files': arm.hashes(context), 'dependencies': []},
                'two': {'context': 'context', 'files': arm.hashes(context), 'dependencies': ['one']},
            }, 'images': {}}))
            info = json.dumps({'OSType': 'linux', 'Architecture': 'aarch64'})
            with patch.object(arm.subprocess, 'check_output', return_value=info), \
                 patch.object(arm, 'inspect_arm64', side_effect=RuntimeError('Missing native kernel')):
                self.assertEqual(arm.build(SimpleNamespace(plan=path, docker_host=None, workers=1, timeout=60)), 1)
            receipt = json.loads((root/'build-receipt.json').read_text())
            self.assertFalse(receipt['complete'])
            self.assertEqual(set(receipt['failed']), {'one', 'two'})
            self.assertFalse((root/'image-cache.json').exists())

    def test_pinning_actual_training_task_without_environment_section(self):
        original = 'version="1.0"\n[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=600\n'
        pinned = ev.tomllib.loads(ev.pin_image(original, IMAGE))
        self.assertEqual(pinned.pop('environment'), {'docker_image': IMAGE})
        self.assertEqual(pinned, ev.tomllib.loads(original))

    def test_successful_native_cache_is_removed_if_recipe_later_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); context = root/'context'; context.mkdir()
            (context/'Dockerfile').write_text('FROM ubuntu:24.04\n')
            expected_files = arm.hashes(context)
            (context/'build.log').write_text('Prior attempt log, excluded from recipe hashes')
            plan = root/'plan.json'
            plan.write_text(json.dumps({'platform': 'linux/arm64', 'errors': {}, 'nodes': {
                'native:one': {'context': 'context', 'files': expected_files, 'dependencies': []}},
                'images': {'rl-swe-agent-dev/one': {'image': 'native:one', 'evaluation': True,
                                                     'original_tag': None, 'source_hashes': {}}}}))
            info = json.dumps({'OSType': 'linux', 'Architecture': 'aarch64'})
            args = SimpleNamespace(plan=plan, docker_host=None, workers=1, timeout=60)
            with patch.object(arm.subprocess, 'check_output', return_value=info), \
                 patch.object(arm, 'inspect_arm64', return_value=IMAGE), \
                 patch.object(arm.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                self.assertEqual(arm.build(args), 0)
                cache = json.loads((root/'image-cache.json').read_text())
                self.assertEqual(cache['rl-swe-agent-dev/one']['platform'], 'linux/arm64')
                (context/'Dockerfile').write_text('FROM modified:unreviewed\n')
                self.assertEqual(arm.build(args), 1)
                self.assertFalse((root/'image-cache.json').exists())

    def test_evaluator_rejects_cached_x86_image_on_arm64(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; task = source/'one'; task.mkdir(parents=True)
            (task/'task.toml').write_text('[environment]\n')
            cache = root/'cache.json'
            cache.write_text(json.dumps({'suite/one': {'image_id': IMAGE, 'original_tag': None,
                                                      'source_hashes': ev.file_hashes(task)}}))
            output = root/'out'; output.mkdir()
            def run(command, **kwargs):
                if command[0] == 'git':
                    return SimpleNamespace(returncode=0, stdout='abc' if 'rev-parse' in command else '')
                return SimpleNamespace(returncode=0, stdout='linux/amd64' if command[-1] == '{{.Os}}/{{.Architecture}}' else IMAGE)
            with patch.object(ev, 'SUITES', {'suite': {'commit': 'abc', 'task_count': 1}}), \
                 patch.object(ev.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(ValueError, 'expected linux/arm64'):
                    ev.prepare_suites({'suite': source}, output, ev.Deadline(3600), {},
                                      image_manifest=cache, task_platform='linux/arm64')

    def test_stage_requires_complete_receipt_and_never_modifies_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'data/tasks/one'; source.mkdir(parents=True)
            (source/'task.toml').write_text('[agent]\ntimeout_sec=600\n')
            (source/'instruction.md').write_text('Unchanged problem')
            before = arm.hashes(source)
            plan = root/'plan.json'; key = 'rl-swe-agent-dev/one'
            plan.write_text(json.dumps({'images': {key: {'image': 'native:one', 'source_hashes': before}}}))
            receipt = {'complete': True, 'plan_sha256': arm.sha(plan.read_bytes()),
                       'built': {'native:one': IMAGE}, 'task_smoke': {key: 'passed'}}
            arm.save(root/'build-receipt.json', receipt)
            args = SimpleNamespace(plan=plan, output=root/'out', training_data=root/'data', swe_tasks=root/'swe')
            arm.stage(args)
            self.assertEqual(arm.hashes(source), before)
            self.assertEqual((args.output/'tasks/rl-swe-agent-train/one/instruction.md').read_text(), 'Unchanged problem')
            receipt['complete'] = False; arm.save(root/'build-receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'complete build/smoke receipt'):
                arm.stage(SimpleNamespace(**{**vars(args), 'output': root/'bad'}))


if __name__ == '__main__':
    unittest.main()
