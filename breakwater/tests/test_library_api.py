"""What a custom launcher and task code may import from Breakwater (SPEC.md, "Custom launchers", "Task code"). A name
renamed or removed here breaks launchers outside the framework, so this list changes only on purpose. No Docker or GPUs.
"""
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRAMEWORK)]

API = {
    'contract': ['FRAMEWORK_REVISION', 'CREDENTIAL_ENV', 'ContractError', 'RulesViolation', 'require', 'read_json',
                 'atomic_json', 'digest', 'inventory', 'seal', 'verify_seal', 'tests_digest', 'overlaps', 'regular',
                 'inspect_boundary', 'check_host_hooks'],
    'spec': ['DEFAULTS', 'SEALED_PIPELINE', 'HOME_MOUNT', 'read', 'resolve', 'load', 'bindings', 'unsupported',
             'recording', 'authoring_plan', 'plan_text', 'model_access_env', 'reserved_names'],
    'task_api': ['Void', 'Evaluation', 'Scoring', 'load', 'load_hook', 'facts', 'score', 'review_views', 'write_json',
                 'jsonl', 'worker_environment'],
    'host_hooks': ['HostPlan', 'Endpoint', 'FromInput', 'FromSecret', 'Text', 'TaskHost', 'Operator'],
    'plan': ['static_endpoints', 'check_health', 'by_role', 'rejects', 'ranges', 'launcher_host'],
    # Evaluation-stage helpers that task code imports (kill_workers, install_tree and InvalidSubmission documented).
    'workers': ['kill_workers', 'install_tree', 'InvalidSubmission', 'worker_prefix', 'clear', 'wipe_worker_files',
                'apptainer'],
    # The Docker launcher's helpers, for a launcher of its own on Docker.
    'run': ['load_config', 'image', 'inspection', 'mount', 'docker', 'gpu_devices', 'check_gpus', 'boundary_probe',
            'observer', 'captured_agent', 'no_lingering_processes', 'runtime_check', 'stop_observer', 'snapshot',
            'service_gateway', 'check_verifier_image'],
}
APPTAINER = {
    'agent_stage': ['AuthoringStage', 'image_environment'],
    'evaluation_stage': ['EvaluationStage'],
    'run': ['verify', 'seed', 'file_sha256', 'read_only', 'SCRATCH', 'PROXY_ENV'],
}


def load(path, name):
    location = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(location)
    location.loader.exec_module(module)
    return module


class LibraryApiTests(unittest.TestCase):
    def test_every_documented_name_exists(self):
        for module, names in API.items():
            loaded = load(FRAMEWORK / f'{module}.py', f'api_{module}')
            for name in names:
                with self.subTest(module=module, name=name):
                    self.assertTrue(hasattr(loaded, name))
        sys.path.insert(0, str(FRAMEWORK / 'apptainer'))
        try:
            for module, names in APPTAINER.items():
                loaded = load(FRAMEWORK / 'apptainer' / f'{module}.py', f'api_apptainer_{module}')
                for name in names:
                    with self.subTest(module='apptainer/' + module, name=name):
                        self.assertTrue(hasattr(loaded, name))
        finally:
            sys.path.remove(str(FRAMEWORK / 'apptainer'))

    def test_the_apptainer_verifier_takes_a_verification_task(self):
        sys.path.insert(0, str(FRAMEWORK / 'apptainer'))
        try:
            launcher = load(FRAMEWORK / 'apptainer' / 'run.py', 'api_apptainer_run_signature')
        finally:
            sys.path.remove(str(FRAMEWORK / 'apptainer'))
        parameters = inspect.signature(launcher.verify).parameters
        self.assertEqual(sorted(parameters), sorted([
            'run', 'task', 'config', 'contract', 'image', 'image_env', 'bindings', 'judge_dir', 'judge_key', 'scratch',
            'identity', 'stages', 'seals', 'authority', 'reports', 'logs', 'extra_roots', 'extra', 'home',
            'judge_concurrency']))
        self.assertTrue(all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()))


if __name__ == '__main__':
    unittest.main()
