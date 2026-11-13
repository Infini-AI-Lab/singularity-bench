"""CPU checks for the shared RL contract, real dataset schemas and reward boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
import train_driver

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import evaluate
import launch
import rewards
import sandbox
from verify_vendor import verify

spec = importlib.util.spec_from_file_location('rl_data', ROOT / 'prepare_data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class Tokenizer:
    def encode(self, text, **_):
        return text.split()


class RecipeTests(unittest.TestCase):
    def test_pinned_base_and_rl_architecture_are_llama_1b(self):
        base = json.loads((ROOT.parents[1] / 'data_sources.json').read_text())['base_model']
        self.assertEqual(base['repo'], 'meta-llama/Llama-3.2-1B')
        self.assertEqual(base['revision'], '4e20de362430cd3b72f300e6b0f18e50e7166e08')
        self.assertEqual(base['path'], 'Llama-3.2-1B')
        self.assertEqual([name for name in base['include'] if 'safetensors' in name], ['model.safetensors'])
        recipe = launch.load_recipe(ROOT / 'recipe.json')
        args = launch.train_args(recipe, Path('/input'), Path('/data'), Path('/output'))
        expected = {'--num-layers': '16', '--hidden-size': '2048', '--ffn-hidden-size': '8192',
                    '--num-attention-heads': '32', '--num-query-groups': '8', '--kv-channels': '64',
                    '--vocab-size': '128256', '--max-position-embeddings': '131072',
                    '--rotary-scaling-factor': '32.0'}
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.assertNotIn('--untie-embeddings-and-output-weights', args)

    def test_five_hour_budget_and_one_optimizer_update_per_rollout(self):
        recipe = launch.load_recipe(ROOT / 'recipe.json')
        args = launch.train_args(recipe, Path('/input'), Path('/data'), Path('/output'))
        value = lambda flag: args[args.index(flag) + 1]
        self.assertNotIn('steps', recipe)
        self.assertEqual(recipe['time_budget_seconds'], 5 * 3600)
        self.assertEqual(value('--num-rollout'), str(launch.ROLLOUT_SAFETY_CEILING))
        self.assertEqual(value('--save-interval'), str(launch.ROLLOUT_SAFETY_CEILING))
        self.assertEqual(value('--custom-megatron-before-train-step-hook-path'), 'wall_clock.wall_clock_lr')
        self.assertEqual(value('--lr-decay-style'), 'constant')  # Hook supplies linear wall-clock decay.
        self.assertEqual(value('--rollout-batch-size'), '128')
        self.assertEqual(value('--n-samples-per-prompt'), '8')
        self.assertEqual(int(value('--rollout-batch-size')) * int(value('--n-samples-per-prompt')),
                         int(value('--global-batch-size')))
        self.assertEqual(value('--global-batch-size'), '1024')
        self.assertEqual(value('--rollout-max-prompt-len'), '2048')
        self.assertEqual(value('--rollout-max-response-len'), '16384')
        self.assertEqual(value('--sglang-context-length'), '18432')
        self.assertEqual(value('--load'), value('--ref-load'))
        self.assertNotIn('--apply-chat-template', args)
        self.assertNotIn('--rollout-global-dataset', args)  # Enabled by default; this positive flag does not exist.
        self.assertEqual(value('--rollout-seed'), str(recipe['seed']))
        self.assertEqual(recipe['counts'], {'math': 19000, 'code': 7000, 'science': 17000})

    def test_copied_slime_has_not_changed(self):
        self.assertGreater(verify(), 200)

    def test_evaluation_launch_reuses_the_recipe_on_two_gpus_without_training(self):
        recipe = launch.load_recipe(ROOT / 'recipe.json')
        args = evaluate.evaluation_command(Path('/input'), Path('/data/amc.jsonl'), Path('/output'))
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(value('--actor-num-gpus-per-node'), '2')
        self.assertEqual(value('--rollout-num-gpus'), '2')
        self.assertEqual(value('--num-gpus-per-node'), '2')
        self.assertIn('--colocate', args)
        self.assertEqual(value('--num-rollout'), '0')
        self.assertEqual(value('--global-batch-size'), '1024')
        self.assertEqual(value('--rollout-max-response-len'), str(recipe['max_generation_tokens']))
        self.assertEqual(value('--rollout-max-context-len'), '20480')
        self.assertEqual(value('--sglang-context-length'), '20480')
        self.assertEqual(value('--seed'), str(recipe['seed']))
        self.assertEqual(value('--sglang-mem-fraction-static'), '0.6')
        self.assertNotIn('--custom-megatron-before-train-step-hook-path', args)
        separate = launch.train_args(recipe, Path('/input'), Path('/data'), Path('/output'), placement='separate')
        self.assertNotIn('--colocate', separate)
        self.assertEqual(separate[separate.index('--actor-num-gpus-per-node') + 1], '2')
        self.assertEqual(separate[separate.index('--rollout-num-gpus') + 1], '2')
        self.assertEqual(separate[separate.index('--num-gpus-per-node') + 1], '4')

    def test_colocation_lifecycle_and_physical_gpu_count(self):
        events = []
        class Remote:
            def __init__(self, name):
                self.name = name
            def remote(self, *args):
                events.append((self.name, *args))
                return 'data'
        manager = SimpleNamespace(**{name: Remote(name) for name in (
            'onload_weights', 'onload_kv', 'generate', 'offload', 'cleanup_rollout_data')})
        class Actor:
            def update_weights(self):
                events.append(('update',))
            def async_train(self, step, data):
                events.append(('train', step))
            def save_model(self, step, force_sync):
                self_outer.assertTrue(force_sync)
                events.append(('save', step))
                Path(args.save_hf.format(rollout_id=step)).mkdir()
        self_outer = self
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=2,
                rollout_num_gpus=2, colocate=True, offload_train=True,
                offload_rollout=True, num_rollout=2, save_hf=directory + '/hf_step_{rollout_id}')
            self.assertEqual(train_driver.allocated_gpu_count(args), 2)
            train_driver.run_training_loop(args, Actor(), manager, lambda value: value)
            complete = json.loads((Path(directory) / 'training_complete.json').read_text())
            self.assertEqual(complete['optimizer_steps'], 2)
            self.assertTrue((Path(directory) / 'hf_step_1/SAVE_COMPLETE').is_file())
        self.assertEqual(events, [
            ('onload_weights',), ('update',), ('onload_kv',),
            ('generate', 0), ('offload',), ('train', 0), ('cleanup_rollout_data', 0),
            ('onload_weights',), ('update',), ('onload_kv',),
            ('generate', 1), ('offload',), ('train', 1), ('save', 1), ('cleanup_rollout_data', 1)])

    def test_sample_is_fixed_unique_and_filters_before_sampling(self):
        rows = [(str(i), {'problem': f'question {i}', 'answer': '1'}) for i in range(20)]
        rows += [('duplicate', rows[0][1]), ('long', {'problem': 'long ' * 100, 'answer': '1'})]
        first, stats = data.reservoir(rows, 10, 'fixed', Tokenizer(), 30, 'math')
        second, _ = data.reservoir(rows, 10, 'fixed', Tokenizer(), 30, 'math')
        self.assertEqual(first, second)
        self.assertEqual(len({r['id'] for r in first}), 10)
        self.assertEqual(stats['duplicate'], 1)
        self.assertEqual(stats['over_length'], 1)
        with self.assertRaisesRegex(ValueError, 'need 30'):
            data.reservoir(rows, 30, 'fixed', Tokenizer(), 30, 'math')

    def test_deepcoder_schemas_preserve_all_tests(self):
        taco = json.dumps({'inputs': ['1\n', '2\n'], 'outputs': ['2\n', '3\n']})
        label = data.normalize_tests(taco)
        self.assertEqual(label['testtype'], 'stdin')
        self.assertEqual(len(label['tests']), 2)
        lcb = json.dumps([{'input': '[1, 2]\n3', 'output': '6', 'testtype': 'functional'}])
        label = data.normalize_tests(lcb, 'add')
        self.assertEqual(label['fn_name'], 'add')
        self.assertEqual(label['testtype'], 'functional')
        with self.assertRaisesRegex(ValueError, 'differ in length'):
            data.normalize_tests({'inputs': ['x'], 'outputs': []})

    def test_science_labels_come_from_explicit_labels_and_not_teacher_reasoning(self):
        row = {'input': 'Q?\nA: alpha\nB: beta',
               'output': r'The answer is \boxed{B}.', 'expected_answer': r'\text{A}'}
        normalized = data.normalize('science', row)
        self.assertEqual(normalized['label'], 'A')
        self.assertEqual(normalized['metadata']['valid_letters'], ['A', 'B'])
        del row['expected_answer']
        with self.assertRaisesRegex(ValueError, 'no explicit'):
            data.normalize('science', row)

    def test_science_reward_uses_only_final_boxed_answer(self):
        self.assertEqual(rewards.science_reward(r'Think B; final \boxed{\text{A}}', 'A', ['A', 'B']), 1.0)
        self.assertEqual(rewards.science_reward(r'\boxed{A} then \boxed{B}', 'A', ['A', 'B']), 0.0)
        self.assertEqual(rewards.science_reward('Maybe A somewhere in the reasoning.', 'A', ['A', 'B']), 0.0)
        self.assertEqual(rewards.science_reward(r'\boxed{C}', 'C', ['A', 'B']), 0.0)

    def test_code_output_and_functional_arguments(self):
        self.assertTrue(sandbox.equivalent('1.0000001 2\n', '1.0 2'))
        self.assertFalse(sandbox.equivalent('nan', '1'))
        self.assertFalse(sandbox.equivalent('1 2 3', '1 2'))
        self.assertTrue(sandbox.equivalent([1, [2, 3]], [1, [2, 3]]))
        # Preserve candidate text verbatim when building the invocation wrapper.
        program, stdin = sandbox.functional_program('def add(xs, x): return sum(xs) + x # FUNCTION SOURCE', 'add', '[1, 2]\n3')
        self.assertIn('# FUNCTION SOURCE', program)
        self.assertEqual(json.loads(stdin), [[1, 2], 3])
        self.assertEqual(rewards.extract_code('```python\nprint(1)\n```\n```python\nprint(2)\n```'), 'print(2)')


if __name__ == '__main__':
    unittest.main()
