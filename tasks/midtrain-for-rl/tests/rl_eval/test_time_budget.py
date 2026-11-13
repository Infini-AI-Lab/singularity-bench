"""Time-budget selection, timeout recovery and LR tests without waiting five hours."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import launch
import train_driver
from completion import validate_completion
from wall_clock import TrainingClock, wall_clock_lr, synchronized_fraction


class TimeBudgetTests(unittest.TestCase):
    def test_actor_ranks_use_rank_zero_learning_rate_fraction(self):
        tensor = SimpleNamespace(value=0.7)
        tensor.item = lambda:tensor.value
        def broadcast(value, src):
            self.assertEqual(src,0)
            value.value=0.25
        dist=SimpleNamespace(is_initialized=lambda:True,get_backend=lambda:'nccl',broadcast=broadcast)
        torch=SimpleNamespace(distributed=dist,float64='float64',tensor=lambda *a,**kw:tensor)
        with patch.dict('sys.modules',{'torch':torch,'torch.distributed':dist}):
            self.assertEqual(synchronized_fraction(0.7),0.25)

    def test_checkpoint_selection_accepts_variable_update_counts(self):
        recipe = launch.load_recipe(launch.ROOT/'recipe.json')
        for steps in (1, 37, 151):
            record = dict(mode='wall_clock', time_budget_seconds=18000, optimizer_steps=steps,
                          elapsed_seconds=17700, checkpoint_elapsed_seconds=17700, stop_reason='wall_clock_reserve')
            validate_completion(record, recipe)
            for changes in ({'elapsed_seconds':100}, {'checkpoint_elapsed_seconds':18001},
                            {'optimizer_steps':0}, {'stop_reason':'step_limit'}, {'time_budget_seconds':60}):
                with self.assertRaises(ValueError):
                    validate_completion(dict(record, **changes), recipe)

    def test_budget_survives_monotonic_float_rounding(self):
        # supervise() passes start and start + budget as strings; near 2**21 s of uptime the difference the driver
        # records is not exactly the recipe's budget.
        recipe = launch.load_recipe(launch.ROOT/'recipe.json')
        start = 2_096_000.123456789
        with patch.dict('os.environ', {'SLIME_JOB_START_MONOTONIC': str(start),
                                       'SLIME_JOB_DEADLINE_MONOTONIC': str(start + recipe['time_budget_seconds'])}):
            clock = TrainingClock.from_environment()
        recorded = clock.deadline - clock.start
        self.assertNotEqual(recorded, recipe['time_budget_seconds'])  # the rounding this test is about
        record = dict(mode='wall_clock', time_budget_seconds=recorded, optimizer_steps=138,
                      elapsed_seconds=17200, checkpoint_elapsed_seconds=17200, stop_reason='wall_clock_reserve')
        validate_completion(record, recipe)
        for bad in (18000.01, 17999.99, 60, None, '18000', True):
            with self.assertRaises(ValueError):
                validate_completion(dict(record, time_budget_seconds=bad), recipe)
        with self.assertRaises(ValueError):
            validate_completion(dict(record, mode='steps'), recipe)

    def test_timeout_recovers_only_a_completed_checkpoint(self):
        recipe = launch.load_recipe(launch.ROOT/'recipe.json')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            old = output/'hf_step_36'; old.mkdir()
            (old/'SAVE_COMPLETE').write_text('optimizer_steps=37\n')
            (output/'hf_step_37').mkdir()  # interrupted save must never be selected
            receipt = dict(checkpoint_path=str(old), optimizer_steps=37, mode='wall_clock',
                           time_budget_seconds=18000, checkpoint_elapsed_seconds=14500)
            (output/'latest_checkpoint.json').write_text(json.dumps(receipt))
            selected, complete = launch.select_checkpoint(output, recipe, 124, 18000.1)
            self.assertEqual(selected,old)
            self.assertEqual(complete['stop_reason'],'hard_deadline')
            with self.assertRaises(RuntimeError):
                launch.select_checkpoint(output,recipe,1,18000.1)
            with self.assertRaises(ValueError):
                launch.select_checkpoint(output,recipe,124,10)
            (old/'SAVE_COMPLETE').unlink()
            with self.assertRaises(RuntimeError):
                launch.select_checkpoint(output,recipe,124,18000.1)

    def test_timed_loop_stops_at_boundary_and_saves_actual_update_count(self):
        current = [0.0]
        events=[]
        class Remote:
            def __init__(self,name):self.name=name
            def remote(self,*args):events.append(self.name);return 'data'
        manager=SimpleNamespace(**{n:Remote(n) for n in ('onload_weights','onload_kv','generate','offload','cleanup_rollout_data')})
        class Actor:
            def update_weights(self):events.append('update')
            def async_train(self,step,data):current[0]+=4;events.append('train')
            def save_model(self,step,force_sync):
                Path(args.save_hf.format(rollout_id=step)).mkdir()
                events.append('save')
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(offload_rollout=True,offload_train=True,num_rollout=1000000,
                                 save_hf=temp+'/hf_step_{rollout_id}')
            result=train_driver.run_training_loop(args,Actor(),manager,lambda x:x,
                clock=TrainingClock(start=0,deadline=10,interval=5,reserve=2),now=lambda:current[0])
            self.assertEqual(result['optimizer_steps'],2)
            self.assertEqual(result['stop_reason'],'wall_clock_reserve')
            self.assertEqual(events.count('train'),2)
            self.assertEqual(events.count('save'),1)
            self.assertTrue((Path(temp)/'hf_step_1/SAVE_COMPLETE').is_file())
            self.assertFalse((Path(temp)/'hf_step_99').exists())

    def test_wall_clock_lr_decays_with_time_not_update_count(self):
        optimizer=SimpleNamespace(param_groups=[{'lr_mult':1.0},{'lr_mult':0.0}])
        scheduler=SimpleNamespace(max_lr=5e-7)
        with patch.dict('os.environ',{'SLIME_JOB_START_MONOTONIC':'0','SLIME_JOB_DEADLINE_MONOTONIC':'18000'}), \
             patch('wall_clock.synchronized_fraction',side_effect=lambda v:v):
            for elapsed in (9000,13500):
                with patch('wall_clock.time.monotonic',return_value=elapsed):
                    wall_clock_lr(SimpleNamespace(lr=5e-7),200,0,None,optimizer,scheduler)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'],5e-7*(1-elapsed/18000))
                self.assertEqual(optimizer.param_groups[1]['lr'],0)


if __name__ == '__main__':
    unittest.main()
