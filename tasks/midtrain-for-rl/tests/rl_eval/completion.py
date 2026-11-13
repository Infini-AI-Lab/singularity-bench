"""Shared, CPU-only eligibility checks for the RL launcher and trusted grader."""
import math


def validate_completion(record, recipe):
    steps = record.get('optimizer_steps')
    if type(steps) is not int or steps < 1:
        raise ValueError('RL must complete at least one optimizer update')
    budget = recipe.get('time_budget_seconds')
    if budget is None:
        if steps != recipe['steps']:
            raise ValueError("RL did not complete the recipe's fixed number of steps")
        return
    # The receipt's budget is deadline - start of two monotonic floats (wall_clock.supervise), so it can differ
    # from the recipe's value by a float rounding error (e.g. 18000.000000000233 near 2**21 s of uptime).
    recorded = record.get('time_budget_seconds')
    if (record.get('mode') != 'wall_clock' or type(recorded) not in (float, int)
            or not math.isclose(recorded, budget, rel_tol=0, abs_tol=1e-6)):
        raise ValueError('RL did not use the fixed wall-clock training budget')
    elapsed, saved = record.get('elapsed_seconds'), record.get('checkpoint_elapsed_seconds')
    if any(type(v) not in (float,int) or not math.isfinite(v) or v < 0 for v in (elapsed,saved)):
        raise ValueError('Invalid RL wall-clock completion receipt')
    if saved > budget or saved > elapsed:
        raise ValueError('Selected checkpoint was not completed within the RL budget')
    reason = record.get('stop_reason')
    if reason == 'wall_clock_reserve':
        if not budget - recipe['final_save_reserve_seconds'] <= elapsed <= budget:
            raise ValueError('RL ended before its wall-clock stopping window or exceeded its budget')
    elif reason == 'hard_deadline':
        if elapsed < budget or record.get('supervisor_exit_code') != 124:
            raise ValueError('RL hard deadline was not reached')
    else:
        raise ValueError('RL ended early instead of using its wall-clock budget')
