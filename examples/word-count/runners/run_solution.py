"""Trusted runner, executed as submitted code: the submitted count_words on each published text, one JSON line each.

It sees only the submission and the published texts, never the scorer or its reference counts.
"""
import importlib.util
import json

spec = importlib.util.spec_from_file_location('solution', '/app/solution.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/eval-inputs/texts.jsonl') as texts:
    for line in texts:
        row = json.loads(line)
        print(json.dumps({'id': row['id'], 'counts': module.count_words(row['text'])}, sort_keys=True), flush=True)
