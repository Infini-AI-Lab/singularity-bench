"""word-count's trusted task code for Breakwater (breakwater/SPEC.md, "Task code").

evaluate: publish the private texts to the submitted code and run it through the trusted runner (runners/).
score: the share of texts whose counts equal the reference counts, recomputed from the sealed outputs.
"""
import json
import string

from contract import atomic_json, require


def evaluate(run):
    run.publish('texts.jsonl', run.asset('texts').read_bytes())
    with (run.raw / 'counts.jsonl').open('wb') as out, (run.raw / 'counts.stderr').open('wb') as err:
        result = run.run_submission(['python3', '/opt/breakwater-drivers/run_solution.py'], stdout=out, stderr=err,
                                    timeout=300)
    atomic_json(run.raw / 'status.json', {'exit_code': result.returncode})


def reference(text):
    counts = {}
    for piece in text.split():
        word = piece.strip(string.punctuation).lower()
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


def score(scoring):
    texts = [json.loads(line) for line in scoring.asset('texts').read_text().splitlines() if line.strip()]
    outputs = {}
    for line in (scoring.raw / 'counts.jsonl').read_text().splitlines():
        row = json.loads(line)
        require(isinstance(row, dict) and set(row) == {'id', 'counts'} and isinstance(row['counts'], dict)
                and all(isinstance(word, str) and type(count) is int for word, count in row['counts'].items()),
                'Malformed output line')
        require(row['id'] not in outputs, 'A text was answered twice')
        outputs[row['id']] = row['counts']
    correct = sum(outputs.get(text['id']) == reference(text['text']) for text in texts)
    return {'reward': correct / len(texts), 'details': {'correct': correct, 'texts': len(texts)},
            'validated': ['every output line is an object with a text id and integer word counts',
                          'no text is answered twice']}
