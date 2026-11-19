"""The word-count example task (examples/word-count) end to end on the Apptainer backend.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

IMAGE.sif is the example's runtime (apptainer build IMAGE.sif docker://python:3.12-slim). A stand-in CLI plays the
agent and writes the reference solution through the real command broker; the evaluation stage runs the task's own
code (tests/breakwater_task.py) with the real coordinator; the verifier recomputes the score with the task's scorer;
a stand-in judge runner answers PASS to every review, so the run ends with the recomputed reward.
"""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests'), str(HERE.parent)]
from test_agent_stage import IMAGE, READY, Provider  # noqa: E402

AGENT = r'''
import json, os, subprocess, sys, urllib.request
def emit(**event):
    print(json.dumps(event), flush=True)
emit(type='system', subtype='init', cwd=os.getcwd())
request = urllib.request.Request('http://127.0.0.1:8001/v1/messages', headers={'Content-Type': 'application/json'},
    data=json.dumps({'model': 'test/author', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'ok'}]}).encode())
emit(type='assistant', reply=json.loads(urllib.request.urlopen(request, timeout=30).read())['content'][0]['text'])
calls = []
def run(command):
    calls.append(f'toolu_{len(calls) + 1}')
    emit(type='assistant', message={'content': [{'type': 'tool_use', 'id': calls[-1], 'name': 'Bash',
                                                 'input': {'command': command}}]})
    result = subprocess.run(['/opt/breakwater-runtime/breakwater-exec', command], capture_output=True, text=True)
    emit(type='user', message={'content': [{'type': 'tool_result', 'tool_use_id': calls[-1], 'content': result.stdout}]})
run('bash -c "$(cat <<\'SH\'\n' + open('/opt/author-cli/solve.sh').read() + '\nSH\n)"; echo written')
run('test -e /private/word-count/texts.jsonl && echo visible || echo hidden')
emit(type='result', subtype='success')
'''

# Answers every review PASS, with the row structure the verifier requires.
JUDGE = r'''#!/usr/bin/env python3
import json, sys
CHECKS = %s
output = sys.argv[sys.argv.index('--output-last-message') + 1]
sys.stdin.buffer.read()
rows = [{'id': check, 'status': 'PASS', 'evidence': 'Stand-in judge for the example test: no issue in this input.'}
        for check in CHECKS]
with open(output, 'w') as stream:
    json.dump({'verdict': 'PASS', 'attempted_cheating': False, 'successful_cheating': False,
               'comments_markdown': 'Stand-in judge runner for the word-count example test; every row passes.',
               'checklist': rows}, stream)
'''


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to the example runtime SIF; the example is NOT verified')
class ExampleTaskTests(unittest.TestCase):
    def run_example(self, *options, agent=AGENT, budget=None):
        """The example end to end with the stand-in agent and judge: (run directory, summary line). `budget` runs a
        copy of the task with that authoring budget in seconds."""
        from contract import CHECKS
        from sandbox import remove_tree
        provider = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        threading.Thread(target=provider.serve_forever, daemon=True).start()
        self.addCleanup(provider.server_close)
        self.addCleanup(provider.shutdown)
        root = Path(tempfile.mkdtemp(prefix='breakwater-example-'))
        self.addCleanup(remove_tree, root)
        cli, judge, run = root / 'author-cli', root / 'judge', root / 'run'
        for folder in (cli, judge):
            folder.mkdir()
        (cli / 'agent.py').write_text(agent)
        (cli / 'solve.sh').write_text((REPO / 'examples/word-count/solution/solve.sh').read_text())
        (judge / 'codex').write_text(JUDGE % json.dumps(list(CHECKS)))
        (judge / 'codex').chmod(0o755)
        task = REPO / 'examples/word-count'
        if budget:
            import shutil
            copy = root / 'word-count'
            shutil.copytree(task, copy)
            toml = (copy / 'task.toml').read_text()
            (copy / 'task.toml').write_text(toml.replace('[agent]\ntimeout_sec = 900.0', f'[agent]\ntimeout_sec = {budget}', 1))
            self.assertIn(f'timeout_sec = {budget}', (copy / 'task.toml').read_text())
            task = copy
        result = subprocess.run(
            [sys.executable, '-I', str(HERE / 'run.py'), '--task', str(task), '--image', IMAGE, '--cli-dir', str(cli),
             '--judge-dir', str(judge), '--run-dir', str(run), '--asset-root', str(REPO / 'examples/assets'),
             '--author-model', 'test/author', '--author-upstream', f'http://127.0.0.1:{provider.server_address[1]}',
             '--author-key-env', 'TEST_PROVIDER_KEY', '--judge-key-env', 'TEST_JUDGE_KEY',
             '--cli-command', json.dumps([['python3', '/opt/author-cli/agent.py'], {'PATH': '/usr/local/bin:/usr/bin:/bin'}]),
             *options],
            capture_output=True, text=True, timeout=1200,
            env={**os.environ, 'TEST_PROVIDER_KEY': 'sk-test-provider-key', 'TEST_JUDGE_KEY': 'fw-test-judge-key'})
        error = run / 'reports/pipeline-error.json'
        self.assertTrue(result.stdout.strip(), result.stderr[-3000:] + (error.read_text() if error.exists() else ''))
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        failure = run / 'reports/verification-error.json'
        self.assertEqual(summary['verifier_exit_code'], 0, failure.read_text() if failure.exists() else summary)
        return run, summary

    def test_word_count_runs_end_to_end_and_scores_its_submission(self):
        from contract import verify_seal
        run, summary = self.run_example()
        self.assertNotIn('smoke', summary)
        # The agent's commands never saw the private texts; the submission holds only the agent's workspace.
        native = [json.loads(line) for line in (run / 'agent-local/native.jsonl').read_text().splitlines()]
        outputs = [item['content'] for line in native if line.get('type') == 'user'
                   for item in line['message']['content']]
        self.assertEqual(outputs, ['written\n', 'hidden\n'])
        launch = verify_seal(run / 'authority', summary['authority_sha256'])
        manifest = verify_seal(run / 'submission', launch['submission_sha256'], launch['identity'])
        self.assertEqual(sorted(manifest['files']), ['instruction.md', 'solution.py'])
        # The task's own code ran in both stages: five texts answered, all correct, judged PASS.
        self.assertEqual(len((run / 'evaluation/raw/counts.jsonl').read_text().splitlines()), 5)
        self.assertEqual(json.loads((run / 'evaluation/raw/status.json').read_text()), {'exit_code': 0})
        self.assertEqual(json.loads((run / 'reports/judge.json').read_text())['verdict'], 'PASS')
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text()), {'reward': 1.0})
        prompt = (run / 'reports/packet-000000.prompt.txt').read_text()
        self.assertIn('"validated": ["every output line is an object with a text id and integer word counts"', prompt)
        self.assertNotIn('SMOKE RUN', prompt)

    def test_an_agent_still_running_at_its_deadline_is_stopped_and_graded(self):
        # agent_deadline = "evaluate": the CLI writes the solution, then never exits. The host stops it at the deadline,
        # the evaluation grades the workspace as it stands, and the verifier accepts the stopped CLI's record.
        from contract import verify_seal
        run, summary = self.run_example(agent=AGENT.replace("emit(type='result', subtype='success')",
                                                            "emit(type='result', subtype='success')\n"
                                                            "import time\ntime.sleep(3600)"), budget=60)
        self.assertIs(summary['deadline_reached'], True)
        self.assertEqual(summary['exit_code'], -9)
        launch = verify_seal(run / 'authority', summary['authority_sha256'])
        self.assertEqual((launch['stages']['agent']['exit_code'], launch['stages']['agent']['deadline_reached']), (-9, True))
        self.assertEqual(json.loads((run / 'reports/mechanical-check.json').read_text())['status'], 'PASS')
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text()), {'reward': 1.0})

    def test_a_smoke_run_is_sealed_and_labeled_as_one(self):
        from contract import verify_seal
        run, summary = self.run_example('--smoke')
        self.assertIs(summary['smoke'], True)
        self.assertIs(verify_seal(run / 'authority', summary['authority_sha256'])['smoke'], True)
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text()), {'reward': 1.0, 'smoke': True})
        self.assertIn('SMOKE RUN', (run / 'reports/packet-000000.prompt.txt').read_text())


if __name__ == '__main__':
    unittest.main()
