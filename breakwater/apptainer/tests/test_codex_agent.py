"""The real pinned Codex CLI as the agent on the Apptainer backend, end to end on the word-count example task.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif BREAKWATER_CODEX_CLI_DIR=DIR \\
      python3 -m unittest discover -s breakwater/apptainer/tests -v

DIR is ``install_cli.py --agent codex --dest DIR`` and IMAGE.sif the example's runtime (apptainer build IMAGE.sif
docker://python:3.12-slim). The scripted stand-in model (scripted_model.py) behind the real author gateway plays the
model: it has Codex run the example's reference solution, add a file with apply_patch, run a command that outlasts
exec_command's 30-second wait (polled with write_stdin until it exits) and run a probe, then finish. Everything else
is real: Codex and its executor in the author sandbox (codex_forward.py forwarding every command to the broker), the
command sandbox, the evaluation stage with the task's own code, and the verifier, whose stand-in judge runner answers
PASS to every review. A second run kills the command sandbox while a command runs, as Slurm does to an out-of-memory
step: the stage replaces it and Codex's next command runs in the replacement.
"""
import http.server
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests'), str(HERE.parent)]
from scripted_model import handler  # noqa: E402
from test_agent_stage import IMAGE, READY  # noqa: E402
from test_example_task import JUDGE  # noqa: E402

CODEX_DIR = os.environ.get('BREAKWATER_CODEX_CLI_DIR')
# The probe: the private texts stay out of the command sandbox, no Codex process runs there (its executor is in the
# author sandbox, which the agent's commands cannot reach), and the file Codex's executor wrote is on the workspace.
PROBE = '''test -e /private/word-count/texts.jsonl && echo visible || echo hidden
echo "codex-processes=$(for p in /proc/[0-9]*; do readlink $p/exe; done 2>/dev/null | grep -c codex)"
cat notes.txt'''
PATCH = '*** Begin Patch\n*** Add File: notes.txt\n+patched by the executor\n*** End Patch\n'


def kill_command_sandbox(run):
    """What Slurm does to an out-of-memory step: kill the command sandbox's whole process tree, from the host."""
    parents, lines = {}, {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text()
                parents[int(entry.name)] = int(fields[fields.rindex(')') + 2:].split()[1])
                lines[int(entry.name)] = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except (OSError, ValueError):
                continue
    victims = {pid for pid, line in lines.items()
               if 'spawner.py' in line and f'{run}/workspace-live:' in line and 'worker-home:/home/worker' in line}
    found = set(victims)
    while True:
        more = {pid for pid, parent in parents.items() if parent in victims and pid not in victims}
        if not more:
            break
        victims |= more
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return found


@unittest.skipUnless(READY and CODEX_DIR, 'set BREAKWATER_APPTAINER_IMAGE (the example runtime SIF) and '
                     'BREAKWATER_CODEX_CLI_DIR (install_cli.py --agent codex); Codex on Apptainer is NOT verified')
class CodexAgentTests(unittest.TestCase):
    def launch(self, steps, watch=None):
        """Run the example task with Codex and a scripted model taking ``steps`` (``watch(run)`` beside it):
        (model, run dir, summary, launcher stderr)."""
        from contract import CHECKS
        from sandbox import remove_tree
        model = handler(steps)
        provider = http.server.ThreadingHTTPServer(('127.0.0.1', 0), model)
        threading.Thread(target=provider.serve_forever, daemon=True).start()
        self.addCleanup(provider.server_close)
        self.addCleanup(provider.shutdown)
        root = Path(tempfile.mkdtemp(prefix='breakwater-codex-'))
        self.addCleanup(remove_tree, root)
        judge, run = root / 'judge', root / 'run'
        judge.mkdir()
        (judge / 'codex').write_text(JUDGE % json.dumps(list(CHECKS)))
        (judge / 'codex').chmod(0o755)
        if watch is not None:
            threading.Thread(target=watch, args=(run,), daemon=True).start()
        result = subprocess.run(
            [sys.executable, '-I', str(HERE / 'run.py'), '--task', str(REPO / 'examples/word-count'), '--image', IMAGE,
             '--cli-dir', CODEX_DIR, '--author-cli', 'codex', '--judge-dir', str(judge), '--run-dir', str(run),
             '--asset-root', str(REPO / 'examples/assets'), '--author-model', 'test/author',
             '--author-upstream', f'http://127.0.0.1:{provider.server_address[1]}',
             '--author-key-env', 'TEST_PROVIDER_KEY', '--judge-key-env', 'TEST_JUDGE_KEY'],
            capture_output=True, text=True, timeout=1200,
            env={**os.environ, 'TEST_PROVIDER_KEY': 'sk-test-provider-key', 'TEST_JUDGE_KEY': 'fw-test-judge-key'})
        error = run / 'reports/pipeline-error.json'
        self.assertTrue(result.stdout.strip(), result.stderr[-3000:] + (error.read_text() if error.exists() else ''))
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        native_err = run / 'agent-local/native.stderr'
        self.assertEqual(summary.get('exit_code'), 0, json.dumps(summary) + '\n' + result.stderr[-3000:]
                         + (native_err.read_text()[-3000:] if native_err.exists() else ''))
        return model, run, summary

    @staticmethod
    def events(run):
        journal = [json.loads(line) for line in (run / 'external/commands/exec.jsonl').read_text().splitlines()]
        native = [json.loads(line) for line in (run / 'agent-local/native.jsonl').read_text().splitlines()]
        outputs = [row['item'] for row in native
                   if row.get('type') == 'item.completed' and row['item'].get('type') == 'command_execution']
        return journal, outputs

    def test_codex_runs_every_command_through_the_broker_in_the_command_sandbox(self):
        from contract import verify_seal
        solve = (REPO / 'examples/word-count/solution/solve.sh').read_text() + '\necho written'
        steps = [solve, {'patch': PATCH}, 'sleep 35 && echo slept', PROBE]
        model, run, summary = self.launch(steps)
        self.assertEqual(summary['agent_cli'], 'codex')
        # The model was reached through the gateway's Responses route, with Codex's tools offered.
        self.assertEqual({request['path'] for request in model.requests}, {'/v1/responses'})
        self.assertIn('exec_command', {tool.get('name') for tool in model.requests[0]['body']['tools']})
        # The long command was polled at least once before it exited.
        polls = [item for request in model.requests for item in request['body'].get('input', [])
                 if isinstance(item, dict) and item.get('name') == 'write_stdin']
        self.assertTrue(polls)
        started = json.loads((run / 'external/gateway/requests.jsonl').read_text().splitlines()[0])
        self.assertEqual((started['event'], started['protocol']), ('gateway_started', 'responses'))
        # Each command the agent ran is one broker command: Codex's shell with the command, with Codex's variables
        # for commands (pager, color, locale). The apply_patch edit is the executor's, not a command.
        journal, outputs = self.events(run)
        commands = [row['command'] for row in journal if row['event'] == 'exec_started']
        self.assertEqual([shlex.split(command)[-1] for command in commands], [solve, 'sleep 35 && echo slept', PROBE])
        self.assertTrue(all('PAGER=cat' in command and ' exec ' in command for command in commands))
        self.assertEqual([row['status'] for row in journal if row['event'] == 'exec_exited'], [0, 0, 0])
        self.assertEqual(len(outputs), 3, outputs)
        self.assertIn('written', outputs[0]['aggregated_output'])
        self.assertIn('slept', outputs[1]['aggregated_output'])
        self.assertEqual(outputs[2]['aggregated_output'].split(),
                         ['hidden', 'codex-processes=0', 'patched', 'by', 'the', 'executor'])
        launch = verify_seal(run / 'authority', summary['authority_sha256'])
        self.assertEqual(launch['stages']['agent']['agent_cli'], 'codex')
        self.assertEqual(launch['stages']['agent']['agent_cli_version'],
                         (Path(CODEX_DIR) / 'version').read_text().strip())
        manifest = verify_seal(run / 'submission', launch['submission_sha256'], launch['identity'])
        self.assertEqual(sorted(manifest['files']), ['instruction.md', 'notes.txt', 'solution.py'])
        # Evaluation and verification as for any agent: five texts, all counted correctly, judged PASS.
        failure = run / 'reports/verification-error.json'
        self.assertEqual(summary['verifier_exit_code'], 0, failure.read_text() if failure.exists() else summary)
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text()), {'reward': 1.0})
        mechanical = json.loads((run / 'reports/mechanical-check.json').read_text())
        self.assertEqual(mechanical['native_agreement']['tool_calls_compared'], len(outputs))
        self.assertIn('forwarded every command', mechanical['established']['capture_complete'])

    def test_a_lost_command_sandbox_is_replaced_and_codex_carries_on(self):
        killed = []

        def watch(run):
            deadline = time.monotonic() + 600
            while not (run / 'workspace-live/kill-me').exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            killed.extend(kill_command_sandbox(run))

        # The second command is killed with its sandbox; the third is asked for once the stage has had time to
        # replace the sandbox, as the next command of an agent would be.
        model, run, summary = self.launch(['echo one', 'touch kill-me; sleep 120',
                                           {'command': 'echo after-restart', 'delay': 10}], watch)
        self.assertEqual(len(killed), 1)
        self.assertEqual(summary['command_sandbox_restarts'], 1)
        journal, outputs = self.events(run)
        self.assertIn('command_sandbox_lost', [row['event'] for row in journal])
        self.assertIn('command_sandbox_restarted', [row['event'] for row in journal])
        self.assertEqual([row['status'] for row in journal if row['event'] == 'exec_exited'], [0, 137, 0])
        self.assertEqual(outputs[-1]['aggregated_output'], 'after-restart\n')
        self.assertEqual(outputs[-1]['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
