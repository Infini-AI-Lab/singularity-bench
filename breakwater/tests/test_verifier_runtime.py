"""Opt-in permission and Codex-feature probe for the actual built verifier image."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'tests'))
from test_docker_boundary import available


@unittest.skipUnless(os.environ.get('BREAKWATER_VERIFIER_TEST_IMAGE') and available(),
                     'Requires an explicitly selected built verifier image')
class VerifierRuntimeTests(unittest.TestCase):
    def test_actual_verifier_is_rootless_readonly_without_gpus_or_execution_tools(self):
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            root = Path(temporary);root.chmod(0o755)
            evidence = root / 'evidence';evidence.mkdir(mode=0o755)
            (evidence / 'canary').write_text('read-only external evidence\n')
            (evidence / 'canary').chmod(0o444)
            program = '''import errno,json,os,re,subprocess
from pathlib import Path
assert os.geteuid()==1102
assert not Path('/dev/nvidia0').exists()
status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert int(status['CapEff'].strip(),16)==0 and status['NoNewPrivs'].strip()=='1'
assert Path('/external-evidence/canary').read_text()=='read-only external evidence\\n'
for file in ('/external-evidence/canary','/opt/breakwater/prohibited-write'):
 try:Path(file).write_text('forbidden')
 except OSError as error:assert error.errno in (errno.EROFS,errno.EACCES,errno.EPERM)
 else:raise AssertionError('write unexpectedly succeeded')
Path('/tmp/verifier-home').mkdir()
features=subprocess.run(['/opt/breakwater/codex','-c','features.shell_tool=false','-c','features.multi_agent=false',
                        'features','list'],capture_output=True,text=True,check=True).stdout
for feature in ('shell_tool','multi_agent'):
 assert re.search(r'^'+feature+r'\\s+.*\\s+false$',features,re.MULTILINE),features
print(json.dumps({'status':'PASS','uid':os.geteuid(),'gpu_devices':0,'external_evidence_write':'DENIED',
                  'rootfs_write':'DENIED','shell_tool':False,'multi_agent':False}))
'''
            result = subprocess.run(['docker','run','--rm','--read-only','--network=none','--ipc=private',
                '--cap-drop=ALL','--security-opt=no-new-privileges:true','--user=1102:1102',
                '--tmpfs','/tmp:rw,nosuid,nodev,mode=1777,size=512m','-e','HOME=/tmp/verifier-home',
                '--mount',f'type=bind,source={evidence},target=/external-evidence,readonly',
                '--entrypoint','python3',os.environ['BREAKWATER_VERIFIER_TEST_IMAGE'],'-I','-B','-c',program],
                capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'],'PASS')
            print(json.dumps(report),flush=True)


if __name__ == '__main__':
    unittest.main()
