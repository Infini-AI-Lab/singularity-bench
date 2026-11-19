"""Codex configuration for the author: no local execution and no mutable extensions.

``texts()`` gives the three files. In the Docker author container ``configure()`` writes them root-owned; the
Apptainer launcher binds them read-only into the author sandbox.
"""
import json
import os
from pathlib import Path

DISABLED = ('apps', 'plugins', 'recommended_plugins', 'tool_suggest', 'multi_agent',
            'multi_agent_v2', 'js_repl', 'code_mode', 'code_mode_host', 'code_mode_only',
            'hooks', 'plugin_hooks', 'shell_snapshot', 'shell_snapshot_v2',
            'responses_websockets', 'responses_websockets_v2')


def feature_config(subagents=False):
    if type(subagents) is not bool:
        raise ValueError('subagents must be a boolean operator selection')
    return '\n'.join(name + ' = ' + ('true' if name == 'multi_agent' and subagents else 'false')
                     for name in DISABLED)


def texts(model, workdir='/app', *, subagents=False, base_url='http://author-gateway:8001/v1',
          executor=('/opt/author-cli/codex-worker', ())):
    """config.toml, environments.toml and requirements.toml. The model is reached at ``base_url`` (the author
    gateway), and every command and file operation goes to the one executor environment, ``executor`` (program,
    arguments), never to the CLI's own machine."""
    features = feature_config(subagents)
    program, arguments = executor
    config = f'''model = {json.dumps(model)}
model_provider = "breakwater"
approval_policy = "never"
sandbox_mode = "danger-full-access"
forced_login_method = "api"
model_reasoning_effort = "high"
model_context_window = 200000
web_search = "disabled"
project_doc_max_bytes = 0
check_for_update_on_startup = false
notify = []

[model_providers.breakwater]
name = "Assigned author model"
base_url = {json.dumps(base_url)}
env_key = "BREAKWATER_AUTHOR_KEY"
wire_api = "responses"
supports_websockets = false

[projects.{json.dumps(workdir)}]
trust_level = "untrusted"

[features]
{features}
unified_exec = true
unified_exec_tty = true
shell_tool = true

[skills]
include_instructions = false
[skills.bundled]
enabled = false
[feedback]
enabled = false
[analytics]
enabled = false
[mcp_servers]
'''
    environments = f'''default = "worker"
include_local = false

[[environments]]
id = "worker"
program = {json.dumps(program)}
args = {json.dumps(list(arguments))}
env = {{}}
cwd = {json.dumps(workdir)}
'''
    requirements = '''allowed_approval_policies = ["never"]
allowed_sandbox_modes = ["read-only", "danger-full-access"]
allow_managed_hooks_only = true
[features]
''' + features + '\n[mcp_servers]\n'
    return {'config.toml': config, 'environments.toml': environments, 'requirements.toml': requirements}


def configure(model, workdir="/app", *, subagents=False):
    """The Docker author container: root-owned files the author can read but not change."""
    files = texts(model, workdir, subagents=subagents)
    private = Path('/run/breakwater/codex')
    private.mkdir()
    os.chown(private, 0, 1100)
    private.chmod(0o3770)
    for name in ('config.toml', 'environments.toml'):
        path = private / name
        path.write_text(files[name])
        os.chown(path, 0, 1100)
        path.chmod(0o640)
    requirements = Path('/etc/codex')
    requirements.mkdir(exist_ok=True)
    (requirements / 'requirements.toml').write_text(files['requirements.toml'])
