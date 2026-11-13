"""Root-owned Codex configuration; no local execution or mutable extensions."""
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


def configure(model, workdir="/app", *, subagents=False):
    features = feature_config(subagents)
    private = Path('/run/breakwater/codex')
    private.mkdir()
    os.chown(private, 0, 1100)
    private.chmod(0o3770)
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
base_url = "http://author-gateway:8001/v1"
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
program = "/opt/author-cli/codex-worker"
args = []
env = {{}}
cwd = {json.dumps(workdir)}
'''
    for name, text in (('config.toml', config), ('environments.toml', environments)):
        path = private / name
        path.write_text(text)
        os.chown(path, 0, 1100)
        path.chmod(0o640)
    requirements = Path('/etc/codex')
    requirements.mkdir(exist_ok=True)
    (requirements / 'requirements.toml').write_text('''allowed_approval_policies = ["never"]
allowed_sandbox_modes = ["read-only", "danger-full-access"]
allow_managed_hooks_only = true
[features]
''' + features + '\n[mcp_servers]\n')
