"""Launch only the pinned, restricted author CLI. No host environment is forwarded."""
import argparse
import json
import os
from pathlib import Path
import sys


def workdir():
    return json.loads(Path("/run/breakwater/runtime.json").read_text()).get("workdir", "/app")


def author_tools(harness, subagents=False):
    """Keep delegation independent of web search, plugins and worker execution."""
    if harness == 'claude-code':
        return ['--tools', 'Bash,Read,Write,Edit' + (',Agent' if subagents else '')]
    raise ValueError('Unsupported tool allowlist profile')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction")
    parser.add_argument("--model")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-thinking-tokens", type=int)
    args = parser.parse_args()
    if os.geteuid() != 1100 or not Path("/run/breakwater/ready").is_file():
        raise SystemExit("Protected author launch requires author uid and completed bootstrap")
    config = json.loads(Path("/run/breakwater/runtime.json").read_text())
    if args.model and args.model != config["model"]:
        raise SystemExit("Author model differs from the gateway's fixed model")
    model = config["model"]
    if config.get('harness') == 'codex':
        return launch_codex(args, parser)
    if config.get("harness") != "claude-code":
        parser.error("Unknown author profile")
    env = {"HOME": "/home/author", "USER": "author", "LOGNAME": "author", "LANG": "C.UTF-8",
           "SHELL": "/bin/bash", "PATH": "/opt/author-tools:/usr/local/bin:/usr/bin:/bin",
           "ANTHROPIC_API_KEY": "gateway-only-not-a-provider-key",
           "ANTHROPIC_BASE_URL": "http://author-gateway:8001", "ANTHROPIC_MODEL": model,
           "CLAUDE_CONFIG_DIR": "/logs/agent/sessions",
           "CLAUDE_CODE_TMPDIR": "/run/breakwater/tmp",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
           "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1", "DISABLE_AUTOUPDATER": "1",
           "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    for key in ("ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        env[key] = model
    if args.max_thinking_tokens is not None:
        env["MAX_THINKING_TOKENS"] = str(args.max_thinking_tokens)
    binary = "/opt/author-cli/claude"
    command = [binary, "--version"] if args.version else [binary, "--verbose", "--output-format=stream-json",
        "--permission-mode=bypassPermissions", "--setting-sources", "", *author_tools('claude-code', config.get('subagents', False)),
        "--model", model]
    if not args.version:
        if args.instruction is None:
            parser.error("--instruction is required")
        for option in ("max_turns", "max_budget_usd", "effort"):
            value = getattr(args, option)
            if value is not None:
                command += ["--" + option.replace("_", "-"), str(value)]
        command += ["--print", "--", args.instruction]
    os.umask(0o002)
    os.chdir(workdir())
    os.execve(binary, command, env)


def launch_codex(args, parser):
    if args.max_budget_usd is not None or args.max_thinking_tokens is not None or args.max_turns is not None:
        parser.error('Codex uses the task clock; dollar, thinking-token and turn limits are unsupported')
    env = {'HOME': '/home/author', 'USER': 'author', 'LOGNAME': 'author', 'LANG': 'C.UTF-8',
           'PATH': '/opt/author-tools:/usr/local/bin:/usr/bin:/bin', 'SHELL': '/bin/bash',
           'CODEX_HOME': '/run/breakwater/codex', 'BREAKWATER_AUTHOR_KEY': 'gateway-only-not-a-provider-key',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
    binary = '/opt/author-cli/codex'
    command = [binary, '--version']
    if not args.version:
        if args.instruction is None:
            parser.error('--instruction is required')
        command = [binary, 'exec', '--strict-config', '--skip-git-repo-check', '--json', '--cd', workdir()]
        if args.effort is not None:
            command += ['-c', 'model_reasoning_effort=' + json.dumps(args.effort)]
        command += ['--', args.instruction]
    os.umask(0o002)
    os.chdir(workdir())
    os.execve(binary, command, env)


if __name__ == "__main__":
    main()
