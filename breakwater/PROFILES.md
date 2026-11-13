# Agent CLI profiles

The protected author image ([author/](author/)) installs exactly one native agent CLI into the task runtime, chosen
with `--build-arg AUTHOR_CLI=...` when the image is built and `--author-cli` at launch. Native binaries are pinned by
SHA-256 for Linux x86-64 and arm64.

| Profile | Pinned version | Provider protocol | Optional run controls |
| --- | --- | --- | --- |
| `claude-code` (Claude Code) | 2.1.284 | Anthropic Messages (Anthropic or an Anthropic-compatible provider such as Fireworks) | turns, dollar ceiling, reasoning effort, thinking tokens |
| `codex` (Codex) | 0.158.0 | OpenAI Responses | reasoning effort |

The task's wall clock applies to both. Model-supplied provider, configuration and extension overrides are rejected.
Native subagents are off: Claude Code gets no `Agent` tool and Codex runs with `multi_agent` disabled.

## Enforced separation

| Process | Identity | Allowed model connection | Credentials |
| --- | --- | --- | --- |
| Author CLI and Claude Code's built-in file tools | `author`, UID 1100 | `author-gateway:8001` | dummy local key only |
| Every command the agent runs, including background commands | `worker`, UID 1101 | task-policy endpoints; `gateway:8000` only in coding-agent-harness | none |
| Author gateway | separate container | the configured provider, assigned model only | author provider key |
| Development gateway (coding-agent-harness) | separate container | two chat models and one embedding model | development provider key |

At start-up the root-only bootstrap ([author/bootstrap.py](author/bootstrap.py)) resolves the gateway addresses,
checks their health and configuration (model and protocol), and installs exact UID/destination/port firewall rules.
Each task's `environment/agent-policy.json` declares its workspace, writable roots, local-service needs and optional
remote fleet endpoints. There is no Docker-subnet, general DNS or provider-host exception; initialization errors stop
the container, and the CLI refuses to start without the bootstrap's ready marker. Provider credentials never enter the
agent container.

The fixed [`author-exec`](author/author-exec.c) entrypoint starts the CLI as `author`. Its Python interpreter is
resolved at build time, checked for root ownership and non-writable parent directories, and compiled into the
entrypoint, so runtime `PATH` and workspace files cannot select it. Under Breakwater, it and the worker entrypoints
first fork a small root-owned stdio supervisor ([capture/stdio_capture.c](capture/stdio_capture.c)) that forwards
bytes and is traced from the external observer container; neither task user can signal it or edit its evidence.
Submitted commands execute only after the worker has dropped privileges. The CLI itself is not traced, so its setuid
worker transitions keep working; its file-tool activity is represented by gateway and CLI events. The recording
contract ([collector/contract.py](collector/contract.py)) distinguishes recording failures from visibility limits such
as shared-memory writes, and the integrity judge assesses those limits in context.

## Claude Code

Root-owned managed settings ([author/managed-settings.json](author/managed-settings.json)) select the fixed
[`worker-exec`](author/worker-exec.c) shell prefix. It permits only the author-to-worker transition, drops saved
identities, clears the inherited environment and non-stdio descriptors, and sets `no_new_privs` before executing the
command. Neither task user has root capabilities or a general-purpose privileged user switch. Workspace files use a
shared group, so file tools and worker commands can edit the same files.

Exec-form hooks would bypass the shell prefix and run as the author, so the profile disables hooks, MCP servers,
user-installed plugins and sideload flags, ignores user/project/local settings, protects the author's home and
startup files, and exposes only `Bash`, `Read`, `Write` and `Edit`. Internal Git calls also pass through the worker
switch (`/opt/author-tools/git`, compiled from the same source), so a workspace Git helper never executes as the
author. The CLI installation is root-owned and inaccessible to the worker, and managed file-tool permissions deny reads
of the CLI and `/proc`. The image contains no other agent CLI or importable agent package
([author/purge.py](author/purge.py) removes any that the task runtime carried).

## Codex

Codex keeps its unmodified native agent loop. Its root-owned `environments.toml` selects only a private stdio executor
(`include_local = false`) built from the pinned upstream execution library ([author/codex-worker.rs](author/codex-worker.rs)).
The executor's small setuid entrypoint drops all identities to the worker, clears the inherited environment and
descriptors, sets `no_new_privs` and disables process dumps before accepting any request, so raw command arrays, native
patches and other filesystem operations have worker rights. Worker commands cannot reach the executor's memory or
control pipe through `/proc`. Background jobs, polling, PTYs and cancellation remain available. The Codex config,
environment selection and managed requirements are root-owned; project config, MCP, hooks and plugins are disabled.

## Model transport

The author gateway ([author_gateway.py](author_gateway.py)) accepts only Anthropic Messages/count-tokens routes or
Codex Responses/compact routes, for the assigned model; the protocol is fixed per run and checked during bootstrap. It
replaces client credentials with its own, rejects duplicate JSON keys and refuses redirects. Messages requests may
carry client tool definitions, supported fields and inline content; server tools, remote MCP/container/skill
configuration, remote media references and unknown beta headers are rejected. Responses requests may carry
client-defined tools; provider-hosted tools, stored responses and cross-response continuation are rejected. Streaming
is relayed, and every request and response is journaled outside the agent's reach. Worker processes cannot reach this
gateway or the provider. The coding-agent-harness task's development models use a separate gateway
([dev_gateway.py](dev_gateway.py)) that accepts only `chat`, `chat-b` and the fixed embedding model.

## Validation

Before upgrading a pinned CLI or exposing another tool, repeat the adversarial checks on the launch host: the worker
must not be able to read, copy or execute the CLI, reach the gateway or provider, or act as the author.
