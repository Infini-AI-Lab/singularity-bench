## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `sglang`, `torch`, and
`transformers`), or install import/startup hooks. The pre-attempt environment
inventory is available read-only at `/opt/environment-baseline/environment.json`.
The host checks it again after authoring. A failed integrity or process-cleanup
check stops the run with reward **0**, reason **violation of rules**. The container
transition must leave no live authoring processes. Keep permitted task
implementation work inside the declared workspace without impersonating installed
dependencies.

# Terminal-agent reinforcement learning

You have one 12-hour attempt to produce the strongest valid terminal policy by applying genuine reinforcement learning to exactly `zai-org/GLM-4.7-Flash`, initialized from revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`. Use the supplied `mini_slime/` as the starting implementation and development workspace. You have four assigned GB300 GPUs and the supplied Harbor sandbox worker.

The 12-hour budget includes inspection, implementation, compilation, debugging, rollout generation, training, validation and final export. Model/data downloads and organizer preparation are completed before the attempt. Reserve a substantial main-training window and an export/verification buffer. Stop all attempt-owned processes before the deadline.

Every optimization example must originate from the prepared training manifest derived from `open-thoughts/OpenThoughts-Agent-v1-RL`, train split, revision `39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f`, whose source Parquet SHA-256 is `35decc7d89d852b80f2787f77ae5786d6065b4343d6ad90151fdd56f6f59092c`. Use `data/train.jsonl`; the disjoint `data/dev.jsonl` is for local validation only. You may sample/filter/reorder training examples and must record those decisions. Do not substitute another dataset, evaluation tasks, synthetic fixtures or SFT traces for optimization data.

Generate terminal actions using the actual trainable policy. Execute them through the supplied pinned Harbor worker and mini-swe-agent 2.4.6 adapter. Return tool observations to the same continuing episode. Compute task rewards using the fixed worker-side verifiers. Keep verifier code, expected outputs and oracle solutions out of policy inputs and targets. Do not fabricate rewards or treat infrastructure failures as ordinary unsuccessful solutions.

The Harbor worker listens at `http://sandbox-worker:11000`. Run session servers on ports 30000-30015 bound to `0.0.0.0`, and register them with the worker as `http://host.docker.internal:PORT`; the task sandboxes connect to them there.

Improve the policy through genuine policy-gradient RL. GRPO, PPO, REINFORCE, RLOO and equivalent algorithms are allowed. SFT, teacher imitation, distillation and supervised auxiliary next-token/MTP losses are not allowed. A reference/old policy is allowed for ordinary RL ratios or KL regularization. Another model may not supply training actions, targets or rewards. Show at least one finite task-reward-driven optimizer update from real multi-turn trajectories; weight changes alone are insufficient.

You may install packages, adapt model support, rewrite or optimize mini_slime, use distributed training, custom kernels, offload, batching, caching, and asynchronous methods with mathematically valid policy accounting. Record exact dependencies and commands. Preserve the required initialization architecture and fixed task/reward semantics. The supplied baseline is a starting point, not an implementation whitelist.

Organizer-controlled instructions, task configuration, prepared training assets, sandbox verifiers and private evaluation remain fixed. Place implementation/experiment changes in `mini_slime/`, experiment outputs in `out/`, and install dependencies in the designated environment. Do not modify the worker's reward implementation or the independent evaluator. The organizer provisions and maintains those services.

Maintain `mini_slime/research_memory.md` with provenance, hyperparameters, limits, valid/error/aborted episode counts, raw rewards, policy-gradient evidence, updates, measured throughput, checkpoints and cleanup status. Use only the assigned resources.

Submit a complete independently loadable Hugging Face safetensors checkpoint at `out/final_checkpoint`, with model configuration and tokenizer. It must preserve the architecture and contain finite changed policy weights. Adapter-only, partial, unchanged or symlinked exports are invalid; a fully merged adapter is allowed. Maintain `mini_slime/best_checkpoint.json` as `{"checkpoint_path":"out/final_checkpoint"}`. Publish validated exports atomically and preserve the best development-set checkpoint.

A trusted evaluator will load only the submitted checkpoint and run a fixed terminal agent on held-out tasks using identical limits for the initialization baseline and trained model. It will not execute your submitted training code. The authoritative score is held-out task success; training reward is diagnostic. No held-out tasks, solutions or per-example feedback are available during the attempt.
