#!/usr/bin/env bash
# Prepare the open-ended-nano-gpt assets under ASSET_ROOT, the directory later passed to
# breakwater/run.py as --asset-root. ASSET_ROOT must lie outside the task directory.
#
#   bash tasks/open-ended-nano-gpt/prepare.sh ASSET_ROOT          # download, tokenize, prepare LAMBADA
#   bash tasks/open-ended-nano-gpt/prepare.sh --check ASSET_ROOT  # validate only; downloads and writes nothing
#
# Layout under ASSET_ROOT (the sources in bindings.example.json):
#   openwebtext_preprocessing/openwebtext_presplit_raw/       pinned presplit OpenWebText snapshot
#   open-ended-nano-gpt/nanoGPT/data/openwebtext_qwen3_full/  train.bin, meta.json: agent, read-only
#   private_validation/openwebtext_qwen3_full/                test.bin, meta.json: evaluation and verifier only
#   private_validation/lambada/                               LAMBADA: evaluation and verifier only
#
# Tokenization runs offline in the task runtime image (build it first with environment/build.sh);
# downloads use the host Hugging Face CLI. Valid outputs are reused.
set -euo pipefail

check=0
if [[ "${1:-}" == --check ]]; then
  check=1
  shift
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 [--check] ASSET_ROOT" >&2
  exit 2
fi
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
asset_root=$(realpath -m -- "$1")
if [[ "${asset_root}/" == "${task_root}/"* ]]; then
  echo "ASSET_ROOT must be outside ${task_root}, which is copied into the agent workspace" >&2
  exit 2
fi
if (( check )); then
  [[ -d "${asset_root}" ]] || { echo "No asset root at ${asset_root}" >&2; exit 1; }
else
  mkdir -p "${asset_root}"
fi

dataset_repo=YangZhoumill/openwebtext-presplit
dataset_revision=b96bd5a6a3ae68fe05593cac6a36b170296e8f04
workers=${OPENWEBTEXT_DOWNLOAD_WORKERS:-16}
batch_size=${OPENWEBTEXT_TOKENIZER_BATCH_SIZE:-512}
task_image=${TASK_IMAGE:-open-ended-nano-gpt:local}
raw_dir=openwebtext_preprocessing/openwebtext_presplit_raw
train_dir=open-ended-nano-gpt/nanoGPT/data/openwebtext_qwen3_full
validation_dir=private_validation/openwebtext_qwen3_full
lambada_dir=private_validation/lambada

if ! docker image inspect "${task_image}" >/dev/null 2>&1; then
  echo "Missing image ${task_image}: build it first with ${task_root}/environment/build.sh" >&2
  exit 1
fi

preprocess() {
  docker run --rm \
    --entrypoint python \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env TOKENIZERS_PARALLELISM=true \
    --volume "${task_root}:/task:ro" \
    --volume "${asset_root}:/assets" \
    --workdir /assets \
    "${task_image}" \
    /task/prepare_presplit_openwebtext_qwen3.py \
    --input-dir "/assets/${raw_dir}" \
    --tokenizer /task/models \
    --train-output-dir "/assets/${train_dir}" \
    --test-output-dir "/assets/${validation_dir}" \
    --batch-size "${batch_size}" \
    "$@"
}

lambada() {
  python3 "${task_root}/tests/prepare_lambada.py" "$@" \
    --output "${asset_root}/${lambada_dir}/lambada_openai.jsonl" \
    --prompts-output "${asset_root}/${lambada_dir}/lambada_prompts.jsonl"
}

if (( check )); then
  preprocess --validate-only
  lambada --check
  exit 0
fi

mkdir -p "${asset_root}/${raw_dir}" "${asset_root}/${train_dir}" \
  "${asset_root}/${validation_dir}" "${asset_root}/${lambada_dir}"

echo "Checking the pinned OpenWebText source snapshot"
if ! preprocess --validate-source-only; then
  echo "Source snapshot is absent or invalid; downloading the pinned revision on the host"
  if ! command -v hf >/dev/null 2>&1; then
    echo 'Host Hugging Face CLI is required: install with python3 -m pip install "huggingface_hub==0.36.0"' >&2
    exit 1
  fi
  HF_HUB_OFFLINE=0 hf download "${dataset_repo}" \
    --repo-type dataset \
    --revision "${dataset_revision}" \
    --local-dir "${asset_root}/${raw_dir}" \
    --max-workers "${workers}"
  preprocess --validate-source-only
fi

echo "Checking the Qwen3-tokenized training and validation splits"
if ! preprocess --validate-only; then
  echo "Tokenized output is absent or incomplete; tokenizing (completed source shards are kept)"
  preprocess
  preprocess --validate-only
fi

echo "Preparing the pinned LAMBADA split on the host"
lambada
echo "Assets are ready under ${asset_root}"
