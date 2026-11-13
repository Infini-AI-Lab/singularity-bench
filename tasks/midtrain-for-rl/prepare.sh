#!/usr/bin/env bash
# Host preparation for midtrain-for-rl: the pinned base model, corpora and development
# evaluation sets; the task runtime image (midtrain-for-rl:local), the RL-preparation image and
# the CPU code-sandbox image; then the RL mixture and post-RL evaluation sets, which only the
# evaluation stage and the verifier see. Idempotent; completed steps are reused.
#
#   bash tasks/midtrain-for-rl/prepare.sh                # everything (needs HF access)
#   bash tasks/midtrain-for-rl/prepare.sh --check        # offline validation of data, images and grading inputs
#   bash tasks/midtrain-for-rl/prepare.sh --data-only [prepare_data.py options]
#   bash tasks/midtrain-for-rl/prepare.sh --images-only
#   bash tasks/midtrain-for-rl/prepare.sh --private-data # grading inputs only; needs the images
set -euo pipefail

TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLS_VENV=${MIDTRAIN_TOOLS_VENV:-${TASK_ROOT}/.tools-venv}
TASK_IMAGE=midtrain-for-rl:local
AGENT_IMAGE=midtrain-for-rl-agent:local
RL_IMAGE=midtrain-for-rl-rl:local
SANDBOX_IMAGE=midtrain-code-sandbox:local
# Gitignored and excluded from the agent workspace (breakwater.json workspace_excludes).
PRIVATE=${TASK_ROOT}/private

mode=all
if [[ "${1:-}" =~ ^--(check|data-only|images-only|private-data)$ ]]; then
  mode=${1#--}
  shift
fi

tools() {
  if [[ ! -x "${TOOLS_VENV}/bin/python" ]]; then
    echo "Creating preparation environment at ${TOOLS_VENV}"
    python3 -m venv "${TOOLS_VENV}"
  fi
  "${TOOLS_VENV}/bin/python" -m pip install --disable-pip-version-check --quiet \
    'huggingface_hub==0.36.0' 'pyarrow==24.0.0'
}

# Gated inputs fail here, before any large download, if access is missing.
require_access() {
  "${TOOLS_VENV}/bin/python" - "$@" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
for spec in sys.argv[1:]:
    kind, repo = spec.split(':', 1)
    try:
        api.auth_check(repo, repo_type=kind)
    except Exception as error:
        raise SystemExit(f"No access to {kind} {repo}: if it is gated, accept its terms on huggingface.co "
                         f"with the account behind HF_TOKEN ({type(error).__name__})")
PY
}

if [[ "${mode}" == check ]]; then
  failed=0
  python=python3
  [[ -x "${TOOLS_VENV}/bin/python" ]] && python=${TOOLS_VENV}/bin/python
  "${python}" "${TASK_ROOT}/prepare_data.py" --check "$@" || failed=1
  for image in "${TASK_IMAGE}" "${AGENT_IMAGE}" "${RL_IMAGE}" "${SANDBOX_IMAGE}"; do
    docker image inspect "${image}" >/dev/null 2>&1 || { echo "FAIL missing image ${image}" >&2; failed=1; }
  done
  # Grading inputs must match their manifests before an attempt spends 12 hours.
  python3 - "${TASK_ROOT}" <<'PY' || failed=1
import hashlib
import json
from pathlib import Path
import sys

task = Path(sys.argv[1])
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()
try:
    rl = json.loads((task / 'private/rl-data/manifest.json').read_text())
    counts = json.loads((task / 'tests/rl_eval/recipe.json').read_text())['counts']
    if rl['counts'] != counts or digest(task / 'private/rl-data/train.jsonl') != rl['sha256']:
        raise ValueError('RL mixture differs from its manifest or the fixed recipe counts')
    evaluation = json.loads((task / 'private/eval-data/manifest.json').read_text())
    expected = json.loads((task / 'tests/rl_eval/eval_sources.json').read_text())
    suites = {name: spec for name, spec in expected.items() if name != 'seed'}
    if evaluation.get('sources') != expected or set(evaluation['datasets']) != set(suites):
        raise ValueError('Evaluation inputs differ from tests/rl_eval/eval_sources.json; prepare a fresh private evaluation directory')
    for name, entry in evaluation['datasets'].items():
        if (entry['items'], entry['samples']) != (suites[name]['items'], suites[name]['samples']):
            raise ValueError(f'{name}: wrong evaluation coverage')
        with (task / f'private/eval-data/{name}.jsonl').open() as stream:
            if sum(bool(line.strip()) for line in stream) != entry['items']:
                raise ValueError(f'{name}: wrong evaluation row count')
        if digest(task / f'private/eval-data/{name}.jsonl') != entry['sha256']:
            raise ValueError(f'{name} evaluation set differs from its manifest')
except (OSError, KeyError, ValueError) as error:
    raise SystemExit(f'FAIL private grading data: {error}')
print('Private RL mixture and post-RL evaluation sets validated')
PY
  if [[ -n "${MIDTRAIN_CODE_SANDBOX_URL:-}" ]]; then
    # Probe from a container, the way the evaluation stage reaches the operator's sandbox.
    docker run --rm --network bridge --add-host host.docker.internal:host-gateway \
      -e MIDTRAIN_CODE_SANDBOX_URL -e MIDTRAIN_CODE_SANDBOX_TOKEN "${RL_IMAGE}" python3 -c '
import json, os, urllib.request
request = urllib.request.Request(os.environ["MIDTRAIN_CODE_SANDBOX_URL"].rstrip("/") + "/health")
if os.environ.get("MIDTRAIN_CODE_SANDBOX_TOKEN"):
    request.add_header("Authorization", "Bearer " + os.environ["MIDTRAIN_CODE_SANDBOX_TOKEN"])
with urllib.request.urlopen(request, timeout=30) as response:
    assert json.load(response).get("status") == "ok"
' || { echo "FAIL code sandbox is unreachable at ${MIDTRAIN_CODE_SANDBOX_URL}" >&2; failed=1; }
  fi
  if (( failed )); then
    echo "midtrain-for-rl is not prepared; run bash ${TASK_ROOT}/prepare.sh" >&2
    exit 1
  fi
  echo "midtrain-for-rl data, images and grading inputs validated"
  exit 0
fi

if [[ "${mode}" == all || "${mode}" == data-only ]]; then
  tools
  require_access model:meta-llama/Llama-3.2-1B dataset:bigcode/starcoderdata
  "${TOOLS_VENV}/bin/python" "${TASK_ROOT}/prepare_data.py" "$@"
fi

if [[ "${mode}" == all || "${mode}" == images-only ]]; then
  for target in agent rl; do
    case "${target}" in
      agent) image=${AGENT_IMAGE} ;;
      rl) image=${RL_IMAGE} ;;
    esac
    docker build --platform linux/arm64 \
      --target "${target}" \
      --file "${TASK_ROOT}/tests/rl_eval/Dockerfile" \
      --tag "${image}" \
      "${TASK_ROOT}/tests/rl_eval"
  done
  # CPU-only image that the operator's sandbox.py runs generated code in.
  docker build --platform linux/arm64 \
    --file "${TASK_ROOT}/tests/rl_eval/Dockerfile.sandbox" \
    --tag "${SANDBOX_IMAGE}" \
    "${TASK_ROOT}/tests/rl_eval"
  # The task runtime image named in task.toml: the agent target, tagged through environment/Dockerfile.
  docker build --platform linux/arm64 --tag "${TASK_IMAGE}" "${TASK_ROOT}/environment"
fi

if [[ "${mode}" == all || "${mode}" == private-data ]]; then
  tools
  # Every pinned evaluation source must be readable (one is gated) before anything is downloaded.
  sources=$("${TOOLS_VENV}/bin/python" - "${TASK_ROOT}/tests/rl_eval/eval_sources.json" <<'PY'
import json
import sys
print(' '.join('dataset:' + spec['repo'] for spec in json.load(open(sys.argv[1])).values() if isinstance(spec, dict)))
PY
)
  read -ra eval_sources <<< "${sources}"
  require_access "${eval_sources[@]}"
  mkdir -p "${PRIVATE}" "${TASK_ROOT}/.downloads/huggingface"
  rl_runtime() {
    docker run --rm --platform linux/arm64 \
      -e HF_TOKEN -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 \
      --mount "type=bind,source=${TASK_ROOT}/artifacts/Llama-3.2-1B,target=/opt/base-model,readonly" \
      -v "${PRIVATE}:/private" \
      -v "${TASK_ROOT}/.downloads/huggingface:/cache/huggingface" \
      "${RL_IMAGE}" "$@"
  }
  # Each output is written once into a fresh directory, then reused.
  for name in rl-data eval-data; do
    if [[ -d "${PRIVATE}/${name}" && ! -f "${PRIVATE}/${name}/manifest.json" ]]; then
      echo "Incomplete ${PRIVATE}/${name} from an interrupted preparation; move it aside and run prepare.sh again" >&2
      exit 1
    fi
  done
  [[ -f "${PRIVATE}/rl-data/manifest.json" ]] || rl_runtime bash /opt/midtrain-rl/download_data.sh \
    --base /opt/base-model --output /private/rl-data --cache-dir /cache/huggingface
  [[ -f "${PRIVATE}/eval-data/manifest.json" ]] || rl_runtime python3 /opt/midtrain-rl/prepare_eval.py \
    --output /private/eval-data --cache-dir /cache/huggingface
fi

echo "midtrain-for-rl preparation complete"
