#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to a trained trace run directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval}"
CHECKPOINT="${CHECKPOINT:-best}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
OTHELLO_EXAMPLES="${OTHELLO_EXAMPLES:-64}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

read -r task architecture < <(
  python -c \
    'import json, sys; config = json.load(open(sys.argv[1])); args = config["args"]; print(args["task"], args["architecture"])' \
    "${RUN_DIR}/config.json"
)

inference_modes=(recompute)
if [[ "${architecture}" != "transformer" ]]; then
  inference_modes+=(append_recurrent)
fi

case "${task}" in
  maze|shortest_path)
    for inference_mode in "${inference_modes[@]}"; do
      python -m experiments.eval_trace \
        --input-run-dir "${RUN_DIR}" \
        --output-dir "${OUTPUT_DIR}/${inference_mode}" \
        --checkpoint "${CHECKPOINT}" \
        --eval-batches "${EVAL_BATCHES}" \
        --inference-mode "${inference_mode}" \
        --token-selection argmax \
        "${runtime_args[@]+"${runtime_args[@]}"}"
    done
    ;;
  othello)
    python -m experiments.eval_othello_prefix \
      --input-run-dir "${RUN_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --examples "${OTHELLO_EXAMPLES}" \
      --evaluation-mode all \
      --inference-modes "${inference_modes[@]}" \
      --token-selection argmax \
      "${runtime_args[@]+"${runtime_args[@]}"}"
    ;;
  *)
    echo "unsupported trace task in ${RUN_DIR}/config.json: ${task}" >&2
    exit 2
    ;;
esac
