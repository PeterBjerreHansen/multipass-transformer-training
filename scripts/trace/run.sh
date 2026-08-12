#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/model_matrix.sh"

# task options: maze othello shortest_path
TASKS="${TASKS:-shortest_path}"
SEEDS="${SEEDS:-1337}"
# architecture options: transformer memory_tape memory_add
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape memory_add}" # default
RESULT_ROOT="${RESULT_ROOT:-results/trace}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

read -r -a task_matrix <<< "${TASKS}"
read -r -a architecture_matrix <<< "${ARCHITECTURES}"
read -r -a seed_matrix <<< "${SEEDS}"
validate_architecture_matrix "${architecture_matrix[@]}"

for task in "${task_matrix[@]}"; do
  if [[ "${task}" != "maze" && "${task}" != "shortest_path" && "${task}" != "othello" ]]; then
    echo "invalid trace task: ${task}" >&2
    echo "valid trace tasks: maze shortest_path othello" >&2
    exit 2
  fi
done

for task in "${task_matrix[@]}"; do
  for architecture in "${architecture_matrix[@]}"; do
    for seed in "${seed_matrix[@]}"; do
      python -m experiments.train_trace \
        --preset "${task}_main" \
        --architecture "${architecture}" \
        --seed "${seed}" \
        --run-dir "${RESULT_ROOT}/${task}/main/${architecture}/seed_${seed}" \
        "${runtime_args[@]+"${runtime_args[@]}"}"
    done
  done
done
