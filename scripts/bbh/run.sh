#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/model_matrix.sh"

# task options: permutation tracking pointer_chasing state_machine
TASKS="${TASKS:-permutation tracking pointer_chasing state_machine}"
SEEDS="${SEEDS:-${SEED:-1337 2027}}"
# architecture options: transformer memory_tape memory_add
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape memory_add}" # default
RESULT_ROOT="${RESULT_ROOT:-results/bbh}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

read -r -a task_matrix <<< "${TASKS}"
read -r -a architecture_matrix <<< "${ARCHITECTURES}"
read -r -a seed_matrix <<< "${SEEDS}"
validate_architecture_matrix "${architecture_matrix[@]}"

for task in "${task_matrix[@]}"; do
  for ARCH in "${architecture_matrix[@]}"; do
    for seed in "${seed_matrix[@]}"; do
      python -m experiments.train_bbh \
        --preset "${task}_main" \
        --architecture "${ARCH}" \
        --seed "${seed}" \
        --run-dir "${RESULT_ROOT}/${task}/${ARCH}/seed_${seed}" \
        "${runtime_args[@]+"${runtime_args[@]}"}"
    done
  done
done
