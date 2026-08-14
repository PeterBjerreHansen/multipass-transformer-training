#!/usr/bin/env bash

# Shared launcher for frozen maze experiment matrices. Callers must enable
# `set -euo pipefail` and set ROOT before sourcing this file.

run_maze_matrix() {
  if [[ $# -ne 5 ]]; then
    printf 'run_maze_matrix expects: condition input target policy architectures\n' >&2
    return 2
  fi

  local condition="$1"
  local input_representation="$2"
  local target_representation="$3"
  local route_policy="$4"
  local architectures="$5"
  local result_root="${RESULT_ROOT:-results/trace/maze}"
  local maze_data_dir="${MAZE_DATA_DIR:-data/maze/searchformer-10}"
  local seeds="${SEEDS:-1337 2027 4099}"
  local runtime_args=()
  [[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

  read -r -a architecture_matrix <<< "${architectures}"
  read -r -a seed_matrix <<< "${seeds}"
  source "${ROOT}/scripts/lib/model_matrix.sh"
  validate_architecture_matrix "${architecture_matrix[@]}"

  for architecture in "${architecture_matrix[@]}"; do
    for seed in "${seed_matrix[@]}"; do
      local run_dir="${result_root}/${condition}/${input_representation}__${target_representation}__${route_policy}/${architecture}/seed_${seed}"
      python -m experiments.train_trace \
        --preset maze_main \
        --architecture "${architecture}" \
        --seed "${seed}" \
        --maze-data-dir "${maze_data_dir}" \
        --maze-input-representation "${input_representation}" \
        --maze-target-representation "${target_representation}" \
        --maze-route-policy "${route_policy}" \
        --run-dir "${run_dir}" \
        "${runtime_args[@]+"${runtime_args[@]}"}"

      RUN_DIR="${run_dir}" \
        OUTPUT_DIR="${run_dir}/eval" \
        bash "${ROOT}/scripts/trace/eval.sh"
    done
  done
}
