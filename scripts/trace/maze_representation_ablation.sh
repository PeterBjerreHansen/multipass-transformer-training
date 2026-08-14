#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/maze_matrix.sh"

# Hold topology, route policy, seeds, and optimizer fixed while varying the
# prompt and target representation. Override ARCHITECTURES after the baseline
# identifies the architectures worth carrying into this larger matrix.
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape}"
for representation in \
  "sparse-cells cell-path" \
  "dense-cells cell-path" \
  "sparse-cells actions" \
  "dense-cells actions"; do
  read -r input_representation target_representation <<< "${representation}"
  run_maze_matrix representation_ablation "${input_representation}" \
    "${target_representation}" astar "${ARCHITECTURES}"
done
