#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/maze_matrix.sh"

# Compare deterministic A* imitation with uniformly sampled shortest routes.
# DFS is intentionally excluded until a separate valid-target metric exists.
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape memory_add latent_feedback}"
for route_policy in astar uniform_shortest; do
  run_maze_matrix policy_ablation sparse-cells cell-path "${route_policy}" \
    "${ARCHITECTURES}"
done
