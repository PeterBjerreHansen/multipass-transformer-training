#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/maze_matrix.sh"

# Primary architecture comparison: fixed Searchformer-style 10x10 mazes.
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape memory_add latent_feedback}"
run_maze_matrix baseline sparse-cells cell-path astar "${ARCHITECTURES}"
