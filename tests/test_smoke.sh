#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RESULT_ROOT="$(mktemp -d /tmp/mpt-smoke.XXXXXX)"
trap 'rm -rf "${RESULT_ROOT}"' EXIT

for ARCH in memory_attention memory_add latent_feedback; do
  python -m experiments.train_bbh \
    --preset pointer_chasing_smoke \
    --architecture "${ARCH}" \
    --run-dir "${RESULT_ROOT}/bbh/${ARCH}" \
    --device cpu

  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture "${ARCH}" \
    --run-dir "${RESULT_ROOT}/trace/${ARCH}" \
    --device cpu
done
