#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RESULT_ROOT="${RESULT_ROOT:-$(mktemp -d /tmp/mpt-shortest-path.XXXXXX)}"
CPU_RUN="${RESULT_ROOT}/cpu"

pytest -q

python -m experiments.train_trace \
  --preset shortest_path_smoke \
  --architecture memory_attention \
  --device cpu \
  --run-dir "${CPU_RUN}"

python -m experiments.train_trace \
  --preset shortest_path_smoke \
  --resume-from "${CPU_RUN}" \
  --train-steps 1 \
  --device cpu \
  --run-dir "${CPU_RUN}"

for inference_mode in recompute append_recurrent; do
  python -m experiments.eval_trace \
    --input-run-dir "${CPU_RUN}" \
    --inference-mode "${inference_mode}" \
    --token-selection argmax \
    --device cpu \
    --eval-batches 1 \
    --output-dir "${CPU_RUN}/eval/${inference_mode}"
done

python -m experiments.diagnose_memory \
  --input-run-dir "${CPU_RUN}" \
  --device cpu \
  --batch-size 2 \
  --eval-batches 1 \
  --extra-passes 2 \
  --schedule-gap-horizon 4 \
  --output "${CPU_RUN}/diagnostics.json"

if python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture memory_attention \
    --device mps \
    --run-dir "${RESULT_ROOT}/mps"
fi

git diff --check
