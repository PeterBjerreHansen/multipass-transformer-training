#!/usr/bin/env bash

# Shared runner for short trace-ablation pilots. Callers are responsible for
# enabling `set -euo pipefail` before sourcing this file.

local_pilot_device() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' "${DEVICE}"
    return
  fi
  python -c 'from experiments.common import auto_device; print(auto_device())'
}

run_trace_pilot_variant() {
  if [[ $# -lt 1 ]]; then
    printf 'run_trace_pilot_variant requires a variant name\n' >&2
    return 2
  fi

  local variant="$1"
  shift
  local preset="${PILOT_PRESET:-shortest_path_main}"
  local architecture="${PILOT_ARCHITECTURE:-memory_attention}"
  local seed="${SEED:-1337}"
  local device
  device="$(local_pilot_device)"
  local train_steps="${TRAIN_STEPS:-250}"
  local eval_interval="${EVAL_INTERVAL:-${train_steps}}"
  local train_eval_batches="${TRAIN_EVAL_BATCHES:-${EVAL_BATCHES:-1}}"
  local final_eval_batches="${FINAL_EVAL_BATCHES:-1}"
  local diagnostic_batches="${DIAGNOSTIC_BATCHES:-1}"
  local batch_size="${BATCH_SIZE:-16}"
  local result_root="${RESULT_ROOT:-results/local_pilots}"
  local run_dir="${result_root}/${variant}/seed_${seed}"

  local schedule_args=()
  if (( train_steps > 1 )); then
    local warmup_steps="${LR_WARMUP_STEPS:-$((train_steps / 50))}"
    if (( warmup_steps < 1 )); then
      warmup_steps=1
    fi
    if (( warmup_steps >= train_steps )); then
      warmup_steps=$((train_steps - 1))
    fi
    schedule_args=(
      --lr-schedule warmup_cosine
      --lr-warmup-steps "${warmup_steps}"
      --lr-decay-steps "${train_steps}"
    )
  else
    schedule_args=(--lr-schedule constant)
  fi

  python -m experiments.train_trace \
    --preset "${preset}" \
    --architecture "${architecture}" \
    --token-selection argmax \
    --train-steps "${train_steps}" \
    --eval-interval "${eval_interval}" \
    --eval-batches "${train_eval_batches}" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --device "${device}" \
    --run-dir "${run_dir}" \
    "${schedule_args[@]}" \
    "$@"

  local modes=(recompute append_recurrent)
  if [[ "${architecture}" == "transformer" ]]; then
    modes=(recompute)
  fi
  for inference_mode in "${modes[@]}"; do
    python -m experiments.eval_trace \
      --input-run-dir "${run_dir}" \
      --output-dir "${run_dir}/drift/${inference_mode}" \
      --inference-mode "${inference_mode}" \
      --token-selection argmax \
      --device "${device}" \
      --eval-batches "${final_eval_batches}" \
      --seed "${seed}"
  done

  if [[ "${RUN_DIAGNOSTICS:-1}" == "1" && "${architecture}" != "transformer" ]]; then
    local diagnostic_batch_size="${batch_size}"
    if (( diagnostic_batch_size < 2 )); then
      diagnostic_batch_size=2
    fi
    python -m experiments.diagnose_memory \
      --input-run-dir "${run_dir}" \
      --device "${device}" \
      --batch-size "${diagnostic_batch_size}" \
      --eval-batches "${diagnostic_batches}" \
      --seed "${seed}" \
      --output "${run_dir}/diagnostics.json"
  fi
}
