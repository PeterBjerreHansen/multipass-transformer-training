"""Small, dependency-light loaders and plotting helpers for result notebooks.

The notebooks deliberately keep figure construction visible.  This module only
normalizes the repository's JSON/JSONL artifacts and provides a few repeated
plotting primitives; it does not save figures or silently select runs.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


ARCHITECTURE_COLORS = {
    "transformer": "#6b7280",
    "memory_attention": "#e68613",
    "memory_add": "#2a9d8f",
    "latent_feedback": "#7c3aed",
    "sandwich_loop": "#2563eb",
}

INFERENCE_MODE_STYLES = {
    "recompute": "-",
    "append_recurrent": "--",
}

PRIMARY_METRICS = {
    "pointer_chasing": "exact_match",
    "tracking": "exact_match",
    "permutation": "exact_match",
    "state_machine": "exact_match",
    "othello": "token_legality",
    "shortest_path": "optimal_path",
    "maze": "optimal_route",
}

METRIC_LABELS = {
    "loss": "Evaluation NLL",
    "train_loss": "Training NLL",
    "exact_match": "Exact-match rate",
    "token_accuracy": "Token accuracy",
    "token_legality": "Legal-token fraction",
    "sequence_legality": "Fully legal sequence rate",
    "mean_legal_len": "Mean legal-prefix length",
    "optimal_path": "Optimal-path rate",
    "optimal_route": "Optimal-route rate",
    "exact_target_route": "Exact target-route rate",
    "optimal_path_short": "Optimal-path rate (short bucket)",
    "optimal_path_medium": "Optimal-path rate (medium bucket)",
    "optimal_path_long": "Optimal-path rate (long bucket)",
    **{
        f"path_step_{step}_accuracy": f"Path step {step} accuracy"
        for step in range(1, 11)
    },
    "legal_move_fraction": "Legal-move fraction",
    "legal_probability_mass": "Probability mass on legal moves",
    "legal_set_nll": "Legal-set NLL",
    "nll_delta": "Append minus recompute NLL",
}


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed process can leave one partial final line.  Earlier
                # complete events remain useful, so ignore only that line.
                continue
    return rows


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _config_context(run_dir: Path, *, root: Path | None = None) -> dict:
    config_path = run_dir / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    args = config.get("args", {})
    model_stats = config.get("model_stats", {})
    context = {
        "run_dir": str(run_dir),
        "relative_run": str(run_dir.relative_to(root)) if root and run_dir.is_relative_to(root) else str(run_dir),
        "task": args.get("task"),
        "architecture": args.get("architecture"),
        "preset": args.get("preset"),
        "seed": args.get("seed"),
        "device": args.get("device"),
        "max_passes": args.get("max_passes"),
        "inference_mode": args.get("inference_mode"),
        "max_level": args.get("max_level"),
        "curriculum_threshold": args.get("curriculum_threshold"),
        "shortest_path_distribution": args.get("shortest_path_distribution"),
        "shortest_path_data_dir": args.get("shortest_path_data_dir"),
        "shortest_path_dataset_id": args.get("shortest_path_dataset_id"),
        "maze_data_dir": args.get("maze_data_dir"),
        "maze_route_policy": args.get("maze_route_policy"),
        "total_parameters": model_stats.get("total_parameters"),
        "non_embedding_parameters": model_stats.get("non_embedding_parameters"),
    }
    return context


def load_training_records(root: str | Path) -> list[dict]:
    """Load one flat record per evaluation event below ``root``."""
    root = Path(root).expanduser().resolve()
    records = []
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        context = _config_context(run_dir, root=root)
        events = read_jsonl(metrics_path)
        for event in events:
            if event.get("event") != "eval":
                continue
            record = {
                **context,
                "step": int(event.get("step", 0)),
                "level": event.get("level"),
                "sampled_train_level": event.get("sampled_train_level"),
                "train_loss": _finite_number(event.get("train_loss")),
                "train_tok_per_s": _finite_number(event.get("train_tok_per_s")),
            }
            for key, value in event.get("metrics", {}).items():
                number = _finite_number(value)
                if number is not None:
                    record[key] = number
            for index, value in enumerate(event.get("pass_losses", []), start=1):
                number = _finite_number(value)
                if number is not None:
                    record[f"pass_{index}_loss"] = number
            for group, summary in (event.get("gradient_norms") or {}).items():
                for statistic in ("mean", "max"):
                    number = _finite_number(summary.get(statistic))
                    if number is not None:
                        record[f"gradient_{group}_{statistic}"] = number
            for key, value in (event.get("resource_stats") or {}).items():
                number = _finite_number(value)
                if number is not None:
                    record[key] = number
            records.append(record)
    return records


def load_drift_records(root: str | Path) -> list[dict]:
    """Load post-training trace-evaluation summaries."""
    root = Path(root).expanduser().resolve()
    records = []
    for summary_path in sorted(root.rglob("summary.json")):
        payload = read_json(summary_path)
        if "metrics" not in payload or "inference_mode" not in payload:
            continue
        input_run = Path(payload.get("input_run_dir", summary_path.parent.parent.parent))
        context = _config_context(input_run, root=root) if input_run.exists() else {}
        record = {
            **context,
            "summary_path": str(summary_path),
            "task": payload.get("task", context.get("task")),
            "architecture": payload.get("architecture", context.get("architecture")),
            "inference_mode": payload.get("effective_inference_mode") or payload.get("inference_mode"),
            "requested_inference_mode": payload.get("inference_mode"),
            "token_selection": payload.get("token_selection"),
            "evaluation_examples": payload.get("evaluation_examples"),
        }
        for key, value in payload.get("metrics", {}).items():
            number = _finite_number(value)
            if number is not None:
                record[key] = number
        records.append(record)
    return records


def load_othello_examples(root: str | Path) -> list[dict]:
    """Load Othello continuation rows emitted by ``eval_othello_prefix``."""
    root = Path(root).expanduser().resolve()
    records = []
    for path in sorted(root.rglob("per_example.jsonl")):
        summary_path = path.parent / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        if summary.get("task") != "othello":
            continue
        input_run = Path(summary.get("input_run_dir", path.parent))
        context = _config_context(input_run, root=root) if input_run.exists() else {}
        for row in read_jsonl(path):
            record = {
                **context,
                "evaluation_dir": str(path.parent),
                "example_index": row.get("example_index"),
                "protocol": row.get("protocol"),
                "inference_mode": row.get("inference_mode"),
                "prompt_moves": row.get("prompt_moves"),
                "prompt_bucket": row.get("prompt_bucket"),
                "reference_suffix_moves": row.get("reference_suffix_moves"),
                "suffix_bucket": row.get("suffix_bucket"),
            }
            for namespace in ("free_generation", "teacher_forced"):
                for key, value in row.get(namespace, {}).items():
                    number = _finite_number(value)
                    if number is not None:
                        record[f"{namespace}.{key}"] = number
            records.append(record)
    return records


def _flatten_numbers(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numbers(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, list):
        return
    else:
        number = _finite_number(value)
        if number is not None:
            output[prefix] = number


def load_diagnostic_records(root: str | Path) -> list[dict]:
    """Load diagnostic summaries while retaining position/pass arrays."""
    root = Path(root).expanduser().resolve()
    records = []
    for path in sorted(root.rglob("diagnostics.json")):
        payload = read_json(path)
        if "memory_interventions" not in payload or "pass_dynamics" not in payload:
            continue
        input_run = Path(payload.get("input_run_dir", path.parent))
        context = _config_context(input_run, root=root) if input_run.exists() else {}
        numeric: dict[str, float] = {}
        _flatten_numbers("", payload, numeric)
        records.append(
            {
                **context,
                **numeric,
                "diagnostics_path": str(path),
                "payload": payload,
            }
        )
    return records


def load_ablation_rows(root: str | Path) -> list[dict]:
    """Load the per-seed table written by ``summarize_ablation``."""
    path = Path(root).expanduser().resolve()
    if path.is_dir():
        path = path / "per_seed.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                if key in {"variant", "seed", "run_dir"}:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def discover_ablation_roots(root: str | Path) -> list[Path]:
    """Return directories containing both per-seed and aggregate summaries."""
    root = Path(root).expanduser().resolve()
    candidates = []
    for summary_path in root.rglob("summary.json"):
        if (summary_path.parent / "per_seed.csv").exists():
            payload = read_json(summary_path)
            if "control" in payload and "variants" in payload:
                candidates.append(summary_path.parent)
    return sorted(candidates, key=lambda path: path.stat().st_mtime)


def filter_records(records: Iterable[dict], **criteria) -> list[dict]:
    return [
        record
        for record in records
        if all(value is None or record.get(key) == value for key, value in criteria.items())
    ]


def unique_values(records: Iterable[dict], key: str) -> list:
    return sorted({record.get(key) for record in records if record.get(key) is not None}, key=str)


def primary_metric(task: str | None) -> str:
    return PRIMARY_METRICS.get(str(task), "exact_match")


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric.split(".")[-1], metric.replace("_", " ").title())


def grouped(records: Iterable[dict], *keys: str) -> dict[tuple, list[dict]]:
    result: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        result[tuple(record.get(key) for key in keys)].append(record)
    return dict(result)


def median_curve(records: Sequence[dict], metric: str) -> tuple[list[float], list[float]]:
    """Median at each observed step; no interpolation or smoothing."""
    by_step: dict[float, list[float]] = defaultdict(list)
    for record in records:
        step = _finite_number(record.get("step"))
        value = _finite_number(record.get(metric))
        if step is not None and value is not None:
            by_step[step].append(value)
    steps = sorted(by_step)
    return steps, [float(median(by_step[step])) for step in steps]


def summarize_curriculum_levels(records: Sequence[dict]) -> list[dict]:
    """Summarize level entry, mastery, and censoring for BBH curriculum runs.

    A level is mastered at its first evaluation meeting the run's curriculum
    threshold. The time spent mastering a level is measured from the previous
    mastered level, avoiding the misleading loss/accuracy discontinuities that
    occur when curriculum difficulty changes.
    """
    summaries = []
    for (run_dir,), run_rows in grouped(records, "run_dir").items():
        ordered = sorted(
            (
                row for row in run_rows
                if _finite_number(row.get("step")) is not None
                and _finite_number(row.get("level")) is not None
            ),
            key=lambda row: float(row["step"]),
        )
        if not ordered:
            continue
        threshold = _finite_number(ordered[0].get("curriculum_threshold"))
        threshold = 0.95 if threshold is None else threshold
        levels = sorted({int(row["level"]) for row in ordered})
        previous_mastery_step = 0.0
        final_level = levels[-1]
        for level in levels:
            level_rows = [row for row in ordered if int(row["level"]) == level]
            mastery_rows = [
                row for row in level_rows
                if (_finite_number(row.get("exact_match")) or 0.0) >= threshold
            ]
            mastery_step = float(mastery_rows[0]["step"]) if mastery_rows else None
            final_row = level_rows[-1]
            summary = {
                **{
                    key: ordered[0].get(key)
                    for key in (
                        "run_dir",
                        "relative_run",
                        "task",
                        "architecture",
                        "preset",
                        "seed",
                        "device",
                        "max_level",
                        "curriculum_threshold",
                    )
                },
                "level": level,
                "entry_step": previous_mastery_step,
                "last_observed_step": float(final_row["step"]),
                "mastery_step": mastery_step,
                "steps_to_mastery": (
                    mastery_step - previous_mastery_step
                    if mastery_step is not None
                    else None
                ),
                "final_exact_match": _finite_number(final_row.get("exact_match")),
                "peak_exact_match": max(
                    (
                        value
                        for row in level_rows
                        if (value := _finite_number(row.get("exact_match"))) is not None
                    ),
                    default=None,
                ),
                "mastered": mastery_step is not None,
                "censored": level == final_level and mastery_step is None,
            }
            summaries.append(summary)
            if mastery_step is not None:
                previous_mastery_step = mastery_step
    return summaries


def plot_seed_and_median_curves(
    ax,
    records: Sequence[dict],
    *,
    metric: str,
    label_key: str = "architecture",
    seed_key: str = "seed",
) -> None:
    """Plot raw seed trajectories faintly and an unsmoothed median prominently."""
    for (label,), label_rows in grouped(records, label_key).items():
        if label is None:
            continue
        color = ARCHITECTURE_COLORS.get(str(label))
        for (_seed,), seed_rows in grouped(label_rows, seed_key).items():
            points = sorted(
                (
                    (float(row["step"]), float(row[metric]))
                    for row in seed_rows
                    if _finite_number(row.get("step")) is not None
                    and _finite_number(row.get(metric)) is not None
                ),
                key=lambda pair: pair[0],
            )
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    alpha=0.18,
                    linewidth=1,
                )
        steps, values = median_curve(label_rows, metric)
        if steps:
            single_point_style = {"marker": "o", "markersize": 4} if len(steps) == 1 else {}
            ax.plot(
                steps,
                values,
                color=color,
                linewidth=2.4,
                label=str(label),
                **single_point_style,
            )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(metric_label(metric))


def paired_values(
    rows: Sequence[dict],
    *,
    control: str,
    treatment: str,
    metric: str,
) -> list[tuple[str, float, float]]:
    by_variant_seed = {
        (str(row.get("variant")), str(row.get("seed"))): _finite_number(row.get(metric))
        for row in rows
    }
    pairs = []
    seeds = sorted(
        {
            seed
            for variant, seed in by_variant_seed
            if variant in {control, treatment}
        }
    )
    for seed in seeds:
        left = by_variant_seed.get((control, seed))
        right = by_variant_seed.get((treatment, seed))
        if left is not None and right is not None:
            pairs.append((seed, left, right))
    return pairs


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 4.8),
            "figure.dpi": 120,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
