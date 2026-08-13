from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

from experiments.common import write_json


QUALITY_MARGIN = 0.01
THROUGHPUT_WIN = 0.10
SIZE_WIN = 0.25
TASK_QUALITY_METRICS = {
    "maze": "drift.append_recurrent.optimal_route",
    "shortest_path": "drift.append_recurrent.optimal_path",
    "othello": "drift.append_recurrent.sequence_legality",
}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Summarize paired ablation runs by seed.", allow_abbrev=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--recommendation-mode",
        choices=["pareto", "quality-only"],
        default="pareto",
    )
    parser.add_argument(
        "--quality-metric",
        default=None,
        help=(
            "Flattened metric used for paired quality deltas. By default it "
            "is inferred from the saved task."
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _flatten(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
        return
    number = _number(value)
    if number is not None:
        output[prefix] = number


def _last_eval(events: Iterable[dict]) -> dict:
    evaluations = [event for event in events if event.get("event") == "eval"]
    return evaluations[-1] if evaluations else {}


def collect_run(run_dir: Path) -> dict[str, float | str]:
    config = _read_json(run_dir / "config.json") or {}
    events = _read_jsonl(run_dir / "metrics.jsonl")
    final_eval = _last_eval(events)
    diagnostics = _read_json(run_dir / "diagnostics.json") or {}
    saved_args = config.get("args", {})
    result: dict[str, float | str] = {
        "run_dir": str(run_dir),
        "task": str(saved_args.get("task", "")),
        "architecture": str(saved_args.get("architecture", "")),
    }

    numeric: dict[str, float] = {}
    _flatten("model", config.get("model_stats", {}), numeric)
    _flatten("model_config", config.get("model_config", {}), numeric)
    _flatten("train", {key: final_eval.get(key) for key in ("step", "train_loss", "train_tok_per_s")}, numeric)
    _flatten("eval", final_eval.get("metrics", {}), numeric)
    _flatten("resource", final_eval.get("resource_stats", {}), numeric)
    _flatten("diagnostics", diagnostics, numeric)

    for mode in ("recompute", "append_recurrent"):
        drift = _read_json(run_dir / "drift" / mode / "summary.json") or {}
        _flatten(f"drift.{mode}", drift.get("metrics", {}), numeric)

    result.update(numeric)
    return result


def discover_variant(root: Path, variant: str) -> dict[str, dict[str, float | str]]:
    variant_dir = root / variant
    return {
        path.name.removeprefix("seed_"): collect_run(path)
        for path in sorted(variant_dir.glob("seed_*"))
        if path.is_dir()
    }


def _median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(median(finite)) if finite else None


def _paired_delta(
    control: dict[str, dict[str, float | str]],
    treatment: dict[str, dict[str, float | str]],
    metric: str,
) -> list[float]:
    values = []
    for seed in sorted(set(control) & set(treatment)):
        left = _number(control[seed].get(metric))
        right = _number(treatment[seed].get(metric))
        if left is not None and right is not None:
            values.append(right - left)
    return values


def _median_ratio(
    control: dict[str, dict[str, float | str]],
    treatment: dict[str, dict[str, float | str]],
    metric: str,
) -> float | None:
    ratios = []
    for seed in sorted(set(control) & set(treatment)):
        left = _number(control[seed].get(metric))
        right = _number(treatment[seed].get(metric))
        if left is not None and right is not None and left > 0:
            ratios.append(right / left)
    return _median(ratios)


def infer_quality_metric(
    control: dict[str, dict[str, float | str]],
) -> str:
    tasks = {
        str(run.get("task"))
        for run in control.values()
        if run.get("task")
    }
    if len(tasks) > 1:
        raise ValueError(
            "control runs contain multiple tasks; pass --quality-metric "
            "explicitly"
        )
    task = next(iter(tasks), None)
    if task not in TASK_QUALITY_METRICS:
        raise ValueError(
            f"no default quality metric for task {task!r}; pass "
            "--quality-metric explicitly"
        )
    return TASK_QUALITY_METRICS[task]


def recommend(
    control: dict[str, dict[str, float | str]],
    treatment: dict[str, dict[str, float | str]],
    *,
    mode: str,
    quality_metric: str,
) -> dict:
    deltas = _paired_delta(control, treatment, quality_metric)
    median_delta = _median(deltas)
    quality_win = bool(
        median_delta is not None
        and median_delta >= QUALITY_MARGIN
        and sum(delta > 0 for delta in deltas) >= 2
    )
    noninferior = bool(
        median_delta is not None
        and median_delta >= -QUALITY_MARGIN
        and sum(delta >= -QUALITY_MARGIN for delta in deltas) >= 2
    )

    train_ratio = _median_ratio(control, treatment, "train.train_tok_per_s")
    eval_ratio = _median_ratio(control, treatment, "drift.append_recurrent.eval_output_tok_per_s")
    parameter_ratio = _median_ratio(control, treatment, "model.non_embedding_parameters")
    tape_ratio = _median_ratio(control, treatment, "model_config.memory_bytes_per_token")
    efficiency_win = bool(
        (train_ratio is not None and train_ratio >= 1.0 + THROUGHPUT_WIN)
        or (eval_ratio is not None and eval_ratio >= 1.0 + THROUGHPUT_WIN)
        or (parameter_ratio is not None and parameter_ratio <= 1.0 - SIZE_WIN)
        or (tape_ratio is not None and tape_ratio <= 1.0 - SIZE_WIN)
    )

    eligible = quality_win if mode == "quality-only" else quality_win or (noninferior and efficiency_win)

    return {
        "recommend_merge": eligible,
        "quality_metric": quality_metric,
        "paired_quality_deltas": deltas,
        "median_quality_delta": median_delta,
        "quality_win": quality_win,
        "quality_noninferior": noninferior,
        "efficiency_win": efficiency_win,
        "median_train_throughput_ratio": train_ratio,
        "median_append_eval_throughput_ratio": eval_ratio,
        "median_parameter_ratio": parameter_ratio,
        "median_tape_bytes_ratio": tape_ratio,
    }


def summarize(
    root: Path,
    control_name: str,
    variants: list[str],
    *,
    mode: str,
    quality_metric: str | None = None,
) -> tuple[list[dict], dict]:
    runs = {name: discover_variant(root, name) for name in [control_name, *variants]}
    control = runs[control_name]
    resolved_quality_metric = (
        quality_metric
        if quality_metric is not None
        else infer_quality_metric(control)
    )
    rows = []
    for variant, per_seed in runs.items():
        for seed, values in per_seed.items():
            rows.append({"variant": variant, "seed": seed, **values})
    summary = {
        "root": str(root),
        "control": control_name,
        "quality_metric": resolved_quality_metric,
        "variants": {
            variant: recommend(
                control,
                runs[variant],
                mode=mode,
                quality_metric=resolved_quality_metric,
            )
            for variant in variants
        },
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    rows, summary = summarize(
        root,
        args.control,
        args.variants,
        mode=args.recommendation_mode,
        quality_metric=args.quality_metric,
    )
    write_csv(output_dir / "per_seed.csv", rows)
    write_json(output_dir / "summary.json", summary)
    print(f"wrote {output_dir / 'per_seed.csv'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
