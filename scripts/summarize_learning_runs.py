#!/usr/bin/env python3
"""Summarize whether local training pilots exhibit a real learning signal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


KEY_METRICS = (
    "token_legality",
    "sequence_legality",
    "optimal_path",
    "optimal_route",
    "exact_target_route",
    "exact_match",
    "token_accuracy",
)
QUALIFICATION_METRICS = ("loss", *KEY_METRICS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize first/final evaluation metrics below a result root."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-relative-loss-drop", type=float, default=0.02)
    return parser.parse_args()


def load_eval_events(path: Path) -> list[dict]:
    events = []
    with path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "eval":
                events.append(payload)
    return events


def load_qualification_metrics(run_dir: Path) -> dict[str, dict]:
    qualification = {}
    for path in sorted((run_dir / "drift").glob("*/summary.json")):
        payload = json.loads(path.read_text())
        mode = payload.get("effective_inference_mode") or payload.get("inference_mode")
        if not mode:
            continue
        metrics = payload.get("metrics", {})
        qualification[str(mode)] = {
            "evaluation_examples": payload.get("evaluation_examples"),
            "metrics": {
                key: float(metrics[key])
                for key in QUALIFICATION_METRICS
                if key in metrics
            },
        }
    return qualification


def summarize_run(path: Path) -> tuple[dict, list[str]]:
    events = load_eval_events(path)
    failures = []
    if len(events) < 2:
        return (
            {"run_dir": str(path.parent), "eval_points": len(events)},
            [f"{path.parent}: expected at least two evaluation points"],
        )

    first = events[0]
    final = events[-1]
    first_metrics = first.get("metrics", {})
    final_metrics = final.get("metrics", {})
    first_loss = float(first_metrics["loss"])
    final_loss = float(final_metrics["loss"])
    if not math.isfinite(first_loss) or not math.isfinite(final_loss):
        failures.append(f"{path.parent}: non-finite evaluation loss")
    relative_drop = (first_loss - final_loss) / max(abs(first_loss), 1e-12)

    config = {}
    config_path = path.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text()).get("args", {})

    summary = {
        "run_dir": str(path.parent),
        "task": config.get("task"),
        "architecture": config.get("architecture"),
        "seed": config.get("seed"),
        "first_step": first.get("step"),
        "final_step": final.get("step"),
        "eval_points": len(events),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "relative_loss_drop": relative_drop,
        "final_metrics": {
            key: float(final_metrics[key])
            for key in KEY_METRICS
            if key in final_metrics
        },
        "qualification": load_qualification_metrics(path.parent),
    }
    return summary, failures


def main() -> None:
    args = parse_args()
    metrics_paths = sorted(args.root.rglob("metrics.jsonl"))
    if not metrics_paths:
        raise SystemExit(f"no metrics.jsonl files found below {args.root}")

    summaries = []
    failures = []
    for path in metrics_paths:
        summary, run_failures = summarize_run(path)
        summary["variant"] = str(path.parent.relative_to(args.root))
        summaries.append(summary)
        failures.extend(run_failures)
        drop = summary.get("relative_loss_drop")
        if args.strict and drop is not None and drop < args.min_relative_loss_drop:
            failures.append(
                f"{summary['run_dir']}: relative loss drop {drop:.3f} is below "
                f"{args.min_relative_loss_drop:.3f}"
            )

    output = {
        "root": str(args.root),
        "min_relative_loss_drop": args.min_relative_loss_drop,
        "strict": args.strict,
        "runs": summaries,
        "failures": failures,
    }
    output_path = args.root / "learning_summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for summary in summaries:
        if "relative_loss_drop" not in summary:
            print(f"{summary['run_dir']}: only {summary['eval_points']} eval point(s)")
            continue
        metrics = " ".join(
            f"{key}={value:.3f}" for key, value in summary["final_metrics"].items()
        )
        print(
            f"{summary['variant']} "
            f"loss={summary['first_loss']:.4f}->{summary['final_loss']:.4f} "
            f"drop={summary['relative_loss_drop']:.1%} {metrics}".rstrip()
        )
        for mode, qualification in summary["qualification"].items():
            count = qualification["evaluation_examples"]
            count_text = "?" if count is None else str(count)
            qualification_metrics = " ".join(
                f"{key}={value:.3f}"
                for key, value in qualification["metrics"].items()
                if key != "loss"
            )
            print(
                f"  qualification[{mode}] n={count_text} "
                f"{qualification_metrics}".rstrip()
            )
    print(f"summary: {output_path}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
