"""Evaluate generation quality for a saved trace-task checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    load_checkpoint_payload,
    resolve_device_arg,
    resolve_evaluation_checkpoint,
    restore_checkpoint_state,
    saved_args_from_run,
    set_seed,
    stable_seed,
    validate_model_args,
    validate_training_args,
    write_json,
)
from experiments.train_trace import (
    build_fixed_eval_batches,
    build_training_objects,
    format_trace_metrics,
    trace_generation_metrics,
    validate_task_args,
)
from tasks.trace.registry import get_trace_task


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate generation for a saved trace-task run.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        choices=EVALUATION_CHECKPOINTS,
        default="best",
        help="Saved checkpoint to evaluate (default: best validation loss).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--shortest-path-data-dir", default=None)
    parser.add_argument("--maze-data-dir", default=None)
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--inference-mode", choices=["recompute", "append_recurrent"], required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def _load_eval_args(cli_args) -> tuple[SimpleNamespace, Path]:
    input_dir = Path(cli_args.input_run_dir).resolve()
    saved = saved_args_from_run(input_dir)
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    if cli_args.eval_batches is not None:
        saved["eval_batches"] = cli_args.eval_batches
    for name in ("shortest_path_data_dir", "maze_data_dir"):
        value = getattr(cli_args, name)
        if value is not None:
            saved[name] = value
    saved["token_selection"] = cli_args.token_selection
    saved["inference_mode"] = cli_args.inference_mode
    saved["seed"] = cli_args.seed
    saved["run_dir"] = str(input_dir)
    saved["resume_from"] = str(input_dir)
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    return args, input_dir


def _default_output_dir(cli_args, args, input_dir: Path) -> Path:
    if cli_args.output_dir:
        return Path(cli_args.output_dir).resolve()
    name = f"{args.architecture}_{cli_args.inference_mode}_{cli_args.token_selection}"
    return Path("results", "eval", args.task, name, input_dir.name).resolve()


def evaluate_run(cli_args) -> Path:
    args, input_dir = _load_eval_args(cli_args)
    output_dir = _default_output_dir(cli_args, args, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    checkpoint_path = resolve_evaluation_checkpoint(
        input_dir,
        cli_args.checkpoint,
    )
    checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")
    block_size, vocab, stoi, _itos, model, _optimizer = build_training_objects(args)
    restore_checkpoint_state(checkpoint, model=model, optimizer=None, device=args.device)
    task = get_trace_task(args.task)
    evaluated_split = task.evaluation_split
    batches = build_fixed_eval_batches(args, stoi, split=evaluated_split)

    set_seed(cli_args.seed)
    metrics = evaluate_prebuilt_batches(
        model,
        args,
        batches,
        generation_metrics_fn=trace_generation_metrics,
        inference_mode=cli_args.inference_mode,
        generation_seed=stable_seed(
            args.seed,
            "drift",
            args.task,
            "paired_generation",
        ),
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(input_dir),
        "checkpoint": cli_args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "task": args.task,
        "architecture": args.architecture,
        "inference_mode": cli_args.inference_mode,
        "effective_inference_mode": effective_inference_mode(args, cli_args.inference_mode),
        "token_selection": args.token_selection,
        "block_size": block_size,
        "vocab_size": len(vocab),
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "evaluation_examples": sum(int(batch.idx.shape[0]) for batch in batches),
        "metrics": metrics,
    }
    summary.update(task.evaluation_metadata(args, split=evaluated_split))
    write_json(summary_path, summary)

    print(
        f"{input_dir.name}: {args.task} | {args.architecture} | {cli_args.inference_mode} | "
        f"{format_trace_metrics(args, metrics)}"
    )
    print(f"output_dir: {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    evaluate_run(parse_args(argv))


if __name__ == "__main__":
    main()
