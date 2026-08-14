"""Run the Othello random-prefix and legal-set evaluation protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    append_jsonl,
    isolated_torch_rng,
    load_checkpoint_payload,
    resolve_device_arg,
    resolve_evaluation_checkpoint,
    restore_checkpoint_state,
    saved_args_from_run,
    stable_seed,
    validate_model_args,
    validate_training_args,
    write_json,
)
from experiments.train_trace import (
    build_training_objects,
    validate_task_args,
)
from model_factory import is_multi_pass_architecture
from tasks.common import EOS_TOKEN
from tasks.trace import othello, othello_eval
from tasks.trace.registry import get_trace_task


INFERENCE_MODES = ("recompute", "append_recurrent")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Othello continuation legality from deterministic "
            "prefix cuts."
        ),
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
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument(
        "--evaluation-mode",
        choices=othello_eval.EVALUATION_MODES,
        default="all",
    )
    parser.add_argument(
        "--inference-modes",
        nargs="+",
        choices=INFERENCE_MODES,
        default=list(INFERENCE_MODES),
    )
    parser.add_argument(
        "--prefix-fractions",
        nargs="+",
        type=float,
        default=list(othello_eval.DEFAULT_PREFIX_FRACTIONS),
    )
    parser.add_argument(
        "--token-selection",
        choices=["argmax", "sample"],
        default="argmax",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def validate_eval_args(args) -> None:
    if args.examples < 1:
        raise ValueError("--examples must be positive")
    if not args.prefix_fractions:
        raise ValueError("--prefix-fractions must not be empty")
    if any(
        not 0.0 < fraction < 1.0
        for fraction in args.prefix_fractions
    ):
        raise ValueError(
            "--prefix-fractions values must be strictly between 0 and 1"
        )
    if len(set(args.inference_modes)) != len(args.inference_modes):
        raise ValueError("--inference-modes must not contain duplicates")


def _load_eval_args(cli_args) -> tuple[SimpleNamespace, Path]:
    run_dir = Path(cli_args.input_run_dir).resolve()
    saved = saved_args_from_run(run_dir)
    if saved.get("task") != "othello":
        raise ValueError(
            "experiments.eval_othello_prefix requires an Othello checkpoint"
        )
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    saved["run_dir"] = str(run_dir)
    saved["resume_from"] = str(run_dir)
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    return args, run_dir


@torch.no_grad()
def _free_generation_metrics(
    model,
    args,
    stoi: dict[str, int],
    example: othello_eval.OthelloEvalExample,
    *,
    inference_mode: str,
    do_sample: bool,
    generation_seed: int,
) -> dict[str, float]:
    prompt_tokens = othello_eval.serialized_prompt(
        stoi,
        example.prefix_move_ids,
    )
    prompt = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=args.device,
    )
    # A game has at most 60 moves. Generate room for any legal continuation
    # plus EOS, rather than only for the sampled reference continuation.
    max_new_tokens = othello.MAX_MOVES - example.cut + 1
    with isolated_torch_rng(generation_seed):
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            inference_mode=inference_mode,
        )
    generated_tokens = generated[0, len(prompt_tokens) :].tolist()
    return othello_eval.score_generated_continuation(
        example.prefix_move_ids,
        generated_tokens,
        eos_id=stoi[EOS_TOKEN],
    )


def evaluate_othello_prefix(cli_args) -> Path:
    validate_eval_args(cli_args)
    args, run_dir = _load_eval_args(cli_args)
    output_dir = (
        Path(cli_args.output_dir).resolve()
        if cli_args.output_dir
        else run_dir / "othello_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "per_example.jsonl"
    if per_example_path.exists():
        per_example_path.unlink()

    checkpoint_path = resolve_evaluation_checkpoint(
        run_dir,
        cli_args.checkpoint,
    )
    checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")
    _block_size, _vocab, stoi, _itos, model, _optimizer = (
        build_training_objects(args)
    )
    restore_checkpoint_state(
        checkpoint,
        model=model,
        optimizer=None,
        device=args.device,
    )

    example_rng = random.Random(
        stable_seed(cli_args.seed, "othello_eval", "examples")
    )
    traces = othello_eval.sample_validation_traces(
        args,
        count=cli_args.examples,
        rng=example_rng,
    )
    examples = othello_eval.build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode=cli_args.evaluation_mode,
        prefix_fractions=cli_args.prefix_fractions,
        rng=example_rng,
    )
    evaluated_modes = list(cli_args.inference_modes)
    if not is_multi_pass_architecture(args.architecture):
        evaluated_modes = [
            mode
            for mode in evaluated_modes
            if mode == "recompute"
        ]
    if not evaluated_modes:
        raise ValueError(
            "the selected architecture has no requested supported "
            "inference mode"
        )

    rows = []
    recompute_cache: dict[tuple[int, ...], torch.Tensor] = {}
    was_training = model.training
    model.eval()
    try:
        for inference_mode in evaluated_modes:
            for example in examples:
                generation_seed = stable_seed(
                    cli_args.seed,
                    "othello_eval",
                    "generation",
                    example.example_index,
                    example.protocol,
                )
                row = {
                    "example_index": example.example_index,
                    "protocol": example.protocol,
                    "inference_mode": inference_mode,
                    "prompt_moves": example.cut,
                    "prompt_bucket": othello_eval.length_bucket(
                        example.cut
                    ),
                    "reference_suffix_moves": len(
                        example.suffix_move_ids
                    ),
                    "suffix_bucket": othello_eval.length_bucket(
                        len(example.suffix_move_ids)
                    ),
                    "free_generation": _free_generation_metrics(
                        model,
                        args,
                        stoi,
                        example,
                        inference_mode=inference_mode,
                        do_sample=(
                            cli_args.token_selection == "sample"
                        ),
                        generation_seed=generation_seed,
                    ),
                    "teacher_forced": (
                        othello_eval.teacher_forced_metrics(
                            model,
                            args,
                            stoi,
                            example,
                            inference_mode=inference_mode,
                            recompute_cache=recompute_cache,
                        )
                    ),
                }
                rows.append(row)
                append_jsonl(per_example_path, row)
    finally:
        model.train(was_training)

    mode_summaries = {}
    for inference_mode in evaluated_modes:
        mode_rows = [
            row
            for row in rows
            if row["inference_mode"] == inference_mode
        ]
        mode_summaries[inference_mode] = {
            "overall": othello_eval.summarize_rows(mode_rows),
            "by_protocol": othello_eval.group_summaries(
                mode_rows,
                "protocol",
            ),
            "by_prompt_bucket": othello_eval.group_summaries(
                mode_rows,
                "prompt_bucket",
            ),
            "by_suffix_bucket": othello_eval.group_summaries(
                mode_rows,
                "suffix_bucket",
            ),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(run_dir),
        "checkpoint": cli_args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "task": args.task,
        "architecture": args.architecture,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "evaluation_mode": cli_args.evaluation_mode,
        "token_selection": cli_args.token_selection,
        "base_trace_count": len(traces),
        "continuation_example_count": len(examples),
        "prefix_fractions": list(cli_args.prefix_fractions),
        "requested_inference_modes": list(cli_args.inference_modes),
        "evaluated_inference_modes": evaluated_modes,
        "modes": mode_summaries,
    }
    task = get_trace_task(args.task)
    payload.update(
        task.evaluation_metadata(args, split=task.evaluation_split)
    )
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    print(f"wrote {summary_path}")
    print(f"wrote {per_example_path}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    evaluate_othello_prefix(parse_args(argv))


if __name__ == "__main__":
    main()
