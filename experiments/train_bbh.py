from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Callable

from experiments.common import (
    FixedPointStatsTracker,
    append_jsonl,
    build_model_and_optimizer,
    clip_gradients,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    format_checkpoint_line,
    format_default_eval_metrics,
    format_gradient_norms,
    format_pass_losses,
    forward_and_loss,
    gradient_norms,
    load_checkpoint_payload,
    model_benchmark_stats,
    prepare_run_artifacts,
    provenance_metadata,
    resolve_device_arg,
    resolve_resume_artifacts,
    restore_checkpoint_state,
    runtime_resource_stats,
    save_best_checkpoint,
    save_latest_checkpoint,
    set_seed,
    stable_seed,
    summarize_gradient_norm_window,
    synchronize_device,
    update_gradient_norm_window,
    validate_model_args,
    validate_training_args,
)
from experiments.pass_schedule import build_pass_scheduler
from experiments.presets import BBH_PRESETS, preset_help_text, resolve_preset_args
from model_factory import (
    ARCHITECTURES,
    supports_pass_override,
    uses_pass_loss_weights,
)
from tasks.bbh import permutation, pointer_chasing, state_machine, tracking


@dataclass(frozen=True)
class BBHTask:
    name: str
    min_level: int
    shape_arg_names: tuple[str, ...]
    vocab_builder: Callable
    block_size_builder: Callable
    batch_builder: Callable

    def shape_kwargs(self, args) -> dict[str, int]:
        return {name: getattr(args, name) for name in self.shape_arg_names}

    def build_vocab(self, args):
        return self.vocab_builder(**self.shape_kwargs(args))

    def required_block_size(self, args, level: int) -> int:
        kwargs = self.shape_kwargs(args)
        level_name = {
            "pointer_chasing": "num_hops",
            "permutation": "num_swaps",
            "tracking": "num_ops",
            "state_machine": "num_steps",
        }[self.name]
        return self.block_size_builder(**kwargs, **{level_name: level})

    def build_batch(self, args, *, batch_size: int, level: int, stoi, rng: random.Random):
        kwargs = self.shape_kwargs(args)
        level_name = {
            "pointer_chasing": "num_hops",
            "permutation": "num_swaps",
            "tracking": "num_ops",
            "state_machine": "num_steps",
        }[self.name]
        return self.batch_builder(
            batch_size=batch_size,
            stoi=stoi,
            device=args.device,
            rng=rng,
            **kwargs,
            **{level_name: level},
        )


BBH_TASKS = {
    "pointer_chasing": BBHTask(
        "pointer_chasing", 1, ("num_nodes",),
        pointer_chasing.build_pointer_chasing_vocab,
        pointer_chasing.required_block_size,
        pointer_chasing.build_pointer_chasing_batch,
    ),
    "permutation": BBHTask(
        "permutation", 0, ("num_objects",),
        permutation.build_permutation_vocab,
        permutation.required_block_size,
        permutation.build_permutation_batch,
    ),
    "tracking": BBHTask(
        "tracking", 1, ("num_objects",),
        tracking.build_tracking_vocab,
        tracking.required_block_size,
        tracking.build_tracking_batch,
    ),
    "state_machine": BBHTask(
        "state_machine", 0, ("num_states", "alphabet_size"),
        state_machine.build_state_machine_vocab,
        state_machine.required_block_size,
        state_machine.build_state_machine_batch,
    ),
}


def _add_override(parser, *names, **kwargs) -> None:
    parser.add_argument(*names, default=argparse.SUPPRESS, **kwargs)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train BBH-style final-answer curricula.",
        allow_abbrev=False,
    )
    _add_override(parser, "--preset", choices=sorted(BBH_PRESETS), help=preset_help_text(BBH_PRESETS))
    _add_override(parser, "--task", choices=sorted(BBH_TASKS))
    _add_override(parser, "--architecture", choices=ARCHITECTURES)
    _add_override(parser, "--model-size", choices=["tiny", "small", "medium", "large"])
    _add_override(parser, "--inference-mode", choices=["recompute", "append_recurrent"])
    _add_override(parser, "--token-selection", choices=["sample", "argmax"])
    _add_override(parser, "--n-layer", type=int)
    _add_override(parser, "--n-head", type=int)
    _add_override(parser, "--n-embd", type=int)
    _add_override(parser, "--max-passes", type=int)
    _add_override(parser, "--min-passes", type=int)
    _add_override(parser, "--train-pass-mode", choices=["fixed", "fixed_point"])
    _add_override(parser, "--fixed-point-memory-tol", type=float)
    _add_override(parser, "--fixed-point-kl-tol", type=float)
    _add_override(parser, "--pass-loss-weights", type=float, nargs="*")
    _add_override(parser, "--memory-width", type=int)
    _add_override(parser, "--memory-read-layers", type=int, nargs="+")
    _add_override(
        parser,
        "--train-pass-schedule",
        nargs="+",
        metavar="START=PASS:WEIGHT,...",
    )
    _add_override(
        parser,
        "--num-nodes",
        type=int,
        help="pointer-chasing label-pool capacity; level L uses 2L+1 active nodes",
    )
    _add_override(parser, "--num-objects", type=int)
    _add_override(parser, "--num-states", type=int)
    _add_override(parser, "--alphabet-size", type=int)
    _add_override(parser, "--curriculum-start-level", type=int)
    _add_override(parser, "--curriculum-threshold", type=float)
    _add_override(parser, "--review-easier-every", type=int)
    _add_override(parser, "--max-level", type=int)
    _add_override(parser, "--batch-size", type=int)
    _add_override(parser, "--train-steps", type=int)
    _add_override(parser, "--lr", type=float)
    _add_override(parser, "--max-grad-norm", type=float)
    _add_override(parser, "--weight-decay", type=float)
    _add_override(parser, "--eval-interval", type=int)
    _add_override(parser, "--eval-batches", type=int)
    _add_override(parser, "--seed", type=int)
    _add_override(parser, "--device")
    _add_override(parser, "--block-size", type=int)
    _add_override(parser, "--run-dir")
    _add_override(parser, "--resume-from")
    raw = parser.parse_args(argv)
    explicit_optimization_overrides = {
        key: getattr(raw, key)
        for key in ("lr", "max_grad_norm")
        if hasattr(raw, key)
    }
    args = resolve_preset_args(
        raw,
        BBH_PRESETS,
        default_preset="pointer_chasing_main",
        parser=parser,
    )
    args._explicit_optimization_overrides = explicit_optimization_overrides
    return args


def validate_task_args(args) -> None:
    if args.task not in BBH_TASKS:
        raise ValueError(f"unsupported BBH task: {args.task}")
    task = BBH_TASKS[args.task]
    if args.curriculum_start_level < task.min_level:
        raise ValueError(f"--curriculum-start-level must be >= {task.min_level}")
    if args.max_level < args.curriculum_start_level:
        raise ValueError("--max-level must be >= --curriculum-start-level")
    if not 0 <= args.curriculum_threshold <= 1:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be non-negative")
    task.build_vocab(args)
    required = task.required_block_size(args, args.max_level)
    if args.block_size is not None and args.block_size < required:
        raise ValueError(f"--block-size must be at least {required} for level {args.max_level}")


def build_training_objects(args):
    task = BBH_TASKS[args.task]
    block_size = args.block_size or task.required_block_size(args, args.max_level)
    vocab, stoi, itos = task.build_vocab(args)
    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return task, block_size, vocab, stoi, itos, model, optimizer


def build_fixed_eval_batches(args, task: BBHTask, stoi, level: int):
    rng = random.Random(stable_seed(args.seed, "bbh", args.task, "eval", level))
    return [
        task.build_batch(args, batch_size=args.batch_size, level=level, stoi=stoi, rng=rng)
        for _ in range(args.eval_batches)
    ]


def choose_train_level(args, task: BBHTask, current_level: int, step: int, rng: random.Random) -> int:
    if (
        args.review_easier_every > 0
        and current_level > args.curriculum_start_level
        and step % args.review_easier_every == 0
    ):
        return rng.randint(task.min_level, current_level - 1)
    return current_level


def _apply_resume_args(args, checkpoint: dict) -> None:
    saved = checkpoint["args"]
    preserve = {
        "resume_from": args.resume_from,
        "run_dir": args.run_dir,
        "train_steps": args.train_steps,
        "device": args.device,
    }
    for key, value in saved.items():
        setattr(args, key, value)
    for key, value in preserve.items():
        if value is not None:
            setattr(args, key, value)


def run_answer_curriculum(args) -> None:
    checkpoint = None
    resume_step = 0
    explicit_optimization_overrides = dict(
        getattr(args, "_explicit_optimization_overrides", {})
    )
    if hasattr(args, "_explicit_optimization_overrides"):
        delattr(args, "_explicit_optimization_overrides")
    if args.resume_from:
        artifacts = resolve_resume_artifacts(args.resume_from)
        checkpoint = load_checkpoint_payload(artifacts.checkpoint_path, device="cpu")
        resume_step = int(checkpoint.get("step", 0))
        _apply_resume_args(args, checkpoint)
        for key, value in explicit_optimization_overrides.items():
            setattr(args, key, value)

    resolve_device_arg(args)
    set_seed(args.seed)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    pass_scheduler = build_pass_scheduler(
        args,
        seed=stable_seed(args.seed, "pass-schedule"),
    )
    fixed_point_tracker = (
        FixedPointStatsTracker() if args.train_pass_mode == "fixed_point" else None
    )
    task, block_size, vocab, stoi, _itos, model, optimizer = build_training_objects(args)
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("bbh", args.task, args.architecture),
        extra_config={"script": "experiments.train_bbh"},
    )

    train_rng = random.Random(stable_seed(args.seed, "bbh", args.task, "train"))
    current_level = args.curriculum_start_level
    promotion_history: list[tuple[int, int, float]] = []
    best_eval_score: tuple[int, float, float] | None = None
    best_eval_step: int | None = None
    if checkpoint is not None:
        restore_checkpoint_state(checkpoint, model=model, optimizer=optimizer, device=args.device)
        extra = checkpoint["extra_state"]
        current_level = int(extra["current_level"])
        promotion_history = [tuple(item) for item in extra["promotion_history"]]
        train_rng.setstate(extra["train_rng_state"])
        if pass_scheduler is not None:
            pass_scheduler.load_state_dict(extra["pass_scheduler_state"])
        best_eval_score = tuple(extra["best_eval_score"])
        saved_best_step = extra["best_eval_step"]
        best_eval_step = None if saved_best_step is None else int(saved_best_step)
        if "lr" in explicit_optimization_overrides:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = explicit_optimization_overrides["lr"]

    print(f"device: {args.device}")
    print(f"task: {args.task}")
    print(f"architecture: {args.architecture}")
    print(f"inference_mode: {effective_inference_mode(args)}")
    print(f"block_size: {block_size}")
    print(f"parameters: {model.get_num_params():,}")
    if supports_pass_override(args.architecture):
        print(f"max_passes: {args.max_passes}")
    if uses_pass_loss_weights(args.architecture):
        if args.train_pass_mode == "fixed_point":
            print(
                "train_pass_mode: fixed_point | "
                f"min_passes {args.min_passes} | max_passes {args.max_passes}"
            )
            print(
                "fixed_point_tolerances: "
                f"memory {args.fixed_point_memory_tol:g} | "
                f"logit_kl {args.fixed_point_kl_tol:g}"
            )
            print("fixed_point_loss: equal first-pass and final adaptive-pass loss")
        else:
            normalized = [weight / sum(args.pass_loss_weights) for weight in args.pass_loss_weights]
            print(f"relative_pass_loss_weights_normalized: {normalized}")
    if pass_scheduler is not None:
        print(f"train_pass_schedule: {args.train_pass_schedule}")
    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_start" if checkpoint is None else "run_resume",
            "step": resume_step,
            "task": args.task,
            "architecture": args.architecture,
            "provenance": provenance_metadata(),
            "config": vars(args),
            "model_stats": model_benchmark_stats(model),
        },
    )

    start_step = resume_step + 1
    final_step = resume_step + args.train_steps if checkpoint is not None else args.train_steps
    window_start = time.perf_counter()
    window_tokens = 0
    gradient_norm_window: dict[str, dict[str, float]] = {}

    for step in range(start_step, final_step + 1):
        model.train()
        sampled_level = choose_train_level(args, task, current_level, step, train_rng)
        batch = task.build_batch(
            args,
            batch_size=args.batch_size,
            level=sampled_level,
            stoi=stoi,
            rng=train_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        sampled_passes = pass_scheduler.sample(step) if pass_scheduler is not None else None
        loss, output, pass_losses = forward_and_loss(
            model,
            batch,
            args,
            passes=sampled_passes,
            fixed_point_training=args.train_pass_mode == "fixed_point",
        )
        if fixed_point_tracker is not None:
            fixed_point_tracker.observe(output)
        loss.backward()
        update_gradient_norm_window(gradient_norm_window, gradient_norms(model))
        clip_gradients(model, args.max_grad_norm)
        optimizer.step()
        window_tokens += int(batch.idx.numel())

        should_eval = step == 1 or step % args.eval_interval == 0 or step == final_step
        if not should_eval:
            continue

        synchronize_device(args.device)
        elapsed = time.perf_counter() - window_start
        tok_per_s = window_tokens / elapsed if elapsed > 0 else 0.0
        fields = [f"loss {loss.item():.4f}", f"tok/s {tok_per_s:.1f}", f"level {current_level}"]
        gradient_summary = summarize_gradient_norm_window(gradient_norm_window)
        fields.append(format_gradient_norms(gradient_summary))
        if uses_pass_loss_weights(args.architecture):
            fields.append(f"pass_losses {format_pass_losses(pass_losses)}")
        if sampled_passes is not None:
            fields.append(f"sampled_passes {sampled_passes}")
        fixed_point_summary = (
            fixed_point_tracker.summary() if fixed_point_tracker is not None else None
        )
        if fixed_point_summary is not None:
            fields.append(f"mean_passes {fixed_point_summary['mean_passes']:.2f}")
            fields.append(
                f"converged {fixed_point_summary['converged_fraction']:.3f}"
            )
        print(format_checkpoint_line(f"step {step}", fields))

        evaluated_level = current_level
        eval_batches = build_fixed_eval_batches(args, task, stoi, evaluated_level)
        metrics = evaluate_prebuilt_batches(
            model,
            args,
            eval_batches,
            inference_mode=args.inference_mode,
            generation_seed=stable_seed(args.seed, "bbh", args.task, "generation", evaluated_level),
        )
        print(
            format_checkpoint_line(
                "eval",
                [f"loss {metrics['loss']:.4f}", f"level {evaluated_level}", format_default_eval_metrics(metrics)],
            )
        )
        exact_match = float(metrics["exact_match"])
        eval_score = (evaluated_level, exact_match, -float(metrics["loss"]))
        is_best_checkpoint = (
            best_eval_score is None or eval_score > best_eval_score
        )
        if is_best_checkpoint:
            best_eval_score = eval_score
            best_eval_step = step
        eval_event = {
            "event": "eval",
            "step": step,
            "level": evaluated_level,
            "sampled_train_level": sampled_level,
            "train_loss": float(loss.item()),
            "pass_losses": [float(item.item()) for item in pass_losses],
            "metrics": metrics,
            "gradient_norms": gradient_summary,
            "train_tok_per_s": tok_per_s,
            "resource_stats": runtime_resource_stats(args.device),
            "is_best_checkpoint": is_best_checkpoint,
            "best_eval_score": best_eval_score,
            "best_eval_step": best_eval_step,
        }
        if pass_scheduler is not None:
            eval_event["pass_schedule"] = pass_scheduler.stats()
        if fixed_point_summary is not None:
            eval_event["fixed_point_training"] = fixed_point_summary
        append_jsonl(artifacts.metrics_path, eval_event)

        if exact_match >= args.curriculum_threshold and current_level < args.max_level:
            promotion_history.append((current_level, step, exact_match))
            current_level += 1
            print(f"curriculum_promote -> level {current_level}")

        checkpoint_extra = {
            "current_level": current_level,
            "evaluated_level": evaluated_level,
            "promotion_history": promotion_history,
            "train_rng_state": train_rng.getstate(),
            "best_eval_score": best_eval_score,
            "best_eval_step": best_eval_step,
        }
        if pass_scheduler is not None:
            checkpoint_extra["pass_scheduler_state"] = pass_scheduler.state_dict()
        save_latest_checkpoint(
            artifacts,
            model=model,
            optimizer=optimizer,
            args=args,
            step=step,
            extra_state=checkpoint_extra,
        )
        if is_best_checkpoint:
            save_best_checkpoint(
                artifacts,
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                extra_state=checkpoint_extra,
            )
            print(
                f"best_checkpoint -> step {step} | level {eval_score[0]} | "
                f"exact_match {eval_score[1]:.4f}"
            )
        synchronize_device(args.device)
        window_start = time.perf_counter()
        window_tokens = 0
        gradient_norm_window = {}
        if fixed_point_tracker is not None:
            fixed_point_tracker.reset()

    append_jsonl(
        artifacts.metrics_path,
        {"event": "run_end", "final_level": current_level, "promotion_history": promotion_history},
    )
    print(f"run_dir: {artifacts.run_dir}")


def main(argv: list[str] | None = None) -> None:
    run_answer_curriculum(parse_args(argv))


if __name__ == "__main__":
    main()
