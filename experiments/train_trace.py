from __future__ import annotations

import argparse
import random
import time

from tasks.trace.registry import TRACE_TASKS, get_trace_task
from tasks.trace.maze import (
    INPUT_REPRESENTATIONS as MAZE_INPUT_REPRESENTATIONS,
    ROUTE_POLICIES as MAZE_ROUTE_POLICIES,
    TARGET_REPRESENTATIONS as MAZE_TARGET_REPRESENTATIONS,
)
from experiments.common import (
    apply_learning_rate,
    append_jsonl,
    build_model_and_optimizer,
    clip_gradients,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    format_checkpoint_line,
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
    synchronize_device,
    summarize_gradient_norm_window,
    update_gradient_norm_window,
    validate_model_args,
    validate_training_args,
)
from experiments.presets import TRACE_PRESETS, preset_help_text, resolve_preset_args
from experiments.pass_schedule import build_pass_scheduler
from model_factory import ARCHITECTURES


_OPTIMIZATION_OVERRIDE_KEYS = (
    "lr",
    "lr_schedule",
    "min_lr",
    "lr_warmup_steps",
    "lr_decay_steps",
    "max_grad_norm",
)


def _add_override(parser, *names, **kwargs) -> None:
    parser.add_argument(*names, default=argparse.SUPPRESS, **kwargs)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train fixed-length trace tasks.", allow_abbrev=False)
    _add_override(parser, "--preset", choices=sorted(TRACE_PRESETS), help=preset_help_text(TRACE_PRESETS))
    _add_override(parser, "--task", choices=TRACE_TASKS)
    _add_override(parser, "--architecture", choices=ARCHITECTURES)
    _add_override(parser, "--model-size", choices=["tiny", "small", "medium", "large"])
    _add_override(parser, "--inference-mode", choices=["recompute", "append_recurrent"])
    _add_override(parser, "--token-selection", choices=["sample", "argmax"])
    _add_override(parser, "--n-layer", type=int)
    _add_override(parser, "--n-head", type=int)
    _add_override(parser, "--n-embd", type=int)
    _add_override(parser, "--n-pass", type=int)
    _add_override(parser, "--pass-loss-weights", type=float, nargs="*")
    _add_override(
        parser,
        "--train-pass-schedule",
        nargs="+",
        metavar="START=PASS:WEIGHT,...",
    )
    _add_override(parser, "--shortest-path-data-dir")
    _add_override(parser, "--maze-data-dir")
    _add_override(
        parser,
        "--maze-input-representation",
        choices=MAZE_INPUT_REPRESENTATIONS,
    )
    _add_override(
        parser,
        "--maze-target-representation",
        choices=MAZE_TARGET_REPRESENTATIONS,
    )
    _add_override(
        parser,
        "--maze-route-policy",
        choices=MAZE_ROUTE_POLICIES,
    )
    _add_override(parser, "--othello-data-dir")
    _add_override(parser, "--othello-train-games", type=int)
    _add_override(parser, "--othello-val-games", type=int)
    _add_override(parser, "--othello-dataset-seed", type=int)
    _add_override(parser, "--batch-size", type=int)
    _add_override(parser, "--train-steps", type=int)
    _add_override(parser, "--lr", type=float)
    _add_override(parser, "--max-grad-norm", type=float)
    _add_override(
        parser,
        "--lr-schedule",
        choices=["constant", "warmup_cosine"],
    )
    _add_override(parser, "--min-lr", type=float)
    _add_override(parser, "--lr-warmup-steps", type=int)
    _add_override(parser, "--lr-decay-steps", type=int)
    _add_override(parser, "--weight-decay", type=float)
    _add_override(parser, "--eval-interval", type=int)
    _add_override(parser, "--eval-batches", type=int)
    _add_override(parser, "--seed", type=int)
    _add_override(parser, "--device")
    _add_override(parser, "--block-size", type=int)
    _add_override(parser, "--run-dir")
    _add_override(parser, "--resume-from")
    raw_args = parser.parse_args(argv)
    explicit_optimization_overrides = {
        key: getattr(raw_args, key)
        for key in _OPTIMIZATION_OVERRIDE_KEYS
        if hasattr(raw_args, key)
    }
    args = resolve_preset_args(
        raw_args,
        TRACE_PRESETS,
        default_preset="shortest_path_main",
        parser=parser,
    )
    args._explicit_optimization_overrides = explicit_optimization_overrides
    return args


def validate_task_args(args) -> None:
    task = get_trace_task(args.task)
    task.build_vocab(args)
    required = task.required_block_size(args)
    if args.block_size is not None and args.block_size < required:
        raise ValueError(f"--block-size must be at least {required} for {args.task}")


def build_training_objects(args):
    task = get_trace_task(args.task)
    block_size = args.block_size or task.required_block_size(args)
    vocab, stoi, itos = task.build_vocab(args)
    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return block_size, vocab, stoi, itos, model, optimizer


def build_task_batch(args, stoi, rng: random.Random, *, split: str):
    return get_trace_task(args.task).build_batch(args, stoi, rng, split=split)


def build_fixed_eval_batches(args, stoi, *, split: str = "val"):
    task = get_trace_task(args.task)
    compiled_batches = task.build_eval_batches(args, stoi, split=split)
    if compiled_batches is not None:
        return compiled_batches
    rng = random.Random(stable_seed(args.seed, "trace", args.task, "eval"))
    return [
        build_task_batch(args, stoi, rng, split=split)
        for _ in range(args.eval_batches)
    ]


def trace_generation_metrics(model, batch, args, *, inference_mode: str | None = None):
    return get_trace_task(args.task).generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


def format_trace_metrics(args, metrics: dict[str, float]) -> str:
    return get_trace_task(args.task).format_metrics(metrics)


def _apply_resume_args(
    args,
    checkpoint: dict,
    *,
    explicit_optimization_overrides: dict[str, object],
) -> None:
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
    for key, value in explicit_optimization_overrides.items():
        setattr(args, key, value)


def run_trace_training(args) -> None:
    checkpoint = None
    resume_step = 0
    explicit_optimization_overrides = dict(
        getattr(args, "_explicit_optimization_overrides", {})
    )
    if hasattr(args, "_explicit_optimization_overrides"):
        delattr(args, "_explicit_optimization_overrides")
    if args.resume_from:
        resume_artifacts = resolve_resume_artifacts(args.resume_from)
        checkpoint = load_checkpoint_payload(resume_artifacts.checkpoint_path, device="cpu")
        resume_step = int(checkpoint.get("step", 0))
        _apply_resume_args(
            args,
            checkpoint,
            explicit_optimization_overrides=explicit_optimization_overrides,
        )

    resolve_device_arg(args)
    set_seed(args.seed)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    pass_scheduler = build_pass_scheduler(
        args,
        seed=stable_seed(args.seed, "latent-feedback", "pass-schedule"),
    )
    block_size, vocab, stoi, _itos, model, optimizer = build_training_objects(args)
    extra_config = {"script": "experiments.train_trace"}
    if args.task == "shortest_path":
        extra_config["training_evaluation_split"] = "validation"
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("trace", args.task, args.architecture),
        extra_config=extra_config,
    )

    train_rng = random.Random(stable_seed(args.seed, "trace", args.task, "train"))
    best_eval_loss = float("inf")
    best_eval_step: int | None = None
    if checkpoint is not None:
        restore_checkpoint_state(checkpoint, model=model, optimizer=optimizer, device=args.device)
        extra = checkpoint["extra_state"]
        best_eval_loss = float(extra["best_eval_loss"])
        saved_best_step = extra["best_eval_step"]
        best_eval_step = None if saved_best_step is None else int(saved_best_step)
        train_rng.setstate(extra["train_rng_state"])
        if pass_scheduler is not None:
            pass_scheduler.load_state_dict(extra["pass_scheduler_state"])
        apply_learning_rate(optimizer, args, resume_step)

    print(f"device: {args.device}")
    print(f"task: {args.task}")
    print(f"architecture: {args.architecture}")
    print(f"inference_mode: {effective_inference_mode(args)}")
    print(f"block_size: {block_size}")
    print(f"parameters: {model.get_num_params():,}")
    if args.lr_schedule == "warmup_cosine":
        print(
            "lr_schedule: warmup_cosine | "
            f"peak {args.lr:.3g} | min {args.min_lr:.3g} | "
            f"warmup_steps {args.lr_warmup_steps} | "
            f"decay_steps {args.lr_decay_steps}"
        )
    else:
        print(f"lr_schedule: constant | lr {args.lr:.3g}")
    if args.architecture != "transformer":
        print(f"n_pass: {args.n_pass}")
        if args.architecture == "latent_feedback":
            print("pass_loss_objective: standard + mean(feedback passes)")
        else:
            total_weight = sum(args.pass_loss_weights)
            print(f"pass_loss_weights_normalized: {[weight / total_weight for weight in args.pass_loss_weights]}")
    if pass_scheduler is not None:
        print(f"train_pass_schedule: {args.train_pass_schedule}")
    run_event = {
        "event": "run_start" if checkpoint is None else "run_resume",
        "step": resume_step,
        "task": args.task,
        "architecture": args.architecture,
        "provenance": provenance_metadata(),
        "config": vars(args),
        "model_stats": model_benchmark_stats(model),
    }
    if args.task == "shortest_path":
        run_event["dataset_split"] = "validation"
    append_jsonl(artifacts.metrics_path, run_event)

    fixed_eval_batches = build_fixed_eval_batches(args, stoi)
    start_step = resume_step + 1
    final_step = resume_step + args.train_steps if checkpoint is not None else args.train_steps
    window_start = time.perf_counter()
    window_tokens = 0
    gradient_norm_window: dict[str, dict[str, float]] = {}

    for step in range(start_step, final_step + 1):
        current_lr = apply_learning_rate(optimizer, args, step)
        model.train()
        batch = build_task_batch(args, stoi, train_rng, split="train")
        optimizer.zero_grad(set_to_none=True)
        sampled_n_pass = pass_scheduler.sample(step) if pass_scheduler is not None else None
        loss, _output, pass_losses = forward_and_loss(
            model,
            batch,
            args,
            n_pass=sampled_n_pass,
        )
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
        fields = [
            f"loss {loss.item():.4f}",
            f"lr {current_lr:.3g}",
            f"tok/s {tok_per_s:.1f}",
        ]
        gradient_summary = summarize_gradient_norm_window(gradient_norm_window)
        fields.append(format_gradient_norms(gradient_summary))
        if args.architecture != "transformer":
            fields.append(f"pass_losses {format_pass_losses(pass_losses)}")
        if sampled_n_pass is not None:
            fields.append(f"sampled_passes {sampled_n_pass}")
        print(format_checkpoint_line(f"step {step}", fields))

        metrics = evaluate_prebuilt_batches(
            model,
            args,
            fixed_eval_batches,
            generation_metrics_fn=trace_generation_metrics,
            inference_mode=args.inference_mode,
            generation_seed=stable_seed(args.seed, "trace", args.task, "generation"),
        )
        print(
            format_checkpoint_line(
                "eval",
                [f"loss {metrics['loss']:.4f}", format_trace_metrics(args, metrics)],
            )
        )
        eval_loss = float(metrics["loss"])
        is_best_checkpoint = eval_loss < best_eval_loss
        if is_best_checkpoint:
            best_eval_loss = eval_loss
            best_eval_step = step
        eval_event = {
            "event": "eval",
            "step": step,
            "learning_rate": current_lr,
            "train_loss": float(loss.item()),
            "pass_losses": [float(item.item()) for item in pass_losses],
            "metrics": metrics,
            "gradient_norms": gradient_summary,
            "train_tok_per_s": tok_per_s,
            "resource_stats": runtime_resource_stats(args.device),
            "is_best_checkpoint": is_best_checkpoint,
            "best_eval_loss": best_eval_loss,
            "best_eval_step": best_eval_step,
        }
        if pass_scheduler is not None:
            eval_event["pass_schedule"] = pass_scheduler.stats()
        if args.task == "shortest_path":
            eval_event["dataset_split"] = "validation"
        append_jsonl(artifacts.metrics_path, eval_event)
        checkpoint_extra = {
            "train_rng_state": train_rng.getstate(),
            "best_eval_loss": best_eval_loss,
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
            print(f"best_checkpoint -> step {step} | eval_loss {best_eval_loss:.4f}")
        synchronize_device(args.device)
        window_start = time.perf_counter()
        window_tokens = 0
        gradient_norm_window = {}

    append_jsonl(artifacts.metrics_path, {"event": "run_end", "task": args.task})
    print(f"run_dir: {artifacts.run_dir}")


def main(argv: list[str] | None = None) -> None:
    run_trace_training(parse_args(argv))


if __name__ == "__main__":
    main()
