"""Task-specific evaluation for shortest-path trace generation."""
from __future__ import annotations

import torch

from model_factory import supports_append_recurrent
from tasks.trace import shortest_path


@torch.no_grad()
def generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    """Evaluate exact optimal-path generation and dataset difficulty."""
    mode = (
        inference_mode or args.inference_mode
        if supports_append_recurrent(args.architecture)
        else "recompute"
    )
    do_sample = getattr(args, "token_selection", "argmax") == "sample"
    totals = {
        "optimal_path": 0.0,
        "mean_target_path_length": 0.0,
    }
    bucket_counts = {
        bucket: 0
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
    }
    bucket_optimal = {
        bucket: 0.0
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
    }
    step_counts = {
        step: 0
        for step in range(
            1,
            shortest_path.get_shortest_path_distribution(
                args.shortest_path_distribution
            ).max_path_length
            + 1,
        )
    }
    step_correct = {step: 0.0 for step in step_counts}

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        target_suffix = batch.targets[
            row,
            prompt_len - 1 : prompt_len - 1 + output_len,
        ].tolist()
        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
        )
        generated_suffix = generated[
            0,
            prompt_len : prompt_len + output_len,
        ].tolist()
        target_path_ids = target_suffix[:-1]
        target_path_length = len(target_path_ids) - 1
        bucket = shortest_path.path_length_bucket(target_path_length)
        optimal_path = generated_suffix == target_suffix
        totals["optimal_path"] += float(optimal_path)
        totals["mean_target_path_length"] += float(target_path_length)
        bucket_counts[bucket] += 1
        bucket_optimal[bucket] += float(optimal_path)
        for step in range(1, len(target_path_ids)):
            step_counts[step] += 1
            step_correct[step] += float(
                generated_suffix[step] == target_path_ids[step]
            )

    count = int(batch.idx.shape[0])
    result = {key: value / count for key, value in totals.items()}
    for bucket in shortest_path.PATH_LENGTH_BUCKETS:
        bucket_count = bucket_counts[bucket]
        if bucket_count:
            result[f"optimal_path_{bucket}"] = (
                bucket_optimal[bucket] / bucket_count
            )
            result[f"optimal_path_{bucket}__weight"] = float(bucket_count)
            result[f"examples_{bucket}__sum"] = float(bucket_count)
    for step, step_count in step_counts.items():
        if step_count:
            metric = f"path_step_{step}_accuracy"
            result[metric] = step_correct[step] / step_count
            result[f"{metric}__weight"] = float(step_count)
            result[f"path_step_{step}_examples__sum"] = float(step_count)
    return result


def format_metrics(metrics: dict[str, float]) -> str:
    fields = [f"optimal {metrics['optimal_path']:.3f}"]
    fields.extend(
        f"{bucket} {metrics[f'optimal_path_{bucket}']:.3f}"
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
        if f"optimal_path_{bucket}" in metrics
    )
    step_numbers = sorted(
        int(key.removeprefix("path_step_").removesuffix("_accuracy"))
        for key in metrics
        if key.startswith("path_step_") and key.endswith("_accuracy")
    )
    step_fields = [
        f"{step}:{metrics[f'path_step_{step}_accuracy']:.3f}"
        for step in step_numbers
    ]
    if step_fields:
        fields.append(f"step_acc [{', '.join(step_fields)}]")
    return " | ".join(fields)


__all__ = [
    "format_metrics",
    "generation_metrics",
]
