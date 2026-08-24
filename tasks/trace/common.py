from __future__ import annotations

from typing import Callable, Sequence

import torch

from model_factory import supports_append_recurrent


@torch.no_grad()
def trace_generation_metrics(
    model,
    batch,
    args,
    *,
    legality_check: Callable[[Sequence[int], Sequence[int]], tuple[int, bool]],
    inference_mode: str | None = None,
) -> dict[str, float]:
    """Evaluate free generation on a trace task.

    The legality function receives the task prompt (without BOS/SEP) and the
    generated non-EOS trace. Padding in target traces is excluded from the legal
    token denominator.
    """
    mode = (
        inference_mode or args.inference_mode
        if supports_append_recurrent(args.architecture)
        else "recompute"
    )
    do_sample = getattr(args, "token_selection", "sample") == "sample"

    token_legalities: list[float] = []
    sequence_legalities: list[float] = []
    legal_lengths: list[float] = []

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        target_suffix = batch.targets[
            row : row + 1,
            prompt_len - 1 : prompt_len - 1 + output_len,
        ]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
        )
        generated_suffix = generated[:, prompt_len : prompt_len + output_len]

        prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
        generated_tokens = generated_suffix[0].tolist()
        generated_trace = generated_tokens[:-1]
        legal_prefix_len, all_legal = legality_check(prompt_tokens, generated_trace)
        legal_lengths.append(float(legal_prefix_len))
        token_legalities.append(float(legal_prefix_len) / float(max(len(generated_trace), 1)))

        eos_ok = generated_tokens[-1] == int(target_suffix[0, -1].item())
        sequence_legalities.append(float(all_legal and eos_ok))

    return {
        "token_legality": sum(token_legalities) / len(token_legalities),
        "sequence_legality": sum(sequence_legalities) / len(sequence_legalities),
        "mean_legal_len": sum(legal_lengths) / len(legal_lengths),
    }


def format_legal_generation_metrics(metrics: dict[str, float]) -> str:
    return (
        f"token_legality {metrics['token_legality']:.3f} | "
        f"sequence_legality {metrics['sequence_legality']:.3f} | "
        f"mean_legal_len {metrics['mean_legal_len']:.2f}"
    )
