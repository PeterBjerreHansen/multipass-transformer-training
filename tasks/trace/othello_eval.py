"""Task-specific evaluation for Othello trace generation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

import torch
import torch.nn.functional as F

from tasks.common import BOS_TOKEN, EOS_TOKEN, SEP_TOKEN
from tasks.trace import othello
from tasks.trace.common import (
    format_legal_generation_metrics,
    trace_generation_metrics,
)


EVALUATION_MODES = ("full-game", "random-prefix", "prefix-grid", "all")
DEFAULT_PREFIX_FRACTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class OthelloEvalExample:
    example_index: int
    protocol: str
    trace_move_ids: tuple[int, ...]
    cut: int

    @property
    def prefix_move_ids(self) -> tuple[int, ...]:
        return self.trace_move_ids[: self.cut]

    @property
    def suffix_move_ids(self) -> tuple[int, ...]:
        return self.trace_move_ids[self.cut :]


def generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    """Measure legality for the standard full-game trace evaluation."""
    return trace_generation_metrics(
        model,
        batch,
        args,
        legality_check=lambda _prompt_tokens, generated_tokens: (
            othello.legal_prefix_length(generated_tokens)
        ),
        inference_mode=inference_mode,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    return format_legal_generation_metrics(metrics)


def build_eval_examples(
    traces: Iterable[list[int]],
    *,
    stoi: dict[str, int],
    evaluation_mode: str,
    prefix_fractions: Iterable[float],
    rng: random.Random,
) -> list[OthelloEvalExample]:
    """Turn validation games into deterministic continuation protocols."""
    examples: list[OthelloEvalExample] = []
    fractions = tuple(prefix_fractions)
    for example_index, square_trace in enumerate(traces):
        move_ids = tuple(
            stoi[othello.move_token(square)]
            for square in square_trace
        )
        if not move_ids:
            raise ValueError(
                "Othello evaluation trace must contain at least one move"
            )

        cuts: list[tuple[str, int]] = []
        if evaluation_mode in {"full-game", "all"}:
            cuts.append(("full-game", 0))
        if evaluation_mode in {"random-prefix", "all"}:
            cuts.append(
                ("random-prefix", rng.randint(1, len(move_ids) - 1))
            )
        if evaluation_mode in {"prefix-grid", "all"}:
            for fraction in fractions:
                cut = min(
                    len(move_ids) - 1,
                    max(1, round(len(move_ids) * fraction)),
                )
                cuts.append((f"prefix-grid-{fraction:g}", cut))

        seen: set[tuple[str, int]] = set()
        for protocol, cut in cuts:
            key = (protocol, cut)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                OthelloEvalExample(
                    example_index=example_index,
                    protocol=protocol,
                    trace_move_ids=move_ids,
                    cut=cut,
                )
            )
    return examples


def legal_set_step_metrics(
    logits: torch.Tensor,
    legal_token_ids: tuple[int, ...],
    gold_token_id: int,
) -> dict[str, float]:
    """Score total probability on the legal next-move set."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab]")
    if not legal_token_ids:
        raise ValueError(
            "legal_token_ids must not be empty at an active move position"
        )
    if gold_token_id not in legal_token_ids:
        raise ValueError("gold Othello move is not in the legal set")
    legal_index = torch.tensor(
        legal_token_ids,
        dtype=torch.long,
        device=logits.device,
    )
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    legal_log_mass = torch.logsumexp(
        log_probabilities.index_select(0, legal_index),
        dim=0,
    )
    return {
        "legal_set_nll": float((-legal_log_mass).item()),
        "gold_move_nll": float((-log_probabilities[gold_token_id]).item()),
        "legal_probability_mass": float(legal_log_mass.exp().item()),
        "top1_legal": float(int(logits.argmax().item()) in legal_token_ids),
        "legal_set_size": float(len(legal_token_ids)),
    }


def sample_validation_traces(
    args,
    *,
    count: int,
    rng: random.Random,
) -> list[list[int]]:
    dataset = othello.load_othello_dataset(
        split="val",
        othello_data_dir=args.othello_data_dir,
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
        othello_dataset_seed=args.othello_dataset_seed,
    )
    if count > len(dataset):
        raise ValueError(
            f"requested {count} validation games without replacement, "
            f"but the dataset contains only {len(dataset)}"
        )
    indices = rng.sample(range(len(dataset)), count)
    return [dataset.trace_at(index) for index in indices]


def serialized_prompt(
    stoi: dict[str, int],
    prefix_move_ids: tuple[int, ...],
) -> list[int]:
    prompt = [stoi[BOS_TOKEN], stoi[SEP_TOKEN]]
    prompt.extend(prefix_move_ids)
    return prompt


def score_generated_continuation(
    prefix_move_ids: tuple[int, ...],
    generated_token_ids: list[int],
    *,
    eos_id: int,
) -> dict[str, float]:
    eos_position = next(
        (
            position
            for position, token_id in enumerate(generated_token_ids)
            if token_id == eos_id
        ),
        None,
    )
    attempted_moves = (
        generated_token_ids
        if eos_position is None
        else generated_token_ids[:eos_position]
    )
    accepted = list(prefix_move_ids)
    legal_prefix_length = 0
    terminal_reached = False
    for token_id in attempted_moves:
        legal_ids = othello.legal_move_token_ids_after_prefix(accepted)
        if not legal_ids or token_id not in legal_ids:
            break
        accepted.append(token_id)
        legal_prefix_length += 1
    else:
        terminal_reached = not othello.legal_move_token_ids_after_prefix(
            accepted
        )

    sequence_legality = float(
        eos_position is not None
        and legal_prefix_length == len(attempted_moves)
        and terminal_reached
    )
    denominator = len(attempted_moves)
    legal_move_fraction = (
        float(legal_prefix_length) / denominator
        if denominator
        else float(sequence_legality)
    )
    return {
        "sequence_legality": sequence_legality,
        "legal_move_fraction": legal_move_fraction,
        "legal_prefix_length": float(legal_prefix_length),
        "generated_move_count": float(len(attempted_moves)),
        "terminal_reached": float(terminal_reached),
        "eos_emitted": float(eos_position is not None),
    }


@torch.no_grad()
def teacher_forced_metrics(
    model,
    args,
    stoi: dict[str, int],
    example: OthelloEvalExample,
    *,
    inference_mode: str,
    recompute_cache: dict[tuple[int, ...], torch.Tensor],
) -> dict[str, float]:
    """Compute legal-set and gold-move metrics on a reference suffix."""
    trace = example.trace_move_ids
    eos_id = stoi[EOS_TOKEN]
    prompt_tokens = serialized_prompt(
        stoi,
        example.prefix_move_ids,
    )
    step_metrics = []

    if inference_mode == "recompute":
        logits_by_position = recompute_cache.get(trace)
        if logits_by_position is None:
            full_input = serialized_prompt(stoi, trace)
            tensor = torch.tensor(
                [full_input],
                dtype=torch.long,
                device=args.device,
            )
            logits_by_position = model(tensor).logits[0].detach()
            recompute_cache[trace] = logits_by_position
        base_length = len(serialized_prompt(stoi, ()))
        for move_index in range(example.cut, len(trace)):
            legal_ids = othello.legal_move_token_ids_after_prefix(
                trace[:move_index]
            )
            step_metrics.append(
                legal_set_step_metrics(
                    logits_by_position[base_length - 1 + move_index],
                    legal_ids,
                    trace[move_index],
                )
            )
        eos_logits = logits_by_position[base_length - 1 + len(trace)]
    elif inference_mode == "append_recurrent":
        state = model.prefill_recurrent(
            torch.tensor(
                [prompt_tokens],
                dtype=torch.long,
                device=args.device,
            )
        )
        for move_index in range(example.cut, len(trace)):
            legal_ids = othello.legal_move_token_ids_after_prefix(
                trace[:move_index]
            )
            step_metrics.append(
                legal_set_step_metrics(
                    state.next_token_logits[0],
                    legal_ids,
                    trace[move_index],
                )
            )
            state = model.recurrent_step(
                state,
                torch.tensor(
                    [[trace[move_index]]],
                    dtype=torch.long,
                    device=args.device,
                ),
            )
        eos_logits = state.next_token_logits[0]
    else:
        raise ValueError(f"unsupported inference mode: {inference_mode}")

    if not step_metrics:
        raise ValueError(
            "Othello prefix cut must leave at least one move to evaluate"
        )
    move_count = len(step_metrics)
    result = {
        key: sum(item[key] for item in step_metrics) / move_count
        for key in step_metrics[0]
    }
    result["move_count"] = float(move_count)
    result["eos_nll"] = float(
        -F.log_softmax(eos_logits.float(), dim=-1)[eos_id].item()
    )
    return result


def length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 15:
        return "1-15"
    if length <= 30:
        return "16-30"
    if length <= 45:
        return "31-45"
    return "46+"


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "teacher_move_count": 0}
    free_fields = tuple(rows[0]["free_generation"])
    teacher_fields = tuple(
        field
        for field in rows[0]["teacher_forced"]
        if field != "move_count"
    )
    teacher_move_count = sum(
        int(row["teacher_forced"]["move_count"])
        for row in rows
    )
    summary = {
        "count": len(rows),
        "teacher_move_count": teacher_move_count,
        "free_generation": {
            field: sum(
                float(row["free_generation"][field])
                for row in rows
            )
            / len(rows)
            for field in free_fields
        },
        "teacher_forced": {},
    }
    for field in teacher_fields:
        if field == "eos_nll":
            summary["teacher_forced"][field] = (
                sum(
                    float(row["teacher_forced"][field])
                    for row in rows
                )
                / len(rows)
            )
        else:
            summary["teacher_forced"][field] = sum(
                float(row["teacher_forced"][field])
                * int(row["teacher_forced"]["move_count"])
                for row in rows
            ) / teacher_move_count
    return summary


def group_summaries(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        group: summarize_rows(items)
        for group, items in sorted(groups.items())
    }


__all__ = [
    "DEFAULT_PREFIX_FRACTIONS",
    "EVALUATION_MODES",
    "OthelloEvalExample",
    "build_eval_examples",
    "format_metrics",
    "generation_metrics",
    "group_summaries",
    "legal_set_step_metrics",
    "length_bucket",
    "sample_validation_traces",
    "score_generated_continuation",
    "serialized_prompt",
    "summarize_rows",
    "teacher_forced_metrics",
]
