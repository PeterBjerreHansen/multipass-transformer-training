"""Solver-based generation evaluation for random-wall mazes."""
from __future__ import annotations

import torch

from tasks.common import EOS_TOKEN
from tasks.trace import maze


def _legal_path_prefix_length(
    cells: list[int | None],
    *,
    size: int,
    walls: frozenset[int],
    start: int,
) -> int:
    if not cells or cells[0] != start:
        return 0
    legal_length = 1
    for previous, current in zip(cells, cells[1:]):
        if (
            previous is None
            or current is None
            or current in walls
            or current not in maze.neighboring_cells(previous, size=size)
        ):
            break
        legal_length += 1
    return legal_length


@torch.no_grad()
def generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    """Evaluate canonical exact match and correctness of any optimal route."""
    mode = (
        "recompute"
        if args.architecture == "transformer"
        else (inference_mode or args.inference_mode)
    )
    do_sample = getattr(args, "token_selection", "argmax") == "sample"
    distribution = maze.get_maze_distribution(args.maze_distribution)
    _vocab, stoi, _itos = maze.build_maze_vocab(args.maze_distribution)
    eos_id = stoi[EOS_TOKEN]
    totals = {
        "optimal_path": 0.0,
        "exact_path": 0.0,
        "goal_reached": 0.0,
        "legal_prefix_fraction": 0.0,
        "mean_target_path_length": 0.0,
        "mean_wall_fraction": 0.0,
        "multiple_shortest_paths": 0.0,
    }

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
        walls, start, goal = maze.parse_maze_prompt(
            prompt_tokens,
            size=distribution.size,
        )
        canonical_path, path_count = maze.solve_maze(
            distribution.size,
            walls,
            start,
            goal,
        )
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

        try:
            eos_position = generated_suffix.index(eos_id)
        except ValueError:
            eos_position = len(generated_suffix)
        generated_cells = [
            maze.token_id_to_cell(token_id, size=distribution.size)
            for token_id in generated_suffix[:eos_position]
        ]
        legal_prefix_length = _legal_path_prefix_length(
            generated_cells,
            size=distribution.size,
            walls=walls,
            start=start,
        )
        complete_legal_path = legal_prefix_length == len(generated_cells)
        eos_ok = eos_position == len(generated_suffix) - 1
        reaches_goal = (
            complete_legal_path
            and bool(generated_cells)
            and generated_cells[-1] == goal
        )
        shortest_path_length = len(canonical_path) - 1
        optimal_path = (
            reaches_goal
            and eos_ok
            and len(generated_cells) - 1 == shortest_path_length
        )

        totals["optimal_path"] += float(optimal_path)
        totals["exact_path"] += float(generated_suffix == target_suffix)
        totals["goal_reached"] += float(reaches_goal)
        totals["legal_prefix_fraction"] += legal_prefix_length / max(
            len(generated_cells),
            1,
        )
        totals["mean_target_path_length"] += float(shortest_path_length)
        totals["mean_wall_fraction"] += len(walls) / float(
            distribution.size * distribution.size
        )
        totals["multiple_shortest_paths"] += float(path_count > 1)

    count = int(batch.idx.shape[0])
    return {key: value / count for key, value in totals.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(
        (
            f"optimal {metrics['optimal_path']:.3f}",
            f"exact {metrics['exact_path']:.3f}",
            f"goal {metrics['goal_reached']:.3f}",
            f"legal_prefix {metrics['legal_prefix_fraction']:.3f}",
            f"path_len {metrics['mean_target_path_length']:.1f}",
        )
    )


__all__ = ["format_metrics", "generation_metrics"]
