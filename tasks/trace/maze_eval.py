"""Solver-based evaluation for offline maze route datasets."""
from __future__ import annotations

import torch

from model_factory import supports_append_recurrent
from tasks.common import EOS_TOKEN
from tasks.trace import maze


def _is_legal_complete_path(
    path: list[int | None],
    problem: maze.MazeProblem,
) -> bool:
    if not path or path[0] != problem.start or path[-1] != problem.goal:
        return False
    if any(cell is None or cell in problem.walls for cell in path):
        return False
    return all(
        current in maze.neighboring_cells(
            int(previous),
            width=problem.width,
            height=problem.height,
        )
        for previous, current in zip(path, path[1:])
        if previous is not None
    )


@torch.no_grad()
def generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    """Measure any optimal route and exact imitation of the selected policy."""
    mode = (
        inference_mode or args.inference_mode
        if supports_append_recurrent(args.architecture)
        else "recompute"
    )
    do_sample = getattr(args, "token_selection", "argmax") == "sample"
    _vocab, stoi, itos = maze.build_maze_vocab(
        args.maze_data_dir,
        args.maze_input_representation,
        args.maze_target_representation,
        args.maze_route_policy,
    )
    eos_id = stoi[EOS_TOKEN]
    optimal_routes = 0.0
    exact_target_routes = 0.0

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        problem = maze.parse_maze_prompt(
            batch.idx[row, 1 : prompt_len - 1].tolist(),
            itos=itos,
            input_representation=args.maze_input_representation,
        )
        shortest_path = maze.solve_maze(problem)
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
        marker_ok, generated_path = maze.decode_maze_target(
            generated_suffix[:eos_position],
            problem=problem,
            itos=itos,
            target_representation=args.maze_target_representation,
        )
        eos_ok = eos_position < len(generated_suffix)
        optimal = (
            marker_ok
            and eos_ok
            and _is_legal_complete_path(generated_path, problem)
            and len(generated_path) == len(shortest_path)
        )
        optimal_routes += float(optimal)
        exact_target_routes += float(generated_suffix == target_suffix)

    count = int(batch.idx.shape[0])
    return {
        "optimal_route": optimal_routes / count,
        "exact_target_route": exact_target_routes / count,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(
        (
            f"optimal_route {metrics['optimal_route']:.3f}",
            f"exact_target_route {metrics['exact_target_route']:.3f}",
        )
    )


__all__ = ["format_metrics", "generation_metrics"]
