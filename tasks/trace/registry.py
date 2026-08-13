from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from tasks.trace import (
    maze,
    maze_eval,
    othello,
    othello_eval,
    shortest_path,
    shortest_path_eval,
)


@dataclass(frozen=True)
class TraceTask:
    """Adapter for the behavior shared by fixed-suffix trace tasks."""

    name: str
    build_vocab_fn: Callable
    required_block_size_fn: Callable
    build_batch_fn: Callable
    generation_metrics_fn: Callable
    format_metrics_fn: Callable

    def build_vocab(self, args):
        return self.build_vocab_fn(args)

    def required_block_size(self, args) -> int:
        return int(self.required_block_size_fn(args))

    def build_batch(self, args, stoi, rng: random.Random, *, split: str):
        return self.build_batch_fn(args, stoi, rng, split)

    def generation_metrics(self, model, batch, args, *, inference_mode: str | None = None):
        return self.generation_metrics_fn(model, batch, args, inference_mode)

    def format_metrics(self, metrics: dict[str, float]) -> str:
        return self.format_metrics_fn(metrics)

def _othello_vocab(args):
    return othello.build_othello_vocab(
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
    )


def _othello_block_size(args) -> int:
    return othello.required_block_size(
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
    )


def _othello_batch(args, stoi, rng: random.Random, split: str):
    return othello.build_othello_batch(
        batch_size=args.batch_size,
        stoi=stoi,
        device=args.device,
        rng=rng,
        split=split,
        othello_data_dir=args.othello_data_dir,
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
        othello_dataset_seed=args.othello_dataset_seed,
    )


def _othello_metrics(model, batch, args, inference_mode: str | None):
    return othello_eval.generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


def _shortest_path_vocab(args):
    return shortest_path.build_shortest_path_vocab(
        args.shortest_path_distribution
    )


def _shortest_path_block_size(args) -> int:
    return shortest_path.required_block_size(
        args.shortest_path_distribution
    )


def _shortest_path_batch(args, stoi, rng: random.Random, _split: str):
    return shortest_path.build_shortest_path_batch(
        batch_size=args.batch_size,
        distribution_name=args.shortest_path_distribution,
        stoi=stoi,
        device=args.device,
        rng=rng,
    )


def _shortest_path_metrics(model, batch, args, inference_mode: str | None):
    return shortest_path_eval.generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


def _maze_vocab(args):
    return maze.build_maze_vocab(
        args.maze_data_dir,
        args.maze_input_representation,
        args.maze_target_representation,
        args.maze_route_policy,
    )


def _maze_block_size(args) -> int:
    return maze.required_block_size(
        args.maze_data_dir,
        args.maze_input_representation,
        args.maze_target_representation,
        args.maze_route_policy,
    )


def _maze_batch(args, stoi, rng: random.Random, _split: str):
    return maze.build_maze_batch(
        batch_size=args.batch_size,
        maze_data_dir=args.maze_data_dir,
        input_representation=args.maze_input_representation,
        target_representation=args.maze_target_representation,
        route_policy=args.maze_route_policy,
        split=_split,
        device=args.device,
        rng=rng,
    )


def _maze_metrics(model, batch, args, inference_mode: str | None):
    return maze_eval.generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


TRACE_TASKS: dict[str, TraceTask] = {
    "maze": TraceTask(
        name="maze",
        build_vocab_fn=_maze_vocab,
        required_block_size_fn=_maze_block_size,
        build_batch_fn=_maze_batch,
        generation_metrics_fn=_maze_metrics,
        format_metrics_fn=maze_eval.format_metrics,
    ),
    "othello": TraceTask(
        name="othello",
        build_vocab_fn=_othello_vocab,
        required_block_size_fn=_othello_block_size,
        build_batch_fn=_othello_batch,
        generation_metrics_fn=_othello_metrics,
        format_metrics_fn=othello_eval.format_metrics,
    ),
    "shortest_path": TraceTask(
        name="shortest_path",
        build_vocab_fn=_shortest_path_vocab,
        required_block_size_fn=_shortest_path_block_size,
        build_batch_fn=_shortest_path_batch,
        generation_metrics_fn=_shortest_path_metrics,
        format_metrics_fn=shortest_path_eval.format_metrics,
    ),
}


def get_trace_task(name: str) -> TraceTask:
    try:
        return TRACE_TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported trace task: {name}") from error


__all__ = ["TRACE_TASKS", "TraceTask", "get_trace_task"]
