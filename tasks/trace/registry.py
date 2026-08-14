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
    build_eval_batches_fn: Callable | None = None

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

    def build_eval_batches(self, args, stoi, *, split: str):
        if self.build_eval_batches_fn is None:
            return None
        return self.build_eval_batches_fn(args, stoi, split)


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


def _bind_shortest_path_dataset(args):
    data_dir = getattr(args, "shortest_path_data_dir", None)
    if not data_dir:
        raise ValueError("shortest_path requires --shortest-path-data-dir")
    bundle = shortest_path.load_shortest_path_bundle(data_dir)
    saved_id = getattr(args, "shortest_path_dataset_id", None)
    if saved_id is not None and saved_id != bundle.dataset_id:
        raise ValueError(
            "shortest-path dataset ID mismatch: saved run uses "
            f"{saved_id}, but {bundle.directory} contains {bundle.dataset_id}"
        )
    saved_distribution = getattr(args, "shortest_path_distribution", None)
    if (
        saved_distribution is not None
        and saved_distribution != bundle.distribution_name
    ):
        raise ValueError(
            "shortest-path distribution mismatch: saved run uses "
            f"{saved_distribution}, but the dataset contains "
            f"{bundle.distribution_name}"
        )
    args.shortest_path_data_dir = str(bundle.directory)
    args.shortest_path_distribution = bundle.distribution_name
    args.shortest_path_dataset_id = bundle.dataset_id
    return bundle


def _shortest_path_vocab(args):
    bundle = _bind_shortest_path_dataset(args)
    return shortest_path.build_shortest_path_vocab(bundle.directory)


def _shortest_path_block_size(args) -> int:
    bundle = _bind_shortest_path_dataset(args)
    return shortest_path.required_block_size(bundle.directory)


def _shortest_path_batch(args, _stoi, rng: random.Random, split: str):
    bundle = _bind_shortest_path_dataset(args)
    return shortest_path.build_shortest_path_batch(
        batch_size=args.batch_size,
        shortest_path_data_dir=bundle.directory,
        split=split,
        device=args.device,
        rng=rng,
    )


def _shortest_path_eval_batches(args, _stoi, split: str):
    bundle = _bind_shortest_path_dataset(args)
    return shortest_path.build_shortest_path_eval_batches(
        batch_size=args.batch_size,
        num_batches=args.eval_batches,
        shortest_path_data_dir=bundle.directory,
        split=split,
        device=args.device,
    )


def _shortest_path_metrics(model, batch, args, inference_mode: str | None):
    return shortest_path_eval.generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


def _maze_vocab(args):
    bundle = _bind_maze_dataset(args)
    return list(bundle.vocab), dict(bundle.stoi), dict(bundle.itos)


def _bind_maze_dataset(args):
    data_dir = getattr(args, "maze_data_dir", None)
    if not data_dir:
        raise ValueError("maze requires --maze-data-dir")
    bundle = maze.load_maze_bundle(
        maze_data_dir=data_dir,
        input_representation=args.maze_input_representation,
        target_representation=args.maze_target_representation,
        route_policy=args.maze_route_policy,
    )
    saved_id = getattr(args, "maze_dataset_id", None)
    if saved_id is not None and saved_id != bundle.dataset_id:
        raise ValueError(
            "maze dataset ID mismatch: saved run uses "
            f"{saved_id}, but {bundle.directory} contains {bundle.dataset_id}"
        )
    args.maze_data_dir = str(bundle.directory)
    args.maze_dataset_id = bundle.dataset_id
    return bundle


def _maze_block_size(args) -> int:
    return int(_bind_maze_dataset(args).manifest["block_size"])


def _maze_batch(args, _stoi, rng: random.Random, split: str):
    bundle = _bind_maze_dataset(args)
    return maze.build_maze_batch(
        batch_size=args.batch_size,
        maze_data_dir=bundle.directory,
        input_representation=args.maze_input_representation,
        target_representation=args.maze_target_representation,
        route_policy=args.maze_route_policy,
        split=split,
        device=args.device,
        rng=rng,
    )


def _maze_eval_batches(args, _stoi, split: str):
    bundle = _bind_maze_dataset(args)
    return maze.build_maze_eval_batches(
        batch_size=args.batch_size,
        num_batches=args.eval_batches,
        maze_data_dir=bundle.directory,
        input_representation=args.maze_input_representation,
        target_representation=args.maze_target_representation,
        route_policy=args.maze_route_policy,
        split=split,
        device=args.device,
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
        build_eval_batches_fn=_maze_eval_batches,
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
        build_eval_batches_fn=_shortest_path_eval_batches,
    ),
}


def get_trace_task(name: str) -> TraceTask:
    try:
        return TRACE_TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported trace task: {name}") from error


__all__ = ["TRACE_TASKS", "TraceTask", "get_trace_task"]
