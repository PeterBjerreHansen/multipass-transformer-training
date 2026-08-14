"""Memory-mapped offline maze datasets for route-learning experiments."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Dict, Sequence, Tuple

import numpy as np

from tasks.common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN, SymbolicBatch
from tasks.trace.compiled import (
    batch_from_compiled_indices,
    chunk_indices,
    deterministic_eval_indices,
    training_indices,
)


INPUT_REPRESENTATIONS = ("dense-cells", "sparse-cells")
TARGET_REPRESENTATIONS = ("cell-path", "actions")
ROUTE_POLICIES = ("astar", "uniform_shortest", "dfs")
COMPILED_FORMAT_VERSION = 1
DEFAULT_DATA_DIR = "data/maze/searchformer-10"
DEFAULT_SMOKE_DATA_DIR = "tests/fixtures/maze-smoke"
HEIGHT_TOKEN = "<height>"
WIDTH_TOKEN = "<width>"
GRID_TOKEN = "<grid>"
START_TOKEN = "<start>"
GOAL_TOKEN = "<goal>"
WALLS_TOKEN = "<walls>"
PATH_TOKEN = "<path>"
ACTIONS_TOKEN = "<actions>"
OPEN_CELL_TOKEN = "."
WALL_CELL_TOKEN = "#"
START_CELL_TOKEN = "S"
GOAL_CELL_TOKEN = "G"
ACTION_DELTAS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
CELL_TOKEN_PATTERN = re.compile(r"r(\d+)c(\d+)")


@dataclass(frozen=True)
class MazeProblem:
    width: int
    height: int
    walls: frozenset[int]
    start: int
    goal: int


@dataclass(frozen=True)
class CompiledMazeDataset:
    tokens: np.ndarray
    sequence_lengths: np.ndarray
    prompt_lengths: np.ndarray
    maze_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

@dataclass(frozen=True)
class CompiledMazeBundle:
    directory: Path
    manifest: dict
    vocab: tuple[str, ...]
    stoi: Dict[str, int]
    itos: Dict[int, str]
    dataset_id: str


_BUNDLE_CACHE: dict[tuple[str, str, str, str], CompiledMazeBundle] = {}
_DATASET_CACHE: dict[tuple[str, str, str, str, str], CompiledMazeDataset] = {}


def _dataset_id(manifest: dict) -> str:
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_name(
    input_representation: str,
    target_representation: str,
    route_policy: str,
) -> str:
    return f"{input_representation}__{target_representation}__{route_policy}"


def resolve_compiled_dir(
    maze_data_dir: str | Path,
    *,
    input_representation: str,
    target_representation: str,
    route_policy: str,
) -> Path:
    root = Path(maze_data_dir).expanduser().resolve()
    if (root / "manifest.json").is_file():
        return root
    return root / _artifact_name(
        input_representation,
        target_representation,
        route_policy,
    )


def load_maze_bundle(
    *,
    maze_data_dir: str | Path,
    input_representation: str,
    target_representation: str,
    route_policy: str,
) -> CompiledMazeBundle:
    if input_representation not in INPUT_REPRESENTATIONS:
        raise ValueError(f"unsupported maze input representation: {input_representation}")
    if target_representation not in TARGET_REPRESENTATIONS:
        raise ValueError(f"unsupported maze target representation: {target_representation}")
    if route_policy not in ROUTE_POLICIES:
        raise ValueError(f"unsupported maze route policy: {route_policy}")
    directory = resolve_compiled_dir(
        maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    key = (
        str(directory),
        input_representation,
        target_representation,
        route_policy,
    )
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "compiled maze dataset not found at "
            f"{directory}. Generate and compile it with maze-data-generator; "
            "multipass never generates maze data online."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "compiled_format_version": COMPILED_FORMAT_VERSION,
        "input_representation": input_representation,
        "target_representation": target_representation,
        "route_policy": route_policy,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"maze dataset manifest {field} is {manifest.get(field)!r}; "
                f"expected {value!r}"
            )
    source_sha256 = manifest.get("source_sha256")
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("compiled maze manifest has an invalid source_sha256")
    vocab_path = directory / str(manifest.get("vocab", "vocab.json"))
    vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = tuple(str(token) for token in vocab_payload["tokens"])
    if len(vocab) != len(set(vocab)) or len(vocab) != int(manifest["vocab_size"]):
        raise ValueError("compiled maze vocabulary is inconsistent")
    stoi = {token: index for index, token in enumerate(vocab)}
    itos = {index: token for token, index in stoi.items()}
    special_tokens = {
        "pad": PAD_TOKEN,
        "bos": BOS_TOKEN,
        "sep": SEP_TOKEN,
        "eos": EOS_TOKEN,
    }
    for token in special_tokens.values():
        if token not in stoi:
            raise ValueError(f"compiled maze vocabulary is missing {token}")
    expected_special_ids = {
        name: stoi[token] for name, token in special_tokens.items()
    }
    if manifest.get("special_token_ids") != expected_special_ids:
        raise ValueError("compiled maze special-token IDs are inconsistent")
    max_sequence_length = int(manifest["max_sequence_length"])
    if int(manifest["block_size"]) != max_sequence_length - 1:
        raise ValueError("compiled maze block size is inconsistent")
    if manifest.get("token_dtype") != "uint16":
        raise ValueError("compiled maze token dtype must be uint16")
    bundle = CompiledMazeBundle(
        directory,
        manifest,
        vocab,
        stoi,
        itos,
        _dataset_id(manifest),
    )
    _BUNDLE_CACHE[key] = bundle
    return bundle


def load_maze_dataset(
    *,
    split: str,
    maze_data_dir: str | Path,
    input_representation: str,
    target_representation: str,
    route_policy: str,
) -> CompiledMazeDataset:
    canonical_split = "validation" if split == "val" else split
    if canonical_split not in {"train", "validation", "test"}:
        raise ValueError("maze split must be train, val, validation, or test")
    bundle = load_maze_bundle(
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    key = (
        str(bundle.directory),
        canonical_split,
        input_representation,
        target_representation,
        route_policy,
    )
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        files = bundle.manifest["splits"][canonical_split]
    except KeyError as error:
        raise ValueError(
            f"compiled maze dataset has no {canonical_split!r} split"
        ) from error
    dataset = CompiledMazeDataset(
        tokens=np.load(bundle.directory / files["tokens"], mmap_mode="r", allow_pickle=False),
        sequence_lengths=np.load(
            bundle.directory / files["sequence_lengths"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        prompt_lengths=np.load(
            bundle.directory / files["prompt_lengths"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        maze_ids=np.load(bundle.directory / files["maze_ids"], mmap_mode="r", allow_pickle=False),
    )
    expected_count = int(files["count"])
    if not (
        expected_count > 0
        and len(dataset) == expected_count
        and dataset.tokens.ndim == 2
        and dataset.tokens.shape[1] == int(bundle.manifest["max_sequence_length"])
        and dataset.sequence_lengths.shape == (expected_count,)
        and dataset.prompt_lengths.shape == (expected_count,)
        and dataset.maze_ids.shape == (expected_count,)
    ):
        raise ValueError("compiled maze split arrays have inconsistent shapes")
    if dataset.tokens.dtype != np.uint16:
        raise ValueError("compiled maze tokens must use uint16")
    if dataset.sequence_lengths.dtype != np.uint16:
        raise ValueError("compiled maze sequence lengths must use uint16")
    if dataset.prompt_lengths.dtype != np.uint16:
        raise ValueError("compiled maze prompt lengths must use uint16")
    if dataset.maze_ids.dtype != np.dtype("S24"):
        raise ValueError("compiled maze IDs must use fixed-width 24-byte strings")
    if np.any(dataset.prompt_lengths >= dataset.sequence_lengths):
        raise ValueError("compiled maze prompt lengths must precede sequence ends")
    if np.any(dataset.sequence_lengths > dataset.tokens.shape[1]):
        raise ValueError("compiled maze sequence lengths exceed the token array")
    _DATASET_CACHE[key] = dataset
    return dataset


def build_maze_vocab(
    maze_data_dir: str | Path,
    input_representation: str = "sparse-cells",
    target_representation: str = "cell-path",
    route_policy: str = "astar",
) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
    bundle = load_maze_bundle(
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    return list(bundle.vocab), dict(bundle.stoi), dict(bundle.itos)


def required_block_size(
    maze_data_dir: str | Path,
    input_representation: str = "sparse-cells",
    target_representation: str = "cell-path",
    route_policy: str = "astar",
) -> int:
    bundle = load_maze_bundle(
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    return int(bundle.manifest["block_size"])


def build_maze_batch(
    batch_size: int,
    maze_data_dir: str | Path,
    input_representation: str,
    target_representation: str,
    route_policy: str,
    *,
    split: str,
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = rng or random.Random()
    dataset = load_maze_dataset(
        split=split,
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    bundle = load_maze_bundle(
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    indices = training_indices(len(dataset), batch_size, rng)
    return batch_from_compiled_indices(
        dataset,
        indices,
        pad_id=bundle.stoi[PAD_TOKEN],
        device=device,
    )


def build_maze_eval_batches(
    *,
    batch_size: int,
    num_batches: int,
    maze_data_dir: str | Path,
    input_representation: str,
    target_representation: str,
    route_policy: str,
    split: str,
    device=None,
) -> list[SymbolicBatch]:
    if batch_size < 1 or num_batches < 1:
        raise ValueError("batch_size and num_batches must be positive")
    dataset = load_maze_dataset(
        split=split,
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    bundle = load_maze_bundle(
        maze_data_dir=maze_data_dir,
        input_representation=input_representation,
        target_representation=target_representation,
        route_policy=route_policy,
    )
    canonical_split = "validation" if split == "val" else split
    indices = deterministic_eval_indices(
        dataset_id=bundle.dataset_id,
        split=canonical_split,
        dataset_size=len(dataset),
        count=batch_size * num_batches,
    )
    return [
        batch_from_compiled_indices(
            dataset,
            batch_indices,
            pad_id=bundle.stoi[PAD_TOKEN],
            device=device,
        )
        for batch_indices in chunk_indices(indices, batch_size)
    ]


def cell_to_coordinate(cell: int, *, width: int, height: int) -> tuple[int, int]:
    if not 0 <= cell < width * height:
        raise ValueError("cell is outside the maze")
    return divmod(cell, width)


def token_to_cell(token: str, *, width: int, height: int) -> int | None:
    match = CELL_TOKEN_PATTERN.fullmatch(token)
    if match is None:
        return None
    row, column = int(match.group(1)), int(match.group(2))
    if not 0 <= row < height or not 0 <= column < width:
        return None
    return row * width + column


def neighboring_cells(cell: int, *, width: int, height: int) -> tuple[int, ...]:
    row, column = cell_to_coordinate(cell, width=width, height=height)
    result = []
    for row_delta, column_delta in ACTION_DELTAS.values():
        next_row, next_column = row + row_delta, column + column_delta
        if 0 <= next_row < height and 0 <= next_column < width:
            result.append(next_row * width + next_column)
    return tuple(result)


def solve_maze(problem: MazeProblem) -> list[int]:
    distances = [-1] * (problem.width * problem.height)
    parents: list[int | None] = [None] * len(distances)
    distances[problem.start] = 0
    queue = deque([problem.start])
    while queue:
        source = queue.popleft()
        for target in neighboring_cells(
            source,
            width=problem.width,
            height=problem.height,
        ):
            if target in problem.walls or distances[target] >= 0:
                continue
            distances[target] = distances[source] + 1
            parents[target] = source
            queue.append(target)
    if distances[problem.goal] < 0:
        raise ValueError("goal is unreachable from start")
    path = [problem.goal]
    while path[-1] != problem.start:
        parent = parents[path[-1]]
        if parent is None:
            raise RuntimeError("maze path reconstruction failed")
        path.append(parent)
    return list(reversed(path))


def parse_maze_prompt(
    prompt_tokens: Sequence[int],
    *,
    itos: Dict[int, str],
    input_representation: str,
) -> MazeProblem:
    try:
        tokens = [itos[int(token_id)] for token_id in prompt_tokens]
    except KeyError as error:
        raise ValueError("maze prompt contains an unknown token") from error
    if len(tokens) < 4 or tokens[0] != HEIGHT_TOKEN or tokens[2] != WIDTH_TOKEN:
        raise ValueError("maze prompt has invalid dimension markers")
    height, width = int(tokens[1]), int(tokens[3])
    if input_representation == "sparse-cells":
        if len(tokens) < 9 or tokens[4] != START_TOKEN or tokens[6] != GOAL_TOKEN or tokens[8] != WALLS_TOKEN:
            raise ValueError("sparse maze prompt has invalid markers")
        start = token_to_cell(tokens[5], width=width, height=height)
        goal = token_to_cell(tokens[7], width=width, height=height)
        wall_cells = [
            token_to_cell(token, width=width, height=height)
            for token in tokens[9:]
        ]
        if start is None or goal is None or any(cell is None for cell in wall_cells):
            raise ValueError("sparse maze prompt contains an invalid cell")
        walls = frozenset(int(cell) for cell in wall_cells if cell is not None)
        if len(walls) != len(wall_cells):
            raise ValueError("sparse maze prompt contains duplicate walls")
    elif input_representation == "dense-cells":
        if len(tokens) != 5 + width * height or tokens[4] != GRID_TOKEN:
            raise ValueError("dense maze prompt has invalid dimensions")
        walls_set = set()
        starts = []
        goals = []
        for cell, symbol in enumerate(tokens[5:]):
            if symbol == WALL_CELL_TOKEN:
                walls_set.add(cell)
            elif symbol == START_CELL_TOKEN:
                starts.append(cell)
            elif symbol == GOAL_CELL_TOKEN:
                goals.append(cell)
            elif symbol != OPEN_CELL_TOKEN:
                raise ValueError("dense maze prompt contains an invalid symbol")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("dense maze prompt must contain one start and goal")
        start, goal, walls = starts[0], goals[0], frozenset(walls_set)
    else:
        raise ValueError(f"unsupported maze input representation: {input_representation}")
    if start == goal or start in walls or goal in walls:
        raise ValueError("maze prompt has invalid endpoints")
    return MazeProblem(width, height, walls, start, goal)


def decode_maze_target(
    token_ids: Sequence[int],
    *,
    problem: MazeProblem,
    itos: Dict[int, str],
    target_representation: str,
) -> tuple[bool, list[int | None]]:
    if not token_ids:
        return False, []
    marker = itos.get(int(token_ids[0]))
    body = token_ids[1:]
    if target_representation == "cell-path":
        path = [
            token_to_cell(
                itos.get(int(token_id), ""),
                width=problem.width,
                height=problem.height,
            )
            for token_id in body
        ]
        return marker == PATH_TOKEN, path
    if target_representation == "actions":
        path: list[int | None] = [problem.start]
        for token_id in body:
            previous = path[-1]
            action = itos.get(int(token_id))
            delta = ACTION_DELTAS.get(action) if action is not None else None
            if previous is None or delta is None:
                path.append(None)
                continue
            row, column = cell_to_coordinate(
                previous,
                width=problem.width,
                height=problem.height,
            )
            row, column = row + delta[0], column + delta[1]
            path.append(
                row * problem.width + column
                if 0 <= row < problem.height and 0 <= column < problem.width
                else None
            )
        return marker == ACTIONS_TOKEN, path
    raise ValueError(f"unsupported maze target representation: {target_representation}")


__all__ = [
    "ACTION_DELTAS",
    "ACTIONS_TOKEN",
    "CompiledMazeBundle",
    "CompiledMazeDataset",
    "DEFAULT_DATA_DIR",
    "DEFAULT_SMOKE_DATA_DIR",
    "INPUT_REPRESENTATIONS",
    "MazeProblem",
    "PATH_TOKEN",
    "ROUTE_POLICIES",
    "TARGET_REPRESENTATIONS",
    "build_maze_batch",
    "build_maze_eval_batches",
    "build_maze_vocab",
    "decode_maze_target",
    "load_maze_bundle",
    "load_maze_dataset",
    "neighboring_cells",
    "parse_maze_prompt",
    "required_block_size",
    "resolve_compiled_dir",
    "solve_maze",
]
