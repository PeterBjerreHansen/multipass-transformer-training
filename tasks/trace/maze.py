"""Searchformer-style random-wall mazes as a fixed-suffix trace task.

Each prompt names the start, goal, and blocked grid cells. The target is one
deterministically selected shortest coordinate path, including both endpoints.
The generator samples an exact wall count from a uniformly sampled density and
rejects mazes that are unsolvable or have a path shorter than the grid width.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, Iterable, Sequence, Tuple

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    make_sequence,
)


START_TOKEN = "<start>"
GOAL_TOKEN = "<goal>"
WALLS_TOKEN = "<walls>"
START_TOKEN_ID = 4
GOAL_TOKEN_ID = 5
WALLS_TOKEN_ID = 6
CELL_TOKEN_OFFSET = 7
MAX_SAMPLING_ATTEMPTS = 10_000


@dataclass(frozen=True)
class MazeDistribution:
    """A named distribution over random blocked-cell mazes."""

    name: str
    size: int
    min_wall_fraction: float
    max_wall_fraction: float
    min_path_length: int


@dataclass(frozen=True)
class MazeInstance:
    """One solver-verified maze, using row-major integer cell indices."""

    distribution_name: str
    size: int
    walls: frozenset[int]
    start: int
    goal: int
    path: tuple[int, ...]
    shortest_path_count: int
    sampled_wall_fraction: float


MAZE_DISTRIBUTIONS = {
    # Small, inexpensive software check. This is not a literature-comparison
    # distribution because the 5x5 grid needs a lower density to remain useful.
    "smoke": MazeDistribution(
        name="smoke",
        size=5,
        min_wall_fraction=0.15,
        max_wall_fraction=0.30,
        min_path_length=5,
    ),
    "searchformer_10": MazeDistribution(
        name="searchformer_10",
        size=10,
        min_wall_fraction=0.30,
        max_wall_fraction=0.50,
        min_path_length=10,
    ),
    "searchformer_20": MazeDistribution(
        name="searchformer_20",
        size=20,
        min_wall_fraction=0.30,
        max_wall_fraction=0.50,
        min_path_length=20,
    ),
    "searchformer_30": MazeDistribution(
        name="searchformer_30",
        size=30,
        min_wall_fraction=0.30,
        max_wall_fraction=0.50,
        min_path_length=30,
    ),
}


def get_maze_distribution(name: str) -> MazeDistribution:
    try:
        return MAZE_DISTRIBUTIONS[name]
    except KeyError as error:
        raise ValueError(f"unsupported maze distribution: {name}") from error


def coordinate_token(row: int, column: int) -> str:
    if row < 0 or column < 0:
        raise ValueError("maze coordinates must be non-negative")
    return f"r{row}c{column}"


def cell_to_coordinate(cell: int, *, size: int) -> tuple[int, int]:
    if size < 2:
        raise ValueError("maze size must be at least 2")
    if not 0 <= cell < size * size:
        raise ValueError("cell is outside the maze")
    return divmod(cell, size)


def build_maze_vocab(
    distribution_name: str,
) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
    distribution = get_maze_distribution(distribution_name)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        START_TOKEN,
        GOAL_TOKEN,
        WALLS_TOKEN,
    ]
    tokens.extend(
        coordinate_token(row, column)
        for row in range(distribution.size)
        for column in range(distribution.size)
    )
    return build_vocab(tokens)


def required_block_size(distribution_name: str) -> int:
    """Return a tight upper bound for every maze sequence in a distribution.

    A maze with ``W`` walls has five prompt markers/coordinates and at most
    ``size**2 - W`` path cells. BOS and SEP occupy two more model positions;
    EOS is the shifted target and does not occupy an input position.
    """
    distribution = get_maze_distribution(distribution_name)
    num_cells = distribution.size * distribution.size
    minimum_wall_count = round(distribution.min_wall_fraction * num_cells)
    # The serialized input has one token per wall and the path can visit at
    # most every open cell. Their sum is therefore bounded by num_cells.
    # Retain the explicit calculation so the bound remains correct if a future
    # distribution changes how endpoints and walls are sampled.
    maximum_path_cells = num_cells - minimum_wall_count
    return 7 + minimum_wall_count + maximum_path_cells


def neighboring_cells(cell: int, *, size: int) -> tuple[int, ...]:
    """Return four-neighbor cells in a fixed up, right, down, left order."""
    row, column = cell_to_coordinate(cell, size=size)
    neighbors = []
    if row > 0:
        neighbors.append(cell - size)
    if column + 1 < size:
        neighbors.append(cell + 1)
    if row + 1 < size:
        neighbors.append(cell + size)
    if column > 0:
        neighbors.append(cell - 1)
    return tuple(neighbors)


def solve_maze(
    size: int,
    walls: Iterable[int],
    start: int,
    goal: int,
) -> tuple[list[int], int]:
    """Return a canonical shortest path and its multiplicity, capped at two."""
    if size < 2:
        raise ValueError("maze size must be at least 2")
    num_cells = size * size
    wall_set = frozenset(int(cell) for cell in walls)
    if any(not 0 <= cell < num_cells for cell in wall_set):
        raise ValueError("wall is outside the maze")
    if not 0 <= start < num_cells or not 0 <= goal < num_cells:
        raise ValueError("start and goal must be valid cells")
    if start == goal:
        raise ValueError("start and goal must differ")
    if start in wall_set or goal in wall_set:
        raise ValueError("start and goal must be open cells")

    distances = [-1] * num_cells
    path_counts = [0] * num_cells
    parents: list[int | None] = [None] * num_cells
    distances[start] = 0
    path_counts[start] = 1
    queue = deque([start])
    while queue:
        source = queue.popleft()
        for target in neighboring_cells(source, size=size):
            if target in wall_set:
                continue
            candidate_distance = distances[source] + 1
            if distances[target] == -1:
                distances[target] = candidate_distance
                path_counts[target] = path_counts[source]
                parents[target] = source
                queue.append(target)
            elif distances[target] == candidate_distance:
                path_counts[target] = min(
                    2,
                    path_counts[target] + path_counts[source],
                )

    if distances[goal] < 0:
        raise ValueError("goal is unreachable from start")
    path = [goal]
    current = goal
    while current != start:
        parent = parents[current]
        if parent is None:
            raise RuntimeError("maze path reconstruction failed")
        path.append(parent)
        current = parent
    path.reverse()
    return path, path_counts[goal]


def sample_random_wall_maze(
    distribution_name: str,
    rng: random.Random,
) -> MazeInstance:
    """Sample one accepted random-wall maze by literal rejection sampling."""
    distribution = get_maze_distribution(distribution_name)
    num_cells = distribution.size * distribution.size
    cells = range(num_cells)
    for _attempt in range(MAX_SAMPLING_ATTEMPTS):
        sampled_fraction = rng.uniform(
            distribution.min_wall_fraction,
            distribution.max_wall_fraction,
        )
        wall_count = min(num_cells - 2, round(sampled_fraction * num_cells))
        walls = frozenset(rng.sample(cells, wall_count))
        open_cells = [cell for cell in cells if cell not in walls]
        start, goal = rng.sample(open_cells, 2)
        try:
            path, path_count = solve_maze(
                distribution.size,
                walls,
                start,
                goal,
            )
        except ValueError:
            # All inputs above are constructed to satisfy solve_maze's static
            # invariants, so only an unreachable goal is an expected rejection.
            continue
        path_length = len(path) - 1
        if path_length < distribution.min_path_length:
            continue
        return MazeInstance(
            distribution_name=distribution.name,
            size=distribution.size,
            walls=walls,
            start=start,
            goal=goal,
            path=tuple(path),
            shortest_path_count=path_count,
            sampled_wall_fraction=sampled_fraction,
        )
    raise RuntimeError(
        f"failed to sample {distribution.name} after "
        f"{MAX_SAMPLING_ATTEMPTS} attempts"
    )


def sample_maze_example(
    distribution_name: str,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], MazeInstance]:
    instance = sample_random_wall_maze(distribution_name, rng)

    def coordinate_id(cell: int) -> int:
        row, column = cell_to_coordinate(cell, size=instance.size)
        return stoi[coordinate_token(row, column)]

    serialized_walls = list(instance.walls)
    # Randomize the set serialization without advancing the topology RNG. This
    # keeps the sequence of sampled mazes stable if representations are added
    # or changed later.
    representation_rng = random.Random()
    representation_rng.setstate(rng.getstate())
    representation_rng.shuffle(serialized_walls)
    prompt = [
        stoi[START_TOKEN],
        coordinate_id(instance.start),
        stoi[GOAL_TOKEN],
        coordinate_id(instance.goal),
        stoi[WALLS_TOKEN],
        *(coordinate_id(cell) for cell in serialized_walls),
    ]
    answer = [coordinate_id(cell) for cell in instance.path]
    return prompt, answer, instance


def build_maze_batch(
    batch_size: int,
    distribution_name: str,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    get_maze_distribution(distribution_name)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _instance = sample_maze_example(
            distribution_name,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def token_id_to_cell(token_id: int, *, size: int) -> int | None:
    cell = int(token_id) - CELL_TOKEN_OFFSET
    return cell if 0 <= cell < size * size else None


def parse_maze_prompt(
    prompt_tokens: Sequence[int],
    *,
    size: int,
) -> tuple[frozenset[int], int, int]:
    """Parse a prompt without BOS or SEP into walls, start, and goal."""
    tokens = [int(token_id) for token_id in prompt_tokens]
    if len(tokens) < 5:
        raise ValueError("maze prompt is too short")
    if (
        tokens[0] != START_TOKEN_ID
        or tokens[2] != GOAL_TOKEN_ID
        or tokens[4] != WALLS_TOKEN_ID
    ):
        raise ValueError("maze prompt has invalid markers")
    start = token_id_to_cell(tokens[1], size=size)
    goal = token_id_to_cell(tokens[3], size=size)
    if start is None or goal is None:
        raise ValueError("maze prompt has an invalid endpoint")
    wall_cells = [token_id_to_cell(token, size=size) for token in tokens[5:]]
    if any(cell is None for cell in wall_cells):
        raise ValueError("maze prompt has an invalid wall coordinate")
    walls = frozenset(int(cell) for cell in wall_cells if cell is not None)
    if len(walls) != len(wall_cells):
        raise ValueError("maze prompt has duplicate walls")
    if start == goal or start in walls or goal in walls:
        raise ValueError("maze prompt has invalid start or goal cells")
    return walls, start, goal


__all__ = [
    "CELL_TOKEN_OFFSET",
    "GOAL_TOKEN",
    "MAZE_DISTRIBUTIONS",
    "MAX_SAMPLING_ATTEMPTS",
    "MazeDistribution",
    "MazeInstance",
    "START_TOKEN",
    "WALLS_TOKEN",
    "build_maze_batch",
    "build_maze_vocab",
    "cell_to_coordinate",
    "coordinate_token",
    "get_maze_distribution",
    "neighboring_cells",
    "parse_maze_prompt",
    "required_block_size",
    "sample_maze_example",
    "sample_random_wall_maze",
    "solve_maze",
    "token_id_to_cell",
]
