"""Unique shortest-path generation as a fixed-suffix trace task.

Each example serializes a shuffled directed acyclic graph, a start node, and a
goal node. The graph is constructed to have exactly one shortest path, and the
target is the complete node sequence from start through goal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_vocab,
)


NODES_TOKEN = "<nodes>"
EDGES_TOKEN = "<edges>"
START_TOKEN = "<start>"
GOAL_TOKEN = "<goal>"
NODE_TOKEN_OFFSET = 8
PATH_LENGTH_BUCKETS = ("short", "medium", "long")
COMPILED_FORMAT_VERSION = 1
DATASET_VERSION = 1
DEFAULT_DATA_DIR = "data/shortest_path/main"
DEFAULT_EASY_DATA_DIR = "data/shortest_path/easy"
DEFAULT_SMOKE_DATA_DIR = "tests/fixtures/shortest-path-smoke"


@dataclass(frozen=True)
class ShortestPathDistribution:
    """A compact distribution over solver-verified shortest-path examples."""

    name: str
    min_nodes: int
    max_nodes: int
    min_path_length: int
    max_path_length: int
    max_out_degree: int
    min_detours: int
    max_detours: int
    max_detour_penalty: int
    min_edge_probability: float
    max_edge_probability: float


SHORTEST_PATH_DISTRIBUTIONS = {
    "easy": ShortestPathDistribution(
        name="easy",
        min_nodes=8,
        max_nodes=12,
        min_path_length=3,
        max_path_length=4,
        max_out_degree=2,
        min_detours=1,
        max_detours=2,
        max_detour_penalty=3,
        min_edge_probability=0.05,
        max_edge_probability=0.25,
    ),
    "main": ShortestPathDistribution(
        name="main",
        min_nodes=16,
        max_nodes=26,
        min_path_length=5,
        max_path_length=10,
        max_out_degree=2,
        min_detours=4,
        max_detours=6,
        max_detour_penalty=2,
        min_edge_probability=0.05,
        max_edge_probability=0.20,
    ),
}


def get_shortest_path_distribution(name: str) -> ShortestPathDistribution:
    try:
        return SHORTEST_PATH_DISTRIBUTIONS[name]
    except KeyError as error:
        raise ValueError(f"unsupported shortest-path distribution: {name}") from error


def path_length_bucket(path_length: int) -> str:
    if path_length < 1:
        raise ValueError("path length must be positive")
    if path_length <= 6:
        return "short"
    if path_length <= 8:
        return "medium"
    return "long"


def node_token(index: int) -> str:
    if index < 0:
        raise ValueError("node index must be non-negative")
    return f"n{index}"


def generation_block_size(distribution_name: str) -> int:
    """Return a safe maximum block size for every example in a distribution."""
    distribution = get_shortest_path_distribution(distribution_name)
    max_edges = distribution.max_out_degree * (distribution.max_nodes - 1)
    prompt_tokens = distribution.max_nodes + 2 * max_edges + 6
    answer_tokens = distribution.max_path_length + 1
    return 2 + prompt_tokens + answer_tokens


def build_generation_vocab(
    distribution_name: str,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    distribution = get_shortest_path_distribution(distribution_name)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        NODES_TOKEN,
        EDGES_TOKEN,
        START_TOKEN,
        GOAL_TOKEN,
    ]
    tokens.extend(node_token(index) for index in range(distribution.max_nodes))
    return build_vocab(tokens)


def solve_shortest_path(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    start: int,
    goal: int,
) -> tuple[list[int], int]:
    """Return one shortest path and the number of shortest paths, capped at two."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if not 0 <= start < num_nodes or not 0 <= goal < num_nodes:
        raise ValueError("start and goal must be valid node indices")
    if start == goal:
        raise ValueError("start and goal must differ")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    seen = set()
    for source, target in edges:
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("edge endpoint must be a valid node index")
        if source == target:
            raise ValueError("self edges are not allowed")
        if (source, target) in seen:
            raise ValueError("duplicate edges are not allowed")
        seen.add((source, target))
        adjacency[source].append(target)
    for targets in adjacency:
        targets.sort()

    distances = [-1] * num_nodes
    path_counts = [0] * num_nodes
    parents: list[int | None] = [None] * num_nodes
    distances[start] = 0
    path_counts[start] = 1
    queue = deque([start])
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            candidate_distance = distances[source] + 1
            if distances[target] == -1:
                distances[target] = candidate_distance
                path_counts[target] = path_counts[source]
                parents[target] = source
                queue.append(target)
            elif distances[target] == candidate_distance:
                path_counts[target] = min(2, path_counts[target] + path_counts[source])

    if distances[goal] < 0:
        raise ValueError("goal is unreachable from start")
    path = [goal]
    current = goal
    while current != start:
        parent = parents[current]
        if parent is None:
            raise RuntimeError("shortest-path reconstruction failed")
        path.append(parent)
        current = parent
    path.reverse()
    return path, path_counts[goal]


def permute_graph_labels(
    edges: Sequence[tuple[int, int]],
    path: Sequence[int],
    permutation: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Apply an arbitrary bijective node relabeling to a graph and its path."""
    num_nodes = len(permutation)
    if sorted(int(node) for node in permutation) != list(range(num_nodes)):
        raise ValueError("permutation must contain every node label exactly once")
    if not path:
        raise ValueError("path must not be empty")
    for source, target in edges:
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("edge endpoint is outside the permutation")
    if any(not 0 <= node < num_nodes for node in path):
        raise ValueError("path node is outside the permutation")
    mapped_edges = [
        (int(permutation[source]), int(permutation[target]))
        for source, target in edges
    ]
    mapped_path = [int(permutation[node]) for node in path]
    return mapped_edges, mapped_path


def sample_shortest_path_graph(
    distribution_name: str,
    rng: random.Random,
    *,
    path_length: int | None = None,
) -> tuple[list[tuple[int, int]], int, int, list[int], int]:
    """Sample a varied DAG with a solver-verified unique shortest path.

    A short route and several longer, randomly shaped alternatives establish
    the task signal. Additional topologically valid edges are sampled with a
    random density, following the distributional spirit of CLRS graph samplers.
    Labels are independently permuted only after graph construction.
    """
    distribution = get_shortest_path_distribution(distribution_name)
    if path_length is None:
        path_length = rng.randint(
            distribution.min_path_length,
            distribution.max_path_length,
        )
    elif not distribution.min_path_length <= path_length <= distribution.max_path_length:
        raise ValueError(
            f"path_length must be in [{distribution.min_path_length}, "
            f"{distribution.max_path_length}] for {distribution.name}"
        )
    # Each minimally sized detour requires one internal node plus one feeder.
    # Sampling path length first keeps every length equally represented while
    # preventing impossible long-path/small-graph combinations.
    minimum_nodes = max(
        distribution.min_nodes,
        path_length + 1 + 2 * distribution.min_detours,
    )
    if minimum_nodes > distribution.max_nodes:
        raise ValueError(
            f"{distribution.name} cannot fit path length {path_length} "
            f"and {distribution.min_detours} detours"
        )
    num_nodes = rng.randint(minimum_nodes, distribution.max_nodes)
    path = list(range(path_length + 1))
    next_node = len(path)
    remaining_nodes = num_nodes - next_node
    detour_count = rng.randint(
        distribution.min_detours,
        min(
            distribution.max_detours,
            path_length,
            remaining_nodes // 2,
        ),
    )
    unused_branches = set(range(path_length))
    detours: dict[int, tuple[list[int], int, int]] = {}
    for detour_index in range(detour_count):
        reserve = 2 * (detour_count - detour_index - 1)
        available_for_detour = num_nodes - next_node - reserve - 1
        feasible = []
        for branch in unused_branches:
            for rejoin in range(branch + 1, path_length + 1):
                direct_span = rejoin - branch
                for penalty in range(1, distribution.max_detour_penalty + 1):
                    internal_nodes = direct_span + penalty - 1
                    if internal_nodes <= available_for_detour:
                        feasible.append(
                            (branch, rejoin, internal_nodes)
                        )
        if not feasible:
            break
        branch, rejoin, internal_count = rng.choice(feasible)
        unused_branches.remove(branch)
        detour_nodes = list(range(next_node, next_node + internal_count))
        next_node += internal_count
        feeder = next_node
        next_node += 1
        detours[branch] = (detour_nodes, rejoin, feeder)

    if len(detours) < distribution.min_detours:
        raise RuntimeError("distribution could not allocate its minimum detours")

    edges = {(path[index], path[index + 1]) for index in range(path_length)}
    for branch, (detour_nodes, rejoin, feeder) in detours.items():
        detour_path = [path[branch], *detour_nodes, path[rejoin]]
        edges.update(zip(detour_path, detour_path[1:]))
        edges.add((feeder, detour_nodes[0]))

    topological_order = []
    for index in range(path_length):
        topological_order.append(path[index])
        if index in detours:
            detour_nodes, _rejoin, feeder = detours[index]
            topological_order.append(feeder)
            topological_order.extend(detour_nodes)
    topological_order.append(path[-1])
    while next_node < num_nodes:
        topological_order.insert(
            rng.randrange(len(topological_order) + 1),
            next_node,
        )
        next_node += 1
    rank = {node: index for index, node in enumerate(topological_order)}

    out_degrees = [0] * num_nodes
    for source, _target in edges:
        out_degrees[source] += 1
    edge_probability = rng.uniform(
        distribution.min_edge_probability,
        distribution.max_edge_probability,
    )
    background_candidates = [
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if rank[source] < rank[target] and (source, target) not in edges
    ]
    rng.shuffle(background_candidates)
    for source, target in background_candidates:
        if (
            out_degrees[source] >= distribution.max_out_degree
            or rng.random() >= edge_probability
        ):
            continue
        candidate_edges = [*edges, (source, target)]
        candidate_path, path_count = solve_shortest_path(
            num_nodes,
            candidate_edges,
            path[0],
            path[-1],
        )
        if path_count != 1 or candidate_path != path:
            continue
        edges.add((source, target))
        out_degrees[source] += 1

    permutation = list(range(num_nodes))
    rng.shuffle(permutation)
    mapped_edges, mapped_path = permute_graph_labels(edges, path, permutation)
    rng.shuffle(mapped_edges)
    solved_path, path_count = solve_shortest_path(
        num_nodes,
        mapped_edges,
        mapped_path[0],
        mapped_path[-1],
    )
    if path_count != 1 or solved_path != mapped_path:
        raise RuntimeError(
            "generated graph failed its final shortest-path verification"
        )
    return (
        mapped_edges,
        mapped_path[0],
        mapped_path[-1],
        mapped_path,
        num_nodes,
    )


def sample_shortest_path_example(
    distribution_name: str,
    stoi: Dict[str, int],
    rng: random.Random,
    *,
    path_length: int | None = None,
) -> tuple[list[int], list[int], list[tuple[int, int]], int, int, list[int]]:
    edges, start, goal, path, num_nodes = sample_shortest_path_graph(
        distribution_name,
        rng,
        path_length=path_length,
    )
    prompt = [stoi[NODES_TOKEN]]
    prompt.extend(stoi[node_token(index)] for index in range(num_nodes))
    prompt.append(stoi[EDGES_TOKEN])
    serialized_edges = list(edges)
    rng.shuffle(serialized_edges)
    for source, target in serialized_edges:
        prompt.extend((stoi[node_token(source)], stoi[node_token(target)]))
    prompt.extend(
        (
            stoi[START_TOKEN],
            stoi[node_token(start)],
            stoi[GOAL_TOKEN],
            stoi[node_token(goal)],
        )
    )
    answer = [stoi[node_token(node)] for node in path]
    return prompt, answer, edges, start, goal, path


@dataclass(frozen=True)
class CompiledShortestPathDataset:
    tokens: np.ndarray
    sequence_lengths: np.ndarray
    prompt_lengths: np.ndarray
    example_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(frozen=True)
class CompiledShortestPathBundle:
    directory: Path
    manifest: dict
    vocab: tuple[str, ...]
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def distribution_name(self) -> str:
        return str(self.manifest["distribution"]["name"])

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])


_BUNDLE_CACHE: dict[str, CompiledShortestPathBundle] = {}
_DATASET_CACHE: dict[tuple[str, str], CompiledShortestPathDataset] = {}


def _canonical_split(split: str) -> str:
    canonical = "validation" if split == "val" else split
    if canonical not in {"train", "validation", "test"}:
        raise ValueError("shortest-path split must be train, val, validation, or test")
    return canonical


def load_shortest_path_bundle(
    shortest_path_data_dir: str | Path,
) -> CompiledShortestPathBundle:
    directory = Path(shortest_path_data_dir).expanduser().resolve()
    key = str(directory)
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "compiled shortest-path dataset not found at "
            f"{directory}. Generate it explicitly with "
            "python -m tasks.trace.shortest_path_data generate; training never "
            "generates shortest-path data online."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("shortest-path manifest must be a JSON object")
    expected = {
        "task": "shortest_path",
        "compiled_format_version": COMPILED_FORMAT_VERSION,
        "dataset_version": DATASET_VERSION,
        "token_dtype": "uint8",
        "length_dtype": "uint8",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"shortest-path manifest {field} is {manifest.get(field)!r}; "
                f"expected {value!r}"
            )
    dataset_id = manifest.get("dataset_id")
    if not (
        isinstance(dataset_id, str)
        and len(dataset_id) == 64
        and all(character in "0123456789abcdef" for character in dataset_id)
    ):
        raise ValueError("shortest-path manifest has an invalid dataset_id")
    distribution_payload = manifest.get("distribution")
    if not isinstance(distribution_payload, dict) or "name" not in distribution_payload:
        raise ValueError("shortest-path manifest is missing its distribution")
    distribution = get_shortest_path_distribution(str(distribution_payload["name"]))
    expected_distribution = {
        field: getattr(distribution, field)
        for field in distribution.__dataclass_fields__
    }
    if distribution_payload != expected_distribution:
        raise ValueError("shortest-path manifest distribution is inconsistent")
    vocab_path = directory / str(manifest.get("vocab", "vocab.json"))
    if vocab_path.parent != directory or vocab_path.name != "vocab.json":
        raise ValueError("shortest-path vocabulary filename is inconsistent")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"shortest-path vocabulary not found: {vocab_path}")
    vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab_payload, dict):
        raise ValueError("shortest-path vocabulary must be a JSON object")
    vocab = tuple(str(token) for token in vocab_payload.get("tokens", ()))
    expected_vocab = tuple(build_generation_vocab(distribution.name)[0])
    if vocab != expected_vocab or len(vocab) != int(manifest.get("vocab_size", -1)):
        raise ValueError("compiled shortest-path vocabulary is inconsistent")
    stoi = {token: index for index, token in enumerate(vocab)}
    itos = {index: token for token, index in stoi.items()}
    expected_special_ids = {
        "pad": stoi[PAD_TOKEN],
        "bos": stoi[BOS_TOKEN],
        "sep": stoi[SEP_TOKEN],
        "eos": stoi[EOS_TOKEN],
    }
    if manifest.get("special_token_ids") != expected_special_ids:
        raise ValueError("compiled shortest-path special-token IDs are inconsistent")
    max_sequence_length = int(manifest.get("max_sequence_length", -1))
    if max_sequence_length != generation_block_size(distribution.name) + 1:
        raise ValueError("compiled shortest-path maximum sequence length is inconsistent")
    if int(manifest.get("block_size", -1)) != max_sequence_length - 1:
        raise ValueError("compiled shortest-path block size is inconsistent")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("compiled shortest-path dataset must contain all three splits")
    dataset_spec = manifest.get("dataset_spec")
    if not isinstance(dataset_spec, dict):
        raise ValueError("shortest-path manifest is missing dataset_spec")
    expected_spec_fields = {
        "name",
        "distribution",
        "train_count",
        "validation_count",
        "test_count",
        "root_seed",
    }
    if set(dataset_spec) != expected_spec_fields:
        raise ValueError("shortest-path dataset_spec fields are inconsistent")
    if not isinstance(dataset_spec["name"], str) or not dataset_spec["name"]:
        raise ValueError("shortest-path dataset name must be non-empty")
    if (
        not isinstance(dataset_spec["root_seed"], int)
        or isinstance(dataset_spec["root_seed"], bool)
    ):
        raise ValueError("shortest-path dataset root seed must be an integer")
    if dataset_spec.get("distribution") != distribution.name:
        raise ValueError("shortest-path dataset specification and distribution disagree")
    expected_count_fields = {
        "train": "train_count",
        "validation": "validation_count",
        "test": "test_count",
    }
    for split, count_field in expected_count_fields.items():
        files = splits[split]
        if not isinstance(files, dict):
            raise ValueError(f"shortest-path {split} split metadata is malformed")
        if set(files) != {
            "count",
            "tokens",
            "sequence_lengths",
            "prompt_lengths",
            "example_ids",
        }:
            raise ValueError(f"shortest-path {split} split files are inconsistent")
        for file_field in (
            "tokens",
            "sequence_lengths",
            "prompt_lengths",
            "example_ids",
        ):
            filename = files[file_field]
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".npy")
            ):
                raise ValueError(
                    f"shortest-path {split} {file_field} filename is invalid"
                )
        count = files.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"shortest-path {split} split count must be positive")
        if dataset_spec.get(count_field) != count:
            raise ValueError(
                f"shortest-path {split} count disagrees with dataset_spec"
            )
    bundle = CompiledShortestPathBundle(directory, manifest, vocab, stoi, itos)
    _BUNDLE_CACHE[key] = bundle
    return bundle


def load_shortest_path_dataset(
    *,
    split: str,
    shortest_path_data_dir: str | Path,
) -> CompiledShortestPathDataset:
    canonical_split = _canonical_split(split)
    bundle = load_shortest_path_bundle(shortest_path_data_dir)
    key = str(bundle.directory), canonical_split
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    files = bundle.manifest["splits"][canonical_split]
    dataset = CompiledShortestPathDataset(
        tokens=np.load(
            bundle.directory / files["tokens"],
            mmap_mode="r",
            allow_pickle=False,
        ),
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
        example_ids=np.load(
            bundle.directory / files["example_ids"],
            mmap_mode="r",
            allow_pickle=False,
        ),
    )
    expected_count = int(files["count"])
    expected_width = int(bundle.manifest["max_sequence_length"])
    if not (
        expected_count > 0
        and dataset.tokens.shape == (expected_count, expected_width)
        and dataset.sequence_lengths.shape == (expected_count,)
        and dataset.prompt_lengths.shape == (expected_count,)
        and dataset.example_ids.shape == (expected_count,)
    ):
        raise ValueError("compiled shortest-path split arrays have inconsistent shapes")
    if dataset.tokens.dtype != np.uint8:
        raise ValueError("compiled shortest-path tokens must use uint8")
    if dataset.sequence_lengths.dtype != np.uint8:
        raise ValueError("compiled shortest-path sequence lengths must use uint8")
    if dataset.prompt_lengths.dtype != np.uint8:
        raise ValueError("compiled shortest-path prompt lengths must use uint8")
    if dataset.example_ids.dtype != np.dtype("S24"):
        raise ValueError("compiled shortest-path IDs must use fixed-width 24-byte strings")
    if np.any(dataset.prompt_lengths < 2):
        raise ValueError("compiled shortest-path prompt lengths are too short")
    if np.any(dataset.sequence_lengths < 4):
        raise ValueError("compiled shortest-path sequence lengths are too short")
    if np.any(dataset.prompt_lengths >= dataset.sequence_lengths):
        raise ValueError("compiled shortest-path prompt lengths must precede sequence ends")
    if np.any(dataset.sequence_lengths > dataset.tokens.shape[1]):
        raise ValueError("compiled shortest-path sequence lengths exceed the token array")
    _DATASET_CACHE[key] = dataset
    return dataset


def build_shortest_path_vocab(
    shortest_path_data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    bundle = load_shortest_path_bundle(shortest_path_data_dir)
    return list(bundle.vocab), dict(bundle.stoi), dict(bundle.itos)


def required_block_size(
    shortest_path_data_dir: str | Path = DEFAULT_DATA_DIR,
) -> int:
    return int(load_shortest_path_bundle(shortest_path_data_dir).manifest["block_size"])


def _batch_from_indices(
    dataset: CompiledShortestPathDataset,
    bundle: CompiledShortestPathBundle,
    indices: Sequence[int],
    *,
    device=None,
) -> SymbolicBatch:
    if not indices:
        raise ValueError("shortest-path batch indices must not be empty")
    selected = list(indices)
    sequence_lengths = np.asarray(dataset.sequence_lengths[selected], dtype=np.int64)
    prompt_lengths = np.asarray(dataset.prompt_lengths[selected], dtype=np.int64)
    max_sequence_length = int(sequence_lengths.max())
    full = np.asarray(
        dataset.tokens[selected, :max_sequence_length],
        dtype=np.int64,
    )
    inputs = full[:, :-1].copy()
    targets = full[:, 1:].copy()
    positions = np.arange(targets.shape[1])[None, :]
    inputs[positions >= (sequence_lengths - 1)[:, None]] = bundle.stoi[PAD_TOKEN]
    targets[positions < (prompt_lengths - 1)[:, None]] = -1
    targets[positions >= (sequence_lengths - 1)[:, None]] = -1
    return SymbolicBatch(
        idx=torch.as_tensor(inputs, dtype=torch.long, device=device),
        targets=torch.as_tensor(targets, dtype=torch.long, device=device),
        prompt_lengths=torch.as_tensor(prompt_lengths, dtype=torch.long, device=device),
        output_lengths=torch.as_tensor(
            sequence_lengths - prompt_lengths,
            dtype=torch.long,
            device=device,
        ),
    )


def build_shortest_path_batch(
    batch_size: int,
    shortest_path_data_dir: str | Path,
    *,
    split: str,
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = rng or random.Random()
    dataset = load_shortest_path_dataset(
        split=split,
        shortest_path_data_dir=shortest_path_data_dir,
    )
    bundle = load_shortest_path_bundle(shortest_path_data_dir)
    indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    return _batch_from_indices(dataset, bundle, indices, device=device)


def build_shortest_path_eval_batches(
    *,
    batch_size: int,
    num_batches: int,
    shortest_path_data_dir: str | Path,
    split: str,
    device=None,
) -> list[SymbolicBatch]:
    if batch_size < 1 or num_batches < 1:
        raise ValueError("batch_size and num_batches must be positive")
    canonical_split = _canonical_split(split)
    dataset = load_shortest_path_dataset(
        split=canonical_split,
        shortest_path_data_dir=shortest_path_data_dir,
    )
    bundle = load_shortest_path_bundle(shortest_path_data_dir)
    count = batch_size * num_batches
    if count > len(dataset):
        raise ValueError(
            f"requested {count} {canonical_split} examples without replacement, "
            f"but the dataset contains only {len(dataset)}"
        )
    selection_seed = int.from_bytes(
        hashlib.sha256(
            f"{bundle.dataset_id}|{canonical_split}|selection-v1".encode("ascii")
        ).digest()[:8],
        "little",
    )
    indices = random.Random(selection_seed).sample(range(len(dataset)), count)
    return [
        _batch_from_indices(
            dataset,
            bundle,
            indices[offset : offset + batch_size],
            device=device,
        )
        for offset in range(0, count, batch_size)
    ]


def parse_prompt_metadata(
    prompt_tokens: Sequence[int],
) -> tuple[list[tuple[int, int]], int, int]:
    tokens = [int(token_id) for token_id in prompt_tokens]
    if not tokens or tokens[0] != 4:
        raise ValueError("prompt must begin with <nodes>")
    try:
        edges_marker = tokens.index(5, 1)
        start_marker = tokens.index(6, edges_marker + 1)
    except ValueError as error:
        raise ValueError("prompt is missing a required marker") from error

    num_nodes = edges_marker - 1
    if num_nodes < 2:
        raise ValueError("prompt must list at least two nodes")
    listed_nodes = [
        token_id_to_node(token_id, num_nodes=num_nodes)
        for token_id in tokens[1:edges_marker]
    ]
    if any(node is None for node in listed_nodes) or set(listed_nodes) != set(range(num_nodes)):
        raise ValueError("prompt node list must contain every node exactly once")

    edge_start = edges_marker + 1
    edge_end = start_marker
    if (edge_end - edge_start) % 2 != 0:
        raise ValueError("prompt edge list must contain source-target pairs")
    edges = []
    for offset in range(edge_start, edge_end, 2):
        source = token_id_to_node(tokens[offset], num_nodes=num_nodes)
        target = token_id_to_node(tokens[offset + 1], num_nodes=num_nodes)
        if source is None or target is None:
            raise ValueError("prompt edge contains an invalid node token")
        edges.append((source, target))
    if len(set(edges)) != len(edges):
        raise ValueError("prompt contains duplicate edges")
    if (
        len(tokens) != start_marker + 4
        or tokens[start_marker] != 6
        or tokens[start_marker + 2] != 7
    ):
        raise ValueError("prompt must end with <start> node <goal> node")
    start = token_id_to_node(tokens[start_marker + 1], num_nodes=num_nodes)
    goal = token_id_to_node(tokens[start_marker + 3], num_nodes=num_nodes)
    if start is None or goal is None:
        raise ValueError("prompt start or goal token is invalid")
    return edges, start, goal


def token_id_to_node(token_id: int, *, num_nodes: int) -> int | None:
    node = int(token_id) - NODE_TOKEN_OFFSET
    return node if 0 <= node < num_nodes else None


__all__ = [
    "COMPILED_FORMAT_VERSION",
    "DATASET_VERSION",
    "DEFAULT_DATA_DIR",
    "DEFAULT_EASY_DATA_DIR",
    "DEFAULT_SMOKE_DATA_DIR",
    "CompiledShortestPathBundle",
    "CompiledShortestPathDataset",
    "SHORTEST_PATH_DISTRIBUTIONS",
    "PATH_LENGTH_BUCKETS",
    "ShortestPathDistribution",
    "build_shortest_path_batch",
    "build_shortest_path_eval_batches",
    "build_shortest_path_vocab",
    "build_generation_vocab",
    "generation_block_size",
    "get_shortest_path_distribution",
    "load_shortest_path_bundle",
    "load_shortest_path_dataset",
    "node_token",
    "path_length_bucket",
    "parse_prompt_metadata",
    "permute_graph_labels",
    "required_block_size",
    "sample_shortest_path_example",
    "sample_shortest_path_graph",
    "solve_shortest_path",
]
