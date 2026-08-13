"""Build and verify finite shortest-path datasets for trace training."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable, Iterator

import numpy as np

from tasks.common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN, make_sequence
from tasks.trace import shortest_path


ROOT_SEED = 20260813
GENERATION_CHUNK_SIZE = 128
MAX_GENERATION_ROUND_SIZE = 8192
MAX_DATASET_WORKERS = 8
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    distribution: str
    train_count: int
    validation_count: int
    test_count: int
    root_seed: int = ROOT_SEED

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            "train": self.train_count,
            "validation": self.validation_count,
            "test": self.test_count,
        }

    def validate(self) -> None:
        shortest_path.get_shortest_path_distribution(self.distribution)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("dataset name must not be empty")
        if (
            not isinstance(self.root_seed, int)
            or isinstance(self.root_seed, bool)
        ):
            raise ValueError("root seed must be an integer")
        if any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for count in self.split_counts.values()
        ):
            raise ValueError("every shortest-path split count must be a positive integer")


DATASET_SPECS = {
    "main": DatasetSpec(
        name="main",
        distribution="main",
        train_count=5_000_000,
        validation_count=8_192,
        test_count=8_192,
    ),
    "easy": DatasetSpec(
        name="easy",
        distribution="easy",
        train_count=1_000_000,
        validation_count=8_192,
        test_count=8_192,
    ),
}


@dataclass(frozen=True)
class GeneratedExample:
    token_ids: tuple[int, ...]
    prompt_length: int
    example_id: str
    num_nodes: int
    edge_count: int
    path_length: int


@dataclass
class SplitStatistics:
    count: int = 0
    node_counts: Counter[int] | None = None
    path_length_counts: Counter[int] | None = None
    edge_count_sum: int = 0
    edge_count_min: int | None = None
    edge_count_max: int | None = None
    prompt_length_sum: int = 0
    prompt_length_min: int | None = None
    prompt_length_max: int | None = None
    sequence_length_sum: int = 0
    sequence_length_min: int | None = None
    sequence_length_max: int | None = None

    def __post_init__(self) -> None:
        self.node_counts = Counter()
        self.path_length_counts = Counter()

    def add(self, example: GeneratedExample) -> None:
        self.count += 1
        assert self.node_counts is not None
        assert self.path_length_counts is not None
        self.node_counts[example.num_nodes] += 1
        self.path_length_counts[example.path_length] += 1
        self.edge_count_sum += example.edge_count
        self.edge_count_min = (
            example.edge_count
            if self.edge_count_min is None
            else min(self.edge_count_min, example.edge_count)
        )
        self.edge_count_max = (
            example.edge_count
            if self.edge_count_max is None
            else max(self.edge_count_max, example.edge_count)
        )
        self.prompt_length_sum += example.prompt_length
        self.prompt_length_min = (
            example.prompt_length
            if self.prompt_length_min is None
            else min(self.prompt_length_min, example.prompt_length)
        )
        self.prompt_length_max = (
            example.prompt_length
            if self.prompt_length_max is None
            else max(self.prompt_length_max, example.prompt_length)
        )
        sequence_length = len(example.token_ids)
        self.sequence_length_sum += sequence_length
        self.sequence_length_min = (
            sequence_length
            if self.sequence_length_min is None
            else min(self.sequence_length_min, sequence_length)
        )
        self.sequence_length_max = (
            sequence_length
            if self.sequence_length_max is None
            else max(self.sequence_length_max, sequence_length)
        )

    @staticmethod
    def _range_summary(total: int, minimum: int | None, maximum: int | None, count: int) -> dict:
        return {
            "mean": total / count,
            "min": minimum,
            "max": maximum,
        }

    def to_dict(self) -> dict:
        if self.count < 1:
            raise ValueError("cannot summarize an empty split")
        assert self.node_counts is not None
        assert self.path_length_counts is not None
        return {
            "count": self.count,
            "node_counts": {str(key): value for key, value in sorted(self.node_counts.items())},
            "path_length_counts": {
                str(key): value for key, value in sorted(self.path_length_counts.items())
            },
            "edge_count": self._range_summary(
                self.edge_count_sum,
                self.edge_count_min,
                self.edge_count_max,
                self.count,
            ),
            "prompt_length": self._range_summary(
                self.prompt_length_sum,
                self.prompt_length_min,
                self.prompt_length_max,
                self.count,
            ),
            "sequence_length": self._range_summary(
                self.sequence_length_sum,
                self.sequence_length_min,
                self.sequence_length_max,
                self.count,
            ),
        }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_seed(root_seed: int, distribution: str, candidate_index: int) -> int:
    payload = f"{root_seed}|{distribution}|{candidate_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def example_id_for(
    *,
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
    start: int,
    goal: int,
) -> str:
    payload = {
        "num_nodes": int(num_nodes),
        "edges": [list(edge) for edge in sorted(edges)],
        "start": int(start),
        "goal": int(goal),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _generate_example(
    distribution_name: str,
    path_length: int,
    seed: int,
) -> GeneratedExample:
    _vocab, stoi, _itos = shortest_path.build_generation_vocab(distribution_name)
    return _generate_example_with_vocab(distribution_name, path_length, seed, stoi)


def _generate_example_with_vocab(
    distribution_name: str,
    path_length: int,
    seed: int,
    stoi: dict[str, int],
) -> GeneratedExample:
    prompt, answer, edges, start, goal, path = shortest_path.sample_shortest_path_example(
        distribution_name,
        stoi,
        random.Random(seed),
        path_length=path_length,
    )
    full, prompt_length, _output_length = make_sequence(prompt, answer, stoi)
    num_nodes = prompt.index(stoi[shortest_path.EDGES_TOKEN]) - 1
    return GeneratedExample(
        token_ids=tuple(full),
        prompt_length=prompt_length,
        example_id=example_id_for(
            num_nodes=num_nodes,
            edges=edges,
            start=start,
            goal=goal,
        ),
        num_nodes=num_nodes,
        edge_count=len(edges),
        path_length=len(path) - 1,
    )


def _generate_chunk(
    payload: tuple[str, tuple[tuple[int, int], ...]],
) -> tuple[GeneratedExample, ...]:
    distribution_name, requests = payload
    _vocab, stoi, _itos = shortest_path.build_generation_vocab(distribution_name)
    return tuple(
        _generate_example_with_vocab(distribution_name, path_length, seed, stoi)
        for path_length, seed in requests
    )


def _balanced_quotas(count: int, path_lengths: tuple[int, ...]) -> dict[int, int]:
    quotient, remainder = divmod(count, len(path_lengths))
    return {
        path_length: quotient + int(index < remainder)
        for index, path_length in enumerate(path_lengths)
    }


def _identity_header(spec: DatasetSpec) -> bytes:
    distribution = shortest_path.get_shortest_path_distribution(spec.distribution)
    payload = {
        "task": "shortest_path",
        "compiled_format_version": shortest_path.COMPILED_FORMAT_VERSION,
        "dataset_version": shortest_path.DATASET_VERSION,
        "dataset_spec": asdict(spec),
        "distribution": asdict(distribution),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")


def _request_chunks(
    *,
    spec: DatasetSpec,
    candidate_start: int,
    count: int,
    path_lengths: tuple[int, ...],
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    requests = [
        (
            path_lengths[(candidate_start + offset) % len(path_lengths)],
            _candidate_seed(
                spec.root_seed,
                spec.distribution,
                candidate_start + offset,
            ),
        )
        for offset in range(count)
    ]
    return [
        (spec.distribution, tuple(requests[offset : offset + GENERATION_CHUNK_SIZE]))
        for offset in range(0, len(requests), GENERATION_CHUNK_SIZE)
    ]


def _iter_generated_chunks(
    payloads: list[tuple[str, tuple[tuple[int, int], ...]]],
    executor: ProcessPoolExecutor | None,
) -> Iterator[tuple[GeneratedExample, ...]]:
    if executor is None:
        return map(_generate_chunk, payloads)
    return executor.map(_generate_chunk, payloads)


def _split_files(split: str) -> dict[str, str]:
    return {
        "tokens": f"{split}_tokens.npy",
        "sequence_lengths": f"{split}_sequence_lengths.npy",
        "prompt_lengths": f"{split}_prompt_lengths.npy",
        "example_ids": f"{split}_example_ids.npy",
    }


def _generate_into_staging(
    spec: DatasetSpec,
    staging: Path,
    *,
    workers: int,
    verbose: bool,
) -> dict:
    spec.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    distribution = shortest_path.get_shortest_path_distribution(spec.distribution)
    vocab, stoi, _itos = shortest_path.build_generation_vocab(spec.distribution)
    max_sequence_length = shortest_path.generation_block_size(spec.distribution) + 1
    if len(vocab) > np.iinfo(np.uint8).max or max_sequence_length > np.iinfo(np.uint8).max:
        raise ValueError("shortest-path artifacts no longer fit uint8")

    split_arrays = {}
    split_payload = {}
    for split, count in spec.split_counts.items():
        files = _split_files(split)
        tokens = np.lib.format.open_memmap(
            staging / files["tokens"],
            mode="w+",
            dtype=np.uint8,
            shape=(count, max_sequence_length),
        )
        tokens[:] = stoi[PAD_TOKEN]
        split_arrays[split] = {
            "tokens": tokens,
            "sequence_lengths": np.lib.format.open_memmap(
                staging / files["sequence_lengths"],
                mode="w+",
                dtype=np.uint8,
                shape=(count,),
            ),
            "prompt_lengths": np.lib.format.open_memmap(
                staging / files["prompt_lengths"],
                mode="w+",
                dtype=np.uint8,
                shape=(count,),
            ),
            "example_ids": np.lib.format.open_memmap(
                staging / files["example_ids"],
                mode="w+",
                dtype="S24",
                shape=(count,),
            ),
        }
        split_payload[split] = {"count": count, **files}

    seen_ids: set[bytes] = set()
    dataset_digest = hashlib.sha256(_identity_header(spec))
    statistics: dict[str, SplitStatistics] = {}
    candidate_index = 0
    candidate_count = 0
    duplicate_rejections = 0
    quota_rejections = 0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for split, count in spec.split_counts.items():
            arrays = split_arrays[split]
            stats = SplitStatistics()
            statistics[split] = stats
            path_lengths = tuple(
                range(distribution.min_path_length, distribution.max_path_length + 1)
            )
            remaining_quota = _balanced_quotas(count, path_lengths)
            row = 0
            next_progress = 100_000
            while row < count:
                remaining = count - row
                round_size = min(
                    MAX_GENERATION_ROUND_SIZE,
                    max(256, remaining * 2),
                )
                payloads = _request_chunks(
                    spec=spec,
                    candidate_start=candidate_index,
                    count=round_size,
                    path_lengths=path_lengths,
                )
                for chunk in _iter_generated_chunks(payloads, executor):
                    for example in chunk:
                        candidate_count += 1
                        if row >= count:
                            continue
                        if remaining_quota[example.path_length] <= 0:
                            quota_rejections += 1
                            continue
                        identity_key = bytes.fromhex(example.example_id)
                        if identity_key in seen_ids:
                            duplicate_rejections += 1
                            continue
                        seen_ids.add(identity_key)
                        remaining_quota[example.path_length] -= 1
                        sequence_length = len(example.token_ids)
                        arrays["tokens"][row, :sequence_length] = np.asarray(
                            example.token_ids,
                            dtype=np.uint8,
                        )
                        arrays["sequence_lengths"][row] = sequence_length
                        arrays["prompt_lengths"][row] = example.prompt_length
                        arrays["example_ids"][row] = example.example_id.encode("ascii")
                        dataset_digest.update(split.encode("ascii"))
                        dataset_digest.update(example.example_id.encode("ascii"))
                        dataset_digest.update(example.prompt_length.to_bytes(1, "little"))
                        dataset_digest.update(bytes(example.token_ids))
                        stats.add(example)
                        row += 1
                        if verbose and row >= next_progress:
                            print(f"{split}: {row:,}/{count:,}", flush=True)
                            next_progress += 100_000
                candidate_index += round_size
            if any(remaining_quota.values()):
                raise RuntimeError(f"failed to fill balanced {split} path-length quotas")
            for array in arrays.values():
                array.flush()
            if verbose:
                print(f"{split}: complete ({count:,})", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    dataset_id = dataset_digest.hexdigest()
    _write_json(staging / "vocab.json", {"tokens": vocab})
    manifest = {
        "task": "shortest_path",
        "compiled_format_version": shortest_path.COMPILED_FORMAT_VERSION,
        "dataset_version": shortest_path.DATASET_VERSION,
        "dataset_id": dataset_id,
        "dataset_spec": asdict(spec),
        "distribution": asdict(distribution),
        "vocab": "vocab.json",
        "vocab_size": len(vocab),
        "token_dtype": "uint8",
        "length_dtype": "uint8",
        "max_sequence_length": max_sequence_length,
        "block_size": max_sequence_length - 1,
        "special_token_ids": {
            "pad": stoi[PAD_TOKEN],
            "bos": stoi[BOS_TOKEN],
            "sep": stoi[SEP_TOKEN],
            "eos": stoi[EOS_TOKEN],
        },
        "splits": split_payload,
    }
    summary = {
        "dataset_id": dataset_id,
        "candidate_count": candidate_count,
        "accepted_count": sum(spec.split_counts.values()),
        "duplicate_rejections": duplicate_rejections,
        "filled_quota_rejections": quota_rejections,
        "splits": {
            split: split_statistics.to_dict()
            for split, split_statistics in statistics.items()
        },
    }
    _write_json(staging / "manifest.json", manifest)
    _write_json(staging / "summary.json", summary)
    return summary


def generate_dataset(
    spec: DatasetSpec,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    verbose: bool = False,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"shortest-path output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_workers = min(
        workers if workers is not None else (os.cpu_count() or 1),
        MAX_DATASET_WORKERS,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        _generate_into_staging(
            spec,
            staging,
            workers=selected_workers,
            verbose=verbose,
        )
        verify_dataset(staging)
        if destination.exists():
            raise FileExistsError(
                f"shortest-path output appeared during generation: {destination}"
            )
        staging.replace(destination)
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
    return destination


def _dataset_spec_from_manifest(manifest: dict) -> DatasetSpec:
    payload = manifest.get("dataset_spec")
    if not isinstance(payload, dict):
        raise ValueError("shortest-path manifest is missing dataset_spec")
    spec = DatasetSpec(**payload)
    spec.validate()
    return spec


def verify_dataset(data_dir: str | Path) -> dict:
    directory = Path(data_dir).expanduser().resolve()
    shortest_path._BUNDLE_CACHE.pop(str(directory), None)
    for split in SPLITS:
        shortest_path._DATASET_CACHE.pop((str(directory), split), None)
    bundle = shortest_path.load_shortest_path_bundle(directory)
    spec = _dataset_spec_from_manifest(bundle.manifest)
    if bundle.distribution_name != spec.distribution:
        raise ValueError("dataset specification and distribution disagree")
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"shortest-path summary not found: {summary_path}")
    recorded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(recorded_summary, dict):
        raise ValueError("shortest-path summary must be a JSON object")
    digest = hashlib.sha256(_identity_header(spec))
    seen_ids: set[bytes] = set()
    split_summaries = {}
    complete_statistics = {}
    distribution = shortest_path.get_shortest_path_distribution(spec.distribution)
    expected_path_lengths = set(
        range(distribution.min_path_length, distribution.max_path_length + 1)
    )
    for split in SPLITS:
        dataset = shortest_path.load_shortest_path_dataset(
            split=split,
            shortest_path_data_dir=directory,
        )
        path_counts: Counter[int] = Counter()
        statistics = SplitStatistics()
        for row in range(len(dataset)):
            sequence_length = int(dataset.sequence_lengths[row])
            prompt_length = int(dataset.prompt_lengths[row])
            token_ids = dataset.tokens[row, :sequence_length].tolist()
            if any(token_id >= len(bundle.vocab) for token_id in token_ids):
                raise ValueError(f"{split} row {row} contains an invalid token ID")
            if (
                token_ids[0] != bundle.stoi[BOS_TOKEN]
                or token_ids[prompt_length - 1] != bundle.stoi[SEP_TOKEN]
                or token_ids[-1] != bundle.stoi[EOS_TOKEN]
            ):
                raise ValueError(f"{split} row {row} has invalid sequence markers")
            if np.any(dataset.tokens[row, sequence_length:] != bundle.stoi[PAD_TOKEN]):
                raise ValueError(f"{split} row {row} has non-padding tokens after EOS")
            prompt = token_ids[1 : prompt_length - 1]
            target = token_ids[prompt_length:-1]
            edges, start, goal = shortest_path.parse_prompt_metadata(prompt)
            num_nodes = prompt.index(bundle.stoi[shortest_path.EDGES_TOKEN]) - 1
            solved, path_count = shortest_path.solve_shortest_path(
                num_nodes,
                edges,
                start,
                goal,
            )
            expected_target = [bundle.stoi[shortest_path.node_token(node)] for node in solved]
            if path_count != 1 or target != expected_target:
                raise ValueError(f"{split} row {row} does not contain its unique shortest path")
            example_id = dataset.example_ids[row].decode("ascii")
            expected_id = example_id_for(
                num_nodes=num_nodes,
                edges=edges,
                start=start,
                goal=goal,
            )
            if example_id != expected_id:
                raise ValueError(f"{split} row {row} has an inconsistent example ID")
            identity_key = example_id.encode("ascii")
            if identity_key in seen_ids:
                raise ValueError(f"duplicate shortest-path example ID: {example_id}")
            seen_ids.add(identity_key)
            digest.update(split.encode("ascii"))
            digest.update(example_id.encode("ascii"))
            digest.update(prompt_length.to_bytes(1, "little"))
            digest.update(bytes(token_ids))
            path_length = len(solved) - 1
            path_counts[path_length] += 1
            statistics.add(
                GeneratedExample(
                    token_ids=tuple(token_ids),
                    prompt_length=prompt_length,
                    example_id=example_id,
                    num_nodes=num_nodes,
                    edge_count=len(edges),
                    path_length=path_length,
                )
            )
        if set(path_counts) != expected_path_lengths:
            raise ValueError(f"{split} does not contain every requested path length")
        if max(path_counts.values()) - min(path_counts.values()) > 1:
            raise ValueError(f"{split} path lengths are not evenly stratified")
        complete_statistics[split] = statistics.to_dict()
        split_summaries[split] = {
            "count": len(dataset),
            "path_length_counts": {
                str(key): value for key, value in sorted(path_counts.items())
            },
        }
    computed_id = digest.hexdigest()
    if computed_id != bundle.dataset_id:
        raise ValueError("shortest-path dataset identity does not match its contents")
    accepted_count = sum(spec.split_counts.values())
    if recorded_summary.get("dataset_id") != computed_id:
        raise ValueError("shortest-path summary dataset ID is inconsistent")
    if recorded_summary.get("accepted_count") != accepted_count:
        raise ValueError("shortest-path summary accepted count is inconsistent")
    for field in (
        "candidate_count",
        "duplicate_rejections",
        "filled_quota_rejections",
    ):
        value = recorded_summary.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"shortest-path summary {field} is invalid")
    if recorded_summary["candidate_count"] < accepted_count:
        raise ValueError("shortest-path candidate count is smaller than accepted count")
    if recorded_summary.get("splits") != complete_statistics:
        raise ValueError("shortest-path summary statistics do not match the arrays")
    return {
        "valid": True,
        "dataset_id": computed_id,
        "unique_example_ids": len(seen_ids),
        "splits": split_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and verify finite shortest-path trace datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--workers", type=int, default=None)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--data-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        output = generate_dataset(
            DATASET_SPECS[args.dataset],
            args.output_dir,
            workers=args.workers,
            verbose=True,
        )
        payload = {
            "output_dir": str(output),
            "dataset_id": json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )["dataset_id"],
            "verified": True,
        }
    else:
        payload = verify_dataset(args.data_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_SPECS",
    "DatasetSpec",
    "ROOT_SEED",
    "example_id_for",
    "generate_dataset",
    "main",
    "verify_dataset",
]
