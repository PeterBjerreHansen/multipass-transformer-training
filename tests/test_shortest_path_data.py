from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.train_trace import validate_task_args
from tasks.trace import shortest_path
from tasks.trace import shortest_path_data


SMOKE_DATA = Path(shortest_path.DEFAULT_SMOKE_DATA_DIR)


def _tiny_spec(*, name: str = "tiny") -> shortest_path_data.DatasetSpec:
    return shortest_path_data.DatasetSpec(
        name=name,
        distribution="easy",
        train_count=17,
        validation_count=7,
        test_count=5,
    )


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _copy_smoke(tmp_path: Path) -> Path:
    destination = tmp_path / "shortest-path"
    shutil.copytree(SMOKE_DATA, destination)
    return destination


def _clear_dataset_caches(directory: Path) -> None:
    resolved = str(directory.resolve())
    shortest_path._BUNDLE_CACHE.pop(resolved, None)
    for split in shortest_path_data.SPLITS:
        shortest_path._DATASET_CACHE.pop((resolved, split), None)


def _rows(batch) -> list[tuple[int, ...]]:
    return [tuple(int(token) for token in row) for row in batch.idx.tolist()]


def test_generation_is_byte_identical_across_worker_counts(tmp_path):
    sequential = shortest_path_data.generate_dataset(
        _tiny_spec(),
        tmp_path / "sequential",
        workers=1,
    )
    parallel = shortest_path_data.generate_dataset(
        _tiny_spec(),
        tmp_path / "parallel",
        workers=2,
    )
    assert _artifact_bytes(sequential) == _artifact_bytes(parallel)


def test_generated_splits_are_balanced_unique_and_solver_verified(tmp_path):
    output = shortest_path_data.generate_dataset(
        _tiny_spec(),
        tmp_path / "dataset",
        workers=1,
    )
    verification = shortest_path_data.verify_dataset(output)
    assert verification["valid"] is True
    assert verification["unique_example_ids"] == 29

    all_ids: set[bytes] = set()
    for split, expected_count in _tiny_spec().split_counts.items():
        dataset = shortest_path.load_shortest_path_dataset(
            split=split,
            shortest_path_data_dir=output,
        )
        ids = set(dataset.example_ids.tolist())
        assert len(ids) == expected_count
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)
        counts = verification["splits"][split]["path_length_counts"]
        assert max(counts.values()) - min(counts.values()) <= 1


def test_global_duplicate_rejection_ignores_serialized_edge_order(
    tmp_path,
    monkeypatch,
):
    assert shortest_path_data.example_id_for(
        num_nodes=3,
        edges=[(0, 1), (1, 2)],
        start=0,
        goal=2,
    ) == shortest_path_data.example_id_for(
        num_nodes=3,
        edges=[(1, 2), (0, 1)],
        start=0,
        goal=2,
    )

    original = shortest_path_data._generate_chunk
    injected = False

    def duplicate_one(payload):
        nonlocal injected
        examples = list(original(payload))
        if not injected and len(examples) >= 3:
            examples[2] = examples[0]
            injected = True
        return tuple(examples)

    monkeypatch.setattr(shortest_path_data, "_generate_chunk", duplicate_one)
    output = shortest_path_data.generate_dataset(
        _tiny_spec(),
        tmp_path / "deduplicated",
        workers=1,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["duplicate_rejections"] >= 1
    assert shortest_path_data.verify_dataset(output)["unique_example_ids"] == 29


def test_compiled_loader_uses_memory_maps_and_seeded_sampling():
    dataset = shortest_path.load_shortest_path_dataset(
        split="train",
        shortest_path_data_dir=SMOKE_DATA,
    )
    assert isinstance(dataset.tokens, np.memmap)
    assert isinstance(dataset.sequence_lengths, np.memmap)
    assert isinstance(dataset.prompt_lengths, np.memmap)
    assert isinstance(dataset.example_ids, np.memmap)

    first = shortest_path.build_shortest_path_batch(
        batch_size=6,
        shortest_path_data_dir=SMOKE_DATA,
        split="train",
        device="cpu",
        rng=random.Random(71),
    )
    second = shortest_path.build_shortest_path_batch(
        batch_size=6,
        shortest_path_data_dir=SMOKE_DATA,
        split="train",
        device="cpu",
        rng=random.Random(71),
    )
    assert torch.equal(first.idx, second.idx)
    assert torch.equal(first.targets, second.targets)


def test_held_out_selection_is_deterministic_without_replacement():
    first = shortest_path.build_shortest_path_eval_batches(
        batch_size=2,
        num_batches=4,
        shortest_path_data_dir=SMOKE_DATA,
        split="validation",
        device="cpu",
    )
    second = shortest_path.build_shortest_path_eval_batches(
        batch_size=2,
        num_batches=4,
        shortest_path_data_dir=SMOKE_DATA,
        split="validation",
        device="cpu",
    )
    first_rows = [row for batch in first for row in _rows(batch)]
    second_rows = [row for batch in second for row in _rows(batch)]
    assert first_rows == second_rows
    assert len(set(first_rows)) == 8

    test_batches = shortest_path.build_shortest_path_eval_batches(
        batch_size=2,
        num_batches=4,
        shortest_path_data_dir=SMOKE_DATA,
        split="test",
        device="cpu",
    )
    test_rows = {row for batch in test_batches for row in _rows(batch)}
    assert set(first_rows).isdisjoint(test_rows)

    with pytest.raises(ValueError, match="without replacement"):
        shortest_path.build_shortest_path_eval_batches(
            batch_size=3,
            num_batches=3,
            shortest_path_data_dir=SMOKE_DATA,
            split="test",
        )


def test_training_loader_never_calls_online_generator(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("online generation was called")

    monkeypatch.setattr(shortest_path, "sample_shortest_path_example", fail)
    shortest_path.build_shortest_path_batch(
        batch_size=2,
        shortest_path_data_dir=SMOKE_DATA,
        split="train",
        rng=random.Random(9),
    )


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    [
        (
            "train_tokens.npy",
            lambda array: array.astype(np.uint16),
            "tokens must use uint8",
        ),
        (
            "validation_prompt_lengths.npy",
            lambda array: array[:-1],
            "inconsistent shapes",
        ),
    ],
)
def test_loader_rejects_wrong_dtypes_and_shapes(
    tmp_path,
    filename,
    mutation,
    message,
):
    directory = _copy_smoke(tmp_path)
    path = directory / filename
    np.save(path, mutation(np.load(path, allow_pickle=False)), allow_pickle=False)
    _clear_dataset_caches(directory)
    split = "train" if filename.startswith("train") else "validation"
    with pytest.raises(ValueError, match=message):
        shortest_path.load_shortest_path_dataset(
            split=split,
            shortest_path_data_dir=directory,
        )


def test_loader_rejects_missing_and_corrupt_manifests(tmp_path):
    with pytest.raises(FileNotFoundError, match="training never generates"):
        shortest_path.load_shortest_path_bundle(tmp_path / "missing")

    directory = _copy_smoke(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["special_token_ids"]["eos"] = 255
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _clear_dataset_caches(directory)
    with pytest.raises(ValueError, match="special-token IDs"):
        shortest_path.load_shortest_path_bundle(directory)


def test_saved_dataset_id_must_match_current_artifact(tmp_path):
    directory = _copy_smoke(tmp_path)
    args = SimpleNamespace(
        task="shortest_path",
        shortest_path_data_dir=str(directory),
        shortest_path_dataset_id="0" * 64,
        shortest_path_distribution="easy",
        block_size=None,
    )
    with pytest.raises(ValueError, match="dataset ID mismatch"):
        validate_task_args(args)


def test_generation_refuses_to_overwrite_existing_artifact(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        shortest_path_data.generate_dataset(_tiny_spec(), output, workers=1)
