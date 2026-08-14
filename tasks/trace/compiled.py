"""Mechanical helpers for finite, array-backed trace datasets."""
from __future__ import annotations

import hashlib
import random
from typing import Protocol, Sequence

import numpy as np
import torch

from tasks.common import SymbolicBatch


class CompiledSequenceArrays(Protocol):
    """The arrays needed to construct a masked trace batch."""

    tokens: np.ndarray
    sequence_lengths: np.ndarray
    prompt_lengths: np.ndarray

    def __len__(self) -> int: ...


def training_indices(
    dataset_size: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample training rows with replacement using the supplied RNG."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    return [rng.randrange(dataset_size) for _ in range(count)]


def deterministic_eval_indices(
    *,
    dataset_id: str,
    split: str,
    dataset_size: int,
    count: int,
) -> list[int]:
    """Select stable evaluation rows without replacement."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    if count > dataset_size:
        raise ValueError(
            f"requested {count} {split} examples without replacement, "
            f"but the dataset contains only {dataset_size}"
        )
    selection_seed = int.from_bytes(
        hashlib.sha256(
            f"{dataset_id}|{split}|selection-v1".encode("utf-8")
        ).digest()[:8],
        "little",
    )
    return random.Random(selection_seed).sample(range(dataset_size), count)


def chunk_indices(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = list(indices)
    if not selected:
        raise ValueError("indices must not be empty")
    return [
        selected[offset : offset + batch_size]
        for offset in range(0, len(selected), batch_size)
    ]


def batch_from_compiled_indices(
    dataset: CompiledSequenceArrays,
    indices: Sequence[int],
    *,
    pad_id: int,
    device=None,
) -> SymbolicBatch:
    """Build a dynamically padded and prompt-masked trace batch."""
    selected = list(indices)
    if not selected:
        raise ValueError("compiled batch indices must not be empty")
    sequence_lengths = np.asarray(
        dataset.sequence_lengths[selected],
        dtype=np.int64,
    )
    prompt_lengths = np.asarray(
        dataset.prompt_lengths[selected],
        dtype=np.int64,
    )
    max_sequence_length = int(sequence_lengths.max())
    full = np.asarray(
        dataset.tokens[selected, :max_sequence_length],
        dtype=np.int64,
    )
    inputs = full[:, :-1].copy()
    targets = full[:, 1:].copy()
    positions = np.arange(targets.shape[1])[None, :]
    inputs[positions >= (sequence_lengths - 1)[:, None]] = pad_id
    targets[positions < (prompt_lengths - 1)[:, None]] = -1
    targets[positions >= (sequence_lengths - 1)[:, None]] = -1
    return SymbolicBatch(
        idx=torch.as_tensor(inputs, dtype=torch.long, device=device),
        targets=torch.as_tensor(targets, dtype=torch.long, device=device),
        prompt_lengths=torch.as_tensor(
            prompt_lengths,
            dtype=torch.long,
            device=device,
        ),
        output_lengths=torch.as_tensor(
            sequence_lengths - prompt_lengths,
            dtype=torch.long,
            device=device,
        ),
    )


__all__ = [
    "CompiledSequenceArrays",
    "batch_from_compiled_indices",
    "chunk_indices",
    "deterministic_eval_indices",
    "training_indices",
]
