"""Permutation task: start from an ordered list of objects, apply a sequence of swaps,
and predict the final arrangement.

Example sequence:
<bos> o0 o1 o2 o3 o4 swap p0 p2 swap p1 p3 <sep> o2 o3 o0 o1 o4 <eos>

With the default five objects, repeated arbitrary transpositions generate the
symmetric group S_5.
"""

from typing import Dict, List, Tuple

import random

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


SWAP_TOKEN = "swap"
DEFAULT_NUM_OBJECTS = 5


def obj_token(index: int) -> str:
    return f"o{index}"


def pos_token(index: int) -> str:
    return f"p{index}"


def required_block_size(num_objects: int, num_swaps: int) -> int:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    if num_swaps < 0:
        raise ValueError("num_swaps must be non-negative")
    return 2 * num_objects + 3 * num_swaps + 2


def build_permutation_vocab(num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, SWAP_TOKEN]
    tokens.extend(obj_token(i) for i in range(num_objects))
    tokens.extend(pos_token(i) for i in range(num_objects))
    return build_vocab(tokens)


def solve_permutation(num_objects: int, swaps: List[tuple[int, int]]) -> list[int]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    state = list(range(num_objects))
    for i, j in swaps:
        if not (0 <= i < num_objects and 0 <= j < num_objects):
            raise ValueError("swap positions must index into the permutation")
        if i == j:
            raise ValueError("swap positions must be distinct")
        state[i], state[j] = state[j], state[i]
    return state


def sample_permutation_example(
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[tuple[int, int]], List[int]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    if num_swaps < 0:
        raise ValueError("num_swaps must be non-negative")

    prompt = [stoi[obj_token(index)] for index in range(num_objects)]
    swaps: List[tuple[int, int]] = []
    for _ in range(num_swaps):
        i, j = rng.sample(range(num_objects), k=2)
        swaps.append((i, j))
        prompt.extend([stoi[SWAP_TOKEN], stoi[pos_token(i)], stoi[pos_token(j)]])

    final_state = solve_permutation(num_objects, swaps)
    answer = [stoi[obj_token(obj)] for obj in final_state]
    return prompt, answer, swaps, final_state


def build_permutation_batch(
    batch_size: int,
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    if num_swaps < 0:
        raise ValueError("num_swaps must be non-negative")
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _ = sample_permutation_example(num_objects, num_swaps, stoi, rng)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "DEFAULT_NUM_OBJECTS",
    "SWAP_TOKEN",
    "build_permutation_batch",
    "build_permutation_vocab",
    "obj_token",
    "pos_token",
    "required_block_size",
    "sample_permutation_example",
    "solve_permutation",
]
