"""Tracking task: start from an ordered list of objects, apply swaps or segment operations,
query one position, and predict which object ends up there.

Example full sequence:
<bos> o0 o1 o2 o3 o4 swap p0 p2 <query> p0 <sep> o2 <eos>
"""

from typing import Dict, List, Sequence, Tuple

import random

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    QUERY_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    make_sequence,
)


SWAP_TOKEN = "swap"
ROTL_TOKEN = "rotl"
REV_TOKEN = "rev"
DEFAULT_NUM_OBJECTS = 5


def obj_token(index: int) -> str:
    return f"o{index}"


def pos_token(index: int) -> str:
    return f"p{index}"


def required_block_size(num_objects: int, num_ops: int) -> int:
    prompt_tokens = num_objects + 3 * num_ops + 2
    answer_len = 1
    return 2 + prompt_tokens + answer_len


def build_tracking_vocab(num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, QUERY_TOKEN, SWAP_TOKEN, ROTL_TOKEN, REV_TOKEN]
    tokens.extend(obj_token(index) for index in range(num_objects))
    tokens.extend(pos_token(index) for index in range(num_objects))
    return build_vocab(tokens)


def apply_tracking_op(state: list[int], op: tuple[str, int, int]):
    name, i, j = op
    lo, hi = sorted((i, j))
    if name == SWAP_TOKEN:
        state[i], state[j] = state[j], state[i]
    elif name == ROTL_TOKEN:
        segment = state[lo : hi + 1]
        state[lo : hi + 1] = segment[1:] + segment[:1]
    elif name == REV_TOKEN:
        state[lo : hi + 1] = reversed(state[lo : hi + 1])
    else:
        raise ValueError(f"Unsupported tracking op: {name}")


def solve_tracking(num_objects: int, ops: Sequence[tuple[str, int, int]]) -> tuple[list[list[int]], list[int]]:
    state = list(range(num_objects))
    trace: list[list[int]] = []
    for op in ops:
        apply_tracking_op(state, op)
        trace.append(list(state))
    return trace, state


def sample_tracking_example(
    num_objects: int,
    num_ops: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[tuple[str, int, int]], int, int]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    if num_ops < 1:
        raise ValueError("num_ops must be at least 1")

    ops: list[tuple[str, int, int]] = []
    prompt = [*(stoi[obj_token(index)] for index in range(num_objects))]
    for _ in range(num_ops):
        op_name = rng.choice((SWAP_TOKEN, ROTL_TOKEN, REV_TOKEN))
        i, j = rng.sample(range(num_objects), k=2)
        ops.append((op_name, i, j))
        prompt.extend([stoi[op_name], stoi[pos_token(i)], stoi[pos_token(j)]])

    _, final_state = solve_tracking(num_objects, ops)
    query_pos = rng.randrange(num_objects)
    final_object = final_state[query_pos]
    prompt.extend([stoi[QUERY_TOKEN], stoi[pos_token(query_pos)]])

    answer = [stoi[obj_token(final_object)]]
    return prompt, answer, ops, query_pos, final_object


def build_tracking_batch(
    batch_size: int,
    num_objects: int,
    num_ops: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_tracking_example(num_objects, num_ops, stoi, rng)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "apply_tracking_op",
    "build_tracking_batch",
    "build_tracking_vocab",
    "required_block_size",
    "sample_tracking_example",
    "solve_tracking",
]
