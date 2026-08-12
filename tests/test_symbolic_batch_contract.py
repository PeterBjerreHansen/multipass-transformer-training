from __future__ import annotations

import random

import torch

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN
from tasks.trace import maze, othello, shortest_path


def _assert_symbolic_batch_contract(
    batch,
    *,
    stoi,
    vocab_size: int,
    required_block_size: int,
) -> None:
    assert batch.idx.shape == batch.targets.shape
    assert batch.idx.size(0) == 3
    assert batch.idx.size(1) <= required_block_size
    assert batch.prompt_lengths.shape == batch.output_lengths.shape == (3,)
    assert int(batch.idx.min()) >= 0
    assert int(batch.idx.max()) < vocab_size

    valid_targets = batch.targets[batch.targets != -1]
    assert valid_targets.numel() > 0
    assert int(valid_targets.min()) >= 0
    assert int(valid_targets.max()) < vocab_size

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row])
        output_len = int(batch.output_lengths[row])
        active_idx_len = prompt_len + output_len - 1
        suffix_start = prompt_len - 1
        suffix_end = suffix_start + output_len

        idx_row = batch.idx[row]
        target_row = batch.targets[row]
        target_suffix = target_row[suffix_start:suffix_end]

        assert int(idx_row[0]) == stoi[BOS_TOKEN]
        assert int(idx_row[prompt_len - 1]) == stoi[SEP_TOKEN]
        assert int(target_suffix[-1]) == stoi[EOS_TOKEN]
        assert not (idx_row[:prompt_len] == stoi[PAD_TOKEN]).any()
        assert not (idx_row[prompt_len:active_idx_len] == stoi[PAD_TOKEN]).any()

        assert torch.equal(target_row[:suffix_start], torch.full((suffix_start,), -1))
        assert torch.equal(target_row[suffix_end:], torch.full_like(target_row[suffix_end:], -1))
        assert torch.equal(
            idx_row[active_idx_len:],
            torch.full_like(idx_row[active_idx_len:], stoi[PAD_TOKEN]),
        )


def test_symbolic_task_batches_follow_shared_contract(tmp_path):
    cases = [
        (
            permutation.build_permutation_vocab(num_objects=4),
            permutation.build_permutation_batch(
                batch_size=3,
                num_objects=4,
                num_swaps=5,
                stoi=permutation.build_permutation_vocab(num_objects=4)[1],
                device="cpu",
                rng=random.Random(2001),
            ),
            permutation.required_block_size(num_objects=4, num_swaps=5),
        ),
        (
            tracking.build_tracking_vocab(num_objects=4),
            tracking.build_tracking_batch(
                batch_size=3,
                num_objects=4,
                num_ops=5,
                stoi=tracking.build_tracking_vocab(num_objects=4)[1],
                device="cpu",
                rng=random.Random(2002),
            ),
            tracking.required_block_size(num_objects=4, num_ops=5),
        ),
        (
            pointer_chasing.build_pointer_chasing_vocab(num_nodes=12),
            pointer_chasing.build_pointer_chasing_batch(
                batch_size=3,
                num_nodes=12,
                num_hops=5,
                stoi=pointer_chasing.build_pointer_chasing_vocab(num_nodes=12)[1],
                device="cpu",
                rng=random.Random(2003),
            ),
            pointer_chasing.required_block_size(num_nodes=12, num_hops=5),
        ),
        (
            state_machine.build_state_machine_vocab(num_states=4, alphabet_size=2),
            state_machine.build_state_machine_batch(
                batch_size=3,
                num_states=4,
                alphabet_size=2,
                num_steps=5,
                stoi=state_machine.build_state_machine_vocab(num_states=4, alphabet_size=2)[1],
                device="cpu",
                rng=random.Random(2004),
            ),
            state_machine.required_block_size(num_states=4, alphabet_size=2, num_steps=5),
        ),
    ]

    for vocab_triplet, batch, block_size in cases:
        vocab, stoi, itos = vocab_triplet
        assert len(vocab) == len(stoi) == len(itos)
        assert set(stoi.values()) == set(range(len(vocab)))
        assert all(itos[index] == token for token, index in stoi.items())
        _assert_symbolic_batch_contract(
            batch,
            stoi=stoi,
            vocab_size=len(vocab),
            required_block_size=block_size,
        )

    distribution_vocab = shortest_path.build_shortest_path_vocab(
        "main"
    )
    distribution_batch = shortest_path.build_shortest_path_batch(
        batch_size=3,
        distribution_name="main",
        stoi=distribution_vocab[1],
        device="cpu",
        rng=random.Random(2028),
    )
    _assert_symbolic_batch_contract(
        distribution_batch,
        stoi=distribution_vocab[1],
        vocab_size=len(distribution_vocab[0]),
        required_block_size=shortest_path.required_block_size("main"),
    )

    maze_vocab = maze.build_maze_vocab("searchformer_10")
    maze_batch = maze.build_maze_batch(
        batch_size=3,
        distribution_name="searchformer_10",
        stoi=maze_vocab[1],
        device="cpu",
        rng=random.Random(2029),
    )
    _assert_symbolic_batch_contract(
        maze_batch,
        stoi=maze_vocab[1],
        vocab_size=len(maze_vocab[0]),
        required_block_size=maze.required_block_size("searchformer_10"),
    )

    kwargs = {
        "othello_data_dir": str(tmp_path / "othello_data"),
        "othello_train_games": 16,
        "othello_val_games": 8,
        "othello_dataset_seed": 11,
    }
    vocab, stoi, itos = othello.build_othello_vocab(**kwargs)
    batch = othello.build_othello_batch(
        batch_size=3,
        stoi=stoi,
        device="cpu",
        rng=random.Random(2007),
        split="val",
        **kwargs,
    )
    assert len(vocab) == len(stoi) == len(itos)
    _assert_symbolic_batch_contract(
        batch,
        stoi=stoi,
        vocab_size=len(vocab),
        required_block_size=othello.required_block_size(**kwargs),
    )
