from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from models import CausalTransformer, TransformerConfig
from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.common import build_batch_from_sequences, make_sequence
from tasks.trace import maze, maze_eval, othello, shortest_path
from tasks.trace import shortest_path_eval


class _ForcedChoiceRandom(random.Random):
    forced_choice: str

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        assert self.forced_choice in population
        return [self.forced_choice] * k


def test_bbh_task_solvers_match_sampled_answers():
    rng = random.Random(3)

    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(9)
    _prompt, _answer, pointers, start, final = pointer_chasing.sample_pointer_chasing_example(9, 4, stoi, rng)
    assert pointer_chasing.solve_pointer_chasing(pointers, start, 4)[1] == final

    _vocab, stoi, _ = permutation.build_permutation_vocab(4)
    _prompt, _answer, swaps, final_state = permutation.sample_permutation_example(4, 5, stoi, rng)
    assert permutation.solve_permutation(4, swaps) == final_state

    _vocab, stoi, _ = tracking.build_tracking_vocab(4)
    _prompt, _answer, ops, query, final_object = tracking.sample_tracking_example(4, 5, stoi, rng)
    assert tracking.solve_tracking(4, ops)[1][query] == final_object

    _vocab, stoi, _ = state_machine.build_state_machine_vocab(4, 2)
    sample = state_machine.sample_state_machine_example(4, 2, 5, stoi, rng)
    _prompt, _answer, table, start, actions, _trace, final = sample
    assert state_machine.solve_state_machine(table, start, actions)[1] == final


def test_state_machine_level_zero_uses_shuffled_full_table_factorizations():
    num_states = 4
    alphabet_size = 2
    _vocab, stoi, _ = state_machine.build_state_machine_vocab(
        num_states,
        alphabet_size,
    )
    prefix_len = 1 + num_states + 1 + alphabet_size + 1
    table_token_len = 3 * num_states * alphabet_size
    canonical_pairs = [
        (source, action)
        for source in range(num_states)
        for action in range(alphabet_size)
    ]

    starts_by_part = {}
    table_orders_by_part = {}
    prompt_lengths = set()
    for part, expected_weight in state_machine.LEVEL_ZERO_PART_WEIGHTS:
        assert expected_weight in {20, 40}
        starts = set()
        orders = set()
        for seed in range(32):
            rng = _ForcedChoiceRandom(seed)
            rng.forced_choice = part
            sample = state_machine.sample_state_machine_example(
                num_states,
                alphabet_size,
                0,
                stoi,
                rng,
            )
            prompt, answer, table, start, actions, trace, final = sample
            prompt_lengths.add(len(prompt))
            starts.add(start)

            table_tokens = prompt[prefix_len : prefix_len + table_token_len]
            pairs = []
            for offset in range(0, table_token_len, 3):
                source_id, action_id, target_id = table_tokens[offset : offset + 3]
                source = source_id - stoi[state_machine.state_token(0)]
                action = action_id - stoi[state_machine.action_token(0)]
                target = target_id - stoi[state_machine.state_token(0)]
                assert table[source][action] == target
                pairs.append((source, action))
            assert sorted(pairs) == canonical_pairs
            orders.add(tuple(pairs))

            assert len(actions) == 1
            assert trace == [final]
            assert answer == [stoi[state_machine.state_token(final)]]
            assert state_machine.solve_state_machine(table, start, actions)[1] == final

            if part == "source_only_full_table":
                assert all(len(set(row)) == 1 for row in table)
            elif part == "action_only_full_table":
                assert all(
                    len({row[action] for row in table}) == 1
                    for action in range(alphabet_size)
                )
            else:
                assert part == "full_lookup"
                assert all(len(set(row)) == alphabet_size for row in table)

        starts_by_part[part] = starts
        table_orders_by_part[part] = orders

    assert dict(state_machine.LEVEL_ZERO_PART_WEIGHTS) == {
        "source_only_full_table": 40,
        "action_only_full_table": 40,
        "full_lookup": 20,
    }
    assert all(starts == set(range(num_states)) for starts in starts_by_part.values())
    assert all(len(orders) > 1 for orders in table_orders_by_part.values())
    assert all(tuple(canonical_pairs) not in orders for orders in table_orders_by_part.values())

    level_one_prompt = state_machine.sample_state_machine_example(
        num_states,
        alphabet_size,
        1,
        stoi,
        random.Random(100),
    )[0]
    assert prompt_lengths == {len(level_one_prompt)}


def test_pointer_chasing_level_scales_odd_cycle_without_shortcuts():
    max_level = 8
    label_pool_size = 2 * max_level + 3
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(label_pool_size)

    for level in range(1, max_level + 1):
        prompt, answer, pointers, start, final = pointer_chasing.sample_pointer_chasing_example(
            label_pool_size,
            level,
            stoi,
            random.Random(100 + level),
        )
        active_nodes = {
            source for source, target in enumerate(pointers)
            if source != target
        }
        trace, solved_final = pointer_chasing.solve_pointer_chasing(
            pointers,
            start,
            level,
        )

        assert len(active_nodes) == 2 * level + 1
        assert active_nodes == set(range(2 * level + 1))
        assert start in active_nodes
        assert solved_final == final
        assert answer == [stoi[pointer_chasing.node_token(final)]]
        assert len({start, *trace}) == level + 1
        assert final not in [start, *trace[:-1]]
        assert pointer_chasing.solve_pointer_chasing(
            pointers,
            final,
            level + 1,
        )[1] == start
        assert prompt.index(stoi["<query>"]) == 3 * len(active_nodes)
        assert pointer_chasing.required_block_size(
            label_pool_size,
            level,
        ) == len(prompt) + 3


def test_pointer_chasing_rejects_too_small_label_pool():
    with pytest.raises(ValueError, match="2 \\* num_hops \\+ 1"):
        pointer_chasing.required_block_size(num_nodes=8, num_hops=4)
    with pytest.raises(ValueError, match="at least 1"):
        pointer_chasing.active_num_nodes(0)
    with pytest.raises(ValueError, match="at least 3"):
        pointer_chasing.build_pointer_chasing_vocab(2)


def test_pointer_chasing_level_one_is_learnable_with_full_vocabulary():
    torch.manual_seed(1337)
    label_pool_size = pointer_chasing.DEFAULT_NUM_NODES
    vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(label_pool_size)
    model = CausalTransformer(
        TransformerConfig(
            block_size=pointer_chasing.required_block_size(
                label_pool_size,
                pointer_chasing.DEFAULT_MAX_LEVEL,
            ),
            vocab_size=len(vocab),
            n_layer=4,
            n_head=4,
            n_embd=128,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    train_rng = random.Random(1337)

    for _step in range(400):
        batch = pointer_chasing.build_pointer_chasing_batch(
            batch_size=64,
            num_nodes=label_pool_size,
            num_hops=1,
            stoi=stoi,
            device="cpu",
            rng=train_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.idx)
        loss = model.calc_loss(output.logits, batch.targets)
        loss.backward()
        optimizer.step()

    eval_batch = pointer_chasing.build_pointer_chasing_batch(
        batch_size=256,
        num_nodes=label_pool_size,
        num_hops=1,
        stoi=stoi,
        device="cpu",
        rng=random.Random(2027),
    )
    with torch.no_grad():
        logits = model(eval_batch.idx).logits
    rows = torch.arange(eval_batch.idx.size(0))
    answer_positions = eval_batch.prompt_lengths - 1
    predictions = logits[rows, answer_positions].argmax(dim=-1)
    targets = eval_batch.targets[rows, answer_positions]
    assert (predictions == targets).float().mean().item() >= 0.99


def test_othello_generated_games_are_legal_and_dataset_cache_is_deterministic(tmp_path):
    for seed in [0, 1, 2, 3, 309]:
        trace = othello.random_game_trace64(seed=seed)
        ids = [square + othello.MOVE_TOKEN_OFFSET for square in trace]
        assert othello.legal_prefix_length(ids) == (len(trace), True)
        cut = len(ids) // 2
        assert othello.legal_prefix_length(
            ids[cut:],
            prefix_move_token_ids=ids[:cut],
        ) == (len(ids) - cut, True)
        assert ids[cut] in othello.legal_move_token_ids_after_prefix(ids[:cut])
        assert othello.legal_prefix_length([*ids, 0]) == (len(ids), False)

    with pytest.raises(ValueError, match="illegal move"):
        othello.legal_move_token_ids_after_prefix([othello.MOVE_TOKEN_OFFSET])

    kwargs = dict(
        othello_data_dir=str(tmp_path),
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=19,
    )
    othello.ensure_othello_datasets(**kwargs)
    first = othello.load_othello_dataset(split="train", **kwargs)
    trace_a = first.sample_trace(random.Random(7))
    othello._DATASET_CACHE.clear()
    second = othello.load_othello_dataset(split="train", **kwargs)
    trace_b = second.sample_trace(random.Random(7))
    assert trace_a == trace_b

    _vocab, stoi, _itos = othello.build_othello_vocab(
        othello_train_games=8,
        othello_val_games=4,
    )
    _prompt, answer, trace = othello.sample_othello_example(
        stoi,
        random.Random(7),
        split="train",
        **kwargs,
    )
    assert answer == [stoi[othello.move_token(square)] for square in trace]
    assert stoi["<pad>"] not in answer


def test_othello_generation_is_partition_invariant():
    seeds = othello.np.random.SeedSequence(123).generate_state(12, dtype=othello.np.uint64)
    whole = othello._generate_trace_dataset_arrays_from_seeds(seeds)
    left = othello._generate_trace_dataset_arrays_from_seeds(seeds[:5])
    right = othello._generate_trace_dataset_arrays_from_seeds(seeds[5:])
    partitioned_traces = othello.np.concatenate((left[0], right[0]), axis=0)
    partitioned_lengths = othello.np.concatenate((left[1], right[1]), axis=0)
    assert othello.np.array_equal(whole[0], partitioned_traces)
    assert othello.np.array_equal(whole[1], partitioned_lengths)


def test_shortest_path_distributions_are_varied_permuted_and_solver_verified():
    for distribution_name in ("easy", "main"):
        distribution = shortest_path.get_shortest_path_distribution(distribution_name)
        _vocab, stoi, _itos = shortest_path.build_generation_vocab(
            distribution_name
        )
        first_rng = random.Random(714)
        second_rng = random.Random(714)
        observed_shapes = set()
        observed_edge_counts = set()
        observed_starts = set()
        for _ in range(500):
            first = shortest_path.sample_shortest_path_example(
                distribution_name,
                stoi,
                first_rng,
            )
            second = shortest_path.sample_shortest_path_example(
                distribution_name,
                stoi,
                second_rng,
            )
            assert first == second
            prompt, answer, edges, start, goal, target_path = first
            parsed_edges, parsed_start, parsed_goal = (
                shortest_path.parse_prompt_metadata(prompt)
            )
            num_nodes = prompt.index(stoi[shortest_path.EDGES_TOKEN]) - 1
            solved_path, path_count = shortest_path.solve_shortest_path(
                num_nodes,
                edges,
                start,
                goal,
            )
            adjacency = [[] for _ in range(num_nodes)]
            for source, target in edges:
                adjacency[source].append(target)

            assert set(parsed_edges) == set(edges)
            assert (parsed_start, parsed_goal) == (start, goal)
            assert path_count == 1
            assert solved_path == target_path
            assert answer == [
                stoi[shortest_path.node_token(node)]
                for node in target_path
            ]
            assert distribution.min_nodes <= num_nodes <= distribution.max_nodes
            assert (
                distribution.min_path_length
                <= len(target_path) - 1
                <= distribution.max_path_length
            )
            assert max(
                sum(source == node for source, _target in edges)
                for node in range(num_nodes)
            ) <= distribution.max_out_degree
            assert sum(len(adjacency[node]) > 1 for node in target_path[:-1]) >= 1
            observed_shapes.add((num_nodes, len(target_path) - 1))
            observed_edge_counts.add(len(edges))
            observed_starts.add(start)

        assert len(observed_edge_counts) > 1
        assert len(observed_starts) >= distribution.max_nodes - 1
        assert len(observed_shapes) > 1


def test_shortest_path_main_uniformly_mixes_feasible_path_lengths():
    _vocab, stoi, _itos = shortest_path.build_generation_vocab("main")
    rng = random.Random(1337)
    path_length_counts = {path_length: 0 for path_length in range(5, 11)}
    for _ in range(6_000):
        prompt, _answer, edges, start, goal, path = (
            shortest_path.sample_shortest_path_example("main", stoi, rng)
        )
        path_length = len(path) - 1
        num_nodes = prompt.index(stoi[shortest_path.EDGES_TOKEN]) - 1
        path_length_counts[path_length] += 1
        adjacency = [[] for _ in range(num_nodes)]
        for source, target in edges:
            adjacency[source].append(target)
        decision_points = sum(len(adjacency[node]) > 1 for node in path[:-1])
        random_legal_probability = 1.0
        for node in path[:-1]:
            random_legal_probability /= len(adjacency[node])
        assert num_nodes >= path_length + 9
        assert decision_points >= 4
        assert random_legal_probability <= 1 / 16

    assert all(count >= 850 for count in path_length_counts.values())
    assert shortest_path.path_length_bucket(5) == "short"
    assert shortest_path.path_length_bucket(7) == "medium"
    assert shortest_path.path_length_bucket(10) == "long"


def test_shortest_path_label_permutation_preserves_the_solution():
    edges = [(0, 1), (1, 3), (0, 2), (2, 1)]
    path = [0, 1, 3]
    permutation = [2, 0, 3, 1]
    mapped_edges, mapped_path = shortest_path.permute_graph_labels(
        edges,
        path,
        permutation,
    )
    solved_path, path_count = shortest_path.solve_shortest_path(
        4,
        mapped_edges,
        mapped_path[0],
        mapped_path[-1],
    )
    assert mapped_path == [2, 0, 1]
    assert path_count == 1
    assert solved_path == mapped_path


def test_shortest_path_step_accuracy_excludes_the_supplied_start_node():
    _vocab, stoi, _itos = shortest_path.build_shortest_path_vocab(
        shortest_path.DEFAULT_SMOKE_DATA_DIR
    )
    batch = shortest_path.build_shortest_path_batch(
        batch_size=1,
        shortest_path_data_dir=shortest_path.DEFAULT_SMOKE_DATA_DIR,
        split="validation",
        device="cpu",
        rng=random.Random(19),
    )
    prompt_len = int(batch.prompt_lengths[0])
    output_len = int(batch.output_lengths[0])
    generated_suffix = batch.targets[
        0,
        prompt_len - 1 : prompt_len - 1 + output_len,
    ].clone()
    generated_suffix[1] = generated_suffix[0]

    class FixedGeneration:
        def generate(self, prompt, **_kwargs):
            return torch.cat((prompt, generated_suffix[None, :]), dim=1)

    metrics = shortest_path_eval.generation_metrics(
        FixedGeneration(),
        batch,
        SimpleNamespace(
            architecture="transformer",
            inference_mode="recompute",
            token_selection="argmax",
            shortest_path_distribution="easy",
        ),
    )
    assert "path_step_0_accuracy" not in metrics
    assert metrics["path_step_1_accuracy"] == 0.0
    assert metrics["path_step_2_accuracy"] == 1.0
    assert metrics["path_step_1_examples__sum"] == 1.0


def test_shortest_path_easy_example_can_be_overfit_and_generated():
    torch.manual_seed(123)
    vocab, stoi, _itos = shortest_path.build_shortest_path_vocab(
        shortest_path.DEFAULT_SMOKE_DATA_DIR
    )
    batch = shortest_path.build_shortest_path_batch(
        batch_size=1,
        shortest_path_data_dir=shortest_path.DEFAULT_SMOKE_DATA_DIR,
        split="train",
        device="cpu",
        rng=random.Random(17),
    )
    model = CausalTransformer(
        TransformerConfig(
            block_size=batch.idx.shape[1],
            vocab_size=len(vocab),
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _step in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.idx)
        loss = model.calc_loss(output.logits, batch.targets)
        loss.backward()
        optimizer.step()

    prompt_len = int(batch.prompt_lengths[0])
    output_len = int(batch.output_lengths[0])
    generated = model.generate(
        batch.idx[:, :prompt_len],
        output_len,
        do_sample=False,
        inference_mode="recompute",
    )
    expected = batch.targets[:, prompt_len - 1 : prompt_len - 1 + output_len]
    assert loss.item() < 0.01
    assert torch.equal(generated[:, prompt_len : prompt_len + output_len], expected)


def test_compiled_maze_datasets_are_memory_mapped_and_solver_verified():
    for input_representation in maze.INPUT_REPRESENTATIONS:
        for target_representation in maze.TARGET_REPRESENTATIONS:
            for route_policy in maze.ROUTE_POLICIES:
                vocab, _stoi, itos = maze.build_maze_vocab(
                    maze.DEFAULT_SMOKE_DATA_DIR,
                    input_representation,
                    target_representation,
                    route_policy,
                )
                dataset = maze.load_maze_dataset(
                    split="train",
                    maze_data_dir=maze.DEFAULT_SMOKE_DATA_DIR,
                    input_representation=input_representation,
                    target_representation=target_representation,
                    route_policy=route_policy,
                )
                assert isinstance(dataset.tokens, np.memmap)
                assert len(dataset) == 18
                sequence_length = int(dataset.sequence_lengths[0])
                prompt_length = int(dataset.prompt_lengths[0])
                tokens = dataset.tokens[0, :sequence_length].tolist()
                problem = maze.parse_maze_prompt(
                    tokens[1 : prompt_length - 1],
                    itos=itos,
                    input_representation=input_representation,
                )
                marker_ok, target_path = maze.decode_maze_target(
                    tokens[prompt_length:-1],
                    problem=problem,
                    itos=itos,
                    target_representation=target_representation,
                )
                shortest = maze.solve_maze(problem)
                assert marker_ok
                assert target_path[0] == problem.start
                assert target_path[-1] == problem.goal
                assert all(
                    current in maze.neighboring_cells(
                        int(previous),
                        width=problem.width,
                        height=problem.height,
                    )
                    for previous, current in zip(target_path, target_path[1:])
                )
                if route_policy != "dfs":
                    assert len(target_path) == len(shortest)
                assert int(dataset.tokens.max()) < len(vocab)


def test_compiled_maze_batch_sampling_is_seeded():
    kwargs = dict(
        batch_size=4,
        maze_data_dir=maze.DEFAULT_SMOKE_DATA_DIR,
        input_representation="dense-cells",
        target_representation="actions",
        route_policy="uniform_shortest",
        split="validation",
        device="cpu",
    )
    first = maze.build_maze_batch(rng=random.Random(827), **kwargs)
    second = maze.build_maze_batch(rng=random.Random(827), **kwargs)
    assert torch.equal(first.idx, second.idx)
    assert torch.equal(first.targets, second.targets)


def test_missing_compiled_maze_dataset_does_not_generate_online(tmp_path):
    with pytest.raises(FileNotFoundError, match="never generates maze data online"):
        maze.build_maze_vocab(tmp_path, "sparse-cells", "cell-path", "astar")


@pytest.mark.parametrize("target_representation", maze.TARGET_REPRESENTATIONS)
def test_maze_evaluation_accepts_an_alternative_optimal_path(target_representation):
    _vocab, stoi, _itos = maze.build_maze_vocab(
        maze.DEFAULT_SMOKE_DATA_DIR,
        "sparse-cells",
        target_representation,
        "astar",
    )

    def cell_id(cell: int) -> int:
        row, column = divmod(cell, 5)
        return stoi[f"r{row}c{column}"]

    # Both 0 -> 1 -> 6 and 0 -> 5 -> 6 are optimal on an open 5x5 grid.
    prompt = [
        stoi[maze.HEIGHT_TOKEN],
        stoi["5"],
        stoi[maze.WIDTH_TOKEN],
        stoi["5"],
        stoi[maze.START_TOKEN],
        cell_id(0),
        stoi[maze.GOAL_TOKEN],
        cell_id(6),
        stoi[maze.WALLS_TOKEN],
    ]
    if target_representation == "cell-path":
        canonical_answer = [
            stoi[maze.PATH_TOKEN],
            *(cell_id(cell) for cell in (0, 1, 6)),
        ]
        alternative_answer = [
            stoi[maze.PATH_TOKEN],
            *(cell_id(cell) for cell in (0, 5, 6)),
        ]
    else:
        canonical_answer = [stoi[maze.ACTIONS_TOKEN], stoi["R"], stoi["D"]]
        alternative_answer = [stoi[maze.ACTIONS_TOKEN], stoi["D"], stoi["R"]]
    batch = build_batch_from_sequences(
        [make_sequence(prompt, canonical_answer, stoi)],
        pad_id=stoi["<pad>"],
        device="cpu",
    )
    alternative_suffix = torch.tensor(
        [*alternative_answer, stoi["<eos>"]]
    )

    class FixedGeneration:
        def generate(self, supplied_prompt, **_kwargs):
            return torch.cat(
                (supplied_prompt, alternative_suffix[None, :]),
                dim=1,
            )

    metrics = maze_eval.generation_metrics(
        FixedGeneration(),
        batch,
        SimpleNamespace(
            architecture="transformer",
            inference_mode="recompute",
            token_selection="argmax",
            maze_data_dir=maze.DEFAULT_SMOKE_DATA_DIR,
            maze_input_representation="sparse-cells",
            maze_target_representation=target_representation,
            maze_route_policy="astar",
        ),
    )
    assert metrics["optimal_route"] == 1.0
    assert metrics["exact_target_route"] == 0.0


def test_maze_evaluation_accepts_early_eos_for_a_shorter_route_than_dfs_target():
    _vocab, stoi, _itos = maze.build_maze_vocab(
        maze.DEFAULT_SMOKE_DATA_DIR,
        "sparse-cells",
        "actions",
        "dfs",
    )
    prompt = [
        stoi[maze.HEIGHT_TOKEN],
        stoi["5"],
        stoi[maze.WIDTH_TOKEN],
        stoi["5"],
        stoi[maze.START_TOKEN],
        stoi["r0c0"],
        stoi[maze.GOAL_TOKEN],
        stoi["r1c0"],
        stoi[maze.WALLS_TOKEN],
    ]
    dfs_answer = [
        stoi[maze.ACTIONS_TOKEN],
        *(stoi[action] for action in "RRRRDLLLL"),
    ]
    batch = build_batch_from_sequences(
        [make_sequence(prompt, dfs_answer, stoi)],
        pad_id=stoi["<pad>"],
        device="cpu",
    )
    shorter_optimal = torch.tensor(
        [stoi[maze.ACTIONS_TOKEN], stoi["D"], stoi["<eos>"]]
    )

    class FixedGeneration:
        def generate(self, supplied_prompt, **_kwargs):
            return torch.cat((supplied_prompt, shorter_optimal[None, :]), dim=1)

    metrics = maze_eval.generation_metrics(
        FixedGeneration(),
        batch,
        SimpleNamespace(
            architecture="transformer",
            inference_mode="recompute",
            token_selection="argmax",
            maze_data_dir=maze.DEFAULT_SMOKE_DATA_DIR,
            maze_input_representation="sparse-cells",
            maze_target_representation="actions",
            maze_route_policy="dfs",
        ),
    )
    assert metrics == {"optimal_route": 1.0, "exact_target_route": 0.0}
