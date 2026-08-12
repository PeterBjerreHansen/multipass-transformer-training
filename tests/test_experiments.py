from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from experiments.common import (
    RunArtifacts,
    clip_gradients,
    evaluate_prebuilt_batches,
    gradient_norms,
    learning_rate_at_step,
    load_checkpoint_payload,
    prepare_run_artifacts,
    restore_checkpoint_state,
    runtime_resource_stats,
    resolve_evaluation_checkpoint,
    save_latest_checkpoint,
)
from experiments.summarize_ablation import infer_quality_metric, recommend
from experiments.diagnose_memory import (
    _effective_rank,
    _memory_stats,
    _relative_linf_residual,
    memory_interventions,
    pass_dynamics,
    teacher_forced_schedule_gap,
)
from experiments.presets import BBH_PRESETS, TRACE_PRESETS
from experiments.train_bbh import BBH_TASKS, build_fixed_eval_batches, parse_args as parse_bbh_args
from experiments.train_trace import parse_args as parse_trace_args
from models import (
    MemoryAddTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MultiPassConfig,
)
from tasks.bbh import pointer_chasing
from tasks.trace import maze, othello, shortest_path
from tasks.trace.othello_eval import (
    build_eval_examples,
    legal_set_step_metrics,
)
from tasks.trace.registry import TRACE_TASKS


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        architecture="memory_tape",
        inference_mode="recompute",
        token_selection="sample",
        pass_loss_weights=[0, 0, 1],
        seed=17,
        device="cpu",
        batch_size=2,
        eval_batches=2,
        task="pointer_chasing",
        num_nodes=5,
        curriculum_start_level=1,
        max_level=2,
    )


def test_fixed_eval_batches_are_identical_every_time():
    args = _args()
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    task = BBH_TASKS["pointer_chasing"]
    a = build_fixed_eval_batches(args, task, stoi, 2)
    b = build_fixed_eval_batches(args, task, stoi, 2)
    for batch_a, batch_b in zip(a, b):
        assert torch.equal(batch_a.idx, batch_b.idx)
        assert torch.equal(batch_a.targets, batch_b.targets)


def test_global_gradient_clipping_caps_the_combined_l2_norm():
    model = torch.nn.Linear(1, 2, bias=False)
    model.weight.grad = torch.tensor([[3.0], [4.0]])

    clip_gradients(model, 2.0)

    assert torch.linalg.vector_norm(model.weight.grad).item() == pytest.approx(2.0)


@pytest.mark.parametrize("parser", [parse_bbh_args, parse_trace_args])
def test_training_presets_default_to_global_gradient_clipping(parser):
    args = parser([])
    assert args.max_grad_norm == pytest.approx(5.0)

    overridden = parser(["--max-grad-norm", "1000000"])
    assert overridden.max_grad_norm == pytest.approx(1_000_000.0)


def test_evaluation_sampling_is_repeatable_and_does_not_change_global_rng():
    args = _args()
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 5, 2, stoi, device="cpu", rng=random.Random(2))
    before = torch.get_rng_state().clone()
    first = evaluate_prebuilt_batches(model, args, [batch], generation_seed=123)
    middle = torch.get_rng_state().clone()
    second = evaluate_prebuilt_batches(model, args, [batch], generation_seed=123)
    after = torch.get_rng_state().clone()
    assert first["exact_match"] == second["exact_match"]
    assert first["token_accuracy"] == second["token_accuracy"]
    assert torch.equal(before, middle)
    assert torch.equal(before, after)


def test_evaluation_aggregates_conditional_metrics_by_example_count():
    args = _args()
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(
        2,
        5,
        2,
        stoi,
        device="cpu",
        rng=random.Random(2),
    )
    batch_metrics = iter(
        (
            {
                "conditional_accuracy": 1.0,
                "conditional_accuracy__weight": 1.0,
                "conditional_examples__sum": 1.0,
            },
            {
                "conditional_accuracy": 0.25,
                "conditional_accuracy__weight": 3.0,
                "conditional_examples__sum": 3.0,
            },
        )
    )

    def generation_metrics(*_args, **_kwargs):
        return next(batch_metrics)

    result = evaluate_prebuilt_batches(
        model,
        args,
        [batch, batch],
        generation_metrics_fn=generation_metrics,
    )
    assert result["conditional_accuracy"] == pytest.approx(0.4375)
    assert result["conditional_examples"] == 4.0
    assert not any(key.endswith(("__weight", "__sum")) for key in result)


def test_memory_interventions_pass_dynamics_and_schedule_gap_return_finite_values():
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 5, 2, stoi, device="cpu", rng=random.Random(2))
    full_output = model(batch.idx)
    interventions = memory_interventions(model, batch, seed=3)
    assert interventions["losses"]["correct"] == model.calc_loss(full_output.logits, batch.targets).item()
    assert set(interventions["losses"]) == {
        "correct", "zero_memory_bank", "cross_example",
        "causal_position_resample", "causal_prefix_mean", "extra_lag"
    }
    assert interventions["loss_deltas"]["correct"] == 0.0
    assert set(interventions["source_memory"]) == {"rms_norm", "effective_rank"}
    dynamics = pass_dynamics(model, batch, extra_passes=2)
    assert len(dynamics["trained_passes"]) == 3
    assert len(dynamics["extra_passes"]) == 2
    assert all(
        set(item["relative_linf_residual"]) == {"mean", "max"}
        for item in (*dynamics["trained_passes"], *dynamics["extra_passes"])
    )
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in dynamics["extra_passes"])

    model.eval()
    schedule_gap = teacher_forced_schedule_gap(model, batch, horizon=2)
    assert schedule_gap["horizon"] == 2
    assert [item["count"] for item in schedule_gap["positions"]] == [2, 2]
    assert schedule_gap["overall"]["count"] == 4
    first = schedule_gap["positions"][0]
    assert first["logit_kl"] < 1e-6
    assert abs(first["nll_delta"]) < 1e-6
    assert first["top1_agreement"] == 1.0
    assert first["memory_rms_delta"] < 1e-6
    for position in schedule_gap["positions"]:
        for name, value in position.items():
            if name not in {"generated_position", "count"}:
                assert torch.isfinite(torch.tensor(value)), name


def test_relative_linf_residual_is_per_example_and_ignores_padding():
    previous = torch.tensor([[[1.0], [0.0]], [[0.0], [0.0]]])
    current = torch.tensor([[[2.0], [100.0]], [[4.0], [2.0]]])
    valid_positions = torch.tensor([[True, False], [True, True]])

    residual = _relative_linf_residual(previous, current, valid_positions)

    assert residual["mean"] == pytest.approx(0.75)
    assert residual["max"] == pytest.approx(1.0)


def test_relative_linf_residual_is_zero_for_identical_tapes():
    memory = torch.randn(2, 3, 4)
    residual = _relative_linf_residual(
        memory,
        memory.clone(),
        torch.ones(2, 3, dtype=torch.bool),
    )
    assert residual == {"mean": 0.0, "max": 0.0}


def test_effective_rank_distinguishes_collapsed_and_full_rank_memory():
    collapsed = torch.ones(16, 1) @ torch.arange(1, 9, dtype=torch.float32)[None, :]
    full_rank = torch.eye(8).repeat(2, 1)

    assert _effective_rank(collapsed) == pytest.approx(0.0, abs=1e-5)
    assert _effective_rank(full_rank) == pytest.approx(7.0, rel=1e-5)
    assert set(_memory_stats(full_rank.reshape(2, 8, 8))) == {
        "rms_norm",
        "effective_rank",
    }


def test_memory_add_diagnostics_return_finite_values():
    model = MemoryAddTransformer(MultiPassConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(
        2,
        5,
        2,
        stoi,
        device="cpu",
        rng=random.Random(2),
    )
    interventions = memory_interventions(model, batch, seed=3)
    assert all(torch.isfinite(torch.tensor(value)) for value in interventions["losses"].values())
    dynamics = pass_dynamics(model, batch, extra_passes=2)
    assert len(dynamics["trained_passes"]) == 3
    assert len(dynamics["extra_passes"]) == 2
    schedule_gap = teacher_forced_schedule_gap(model, batch, horizon=2)
    assert schedule_gap["overall"]["count"] == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in schedule_gap["overall"].values())


def test_gradient_norms_cover_memory_subsystems_after_backward():
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 5, 2, stoi, device="cpu", rng=random.Random(2))
    output = model(batch.idx)
    loss = model.calc_total_loss(output, batch.targets, [0, 0, 1]).loss
    loss.backward()
    norms = gradient_norms(model)
    assert {"global", "backbone", "memory_writer", "memory_attention"} <= set(norms)
    assert "memory_gate" not in norms
    assert all(torch.isfinite(torch.tensor(value)) and value > 0 for value in norms.values())


def test_memory_add_gradient_norms_report_fusion_branch():
    model = MemoryAddTransformer(MultiPassConfig(24, 12, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    batch = pointer_chasing.build_pointer_chasing_batch(
        2,
        5,
        2,
        stoi,
        device="cpu",
        rng=random.Random(2),
    )
    loss = model.calc_total_loss(model(batch.idx), batch.targets, [0, 0, 1]).loss
    loss.backward()
    norms = gradient_norms(model)
    assert norms["memory_fusion"] > 0
    assert norms["memory_writer"] > 0


def _one_step(model, optimizer, tokens, targets):
    optimizer.zero_grad(set_to_none=True)
    output = model(tokens)
    loss = model.calc_total_loss(output, targets, [0, 0, 1]).loss
    loss.backward()
    optimizer.step()


def test_checkpoint_resume_reproduces_next_optimizer_step(tmp_path):
    torch.manual_seed(101)
    config = MemoryTapeConfig(8, 13, 1, 1, 8, 3)
    model = MemoryTapeTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(55)
    batch1 = torch.randint(0, 13, (2, 6), generator=gen)
    target1 = torch.randint(0, 13, (2, 6), generator=gen)
    batch2 = torch.randint(0, 13, (2, 6), generator=gen)
    target2 = torch.randint(0, 13, (2, 6), generator=gen)

    _one_step(model, optimizer, batch1, target1)
    artifacts = RunArtifacts(tmp_path, tmp_path / "config.json", tmp_path / "metrics.jsonl", tmp_path / "latest.pt")
    args = SimpleNamespace(example=True)
    local_rng = random.Random(9)
    save_latest_checkpoint(
        artifacts,
        model=model,
        optimizer=optimizer,
        args=args,
        step=1,
        extra_state={"local_rng": local_rng.getstate()},
    )
    _one_step(model, optimizer, batch2, target2)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    torch.manual_seed(999)
    restored_model = MemoryTapeTransformer(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint_payload(tmp_path / "latest.pt", device="cpu")
    restore_checkpoint_state(checkpoint, model=restored_model, optimizer=restored_optimizer, device="cpu")
    _one_step(restored_model, restored_optimizer, batch2, target2)
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_warmup_cosine_learning_rate_uses_absolute_steps():
    schedule = {
        "schedule": "warmup_cosine",
        "peak_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 4_000,
        "decay_steps": 200_000,
    }
    assert learning_rate_at_step(1, **schedule) == pytest.approx(3e-4 / 4_000)
    assert learning_rate_at_step(4_000, **schedule) == pytest.approx(3e-4)
    assert learning_rate_at_step(102_000, **schedule) == pytest.approx(1.65e-4)
    assert learning_rate_at_step(200_000, **schedule) == pytest.approx(3e-5)
    assert learning_rate_at_step(220_000, **schedule) == pytest.approx(3e-5)

    rates = [
        learning_rate_at_step(step, **schedule)
        for step in range(4_000, 200_001, 1_000)
    ]
    assert all(left >= right for left, right in zip(rates, rates[1:]))


def test_run_directory_cannot_be_reused_without_resume(tmp_path):
    run_dir = tmp_path / "run"
    model = MemoryTapeTransformer(MemoryTapeConfig(8, 13, 1, 1, 8, 3))
    args = SimpleNamespace(run_dir=str(run_dir), resume_from=None)
    prepare_run_artifacts(args, model=model, default_root_parts=("test",))

    with pytest.raises(FileExistsError, match="run directory is not empty"):
        prepare_run_artifacts(
            SimpleNamespace(run_dir=str(run_dir), resume_from=None),
            model=model,
            default_root_parts=("test",),
        )


def test_resume_preserves_original_run_configuration(tmp_path):
    run_dir = tmp_path / "run"
    model = MemoryTapeTransformer(MemoryTapeConfig(8, 13, 1, 1, 8, 3))
    prepare_run_artifacts(
        SimpleNamespace(run_dir=str(run_dir), resume_from=None, marker="original"),
        model=model,
        default_root_parts=("test",),
    )
    original_config = (run_dir / "config.json").read_text(encoding="utf-8")

    prepare_run_artifacts(
        SimpleNamespace(
            run_dir=str(run_dir),
            resume_from=str(run_dir),
            marker="resume",
        ),
        model=model,
        default_root_parts=("test",),
    )
    assert (run_dir / "config.json").read_text(encoding="utf-8") == original_config


def test_cli_has_only_two_inference_modes_and_no_cache_source():
    args = parse_bbh_args([
        "--preset", "pointer_chasing_smoke",
        "--architecture", "memory_tape",
        "--inference-mode", "append_recurrent",
    ])
    assert args.inference_mode == "append_recurrent"
    assert not hasattr(args, "cache_source")
    assert not hasattr(args, "memory_tape_gate")


def test_shortest_path_cli_exposes_only_easy_and_main_distributions():
    main = parse_trace_args(["--preset", "shortest_path_main"])
    easy = parse_trace_args(["--preset", "shortest_path_easy"])
    smoke = parse_trace_args(["--preset", "shortest_path_smoke"])
    assert main.shortest_path_distribution == "main"
    assert easy.shortest_path_distribution == "easy"
    assert easy.train_steps == 50_000
    assert main.train_steps == 200_000
    assert smoke.shortest_path_distribution == "easy"
    with pytest.raises(SystemExit):
        parse_trace_args(
            [
                "--preset",
                "shortest_path_main",
                "--distractor-edges",
                "5",
            ]
        )


def test_shortest_path_main_uses_warmup_cosine_learning_rate():
    path = TRACE_PRESETS["shortest_path_main"].values
    assert path["lr"] == 5e-4
    assert path["lr_schedule"] == "warmup_cosine"
    assert path["min_lr"] == 1e-5
    assert path["lr_warmup_steps"] == 4_000
    assert path["lr_decay_steps"] == path["train_steps"] == 200_000
    for name, preset in TRACE_PRESETS.items():
        if name not in {"maze_main", "shortest_path_main"}:
            assert preset.values["lr_schedule"] == "constant"


def test_maze_cli_exposes_named_searchformer_distributions():
    main = parse_trace_args(["--preset", "maze_main"])
    smoke = parse_trace_args(["--preset", "maze_smoke"])
    assert main.task == "maze"
    assert main.maze_distribution == "searchformer_10"
    assert smoke.maze_distribution == "smoke"
    for distribution_name in (
        "searchformer_10",
        "searchformer_20",
        "searchformer_30",
    ):
        args = parse_trace_args(
            [
                "--preset",
                "maze_main",
                "--maze-distribution",
                distribution_name,
            ]
        )
        assert args.maze_distribution == distribution_name
    assert maze.required_block_size("searchformer_10") == 107
    assert maze.required_block_size("searchformer_20") == 407
    assert maze.required_block_size("searchformer_30") == 907


def test_main_presets_use_declared_experiment_scales():
    from experiments.presets import BBH_PRESETS, TRACE_PRESETS

    pointer = BBH_PRESETS["pointer_chasing_main"].values
    assert pointer["num_nodes"] == 65
    assert pointer["curriculum_start_level"] == 1
    assert pointer["max_level"] == 32
    assert pointer_chasing.required_block_size(
        pointer["num_nodes"],
        pointer["max_level"],
    ) == 232
    assert pointer["lr"] == 1e-4
    assert pointer["eval_interval"] == 200
    assert pointer["batch_size"] == 64

    othello_main = TRACE_PRESETS["othello_main"].values
    assert othello_main["othello_train_games"] == 5_000_000
    assert othello_main["othello_val_games"] == 1_024
    assert othello_main["batch_size"] == 128
    assert othello_main["eval_interval"] == 5_000

    path = TRACE_PRESETS["shortest_path_main"].values
    assert path["shortest_path_distribution"] == "main"
    assert not any(
        key in path
        for key in (
            "num_nodes",
            "shortest_path_length",
            "branching_factor",
            "distractor_edges",
        )
    )
    assert shortest_path.required_block_size("easy") == 69
    assert shortest_path.required_block_size("main") == 145


def test_othello_prefix_examples_and_legal_set_metrics_are_deterministic():
    _vocab, stoi, _itos = othello.build_othello_vocab(
        othello_train_games=1,
        othello_val_games=1,
    )
    traces = [othello.random_game_trace64(seed=7), othello.random_game_trace64(seed=8)]
    first = build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode="all",
        prefix_fractions=(0.25, 0.5, 0.75),
        rng=random.Random(99),
    )
    second = build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode="all",
        prefix_fractions=(0.25, 0.5, 0.75),
        rng=random.Random(99),
    )
    assert first == second
    assert {example.protocol for example in first} == {
        "full-game",
        "random-prefix",
        "prefix-grid-0.25",
        "prefix-grid-0.5",
        "prefix-grid-0.75",
    }
    assert all(0 <= example.cut < len(example.trace_move_ids) for example in first)

    legal_ids = othello.legal_move_token_ids_after_prefix(())
    logits = torch.zeros(len(stoi))
    metrics = legal_set_step_metrics(logits, legal_ids, legal_ids[0])
    assert metrics["legal_set_size"] == len(legal_ids)
    assert metrics["legal_probability_mass"] == pytest.approx(len(legal_ids) / len(stoi))
    assert metrics["legal_set_nll"] == pytest.approx(
        -torch.log(torch.tensor(len(legal_ids) / len(stoi))).item()
    )


def test_trace_registry_preserves_seeded_task_behavior(tmp_path):
    othello_args = SimpleNamespace(
        batch_size=3,
        device="cpu",
        othello_data_dir=str(tmp_path / "othello"),
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=31,
    )
    othello_task = TRACE_TASKS["othello"]
    direct_othello_vocab = othello.build_othello_vocab(
        othello_train_games=8,
        othello_val_games=4,
    )
    assert othello_task.build_vocab(othello_args) == direct_othello_vocab
    assert othello_task.required_block_size(othello_args) == othello.required_block_size(
        othello_train_games=8,
        othello_val_games=4,
    )
    direct_othello_batch = othello.build_othello_batch(
        batch_size=3,
        stoi=direct_othello_vocab[1],
        device="cpu",
        rng=random.Random(2027),
        split="val",
        othello_data_dir=othello_args.othello_data_dir,
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=31,
    )
    registered_othello_batch = othello_task.build_batch(
        othello_args,
        direct_othello_vocab[1],
        random.Random(2027),
        split="val",
    )
    assert torch.equal(registered_othello_batch.idx, direct_othello_batch.idx)
    assert torch.equal(registered_othello_batch.targets, direct_othello_batch.targets)


def test_runtime_resource_stats_reports_peak_rss():
    stats = runtime_resource_stats("cpu")
    assert stats["process_peak_rss_bytes"] > 0


def test_main_bbh_preset_contract_is_frozen():
    common = {
        "model_size": "small",
        "n_pass": 4,
        "pass_loss_weights": [0.0, 0.0, 1.0, 1.0],
        "batch_size": 64,
        "train_steps": 50_000,
        "lr": 1e-4,
        "max_grad_norm": 5.0,
        "weight_decay": 0.0,
        "eval_interval": 200,
        "eval_batches": 4,
        "seed": 1337,
        "max_level": 64,
        "curriculum_threshold": 0.95,
        "review_easier_every": 2,
        "token_selection": "argmax",
        "inference_mode": "recompute",
    }
    task_contracts = {
        "pointer_chasing_main": {
            "task": "pointer_chasing",
            "num_nodes": 65,
            "curriculum_start_level": 1,
            "max_level": 32,
        },
        "tracking_main": {
            "task": "tracking",
            "num_objects": 4,
            "curriculum_start_level": 1,
        },
        "permutation_main": {
            "task": "permutation",
            "num_objects": 4,
            "curriculum_start_level": 1,
        },
        "state_machine_main": {
            "task": "state_machine",
            "num_states": 4,
            "alphabet_size": 2,
            "curriculum_start_level": 0,
        },
    }
    for name, task_values in task_contracts.items():
        values = BBH_PRESETS[name].values
        for key, expected in {**common, **task_values}.items():
            assert values[key] == expected, f"{name}.{key} changed"


def test_main_trace_preset_contract_is_frozen():
    common = {
        "model_size": "small",
        "n_pass": 4,
        "pass_loss_weights": [0.0, 0.0, 1.0, 1.0],
        "max_grad_norm": 5.0,
        "weight_decay": 0.0,
        "seed": 1337,
        "inference_mode": "append_recurrent",
    }
    contracts = {
        "othello_main": {
            "task": "othello",
            "batch_size": 128,
            "train_steps": 500_000,
            "lr": 1e-4,
            "lr_schedule": "constant",
            "min_lr": 1e-4,
            "lr_warmup_steps": 0,
            "lr_decay_steps": 0,
            "eval_interval": 5_000,
            "eval_batches": 1,
            "othello_train_games": 5_000_000,
            "othello_val_games": 1_024,
        },
        "shortest_path_main": {
            "task": "shortest_path",
            "batch_size": 64,
            "train_steps": 200_000,
            "lr": 5e-4,
            "lr_schedule": "warmup_cosine",
            "min_lr": 1e-5,
            "lr_warmup_steps": 4_000,
            "lr_decay_steps": 200_000,
            "eval_interval": 5_000,
            "eval_batches": 4,
            "shortest_path_distribution": "main",
        },
        "maze_main": {
            "task": "maze",
            "batch_size": 64,
            "train_steps": 200_000,
            "lr": 5e-4,
            "lr_schedule": "warmup_cosine",
            "min_lr": 1e-5,
            "lr_warmup_steps": 4_000,
            "lr_decay_steps": 200_000,
            "eval_interval": 5_000,
            "eval_batches": 4,
            "maze_distribution": "searchformer_10",
        },
    }
    for name, contract in contracts.items():
        values = TRACE_PRESETS[name].values
        for key, expected in {**common, **contract}.items():
            assert values[key] == expected, f"{name}.{key} changed"


def test_ablation_recommendation_accepts_noninferior_efficiency_win():
    control = {
        str(seed): {
            "drift.append_recurrent.token_legality": 0.80,
            "train.train_tok_per_s": 100.0,
            "model.non_embedding_parameters": 1000.0,
        }
        for seed in range(3)
    }
    treatment = {
        str(seed): {
            "drift.append_recurrent.token_legality": 0.795,
            "train.train_tok_per_s": 115.0,
            "model.non_embedding_parameters": 1000.0,
        }
        for seed in range(3)
    }
    result = recommend(
        control,
        treatment,
        mode="pareto",
        quality_metric="drift.append_recurrent.token_legality",
    )
    assert result["quality_noninferior"]
    assert result["efficiency_win"]
    assert result["recommend_merge"]


def test_ablation_recommendation_accepts_task_specific_quality_metric():
    control = {
        str(seed): {"drift.append_recurrent.optimal_path": 0.30}
        for seed in range(3)
    }
    treatment = {
        str(seed): {"drift.append_recurrent.optimal_path": value}
        for seed, value in enumerate((0.32, 0.31, 0.30))
    }
    result = recommend(
        control,
        treatment,
        mode="quality-only",
        quality_metric="drift.append_recurrent.optimal_path",
    )
    assert result["quality_metric"] == "drift.append_recurrent.optimal_path"
    assert result["quality_win"]
    assert result["recommend_merge"]


def test_ablation_quality_metric_is_inferred_from_task():
    shortest_path = {
        str(seed): {"task": "shortest_path"}
        for seed in range(3)
    }
    assert (
        infer_quality_metric(shortest_path)
        == "drift.append_recurrent.optimal_path"
    )
    assert (
        infer_quality_metric({"1337": {"task": "othello"}})
        == "drift.append_recurrent.sequence_legality"
    )
    with pytest.raises(ValueError, match="pass --quality-metric"):
        infer_quality_metric({"1337": {"task": "unknown"}})


def test_evaluation_checkpoint_selection_is_explicit(tmp_path):
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    best.touch()
    latest.touch()
    assert resolve_evaluation_checkpoint(tmp_path) == best
    assert resolve_evaluation_checkpoint(tmp_path, "latest") == latest
    with pytest.raises(ValueError, match="checkpoint must be one of"):
        resolve_evaluation_checkpoint(tmp_path, "other")
