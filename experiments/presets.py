from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.trace import maze, othello, shortest_path


@dataclass(frozen=True)
class ExperimentPreset:
    description: str
    values: dict[str, object]


def _base_defaults(
    *,
    task: str,
    smoke: bool,
    inference_mode: str,
    token_selection: str,
) -> dict[str, object]:
    n_pass = 3 if smoke else 4
    return {
        "task": task,
        "architecture": "transformer",
        "model_size": "tiny" if smoke else "small",
        # Smoke overrides stay deliberately small and uniform. Main presets use
        # the model-size preset so that width/depth remain centralized.
        "n_layer": 1 if smoke else None,
        "n_head": 1 if smoke else None,
        "n_embd": 16 if smoke else None,
        "n_pass": n_pass,
        "pass_loss_weights": [0.0] * (n_pass - 1) + [1.0]
        if smoke
        else [0.0, 0.0, 1.0, 1.0],
        "inference_mode": inference_mode,
        "token_selection": token_selection,
        "batch_size": 1 if smoke else 64,
        "train_steps": 1 if smoke else 50_000,
        "max_grad_norm": 5.0,
        "eval_batches": 1 if smoke else 4,
        "weight_decay": 0.0,
        "seed": 1337,
        "run_dir": None,
        "resume_from": None,
        "device": None,
        "block_size": None,
    }


def _bbh_defaults(*, task: str, smoke: bool) -> dict[str, object]:
    values = _base_defaults(
        task=task,
        smoke=smoke,
        inference_mode="recompute",
        token_selection="argmax",
    )
    values.update(
        # Keep the established curriculum settings for the main comparison.
        lr=1e-4,
        eval_interval=1 if smoke else 200,
        max_level=2 if smoke else 64,
        curriculum_threshold=0.95,
        review_easier_every=2,
    )
    return values


def _trace_defaults(
    *,
    task: str,
    smoke: bool,
    token_selection: str,
) -> dict[str, object]:
    values = _base_defaults(
        task=task,
        smoke=smoke,
        inference_mode="append_recurrent",
        token_selection=token_selection,
    )
    values.update(
        lr=1e-4,
        lr_schedule="constant",
        min_lr=1e-4,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        eval_interval=1 if smoke else 5_000,
    )
    return values


TRACE_PRESETS: dict[str, ExperimentPreset] = {}

othello_main = _trace_defaults(task="othello", smoke=False, token_selection="sample")
othello_main.update(
    batch_size=128,
    train_steps=500_000,
    eval_interval=5_000,
    eval_batches=1,
    othello_data_dir=othello.DEFAULT_DATA_DIR,
    # Keep the principal Othello experiment sizes explicit rather than
    # inheriting potentially changing dataset defaults.
    othello_train_games=5_000_000,
    othello_val_games=1_024,
    othello_dataset_seed=othello.DEFAULT_DATASET_SEED,
)
TRACE_PRESETS["othello_main"] = ExperimentPreset("Main Othello trace setup.", othello_main)

othello_smoke = _trace_defaults(task="othello", smoke=True, token_selection="argmax")
othello_smoke.update(
    othello_data_dir="data/othello_smoke",
    othello_train_games=16,
    othello_val_games=8,
    othello_dataset_seed=9,
)
TRACE_PRESETS["othello_smoke"] = ExperimentPreset(
    "Tiny deterministic Othello smoke setup.",
    othello_smoke,
)

shortest_path_main = _trace_defaults(
    task="shortest_path",
    smoke=False,
    token_selection="argmax",
)
shortest_path_main.update(
    shortest_path_data_dir=shortest_path.DEFAULT_DATA_DIR,
    train_steps=200_000,
    lr=5e-4,
    lr_schedule="warmup_cosine",
    min_lr=1e-5,
    lr_warmup_steps=4_000,
    lr_decay_steps=200_000,
)
TRACE_PRESETS["shortest_path_main"] = ExperimentPreset(
    "Main mixed-difficulty, solver-verified shortest-path dataset.",
    shortest_path_main,
)

shortest_path_easy = _trace_defaults(
    task="shortest_path",
    smoke=False,
    token_selection="argmax",
)
shortest_path_easy.update(
    shortest_path_data_dir=shortest_path.DEFAULT_EASY_DATA_DIR,
)
TRACE_PRESETS["shortest_path_easy"] = ExperimentPreset(
    "Full training setup for the easy shortest-path dataset.",
    shortest_path_easy,
)

shortest_path_smoke = _trace_defaults(
    task="shortest_path",
    smoke=True,
    token_selection="argmax",
)
shortest_path_smoke.update(
    shortest_path_data_dir=shortest_path.DEFAULT_SMOKE_DATA_DIR,
)
TRACE_PRESETS["shortest_path_smoke"] = ExperimentPreset(
    "One-step software check using the compiled shortest-path fixture.",
    shortest_path_smoke,
)

maze_main = _trace_defaults(
    task="maze",
    smoke=False,
    token_selection="argmax",
)
maze_main.update(
    maze_data_dir=maze.DEFAULT_DATA_DIR,
    maze_input_representation="sparse-cells",
    maze_target_representation="cell-path",
    maze_route_policy="astar",
    train_steps=200_000,
    lr=5e-4,
    lr_schedule="warmup_cosine",
    min_lr=1e-5,
    lr_warmup_steps=4_000,
    lr_decay_steps=200_000,
)
TRACE_PRESETS["maze_main"] = ExperimentPreset(
    "Searchformer-style 10x10 random-wall maze paths.",
    maze_main,
)

maze_smoke = _trace_defaults(
    task="maze",
    smoke=True,
    token_selection="argmax",
)
maze_smoke.update(
    maze_data_dir=maze.DEFAULT_SMOKE_DATA_DIR,
    maze_input_representation="sparse-cells",
    maze_target_representation="cell-path",
    maze_route_policy="astar",
)
TRACE_PRESETS["maze_smoke"] = ExperimentPreset(
    "One-step software check using small random-wall mazes.",
    maze_smoke,
)


BBH_PRESETS: dict[str, ExperimentPreset] = {}


def _add_bbh_pair(
    task: str,
    main_values: dict[str, object],
    smoke_values: dict[str, object],
) -> None:
    main_values = dict(main_values)
    smoke_values = dict(smoke_values)

    main = _bbh_defaults(task=task, smoke=False)
    main.update(
        curriculum_start_level=main_values.pop("curriculum_start_level"),
        **main_values,
    )
    smoke = _bbh_defaults(task=task, smoke=True)
    smoke.update(
        curriculum_start_level=smoke_values.pop("curriculum_start_level"),
        **smoke_values,
    )
    BBH_PRESETS[f"{task}_main"] = ExperimentPreset(
        f"Main {task} curriculum setup.",
        main,
    )
    BBH_PRESETS[f"{task}_smoke"] = ExperimentPreset(
        f"Tiny {task} smoke setup.",
        smoke,
    )


_add_bbh_pair(
    "pointer_chasing",
    {
        "num_nodes": pointer_chasing.DEFAULT_NUM_NODES,
        "curriculum_start_level": 1,
        "max_level": pointer_chasing.DEFAULT_MAX_LEVEL,
    },
    {"num_nodes": 5, "curriculum_start_level": 1},
)
_add_bbh_pair(
    "tracking",
    {"num_objects": tracking.DEFAULT_NUM_OBJECTS, "curriculum_start_level": 1},
    {"num_objects": 5, "curriculum_start_level": 1},
)
_add_bbh_pair(
    "permutation",
    {"num_objects": permutation.DEFAULT_NUM_OBJECTS, "curriculum_start_level": 1},
    {"num_objects": permutation.DEFAULT_NUM_OBJECTS, "curriculum_start_level": 1},
)
_add_bbh_pair(
    "state_machine",
    {
        "num_states": state_machine.DEFAULT_NUM_STATES,
        "alphabet_size": state_machine.DEFAULT_ALPHABET_SIZE,
        "curriculum_start_level": 0,
    },
    {"num_states": 4, "alphabet_size": 2, "curriculum_start_level": 0},
)


def resolve_preset_args(
    raw_args: argparse.Namespace,
    presets: dict[str, ExperimentPreset],
    *,
    default_preset: str,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    overrides = vars(raw_args).copy()
    preset_name = str(overrides.pop("preset", default_preset))
    if preset_name not in presets:
        parser.error(f"unknown preset: {preset_name}")
    values = deepcopy(presets[preset_name].values)
    values.update(overrides)
    values["preset"] = preset_name
    return argparse.Namespace(**values)


def preset_help_text(presets: dict[str, ExperimentPreset]) -> str:
    return " ".join(
        f"{name}: {preset.description}"
        for name, preset in sorted(presets.items())
    )
