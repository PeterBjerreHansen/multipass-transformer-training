from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_plotting_architecture_registry_matches_model_factory():
    pytest.importorskip("matplotlib")
    from figures.plotting_utils import ARCHITECTURE_COLORS
    from model_factory import ARCHITECTURES

    assert tuple(ARCHITECTURE_COLORS) == ARCHITECTURES
    for name in (
        "01_bbh_curricula.ipynb",
        "02_trace_learning.ipynb",
        "03_deployment_and_othello.ipynb",
    ):
        source = (ROOT / "figures" / name).read_text(encoding="utf-8")
        assert "ARCHITECTURES = list(ARCHITECTURE_COLORS)" in source


def test_readme_uses_local_figure_paths_and_no_fetch_helper_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = {
        "bbh_permutation_frontier.png",
        "inference_pattern_fig.png",
        "generation_fig.png",
        "multipass_training_fig.png",
        "mismatch_fig.png",
        "trace_plot_figs.png",
    }
    for filename in expected:
        assert f"figures/{filename}" in readme
        figure_path = ROOT / "figures" / filename
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0
    assert "drift_plots_othello.png" not in readme
    assert not (ROOT / "figures" / "drift_plots_othello.png").exists()


def test_plotting_notebooks_are_valid_output_free_python():
    expected = {
        "01_bbh_curricula.ipynb",
        "02_trace_learning.ipynb",
        "03_deployment_and_othello.ipynb",
        "04_ablation_diagnostics.ipynb",
    }
    paths = {path.name: path for path in (ROOT / "figures").glob("*.ipynb")}
    assert expected <= paths.keys()

    for name in expected:
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")

    bbh = json.loads(
        (ROOT / "figures" / "01_bbh_curricula.ipynb").read_text(
            encoding="utf-8"
        )
    )
    bbh_source = "\n".join(
        "".join(cell["source"])
        for cell in bbh["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'TASK = "permutation"' in bbh_source
    assert "Cost of mastering each level" not in bbh_source
    assert "steps_to_mastery" not in bbh_source
    assert 'ylabel="Number of swaps"' in bbh_source
    assert 'title="S₅ permutation tracking"' in bbh_source
    assert 'fig.suptitle(TASK.replace' not in bbh_source
    assert "def find_repo_root(start):" in bbh_source
    assert bbh_source.count("plt.show()") == 3

    trace = json.loads(
        (ROOT / "figures" / "02_trace_learning.ipynb").read_text(
            encoding="utf-8"
        )
    )
    trace_source = "\n".join(
        "".join(cell["source"])
        for cell in trace["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'RESULT_ROOT = REPO_ROOT / "results" / "trace"' in trace_source
    assert 'SHORTEST_PATH_DISTRIBUTION = "main"' in trace_source
    assert "shortest_path_distribution=SHORTEST_PATH_DISTRIBUTION" in trace_source
    assert 'f"path_step_{step}_accuracy"' in trace_source
    assert "Final checkpoint: accuracy by path step" in trace_source
    assert "row.get(\"level\") is None" in trace_source


def test_plotting_loaders_follow_current_artifact_schemas(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.plotting_utils import (
        load_ablation_rows,
        load_diagnostic_records,
        load_drift_records,
        load_othello_examples,
        load_training_records,
        plot_seed_and_median_curves,
        summarize_curriculum_levels,
    )

    run_dir = tmp_path / "control" / "seed_1337"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {
                    "task": "shortest_path",
                    "architecture": "memory_tape",
                    "preset": "shortest_path_main",
                    "seed": 1337,
                    "device": "cpu",
                    "max_passes": 4,
                    "shortest_path_distribution": "main",
                    "shortest_path_data_dir": "data/shortest_path/main",
                    "shortest_path_dataset_id": "a" * 64,
                },
                "model_stats": {
                    "total_parameters": 1200,
                    "non_embedding_parameters": 1000,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "event": "eval",
                "step": 10,
                "train_loss": 1.5,
                "pass_losses": [1.8, 1.6, 1.5, 1.4],
                "train_tok_per_s": 100.0,
                "metrics": {"loss": 1.4, "optimal_path": 0.5},
                "gradient_norms": {"memory_writer": {"mean": 0.2, "max": 0.3}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    drift_dir = run_dir / "drift" / "append_recurrent"
    drift_dir.mkdir(parents=True)
    (drift_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_run_dir": str(run_dir),
                "task": "shortest_path",
                "architecture": "memory_tape",
                "inference_mode": "append_recurrent",
                "effective_inference_mode": "append_recurrent",
                "evaluation_examples": 16,
                "metrics": {
                    "optimal_path": 0.5,
                    "path_step_1_accuracy": 0.75,
                    "eval_output_tok_per_s": 25.0,
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostics = {
        "input_run_dir": str(run_dir),
        "task": "shortest_path",
        "architecture": "memory_tape",
        "memory_interventions": {
            "loss_deltas": {"correct": 0.0, "zero_memory_bank": 0.2}
        },
        "pass_dynamics": {
            "trained_passes": [{"pass": 1, "loss": 1.8}],
            "extra_passes": [],
        },
        "teacher_forced_schedule_gap": {
            "positions": [
                {
                    "generated_position": 1,
                    "count": 16,
                    "nll_delta": 0.1,
                    "memory_rms_delta": 0.2,
                }
            ],
            "overall": {"count": 16, "nll_delta": 0.1},
        },
    }
    (run_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics),
        encoding="utf-8",
    )

    othello_dir = run_dir / "othello_eval"
    othello_dir.mkdir()
    (othello_dir / "summary.json").write_text(
        json.dumps({"task": "othello", "input_run_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (othello_dir / "per_example.jsonl").write_text(
        json.dumps(
            {
                "example_index": 0,
                "protocol": "random-prefix",
                "inference_mode": "append_recurrent",
                "prompt_moves": 12,
                "prompt_bucket": "1-15",
                "reference_suffix_moves": 40,
                "suffix_bucket": "31-45",
                "free_generation": {"legal_move_fraction": 0.8},
                "teacher_forced": {
                    "legal_probability_mass": 0.9,
                    "move_count": 40.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (tmp_path / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "seed", "drift.append_recurrent.optimal_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "control",
                "seed": "1337",
                "drift.append_recurrent.optimal_path": "0.5",
            }
        )

    training = load_training_records(tmp_path)
    assert training[0]["shortest_path_distribution"] == "main"
    assert training[0]["shortest_path_data_dir"] == "data/shortest_path/main"
    assert training[0]["shortest_path_dataset_id"] == "a" * 64
    assert training[0]["pass_4_loss"] == 1.4
    assert training[0]["gradient_memory_writer_mean"] == 0.2
    figure, axis = plt.subplots()
    plot_seed_and_median_curves(axis, training, metric="optimal_path")
    assert len(axis.lines) == 2
    assert axis.lines[-1].get_ydata().tolist() == [0.5]
    assert axis.lines[-1].get_marker() == "o"
    plt.close(figure)

    curriculum_records = [
        {
            "run_dir": "run-a",
            "task": "pointer_chasing",
            "architecture": "memory_tape",
            "seed": 1337,
            "device": "cpu",
            "max_level": 4,
            "curriculum_threshold": 0.95,
            "step": 100,
            "level": 1,
            "exact_match": 0.8,
        },
        {
            "run_dir": "run-a",
            "task": "pointer_chasing",
            "architecture": "memory_tape",
            "seed": 1337,
            "device": "cpu",
            "max_level": 4,
            "curriculum_threshold": 0.95,
            "step": 200,
            "level": 1,
            "exact_match": 0.96,
        },
        {
            "run_dir": "run-a",
            "task": "pointer_chasing",
            "architecture": "memory_tape",
            "seed": 1337,
            "device": "cpu",
            "max_level": 4,
            "curriculum_threshold": 0.95,
            "step": 300,
            "level": 2,
            "exact_match": 0.4,
        },
    ]
    curriculum = summarize_curriculum_levels(curriculum_records)
    assert curriculum[0]["steps_to_mastery"] == 200
    assert curriculum[0]["mastered"] is True
    assert curriculum[0]["censored"] is False
    assert curriculum[1]["steps_to_mastery"] is None
    assert curriculum[1]["mastered"] is False
    assert curriculum[1]["censored"] is True

    drift = load_drift_records(tmp_path)
    assert drift[0]["optimal_path"] == 0.5
    assert drift[0]["path_step_1_accuracy"] == 0.75

    loaded_diagnostics = load_diagnostic_records(tmp_path)
    assert loaded_diagnostics[0][
        "memory_interventions.loss_deltas.zero_memory_bank"
    ] == 0.2
    assert loaded_diagnostics[0]["payload"]["pass_dynamics"]["trained_passes"][0]["pass"] == 1

    othello = load_othello_examples(tmp_path)
    assert othello[0]["free_generation.legal_move_fraction"] == 0.8
    assert othello[0]["teacher_forced.legal_probability_mass"] == 0.9

    ablation = load_ablation_rows(tmp_path)
    assert ablation[0]["drift.append_recurrent.optimal_path"] == 0.5


def test_training_loader_preserves_every_logged_evaluation(tmp_path):
    from figures.plotting_utils import load_training_records

    run_dir = tmp_path / "pointer_chasing" / "transformer" / "seed_1337"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {
                    "task": "pointer_chasing",
                    "architecture": "transformer",
                    "seed": 1337,
                    "max_level": 32,
                    "curriculum_threshold": 0.95,
                }
            }
        ),
        encoding="utf-8",
    )
    events = [
        {"event": "run_start"},
        {
            "event": "eval",
            "step": 100,
            "level": 7,
            "metrics": {"exact_match": 0.99},
        },
        {"event": "run_start"},
        {
            "event": "eval",
            "step": 100,
            "level": 2,
            "metrics": {"exact_match": 0.60},
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    records = load_training_records(tmp_path)
    assert [record["level"] for record in records] == [7, 2]
    assert [record["exact_match"] for record in records] == [0.99, 0.60]
