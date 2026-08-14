from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)})
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_bbh_training_cli_writes_restorable_checkpoint(tmp_path):
    run_dir = tmp_path / "bbh"
    result = _run(
        "-m", "experiments.train_bbh",
        "--preset", "pointer_chasing_smoke",
        "--architecture", "memory_tape",
        "--curriculum-threshold", "0",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert "architecture: memory_tape" in result.stdout
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["extra_state"]["current_level"] == 2
    assert checkpoint["extra_state"]["evaluated_level"] == 1
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert "gradient_norms" in evaluation
    assert evaluation["gradient_norms"]["global"]["max"] > 0

    diagnostics = tmp_path / "bbh_diagnostics.json"
    _run(
        "-m", "experiments.diagnose_memory",
        "--input-run-dir", str(run_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--eval-batches", "1",
        "--extra-passes", "0",
        "--schedule-gap-horizon", "1",
        "--output", str(diagnostics),
    )
    assert json.loads(diagnostics.read_text(encoding="utf-8"))["evaluated_level"] == 1


def test_memory_add_bbh_cli_reports_fusion_gradients(tmp_path):
    architecture = "memory_add"
    run_dir = tmp_path / "memory_add_bbh"
    result = _run(
        "-m", "experiments.train_bbh",
        "--preset", "pointer_chasing_smoke",
        "--architecture", architecture,
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert f"architecture: {architecture}" in result.stdout
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert evaluation["gradient_norms"]["memory_fusion"]["max"] > 0
    assert (run_dir / "latest.pt").exists()


def test_trace_training_evaluation_and_diagnostics_cli(tmp_path):
    run_dir = tmp_path / "trace"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--architecture", "memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )

    diagnostics = tmp_path / "diagnostics.json"
    _run(
        "-m", "experiments.diagnose_memory",
        "--input-run-dir", str(run_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--eval-batches", "1",
        "--extra-passes", "2",
        "--output", str(diagnostics),
    )
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "best"
    assert payload["checkpoint_path"] == str(run_dir / "best.pt")
    assert "memory_interventions" in payload
    assert payload["memory_interventions"]["source_memory"]["effective_rank"] >= 0
    assert len(payload["pass_dynamics"]["extra_passes"]) == 2
    assert set(payload["pass_dynamics"]["trained_passes"][0]["relative_linf_residual"]) == {
        "mean",
        "max",
    }
    assert "logit_kl_from_previous" in payload["pass_dynamics"]["trained_passes"][1]
    assert payload["teacher_forced_schedule_gap"]["horizon"] == 16
    assert payload["teacher_forced_schedule_gap"]["overall"]["count"] > 0

    trace_events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    trace_evaluation = next(event for event in trace_events if event["event"] == "eval")
    assert trace_evaluation["gradient_norms"]["global"]["mean"] > 0
    assert trace_evaluation["dataset_split"] == "validation"

    eval_dir = tmp_path / "eval"
    _run(
        "-m", "experiments.eval_trace",
        "--input-run-dir", str(run_dir),
        "--inference-mode", "append_recurrent",
        "--token-selection", "argmax",
        "--device", "cpu",
        "--eval-batches", "1",
        "--output-dir", str(eval_dir),
    )
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"] == "best"
    assert summary["checkpoint_path"] == str(run_dir / "best.pt")
    assert summary["effective_inference_mode"] == "append_recurrent"
    assert summary["eval_batches"] == 1
    assert summary["evaluation_examples"] == 1
    assert summary["dataset_split"] == "test"
    assert len(summary["shortest_path_dataset_id"]) == 64
    assert "path_step_1_accuracy" in summary["metrics"]


def test_othello_random_prefix_evaluation_cli(tmp_path):
    run_dir = tmp_path / "othello"
    data_dir = tmp_path / "othello_data"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "othello_smoke",
        "--architecture", "memory_tape",
        "--othello-data-dir", str(data_dir),
        "--othello-train-games", "8",
        "--othello-val-games", "4",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )

    output_dir = tmp_path / "othello_eval"
    _run(
        "-m", "experiments.eval_othello_prefix",
        "--input-run-dir", str(run_dir),
        "--output-dir", str(output_dir),
        "--evaluation-mode", "random-prefix",
        "--inference-modes", "recompute", "append_recurrent",
        "--token-selection", "argmax",
        "--examples", "1",
        "--device", "cpu",
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"] == "best"
    assert summary["checkpoint_path"] == str(run_dir / "best.pt")
    assert summary["evaluated_inference_modes"] == ["recompute", "append_recurrent"]
    for mode in summary["evaluated_inference_modes"]:
        overall = summary["modes"][mode]["overall"]
        assert overall["count"] == 1
        assert overall["teacher_move_count"] > 0
        assert 0.0 <= overall["teacher_forced"]["legal_probability_mass"] <= 1.0
    rows = [
        json.loads(line)
        for line in (output_dir / "per_example.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2


def test_maze_training_and_solver_based_evaluation_cli(tmp_path):
    run_dir = tmp_path / "maze"
    result = _run(
        "-m", "experiments.train_trace",
        "--preset", "maze_smoke",
        "--architecture", "memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert "task: maze" in result.stdout
    assert (run_dir / "best.pt").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert evaluation["dataset_split"] == "validation"
    for metric in ("optimal_route", "exact_target_route"):
        assert metric in evaluation["metrics"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["training_evaluation_split"] == "validation"
    assert len(config["args"]["maze_dataset_id"]) == 64

    eval_dir = tmp_path / "maze_eval"
    _run(
        "-m", "experiments.eval_trace",
        "--input-run-dir", str(run_dir),
        "--inference-mode", "append_recurrent",
        "--token-selection", "argmax",
        "--device", "cpu",
        "--eval-batches", "1",
        "--maze-data-dir", config["args"]["maze_data_dir"],
        "--output-dir", str(eval_dir),
    )
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "maze"
    assert summary["dataset_split"] == "test"
    assert summary["maze_dataset_id"] == config["args"]["maze_dataset_id"]
    assert summary["effective_inference_mode"] == "append_recurrent"
    assert "optimal_route" in summary["metrics"]
    assert "exact_target_route" in summary["metrics"]


def test_shortest_path_training_resume_evaluation_and_diagnostics_cli(
    tmp_path,
):
    run_dir = tmp_path / "shortest_path"
    result = _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--architecture", "memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert "task: shortest_path" in result.stdout
    assert (run_dir / "best.pt").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert evaluation["learning_rate"] == pytest.approx(1e-4)
    assert evaluation["dataset_split"] == "validation"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["training_evaluation_split"] == "validation"
    assert config["args"]["shortest_path_distribution"] == "easy"
    assert len(config["args"]["shortest_path_dataset_id"]) == 64
    for metric in (
        "optimal_path",
        "optimal_path_short",
        "examples_short",
        "path_step_1_accuracy",
        "path_step_1_examples",
    ):
        assert metric in evaluation["metrics"]

    _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--resume-from", str(run_dir),
        "--train-steps", "1",
        "--lr", "0.00005",
        "--max-grad-norm", "1000000",
        "--shortest-path-data-dir", config["args"]["shortest_path_data_dir"],
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    resumed_events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resumed_evaluation = next(
        event
        for event in resumed_events
        if event.get("event") == "eval" and event.get("step") == 2
    )
    resume_event = next(
        event for event in resumed_events if event.get("event") == "run_resume"
    )
    assert resume_event["provenance"]["git"]["commit"]
    assert "train_trace.py" in resume_event["provenance"]["command"]
    assert resumed_evaluation["learning_rate"] == pytest.approx(0.00005)
    resumed_checkpoint = torch.load(
        run_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_checkpoint["args"]["lr"] == pytest.approx(0.00005)
    assert resumed_checkpoint["args"]["lr_schedule"] == "constant"
    assert resumed_checkpoint["args"]["max_grad_norm"] == pytest.approx(1_000_000.0)
    assert all(
        group["lr"] == pytest.approx(0.00005)
        for group in resumed_checkpoint["optimizer_state_dict"]["param_groups"]
    )

    for inference_mode in ("recompute", "append_recurrent"):
        eval_dir = tmp_path / f"eval_{inference_mode}"
        _run(
            "-m", "experiments.eval_trace",
            "--input-run-dir", str(run_dir),
            "--checkpoint", "latest",
            "--inference-mode", inference_mode,
            "--token-selection", "argmax",
            "--device", "cpu",
            "--eval-batches", "1",
            "--shortest-path-data-dir", config["args"]["shortest_path_data_dir"],
            "--output-dir", str(eval_dir),
        )
        summary = json.loads(
            (eval_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["task"] == "shortest_path"
        assert summary["checkpoint"] == "latest"
        assert summary["checkpoint_step"] == 2
        assert summary["dataset_split"] == "test"
        assert (
            summary["shortest_path_dataset_id"]
            == config["args"]["shortest_path_dataset_id"]
        )
        assert "optimal_path" in summary["metrics"]

    diagnostics = tmp_path / "shortest_path_diagnostics.json"
    _run(
        "-m", "experiments.diagnose_memory",
        "--input-run-dir", str(run_dir),
        "--checkpoint", "latest",
        "--device", "cpu",
        "--batch-size", "2",
        "--eval-batches", "1",
        "--shortest-path-data-dir", config["args"]["shortest_path_data_dir"],
        "--extra-passes", "1",
        "--schedule-gap-horizon", "4",
        "--output", str(diagnostics),
    )
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert payload["task"] == "shortest_path"
    assert payload["checkpoint"] == "latest"
    assert payload["checkpoint_step"] == 2
    assert payload["dataset_split"] == "validation"
    assert (
        payload["shortest_path_dataset_id"]
        == config["args"]["shortest_path_dataset_id"]
    )
    assert payload["teacher_forced_schedule_gap"]["overall"]["count"] > 0


def test_trace_cosine_schedule_resume_matches_uninterrupted_training(tmp_path):
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    schedule_args = (
        "--preset", "shortest_path_smoke",
        "--architecture", "transformer",
        "--device", "cpu",
        "--eval-interval", "1",
        "--lr", "0.0003",
        "--lr-schedule", "warmup_cosine",
        "--min-lr", "0.00003",
        "--lr-warmup-steps", "2",
        "--lr-decay-steps", "6",
    )
    _run(
        "-m", "experiments.train_trace",
        *schedule_args,
        "--train-steps", "4",
        "--run-dir", str(uninterrupted_dir),
    )
    _run(
        "-m", "experiments.train_trace",
        *schedule_args,
        "--train-steps", "2",
        "--run-dir", str(resumed_dir),
    )
    _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--resume-from", str(resumed_dir),
        "--train-steps", "2",
        "--device", "cpu",
        "--run-dir", str(resumed_dir),
    )

    uninterrupted = torch.load(
        uninterrupted_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        resumed_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert uninterrupted["step"] == resumed["step"] == 4
    for name, value in uninterrupted["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name]), name
    assert resumed["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(
        1.65e-4
    )
    events = [
        json.loads(line)
        for line in (resumed_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step_four = next(
        event
        for event in events
        if event.get("event") == "eval" and event.get("step") == 4
    )
    assert step_four["learning_rate"] == pytest.approx(1.65e-4)
