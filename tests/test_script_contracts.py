import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_all_project_workflows_live_under_scripts():
    assert not (ROOT / "runs").exists()
    assert (ROOT / "scripts" / "bbh" / "run.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "run.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "eval.sh").is_file()
    assert (ROOT / "tests" / "test_shortest_path.sh").is_file()
    assert (ROOT / "tests" / "test_smoke.sh").is_file()
    assert not (ROOT / "scripts" / "local").exists()
    assert not (ROOT / "scripts" / "drift").exists()
    assert not (ROOT / "scripts" / "ablations").exists()


def test_launcher_architecture_registry_matches_model_factory():
    from model_factory import ARCHITECTURES

    env = os.environ.copy()
    env["MATRIX_LIB"] = str(ROOT / "scripts" / "lib" / "model_matrix.sh")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "${MATRIX_LIB}"; printf "%s\\n" "${MPT_ARCHITECTURES[@]}"',
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(result.stdout.splitlines()) == ARCHITECTURES


def test_canonical_training_launchers_do_not_accept_scientific_overrides():
    launchers = [
        ROOT / "scripts" / "bbh" / "run.sh",
        ROOT / "scripts" / "trace" / "run.sh",
    ]
    prohibited = (
        "TRAIN_STEPS",
        "EVAL_INTERVAL",
        "EVAL_BATCHES",
        "BATCH_SIZE",
        "MEMORY_GATE_INIT",
        "TOKEN_SELECTION",
    )
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "memory_add" in text, f"{launcher} omits memory_add from its default matrix"
        for variable in prohibited:
            assert variable not in text, f"{launcher} accepts scientific override {variable}"

    trace_text = launchers[1].read_text(encoding="utf-8")
    assert 'TASKS="${TASKS:-shortest_path}"' in trace_text
    assert (
        'ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape memory_add}"'
        in trace_text
    )


def test_trace_launcher_runs_task_matrix_into_task_first_results(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "DEVICE": "cpu",
            "TASKS": "shortest_path maze othello",
            "ARCHITECTURES": "memory_add",
            "SEEDS": "1337",
            "RESULT_ROOT": str(tmp_path / "results"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "trace" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    training_calls = [
        call for call in calls if "-m experiments.train_trace" in call
    ]
    assert len(training_calls) == 3
    assert "--preset shortest_path_main" in training_calls[0]
    assert "--preset maze_main" in training_calls[1]
    assert "--preset othello_main" in training_calls[2]
    assert all("--architecture memory_add" in call for call in training_calls)
    assert (
        f"--run-dir {tmp_path}/results/shortest_path/main/memory_add/seed_1337"
        in training_calls[0]
    )
    assert (
        f"--run-dir {tmp_path}/results/maze/main/memory_add/seed_1337"
        in training_calls[1]
    )
    assert (
        f"--run-dir {tmp_path}/results/othello/main/memory_add/seed_1337"
        in training_calls[2]
    )


def test_trace_eval_routes_to_task_specific_evaluator(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "evaluation"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "RUN_DIR": str(run_dir),
            "OUTPUT_DIR": str(output_dir),
            "DEVICE": "cpu",
        }
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {"args": {"task": "shortest_path", "architecture": "transformer"}}
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "trace" / "eval.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "-m experiments.eval_trace" in calls[0]
    assert "--checkpoint best" in calls[0]
    assert "--eval-batches 64" in calls[0]
    assert "--inference-mode recompute" in calls[0]
    assert "--inference-mode append_recurrent" not in calls[0]

    log_path.unlink()
    (run_dir / "config.json").write_text(
        json.dumps(
            {"args": {"task": "othello", "architecture": "memory_tape"}}
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "trace" / "eval.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "-m experiments.eval_othello_prefix" in calls[0]
    assert "--checkpoint best" in calls[0]
    assert "--examples 64" in calls[0]
    assert "--inference-modes recompute append_recurrent" in calls[0]


def _fake_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_args.txt"
    executable = bin_dir / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-c" ]]; then\n'
        '  exec "${REAL_PYTHON:?}" "$@"\n'
        "fi\n"
        f"printf '%s\\n' \"$*\" >> {log_path!s}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, log_path


def test_local_trace_pilot_scales_schedule_and_runs_final_checks(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "DEVICE": "cpu",
            "TRAIN_STEPS": "250",
            "RESULT_ROOT": str(tmp_path / "results"),
        }
    )
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail; "
                "source scripts/lib/local_pilot.sh; "
                "run_trace_pilot_variant control"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert "-m experiments.train_trace" in calls[0]
    assert "--train-steps 250" in calls[0]
    assert "--lr-warmup-steps 5" in calls[0]
    assert "--lr-decay-steps 250" in calls[0]
    assert sum("-m experiments.eval_trace" in call for call in calls) == 2
    assert "-m experiments.diagnose_memory" in calls[-1]


def test_bbh_launcher_passes_supported_architecture_names(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "TASKS": "permutation",
            "ARCHITECTURES": "memory_add memory_tape",
            "SEEDS": "1337",
            "RESULT_ROOT": str(tmp_path / "results"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "bbh" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--architecture memory_add" in calls[0]
    assert "--architecture memory_tape" in calls[1]


def test_bbh_launcher_rejects_entire_bad_matrix_before_starting(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "TASKS": "permutation",
            "ARCHITECTURES": "memory_add memory_tape#",
            "SEEDS": "1337",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "bbh" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid architecture in matrix: memory_tape#" in result.stderr
    assert not log_path.exists()
