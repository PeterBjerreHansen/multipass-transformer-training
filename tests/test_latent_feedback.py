from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.pass_schedule import (
    ProbabilisticPassScheduler,
    build_pass_scheduler,
    parse_pass_schedule,
)
from experiments.train_bbh import parse_args as parse_bbh_args
from experiments.train_trace import parse_args as parse_trace_args
from models import LatentFeedbackTransformer, MultiPassConfig


def tiny_model(*, max_passes: int = 3) -> LatentFeedbackTransformer:
    torch.manual_seed(11)
    return LatentFeedbackTransformer(
        MultiPassConfig(
            block_size=10,
            vocab_size=17,
            n_layer=2,
            n_head=2,
            n_embd=8,
            max_passes=max_passes,
        )
    )


def test_glu_feedback_forward_matches_paper_equation():
    model = tiny_model()
    tokens = torch.randint(0, 17, (2, 6))
    embeddings = model.embed_tokens(tokens)
    read_memory = torch.randn_like(embeddings)
    read_memory[:, 0] = 0

    expected = model.feedback_value(read_memory) * torch.sigmoid(
        model.feedback_gate(model.feedback_input_ln(embeddings))
    )
    expected = model.feedback_input_ln(expected)
    expected[:, 0] = embeddings[:, 0]
    for block in model.transformer.h:
        expected = block(expected)
    expected = model.transformer.ln_f(expected)

    actual = model.forward_pass(embeddings, read_memory)
    assert torch.equal(actual.hidden_states, expected)
    assert torch.equal(actual.memory_states, expected)


def test_first_pass_is_plain_transformer_and_memory_is_top_layer_state():
    model = tiny_model()
    tokens = torch.randint(0, 17, (2, 6))
    token_stream = model.embed_tokens(tokens)
    expected_hidden = token_stream
    for block in model.transformer.h:
        expected_hidden = block(expected_hidden)
    expected_hidden = model.transformer.ln_f(expected_hidden)

    output = model(tokens, passes=1)

    assert len(output.passes) == 1
    assert torch.equal(output.hidden_states, expected_hidden)
    assert torch.equal(output.final_memory, output.hidden_states)
    assert not hasattr(model, "mem_head")
    assert not hasattr(model, "ln_mem")


def test_forward_uses_requested_fixed_pass_count_without_mutating_config():
    model = tiny_model(max_passes=3)
    tokens = torch.randint(0, 17, (2, 6))

    assert len(model(tokens).passes) == 3
    assert len(model(tokens, passes=1).passes) == 1
    assert len(model(tokens, passes=2).passes) == 2
    assert model.config.max_passes == 3
    with pytest.raises(ValueError, match="between 1 and 3"):
        model(tokens, passes=0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        model(tokens, passes=4)


def test_latent_feedback_uses_generic_relative_pass_weights():
    model = tiny_model(max_passes=3)
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))

    output = model(tokens)
    result = model.calc_total_loss(output, targets, [1, 1])
    expected = torch.stack(result.pass_losses[-2:]).mean()

    assert torch.equal(result.loss, expected)
    one_pass = model.calc_total_loss(model(tokens, passes=1), targets, [1, 1])
    assert torch.equal(one_pass.loss, one_pass.pass_losses[0])


def test_feedback_loss_backpropagates_through_earlier_pass_state():
    model = tiny_model(max_passes=2)
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))
    output = model(tokens)

    second_pass_only = model.calc_loss(output.passes[1].logits, targets)
    second_pass_only.backward()

    assert model.feedback_value.weight.grad is not None
    assert model.feedback_value.weight.grad.abs().sum() > 0
    assert model.feedback_gate.weight.grad is not None
    assert model.feedback_gate.weight.grad.abs().sum() > 0
    first_block = model.transformer.h[0].attn.c_attn.weight
    assert first_block.grad is not None
    assert first_block.grad.abs().sum() > 0


def test_latent_feedback_is_causal_at_every_pass():
    model = tiny_model(max_passes=3)
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    first = torch.cat((prefix, torch.tensor([[5, 6]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[7, 8]])), dim=1)

    output_a = model(first)
    output_b = model(second)

    for pass_a, pass_b in zip(output_a.passes, output_b.passes):
        assert torch.allclose(pass_a.logits[:, :4], pass_b.logits[:, :4], atol=1e-6, rtol=0)
        assert torch.allclose(
            pass_a.hidden_states[:, :4],
            pass_b.hidden_states[:, :4],
            atol=1e-6,
            rtol=0,
        )


def test_scheduler_phases_are_normalized_and_selected_by_step():
    phases = parse_pass_schedule(["1=1:3,2:1", "10=1:3,2:0.88,3:0.12"])

    assert phases[0].start_step == 1
    assert phases[0].probabilities == ((1, 0.75), (2, 0.25))
    assert sum(weight for _, weight in phases[1].probabilities) == pytest.approx(1.0)
    scheduler = ProbabilisticPassScheduler(
        ["1=1:1", "10=3:1"],
        seed=7,
    )
    assert scheduler.sample(9) == 1
    assert scheduler.sample(10) == 3


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (["2=1:1"], "start at step 1"),
        (["1=0:1"], "pass counts must be positive"),
        (["1=1:0"], "weights must be finite and positive"),
        (["1=1:1", "1=2:1"], "strictly increasing"),
        (["1=1:1,1:1"], "duplicate pass count"),
    ],
)
def test_scheduler_rejects_invalid_schedules(schedule, message):
    with pytest.raises(ValueError, match=message):
        parse_pass_schedule(schedule)


def test_scheduler_state_restores_the_exact_sampling_sequence():
    schedule = ["1=1:0.75,2:0.22,3:0.03"]
    first = ProbabilisticPassScheduler(schedule, seed=123)
    for step in range(1, 20):
        first.sample(step)
    state = first.state_dict()

    restored = ProbabilisticPassScheduler(schedule, seed=999)
    restored.load_state_dict(state)

    expected = [first.sample(step) for step in range(20, 50)]
    actual = [restored.sample(step) for step in range(20, 50)]
    assert actual == expected
    assert restored.stats() == first.stats()


def test_scheduler_supports_multipass_models_and_respects_configured_maximum():
    args = SimpleNamespace(
        architecture="latent_feedback",
        max_passes=3,
        train_pass_schedule=["1=1:1,3:1"],
    )
    assert build_pass_scheduler(args, seed=1) is not None

    args.architecture = "memory_tape"
    assert build_pass_scheduler(args, seed=1) is not None
    args.architecture = "transformer"
    with pytest.raises(ValueError, match="requires a multi-pass architecture"):
        build_pass_scheduler(args, seed=1)
    args.architecture = "latent_feedback"
    args.max_passes = 2
    with pytest.raises(ValueError, match="cannot exceed --max-passes"):
        build_pass_scheduler(args, seed=1)


def test_both_training_clis_parse_pass_schedules():
    options = [
        "--architecture",
        "latent_feedback",
        "--train-pass-schedule",
        "1=1:1",
        "100=1:.75,2:.25",
    ]
    assert parse_bbh_args(["--preset", "permutation_smoke", *options]).train_pass_schedule == options[-2:]
    assert parse_trace_args(["--preset", "maze_smoke", *options]).train_pass_schedule == options[-2:]


def test_bbh_smoke_checkpoint_contains_scheduler_state(tmp_path: Path):
    run_dir = tmp_path / "latent_feedback"
    command = [
        sys.executable,
        "-m",
        "experiments.train_bbh",
        "--preset",
        "permutation_smoke",
        "--architecture",
        "latent_feedback",
        "--max-passes",
        "3",
        "--train-pass-schedule",
        "1=2:1",
        "--device",
        "cpu",
        "--run-dir",
        str(run_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "sampled_passes 2" in result.stdout
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    schedule_state = checkpoint["extra_state"]["pass_scheduler_state"]
    assert schedule_state["sample_count"] == 1
    assert schedule_state["histogram"] == {2: 1}
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert evaluation["pass_schedule"]["histogram"] == {"2": 1}
