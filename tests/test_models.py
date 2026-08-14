from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from model_factory import build_model
from models import (
    CausalCrossAttention,
    CausalTransformer,
    LatentFeedbackTransformer,
    LayerNorm,
    MemoryAddTransformer,
    MemoryBlock,
    MultiPassConfig,
    MemoryTapeTransformer,
    TransformerConfig,
    normalize_pass_weights,
    sample_next_token,
    shift_right,
)


def tiny_memory_model(*, block_size: int = 12, max_passes: int = 3) -> MemoryTapeTransformer:
    torch.manual_seed(7)
    return MemoryTapeTransformer(
        MultiPassConfig(
            block_size=block_size,
            vocab_size=19,
            n_layer=2,
            n_head=2,
            n_embd=8,
            max_passes=max_passes,
        )
    )


def test_shift_right_is_exact():
    memory = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    shifted = shift_right(memory)
    assert torch.equal(shifted[:, 0], torch.zeros_like(shifted[:, 0]))
    assert torch.equal(shifted[:, 1:], memory[:, :-1])


def test_pass_weights_are_always_normalized():
    a = normalize_pass_weights([0, 0, 1, 1], 4, device=torch.device("cpu"), dtype=torch.float32)
    b = normalize_pass_weights([0, 0, 0.5, 0.5], 4, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(a, b)
    assert a.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_pass_weights([0, 0, 0], 3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        normalize_pass_weights([1, -1], 2, device=torch.device("cpu"), dtype=torch.float32)


def test_relative_pass_weights_follow_the_active_depth():
    device = torch.device("cpu")
    four = normalize_pass_weights([1, 1], 4, device=device, dtype=torch.float32)
    three = normalize_pass_weights([1, 1], 3, device=device, dtype=torch.float32)
    two = normalize_pass_weights([1, 1], 2, device=device, dtype=torch.float32)
    one = normalize_pass_weights([1, 1], 1, device=device, dtype=torch.float32)

    assert torch.equal(four, torch.tensor([0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(three, torch.tensor([0.0, 0.5, 0.5]))
    assert torch.equal(two, torch.tensor([0.5, 0.5]))
    assert torch.equal(one, torch.tensor([1.0]))


def test_equivalent_relative_pass_weights_give_identical_loss():
    model = tiny_memory_model(max_passes=4)
    tokens = torch.randint(0, 19, (2, 7))
    targets = torch.randint(0, 19, (2, 7))
    output = model(tokens)
    loss_a = model.calc_total_loss(output, targets, [0, 0, 1, 1]).loss
    loss_b = model.calc_total_loss(output, targets, [0, 0, 0.5, 0.5]).loss
    assert torch.equal(loss_a, loss_b)


def test_zero_memory_produces_exact_zero_cross_attention_output():
    config = MultiPassConfig(8, 17, 1, 2, 8, 2)
    attention = CausalCrossAttention(config)
    query = torch.randn(2, 6, 8)
    output = attention(query, torch.zeros_like(query))
    assert torch.equal(output, torch.zeros_like(output))


def test_cross_attention_manual_and_sdpa_paths_agree():
    config = MultiPassConfig(8, 17, 1, 2, 8, 2)
    attention = CausalCrossAttention(config)
    if not attention.flash:
        pytest.skip("scaled_dot_product_attention is unavailable")
    query = torch.randn(2, 6, 8)
    memory = torch.randn(2, 6, 8)
    expected = attention(query, memory)
    attention.flash = False
    actual = attention(query, memory)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_memory_block_has_no_first_pass_intercept():
    config = MultiPassConfig(8, 17, 1, 2, 8, 2)
    block = MemoryBlock(config)
    block.eval()
    hidden = torch.randn(2, 6, 8)
    after_self = hidden + block.attn(block.ln_self(hidden))
    expected = after_self + block.mlp(block.ln_mlp(after_self))
    actual = block(hidden, torch.zeros_like(hidden))
    assert torch.equal(actual, expected)


def test_memory_tape_uses_standard_memory_normalization_and_writer():
    model = tiny_memory_model()
    assert isinstance(model.transformer.h[0].ln_mem_kv, LayerNorm)
    tokens = torch.randint(0, 19, (2, 6))
    output = model.forward_pass(model.embed_tokens(tokens), None)
    assert torch.equal(output.memory_states, model.mem_head(model.ln_mem(output.hidden_states)))


def test_causal_transformer_structured_output_and_generation():
    model = CausalTransformer(TransformerConfig(8, 17, 1, 1, 8))
    tokens = torch.randint(0, 17, (2, 6))
    output = model(tokens)
    assert len(output.passes) == 1
    assert output.logits.shape == (2, 6, 17)
    generated = model.generate(tokens[:, :4], 2, do_sample=False)
    assert generated.shape == (2, 6)
    with pytest.raises(ValueError):
        model.generate(tokens[:, :4], 1, inference_mode="append_recurrent")


@pytest.mark.parametrize(
    ("model_class", "config"),
    [
        (MemoryTapeTransformer, MultiPassConfig(8, 17, 1, 1, 8, 3)),
        (MemoryAddTransformer, MultiPassConfig(8, 17, 1, 1, 8, 3)),
        (LatentFeedbackTransformer, MultiPassConfig(8, 17, 1, 1, 8, 3)),
    ],
)
def test_multipass_models_return_all_passes_and_finite_losses(model_class, config):
    model = model_class(config)
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))
    output = model(tokens)
    assert len(output.passes) == 3
    assert all(item.memory_states is not None for item in output.passes)
    assert torch.isfinite(model.calc_total_loss(output, targets, [1]).loss)


def test_memory_tape_is_causal_in_tokens_and_emitted_memory():
    model = tiny_memory_model()
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    a = torch.cat((prefix, torch.tensor([[5, 6, 7, 8]])), dim=1)
    b = torch.cat((prefix, torch.tensor([[9, 10, 11, 12]])), dim=1)
    out_a = model(a)
    out_b = model(b)
    for pass_a, pass_b in zip(out_a.passes, out_b.passes):
        assert torch.allclose(pass_a.logits[:, :4], pass_b.logits[:, :4], atol=1e-6, rtol=0)
        assert pass_a.memory_states is not None and pass_b.memory_states is not None
        assert torch.allclose(pass_a.memory_states[:, :4], pass_b.memory_states[:, :4], atol=1e-6, rtol=0)


def test_aligned_memory_at_t_cannot_affect_earlier_positions():
    model = tiny_memory_model(max_passes=2)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    token_stream = model.embed_tokens(tokens)
    base = torch.zeros_like(token_stream)
    changed = base.clone()
    changed[:, 2, :] = torch.randn_like(changed[:, 2, :])
    out_base = model.forward_pass(token_stream, base)
    out_changed = model.forward_pass(token_stream, changed)
    assert torch.allclose(out_base.logits[:, :2], out_changed.logits[:, :2], atol=1e-6, rtol=0)
    assert not torch.allclose(out_base.logits[:, 2:], out_changed.logits[:, 2:])


def test_final_pass_loss_reaches_memory_writer_and_reader():
    model = tiny_memory_model(max_passes=3)
    tokens = torch.randint(0, 19, (2, 8))
    targets = torch.randint(0, 19, (2, 8))
    loss = model.calc_total_loss(model(tokens), targets, [0, 0, 1]).loss
    loss.backward()
    assert model.mem_head.weight.grad is not None
    assert model.mem_head.weight.grad.abs().sum().item() > 0
    reader = model.transformer.h[0].cross_attn.c_kv.weight
    assert reader.grad is not None
    assert reader.grad.abs().sum().item() > 0


def test_memory_add_starts_with_a_small_memory_residual():
    model = MemoryAddTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3))
    tokens = torch.randint(0, 17, (2, 6))
    output = model(tokens)

    projection = model.memory_projection.weight
    assert torch.count_nonzero(projection).item() > 0
    assert projection.std().item() == pytest.approx(1e-3, rel=0.3)

    token_stream = model.embed_tokens(tokens)
    random_memory = torch.randn_like(token_stream)
    baseline = model.forward_pass(token_stream, torch.zeros_like(token_stream))
    with_memory = model.forward_pass(token_stream, random_memory)
    assert not torch.equal(with_memory.hidden_states, baseline.hidden_states)
    assert not torch.equal(with_memory.logits, baseline.logits)


def test_memory_add_projection_and_writer_receive_gradients():
    model = MemoryAddTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))

    first = model.calc_total_loss(model(tokens), targets, [0, 0, 1]).loss
    first.backward()
    assert model.memory_projection.weight.grad is not None
    assert model.memory_projection.weight.grad.abs().sum().item() > 0
    assert model.mem_head.weight.grad is not None
    assert model.mem_head.weight.grad.abs().sum().item() > 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model.calc_total_loss(model(tokens), targets, [0, 0, 1]).loss
    second.backward()
    assert model.mem_head.weight.grad is not None
    assert model.mem_head.weight.grad.abs().sum().item() > 0


def test_memory_add_shifted_memory_is_causal():
    model = MemoryAddTransformer(MultiPassConfig(8, 17, 1, 1, 8, 2))
    model.eval()
    with torch.no_grad():
        model.memory_projection.weight.copy_(torch.eye(model.config.n_embd))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    token_stream = model.embed_tokens(tokens)
    base = torch.zeros_like(token_stream)
    changed = base.clone()
    changed[:, 2, :] = torch.randn_like(changed[:, 2, :])

    output_base = model.forward_pass(token_stream, base)
    output_changed = model.forward_pass(token_stream, changed)
    assert torch.allclose(output_base.logits[:, :2], output_changed.logits[:, :2], atol=1e-6, rtol=0)
    assert not torch.allclose(output_base.logits[:, 2:], output_changed.logits[:, 2:])


def test_final_pass_can_be_reproduced_from_previous_pass_memory_input():
    model = tiny_memory_model()
    tokens = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    previous = output.passes[-2].memory_states
    assert previous is not None
    reproduced = model.forward_pass(model.embed_tokens(tokens), shift_right(previous))
    assert torch.equal(reproduced.logits, output.logits)
    assert reproduced.memory_states is not None
    assert torch.equal(reproduced.memory_states, output.final_memory)


def test_recurrent_prefill_uses_last_pass_memory_and_append_is_immutable():
    model = tiny_memory_model(block_size=10)
    tokens = torch.randint(0, 19, (2, 5))
    output = model(tokens)
    state = model.prefill_recurrent(tokens)
    assert torch.equal(state.memory_states, output.final_memory)
    old_memory = state.memory_states.clone()
    next_token = state.next_token_logits.argmax(dim=-1, keepdim=True)
    next_state = model.recurrent_step(state, next_token)
    assert next_state.memory_states.shape[1] == old_memory.shape[1] + 1
    assert torch.equal(next_state.memory_states[:, :-1], old_memory)


@pytest.mark.parametrize(
    "model",
    [
        MemoryTapeTransformer(MultiPassConfig(10, 17, 1, 1, 8, 3)),
        MemoryAddTransformer(MultiPassConfig(10, 17, 1, 1, 8, 3)),
        LatentFeedbackTransformer(MultiPassConfig(10, 17, 1, 1, 8, 3)),
    ],
)
def test_append_recurrent_reads_and_appends_the_frozen_emitted_tape(model):
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = model.prefill_recurrent(prompt)
    next_token = torch.tensor([[5]])

    extended_tokens = torch.cat((prompt, next_token), dim=1)
    placeholder = torch.zeros_like(state.memory_states[:, :1, :])
    extended_read_memory = shift_right(
        torch.cat((state.memory_states, placeholder), dim=1)
    )
    manual = model.forward_pass(model.embed_tokens(extended_tokens), extended_read_memory)
    next_state = model.recurrent_step(state, next_token)

    assert torch.equal(next_state.memory_states[:, :-1], state.memory_states)
    assert torch.equal(next_state.memory_states[:, -1:], manual.memory_states[:, -1:])
    assert torch.equal(next_state.next_token_logits, manual.logits[:, -1, :])


def test_append_recurrent_matches_manual_two_token_rollout():
    model = tiny_memory_model(block_size=10)
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = model.prefill_recurrent(prompt)
    first = state.next_token_logits.argmax(dim=-1, keepdim=True)
    state = model.recurrent_step(state, first)
    second = state.next_token_logits.argmax(dim=-1, keepdim=True)
    expected = torch.cat((prompt, first, second), dim=1)
    actual = model.generate(prompt, 2, do_sample=False, inference_mode="append_recurrent")
    assert torch.equal(actual, expected)


def test_append_recurrent_context_guard_allows_final_unprocessed_token():
    model = tiny_memory_model(block_size=8)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    allowed = model.generate(prompt, 3, do_sample=False, inference_mode="append_recurrent")
    assert allowed.shape[1] == 9
    with pytest.raises(ValueError, match="prompt_length"):
        model.generate(prompt, 4, do_sample=False, inference_mode="append_recurrent")


def test_generation_restores_mode_and_validates_sampling_even_for_zero_tokens():
    model = tiny_memory_model()
    model.train()
    tokens = torch.tensor([[1, 2]])
    returned = model.generate(tokens, 0, inference_mode="append_recurrent")
    assert returned is tokens
    assert model.training
    with pytest.raises(ValueError, match="temperature"):
        model.generate(tokens, 0, temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(torch.randn(1, 3), top_k=0)


@pytest.mark.parametrize(
    "model",
    [
        MemoryTapeTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3)),
        MemoryAddTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3)),
        LatentFeedbackTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3)),
    ],
)
def test_all_multipass_models_support_append_recurrent(model):
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, 2, do_sample=False, inference_mode="append_recurrent")
    assert generated.shape == (1, 5)


def test_model_factory_constructs_supported_architectures():
    base = dict(
        n_layer=1,
        n_head=1,
        n_embd=8,
        max_passes=3,
        pass_loss_weights=[1],
    )
    expected = {
        "transformer": CausalTransformer,
        "memory_tape": MemoryTapeTransformer,
        "memory_add": MemoryAddTransformer,
        "latent_feedback": LatentFeedbackTransformer,
    }
    for architecture, model_class in expected.items():
        model = build_model(SimpleNamespace(architecture=architecture, **base), 17, 8, "cpu")
        assert isinstance(model, model_class)


@pytest.mark.parametrize("architecture", ["memory_tape", "memory_add"])
def test_canonical_memory_models_have_no_gate_parameters(architecture):
    args = SimpleNamespace(
        architecture=architecture,
        n_layer=2,
        n_head=1,
        n_embd=8,
        max_passes=3,
    )
    model = build_model(args, 17, 8, "cpu")
    assert not any("gate" in name for name, _parameter in model.named_parameters())
