from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Outputs and recurrent state
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PassOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    memory_states: torch.Tensor | None = None


@dataclass(frozen=True)
class PassHaltingStats:
    pass_counts: torch.Tensor
    converged: torch.Tensor
    relative_linf_residual: torch.Tensor
    logit_kl: torch.Tensor


@dataclass(frozen=True)
class MultiPassOutput:
    passes: tuple[PassOutput, ...]
    halting: PassHaltingStats | None = None

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("MultiPassOutput requires at least one pass")

    @property
    def logits(self) -> torch.Tensor:
        return self.passes[-1].logits

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.passes[-1].hidden_states

    @property
    def final_memory(self) -> torch.Tensor:
        memory = self.passes[-1].memory_states
        if memory is None:
            raise RuntimeError("this model output does not contain memory states")
        return memory

    @property
    def logits_per_pass(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.logits for item in self.passes)


@dataclass(frozen=True)
class RecurrentState:
    tokens: torch.Tensor
    memory_states: torch.Tensor
    next_token_logits: torch.Tensor


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    pass_losses: tuple[torch.Tensor, ...]


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    position_encoding: str = field(default="learned_absolute", kw_only=True)
    rope_theta: float = field(default=10_000.0, kw_only=True)

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.position_encoding not in {"learned_absolute", "rope"}:
            raise ValueError("position_encoding must be 'learned_absolute' or 'rope'")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if self.position_encoding == "rope" and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even attention-head dimension")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiPassConfig(TransformerConfig):
    max_passes: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_passes < 2:
            raise ValueError(
                f"max_passes ({self.max_passes}) must be at least 2 for multi-pass models"
            )


@dataclass
class MemoryAttentionConfig(MultiPassConfig):
    """MemoryAttention-specific reader placement and memory-vector width."""

    memory_width: int | None = None
    memory_read_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.memory_width is not None and self.memory_width < 1:
            raise ValueError("memory_width must be positive when specified")
        if self.memory_read_layers is not None:
            if not self.memory_read_layers:
                raise ValueError("memory_read_layers must not be empty")
            if len(set(self.memory_read_layers)) != len(self.memory_read_layers):
                raise ValueError("memory_read_layers must not contain duplicates")
            if any(
                layer < 0 or layer >= self.n_layer
                for layer in self.memory_read_layers
            ):
                raise ValueError("memory_read_layers contains an out-of-range layer")


# -----------------------------------------------------------------------------
# Shared components
# -----------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm with a learned scale and no learned bias."""

    def __init__(self, ndim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, theta: float):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even attention-head dimension")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=x.device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=x.dtype), angles.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_position_embeddings(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, theta=config.rope_theta)
            if config.position_encoding == "rope"
            else None
        )
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v)
            q, k = apply_rotary_position_embeddings(q, k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            y = F.softmax(scores, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalCrossAttention(nn.Module):
    """Causal cross-attention into an already right-shifted memory state."""

    def __init__(self, config: TransformerConfig, *, memory_dim: int | None = None):
        super().__init__()
        self.memory_dim = config.n_embd if memory_dim is None else memory_dim
        if self.memory_dim < 1:
            raise ValueError("memory_dim must be positive")
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_kv = nn.Linear(self.memory_dim, 2 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        memory_batch, memory_len, memory_dim = memory.shape
        if (
            batch_size != memory_batch
            or dim != self.n_embd
            or memory_dim != self.memory_dim
        ):
            raise ValueError("x and memory have incompatible batch or embedding dimensions")
        if memory_len != seq_len:
            raise ValueError("x and memory must share sequence length")

        q = self.c_q(x)
        k, v = self.c_kv(memory).split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, memory_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, memory_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(
                torch.ones(seq_len, memory_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            y = F.softmax(scores, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.c_proj(y)


def shift_right(memory: torch.Tensor) -> torch.Tensor:
    if memory.ndim != 3:
        raise ValueError("memory must have shape [B, T, D]")
    shifted = torch.zeros_like(memory)
    if memory.shape[1] > 1:
        shifted[:, 1:, :] = memory[:, :-1, :]
    return shifted


def normalize_pass_weights(
    weights: Sequence[float] | None,
    passes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if passes < 1:
        raise ValueError("passes must be positive")
    if weights is None:
        result = torch.ones(passes, device=device, dtype=dtype)
    else:
        if not weights:
            raise ValueError("loss_weights must not be empty")
        if len(weights) > passes:
            selected = weights[-passes:]
        else:
            selected = [0.0] * (passes - len(weights)) + list(weights)
        result = torch.as_tensor(selected, device=device, dtype=dtype)

    if not torch.isfinite(result).all():
        raise ValueError("loss_weights must be finite")
    if (result < 0).any():
        raise ValueError("loss_weights must be non-negative")
    total = result.sum()
    if total <= 0:
        raise ValueError("at least one loss weight must be positive")
    return result / total


def relative_linf_residual_per_example(
    previous: torch.Tensor,
    current: torch.Tensor,
    valid_positions: torch.Tensor,
) -> torch.Tensor:
    """Return a detached relative L-infinity memory residual per example."""
    if previous.shape != current.shape or current.ndim != 3:
        raise ValueError("states must have matching [B, T, D] shapes")
    if valid_positions.shape != current.shape[:2]:
        raise ValueError("valid_positions must have shape [B, T]")
    valid = valid_positions.to(device=current.device, dtype=torch.bool).unsqueeze(-1)
    if not valid.squeeze(-1).any(dim=1).all():
        raise ValueError("each example must contain at least one valid position")
    previous_float = previous.detach().float().masked_fill(~valid, 0.0)
    current_float = current.detach().float().masked_fill(~valid, 0.0)
    numerator = (current_float - previous_float).abs().amax(dim=(1, 2))
    denominator = current_float.abs().amax(dim=(1, 2))
    return numerator / (denominator + 1e-8)


def logit_kl_per_example(
    previous: torch.Tensor,
    current: torch.Tensor,
    valid_positions: torch.Tensor,
) -> torch.Tensor:
    """Return detached mean tokenwise KL(previous || current) per example."""
    if previous.shape != current.shape or current.ndim != 3:
        raise ValueError("logits must have matching [B, T, V] shapes")
    if valid_positions.shape != current.shape[:2]:
        raise ValueError("valid_positions must have shape [B, T]")
    valid = valid_positions.to(device=current.device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("each example must contain at least one valid position")
    previous_log = F.log_softmax(previous.detach().float(), dim=-1)
    current_log = F.log_softmax(current.detach().float(), dim=-1)
    token_kl = (previous_log.exp() * (previous_log - current_log)).sum(dim=-1)
    value = (token_kl * valid).sum(dim=1) / valid.sum(dim=1)
    return value.clamp_min(0.0)


def _validate_sampling_args(temperature: float, top_k: int | None) -> None:
    if temperature < 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V], got {tuple(logits.shape)}")
    _validate_sampling_args(temperature, top_k)

    if not do_sample or temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature
    if top_k is not None:
        k = min(top_k, scaled.shape[-1])
        cutoff = torch.topk(scaled, k=k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    probabilities = F.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


# -----------------------------------------------------------------------------
# Causal transformer baseline
# -----------------------------------------------------------------------------


class CausalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": (
                    nn.Embedding(config.block_size, config.n_embd)
                    if config.position_encoding == "learned_absolute"
                    else nn.Identity()
                ),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=std)

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding and isinstance(self.transformer.wpe, nn.Embedding):
            count -= self.transformer.wpe.weight.numel()
        return count

    def embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [B, T]")
        seq_len = idx.shape[1]
        if seq_len < 1:
            raise ValueError("input sequence must be non-empty")
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        hidden = self.transformer.wte(idx)
        if self.config.position_encoding == "learned_absolute":
            positions = torch.arange(seq_len, device=idx.device)
            hidden = hidden + self.transformer.wpe(positions)[None, :, :]
        return hidden

    def forward(self, idx: torch.Tensor) -> MultiPassOutput:
        hidden = self.embed_tokens(idx)
        for block in self.transformer.h:
            hidden = block(hidden)
        hidden = self.transformer.ln_f(hidden)
        logits = self.lm_head(hidden)
        return MultiPassOutput((PassOutput(logits=logits, hidden_states=hidden),))

    @staticmethod
    def calc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int | None = None,
        inference_mode: str = "recompute",
    ) -> torch.Tensor:
        if inference_mode != "recompute":
            raise ValueError("CausalTransformer only supports inference_mode='recompute'")
        _validate_generation_inputs(ids, max_new_tokens)
        _validate_sampling_args(temperature, top_k)
        if max_new_tokens == 0:
            return ids
        was_training = self.training
        self.eval()
        try:
            result = ids
            for _ in range(max_new_tokens):
                context = result[:, -self.config.block_size :]
                logits = self(context).logits[:, -1, :]
                next_token = sample_next_token(
                    logits,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                )
                result = torch.cat((result, next_token), dim=1)
            return result
        finally:
            self.train(was_training)


class SandwichLoopTransformer(CausalTransformer):
    """Depth-recurrent decoder with a one-shot prelude and coda.

    The first block contextualizes the token stream once, the middle blocks
    form a weight-tied recurrent core, and the final block produces the
    readout. This is depth recurrence rather than append-recurrent token
    memory, so generation always recomputes the complete prefix.
    """

    def __init__(self, config: MultiPassConfig):
        if config.n_layer < 3:
            raise ValueError("sandwich_loop requires at least three layers")
        super().__init__(config)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        passes: int | None = None,
    ) -> MultiPassOutput:
        effective_passes = self.config.max_passes if passes is None else int(passes)
        if not 1 <= effective_passes <= self.config.max_passes:
            raise ValueError(f"passes must be between 1 and {self.config.max_passes}")

        hidden = self.transformer.h[0](self.embed_tokens(idx))
        for _ in range(effective_passes):
            for block in self.transformer.h[1:-1]:
                hidden = block(hidden)
        hidden = self.transformer.h[-1](hidden)
        hidden = self.transformer.ln_f(hidden)
        logits = self.lm_head(hidden)
        return MultiPassOutput((PassOutput(logits=logits, hidden_states=hidden),))


# -----------------------------------------------------------------------------
# Multi-pass base and variants
# -----------------------------------------------------------------------------


class MultiPassTransformer(nn.Module):
    block_cls: type[nn.Module] | None = None

    def build_blocks(self, config: MultiPassConfig) -> nn.ModuleList:
        if self.block_cls is None:
            raise ValueError(f"{type(self).__name__} must define block_cls")
        return nn.ModuleList([self.block_cls(config) for _ in range(config.n_layer)])

    def __init__(self, config: MultiPassConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": (
                    nn.Embedding(config.block_size, config.n_embd)
                    if config.position_encoding == "learned_absolute"
                    else nn.Identity()
                ),
                "h": self.build_blocks(config),
                "ln_f": LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @property
    def memory_dim(self) -> int:
        return self.config.n_embd

    def finish_initialization(self) -> None:
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding and isinstance(self.transformer.wpe, nn.Embedding):
            count -= self.transformer.wpe.weight.numel()
        return count

    def embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [B, T]")
        seq_len = idx.shape[1]
        if seq_len < 1:
            raise ValueError("input sequence must be non-empty")
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        hidden = self.transformer.wte(idx)
        if self.config.position_encoding == "learned_absolute":
            positions = torch.arange(seq_len, device=idx.device)
            hidden = hidden + self.transformer.wpe(positions)[None, :, :]
        return hidden

    def forward_pass(
        self,
        token_stream: torch.Tensor,
        read_memory: torch.Tensor | None,
    ) -> PassOutput:
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, *, passes: int | None = None) -> MultiPassOutput:
        effective_passes = self.config.max_passes if passes is None else int(passes)
        if not 1 <= effective_passes <= self.config.max_passes:
            raise ValueError(f"passes must be between 1 and {self.config.max_passes}")
        token_stream = self.embed_tokens(idx)
        initial = self.forward_pass(token_stream, None)
        outputs = [initial]
        previous_memory = initial.memory_states
        if previous_memory is None:
            raise RuntimeError("multi-pass model failed to emit memory states")
        for _ in range(1, effective_passes):
            output = self.forward_pass(token_stream, shift_right(previous_memory))
            outputs.append(output)
            previous_memory = output.memory_states
            if previous_memory is None:
                raise RuntimeError("multi-pass model failed to emit memory states")
        return MultiPassOutput(tuple(outputs))

    def forward_fixed_point(
        self,
        idx: torch.Tensor,
        *,
        min_passes: int,
        max_passes: int,
        memory_tolerance: float,
        kl_tolerance: float,
        memory_positions: torch.Tensor,
        logit_positions: torch.Tensor,
    ) -> MultiPassOutput:
        """Run recurrence with detached, per-example fixed-point halting.

        Halted examples keep their last tensors while the remaining active
        sub-batch continues. The final pass therefore contains every example
        at its own selected depth, with gradients through all executed passes.
        """
        if not 2 <= min_passes <= max_passes <= self.config.max_passes:
            raise ValueError(
                "pass limits must satisfy 2 <= min_passes <= max_passes <= configured max_passes"
            )
        for name, value in (
            ("memory_tolerance", memory_tolerance),
            ("kl_tolerance", kl_tolerance),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if memory_positions.shape != idx.shape or logit_positions.shape != idx.shape:
            raise ValueError("position masks must have the same [B, T] shape as idx")
        memory_positions = memory_positions.to(device=idx.device, dtype=torch.bool)
        logit_positions = logit_positions.to(device=idx.device, dtype=torch.bool)
        if not memory_positions.any(dim=1).all() or not logit_positions.any(dim=1).all():
            raise ValueError("each example must contain valid memory and logit positions")

        token_stream = self.embed_tokens(idx)
        batch_size = token_stream.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=idx.device)
        pass_counts = torch.zeros(batch_size, dtype=torch.long, device=idx.device)
        converged = torch.zeros_like(active)
        final_residual = torch.full(
            (batch_size,),
            float("inf"),
            device=idx.device,
            dtype=torch.float32,
        )
        final_kl = torch.full_like(final_residual, float("inf"))
        outputs: list[PassOutput] = []
        previous_memory: torch.Tensor | None = None
        previous_logits: torch.Tensor | None = None
        previous_hidden: torch.Tensor | None = None

        for pass_index in range(1, max_passes + 1):
            active_indices = active.nonzero(as_tuple=False).flatten()
            active_tokens = token_stream.index_select(0, active_indices)
            read_memory = None
            active_memory = None
            if previous_memory is not None:
                active_memory = previous_memory.index_select(0, active_indices)
                read_memory = shift_right(active_memory)
            raw = self.forward_pass(active_tokens, read_memory)
            if raw.memory_states is None:
                raise RuntimeError("multi-pass model failed to emit memory states")

            pass_counts = pass_counts + active.to(dtype=torch.long)
            if previous_memory is None or previous_logits is None or previous_hidden is None:
                current_memory = raw.memory_states
                current_logits = raw.logits
                current_hidden = raw.hidden_states
            else:
                current_memory = previous_memory.index_copy(
                    0,
                    active_indices,
                    raw.memory_states,
                )
                current_logits = previous_logits.index_copy(0, active_indices, raw.logits)
                current_hidden = previous_hidden.index_copy(
                    0,
                    active_indices,
                    raw.hidden_states,
                )
                if active_memory is None:
                    raise RuntimeError("active memory is missing after the first pass")
                residual = relative_linf_residual_per_example(
                    active_memory,
                    raw.memory_states,
                    memory_positions.index_select(0, active_indices),
                )
                logit_kl = logit_kl_per_example(
                    previous_logits.index_select(0, active_indices),
                    raw.logits,
                    logit_positions.index_select(0, active_indices),
                )
                final_residual = final_residual.index_copy(0, active_indices, residual)
                final_kl = final_kl.index_copy(0, active_indices, logit_kl)

                if pass_index >= min_passes:
                    newly_converged = (residual <= memory_tolerance) & (
                        logit_kl <= kl_tolerance
                    )
                    converged_indices = active_indices[newly_converged]
                    converged = converged.index_fill(0, converged_indices, True)
                    active = active.index_fill(0, converged_indices, False)

            outputs.append(
                PassOutput(
                    logits=current_logits,
                    hidden_states=current_hidden,
                    memory_states=current_memory,
                )
            )
            previous_memory = current_memory
            previous_logits = current_logits
            previous_hidden = current_hidden
            if not bool(active.any().item()):
                break

        return MultiPassOutput(
            tuple(outputs),
            halting=PassHaltingStats(
                pass_counts=pass_counts.detach(),
                converged=converged.detach(),
                relative_linf_residual=final_residual.detach(),
                logit_kl=final_kl.detach(),
            ),
        )

    @staticmethod
    def calc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

    def calc_total_loss(
        self,
        output: MultiPassOutput,
        targets: torch.Tensor,
        loss_weights: Sequence[float] | None = None,
    ) -> LossOutput:
        losses = tuple(self.calc_loss(logits, targets) for logits in output.logits_per_pass)
        weights = normalize_pass_weights(
            loss_weights,
            len(losses),
            device=losses[0].device,
            dtype=losses[0].dtype,
        )
        total = torch.stack(losses).mul(weights).sum()
        return LossOutput(loss=total, pass_losses=losses)

    @torch.no_grad()
    def prefill_recurrent(self, ids: torch.Tensor) -> RecurrentState:
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("ids must have shape [B, T] with T >= 1")
        if ids.shape[1] > self.config.block_size:
            raise ValueError("prompt length exceeds block_size")
        output = self(ids)
        return RecurrentState(
            tokens=ids,
            memory_states=output.final_memory,
            next_token_logits=output.logits[:, -1, :],
        )

    @torch.no_grad()
    def recurrent_step(self, state: RecurrentState, next_token: torch.Tensor) -> RecurrentState:
        if state.tokens.ndim != 2 or state.memory_states.ndim != 3:
            raise ValueError("invalid recurrent state shapes")
        if state.memory_states.shape[:2] != state.tokens.shape:
            raise ValueError("recurrent memory must align with recurrent tokens")
        if state.memory_states.shape[2] != self.memory_dim:
            raise ValueError("recurrent memory has the wrong embedding dimension")
        if next_token.ndim != 2 or next_token.shape != (state.tokens.shape[0], 1):
            raise ValueError("next_token must have shape [B, 1]")
        tokens = torch.cat((state.tokens, next_token), dim=1)
        if tokens.shape[1] > self.config.block_size:
            raise ValueError("append_recurrent cannot exceed block_size")

        placeholder = torch.zeros(
            state.memory_states.shape[0],
            1,
            state.memory_states.shape[2],
            device=state.memory_states.device,
            dtype=state.memory_states.dtype,
        )
        previous_memory = torch.cat((state.memory_states, placeholder), dim=1)
        token_stream = self.embed_tokens(tokens)
        output = self.forward_pass(token_stream, shift_right(previous_memory))
        if output.memory_states is None:
            raise RuntimeError("recurrent pass failed to emit memory states")
        appended_memory = output.memory_states[:, -1:, :]
        memory_states = torch.cat((state.memory_states, appended_memory), dim=1)
        return RecurrentState(
            tokens=tokens,
            memory_states=memory_states,
            next_token_logits=output.logits[:, -1, :],
        )

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int | None = None,
        inference_mode: str = "recompute",
    ) -> torch.Tensor:
        _validate_generation_inputs(ids, max_new_tokens)
        _validate_sampling_args(temperature, top_k)
        if inference_mode not in {"recompute", "append_recurrent"}:
            raise ValueError("inference_mode must be 'recompute' or 'append_recurrent'")
        if max_new_tokens == 0:
            return ids

        was_training = self.training
        self.eval()
        try:
            if inference_mode == "recompute":
                result = ids
                for _ in range(max_new_tokens):
                    context = result[:, -self.config.block_size :]
                    logits = self(context).logits[:, -1, :]
                    next_token = sample_next_token(
                        logits,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                    )
                    result = torch.cat((result, next_token), dim=1)
                return result

            # The final sampled token does not need a recurrent update.  Every
            # context used to compute logits must fit, so a returned sequence may
            # be one token longer than block_size.
            if ids.shape[1] + max_new_tokens - 1 > self.config.block_size:
                raise ValueError(
                    "append_recurrent requires prompt_length + max_new_tokens - 1 <= block_size"
                )
            state = self.prefill_recurrent(ids)
            for step in range(max_new_tokens):
                next_token = sample_next_token(
                    state.next_token_logits,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                )
                if step == max_new_tokens - 1:
                    return torch.cat((state.tokens, next_token), dim=1)
                state = self.recurrent_step(state, next_token)
            return state.tokens
        finally:
            self.train(was_training)

class MemoryAddTransformer(MultiPassTransformer):

    block_cls = Block

    def __init__(self, config: MultiPassConfig):
        super().__init__(config)
        self.ln_mem = LayerNorm(config.n_embd)
        self.mem_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.mem_in_ln = LayerNorm(config.n_embd)
        self.memory_projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.finish_initialization()
        nn.init.normal_(self.memory_projection.weight, mean=0.0, std=1e-3)

    def forward_pass(
        self,
        token_stream: torch.Tensor,
        read_memory: torch.Tensor | None,
    ) -> PassOutput:
        if read_memory is None:
            read_memory = torch.zeros_like(token_stream)
        if token_stream.shape != read_memory.shape:
            raise ValueError("token_stream and read_memory must have the same shape")
        hidden = token_stream + self.memory_projection(self.mem_in_ln(read_memory))
        for block in self.transformer.h:
            hidden = block(hidden)
        hidden = self.transformer.ln_f(hidden)
        return PassOutput(
            logits=self.lm_head(hidden),
            hidden_states=hidden,
            memory_states=self.mem_head(self.ln_mem(hidden)),
        )


class LatentFeedbackTransformer(MultiPassTransformer):
    """Top-layer latent feedback from Wang et al., arXiv:2608.08888."""

    block_cls = Block

    def __init__(self, config: MultiPassConfig):
        super().__init__(config)
        self.feedback_input_ln = LayerNorm(config.n_embd)
        self.feedback_value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.feedback_gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.finish_initialization()

    def forward_pass(
        self,
        token_stream: torch.Tensor,
        read_memory: torch.Tensor | None,
    ) -> PassOutput:
        if read_memory is None:
            hidden = token_stream
        else:
            if token_stream.shape != read_memory.shape:
                raise ValueError("token_stream and read_memory must have the same shape")
            hidden = self.feedback_value(read_memory) * torch.sigmoid(
                self.feedback_gate(self.feedback_input_ln(token_stream))
            )
            hidden = self.feedback_input_ln(hidden)
            hidden = torch.cat((token_stream[:, :1, :], hidden[:, 1:, :]), dim=1)
        for block in self.transformer.h:
            hidden = block(hidden)
        hidden = self.transformer.ln_f(hidden)
        return PassOutput(
            logits=self.lm_head(hidden),
            hidden_states=hidden,
            memory_states=hidden,
        )


class MemoryBlock(nn.Module):
    def __init__(self, config: MultiPassConfig, *, read_memory: bool = True):
        super().__init__()
        self.ln_self = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        memory_dim = getattr(config, "memory_width", None) or config.n_embd
        self.ln_mem_q = LayerNorm(config.n_embd) if read_memory else None
        self.ln_mem_kv = LayerNorm(memory_dim) if read_memory else None
        self.cross_attn = (
            CausalCrossAttention(config, memory_dim=memory_dim)
            if read_memory
            else None
        )
        self.ln_mlp = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_self(x))
        if self.cross_attn is not None:
            if self.ln_mem_q is None or self.ln_mem_kv is None:
                raise RuntimeError("memory reader normalization is missing")
            memory_delta = self.cross_attn(
                self.ln_mem_q(x),
                self.ln_mem_kv(memory_states),
            )
            x = x + memory_delta
        x = x + self.mlp(self.ln_mlp(x))
        return x


class MemoryAttentionTransformer(MultiPassTransformer):
    block_cls = MemoryBlock

    def build_blocks(self, config: MultiPassConfig) -> nn.ModuleList:
        if not isinstance(config, MemoryAttentionConfig):
            raise TypeError("MemoryAttentionTransformer requires MemoryAttentionConfig")
        enabled = (
            set(range(config.n_layer))
            if config.memory_read_layers is None
            else set(config.memory_read_layers)
        )
        return nn.ModuleList(
            [
                MemoryBlock(config, read_memory=layer in enabled)
                for layer in range(config.n_layer)
            ]
        )

    def __init__(self, config: MultiPassConfig):
        if not isinstance(config, MemoryAttentionConfig):
            config = MemoryAttentionConfig(**config.to_dict())
        super().__init__(config)
        self.ln_mem = LayerNorm(config.n_embd)
        self.mem_head = nn.Linear(config.n_embd, self.memory_dim, bias=False)
        self.finish_initialization()

    @property
    def memory_dim(self) -> int:
        config = self.config
        if not isinstance(config, MemoryAttentionConfig):
            raise TypeError("MemoryAttentionTransformer requires MemoryAttentionConfig")
        return config.memory_width or config.n_embd

    @property
    def memory_read_layers(self) -> tuple[int, ...]:
        return tuple(
            layer
            for layer, block in enumerate(self.transformer.h)
            if isinstance(block, MemoryBlock) and block.cross_attn is not None
        )

    def forward_pass(
        self,
        token_stream: torch.Tensor,
        read_memory: torch.Tensor | None,
    ) -> PassOutput:
        if read_memory is None:
            read_memory = token_stream.new_zeros(
                token_stream.shape[0],
                token_stream.shape[1],
                self.memory_dim,
            )
        if token_stream.ndim != 3 or read_memory.ndim != 3:
            raise ValueError("token_stream and read_memory must have shape [B, T, D]")
        if token_stream.shape[:2] != read_memory.shape[:2]:
            raise ValueError("token_stream and read_memory must align in batch and sequence")
        if (
            token_stream.shape[2] != self.config.n_embd
            or read_memory.shape[2] != self.memory_dim
        ):
            raise ValueError("token_stream or read_memory has the wrong embedding dimension")
        hidden = token_stream
        for block in self.transformer.h:
            hidden = block(hidden, read_memory)
        hidden = self.transformer.ln_f(hidden)
        return PassOutput(
            logits=self.lm_head(hidden),
            hidden_states=hidden,
            memory_states=self.mem_head(self.ln_mem(hidden)),
        )


def _validate_generation_inputs(ids: torch.Tensor, max_new_tokens: int) -> None:
    if ids.ndim != 2:
        raise ValueError("ids must have shape [B, T]")
    if ids.shape[1] < 1:
        raise ValueError("generation requires a non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
