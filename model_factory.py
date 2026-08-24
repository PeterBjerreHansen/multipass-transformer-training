from __future__ import annotations

from models import (
    CausalTransformer,
    LatentFeedbackTransformer,
    MemoryAddTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MultiPassConfig,
    SandwichLoopTransformer,
    TransformerConfig,
)

ARCHITECTURES = (
    "transformer",
    "memory_tape",
    "memory_add",
    "latent_feedback",
    "sandwich_loop",
)


PASS_LOSS_ARCHITECTURES = frozenset(
    {"memory_tape", "memory_add", "latent_feedback"}
)
PASS_OVERRIDE_ARCHITECTURES = PASS_LOSS_ARCHITECTURES | {"sandwich_loop"}
APPEND_RECURRENT_ARCHITECTURES = PASS_LOSS_ARCHITECTURES
MEMORY_DIAGNOSTIC_ARCHITECTURES = PASS_LOSS_ARCHITECTURES


def is_multi_pass_architecture(architecture: str) -> bool:
    """Compatibility predicate for models that emit and weight every pass."""
    return architecture in PASS_LOSS_ARCHITECTURES


def supports_pass_override(architecture: str) -> bool:
    return architecture in PASS_OVERRIDE_ARCHITECTURES


def uses_pass_loss_weights(architecture: str) -> bool:
    return architecture in PASS_LOSS_ARCHITECTURES


def supports_fixed_point_training(architecture: str) -> bool:
    return architecture in PASS_LOSS_ARCHITECTURES


def supports_append_recurrent(architecture: str) -> bool:
    return architecture in APPEND_RECURRENT_ARCHITECTURES


def supports_memory_diagnostics(architecture: str) -> bool:
    return architecture in MEMORY_DIAGNOSTIC_ARCHITECTURES


def build_model(args, vocab_size: int, block_size: int, device: str):
    common = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        position_encoding=getattr(args, "position_encoding", "learned_absolute"),
        rope_theta=getattr(args, "rope_theta", 10_000.0),
    )

    if args.architecture == "transformer":
        model = CausalTransformer(TransformerConfig(**common))
    elif args.architecture == "memory_tape":
        model = MemoryTapeTransformer(
            MemoryTapeConfig(
                **common,
                max_passes=args.max_passes,
                memory_width=getattr(args, "memory_width", None),
                memory_read_layers=(
                    tuple(args.memory_read_layers)
                    if getattr(args, "memory_read_layers", None) is not None
                    else None
                ),
            )
        )
    elif args.architecture == "memory_add":
        model = MemoryAddTransformer(
            MultiPassConfig(
                **common,
                max_passes=args.max_passes,
            )
        )
    elif args.architecture == "latent_feedback":
        model = LatentFeedbackTransformer(
            MultiPassConfig(
                **common,
                max_passes=args.max_passes,
            )
        )
    elif args.architecture == "sandwich_loop":
        model = SandwichLoopTransformer(
            MultiPassConfig(
                **common,
                max_passes=args.max_passes,
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
