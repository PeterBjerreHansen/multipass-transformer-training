from __future__ import annotations

from models import (
    CausalTransformer,
    LatentFeedbackTransformer,
    MemoryAddTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MultiPassConfig,
    TransformerConfig,
)


ARCHITECTURES = (
    "transformer",
    "memory_tape",
    "memory_add",
    "latent_feedback",
)


def is_multi_pass_architecture(architecture: str) -> bool:
    return architecture != "transformer"


def build_model(args, vocab_size: int, block_size: int, device: str):
    common = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    if args.architecture == "transformer":
        model = CausalTransformer(TransformerConfig(**common))
    elif args.architecture == "memory_tape":
        model = MemoryTapeTransformer(
            MemoryTapeConfig(
                **common,
                max_passes=args.max_passes,
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
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
