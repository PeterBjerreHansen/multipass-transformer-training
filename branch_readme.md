# Latent-feedback experiment

This branch adapts the latent-feedback architecture from Wang et al.,
[*Full-bandwidth transformer*](https://arxiv.org/abs/2608.08888), to this
repository's small decoder-only transformer.

## Architecture

The first pass is an ordinary transformer pass. Later passes shift the previous
top-layer states by one position and fuse them with the fixed token embeddings:

```text
input_t = W_U h_(t-1) * sigmoid(W_G norm(e_t))
```

The fused inputs pass through the same transformer stack. The resulting
top-layer hidden states are the next memory tape, so the architecture has no
separate memory writer. This is paper-inspired rather than a full reproduction:
the repository retains its existing backbone, tasks, loss controls, and
evaluation protocol.

Append-recurrent inference uses the same memory contract as the other
multi-pass models. Prefill runs `max_passes` times and freezes the final emitted
memory tape. Each generated token reads that tape, appends one new memory, and
leaves the old tape unchanged. Recomputing old positions is an implementation
detail; their newly emitted memories are discarded.

## Pass scheduling

`--train-pass-schedule` samples the number of passes once per optimizer step.
Each phase uses `START=PASS:WEIGHT,...`. The scheduler is available to every
multi-pass architecture and has its own checkpointed random-number generator.

```bash
python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture latent_feedback \
  --max-passes 3 \
  --pass-loss-weights 1 1 \
  --train-pass-schedule \
    "1=1:1" \
    "25001=1:3,2:1" \
    "37501=1:5,2:3,3:2" \
  --device mps \
  --run-dir results/bbh/permutation/latent_feedback/seed_1337
```

Pass-loss weights are relative to the active depth. With weights `1 1`, one
active pass receives weight `[1]`, two receive `[1, 1]`, and four receive
`[0, 0, 1, 1]`. Evaluation always uses `--max-passes`.

## Code review and merging

The architecture adds two square projections and one normalization layer. It
uses the shared multi-pass loss, scheduler, generation, and diagnostics paths.
MemoryAdd and MemoryTape retain their existing append-recurrent semantics.

The branch can be merged without another model abstraction. The empirical
question is whether latent feedback improves fixed-pass or append-recurrent
evaluation enough to justify maintaining a fourth architecture.
