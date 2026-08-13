# Latent-feedback experiment

This branch adapts the latent-feedback method from Wang et al.,
[*Full-bandwidth transformer*](https://arxiv.org/abs/2608.08888), to this
repository's small decoder-only transformer.

## Method

Pass 1 is an ordinary transformer pass over token and position embeddings. On
each later pass, the previous top-layer state is shifted one position to the
right and fused with the fixed token embedding:

```text
fused_t = W_U h_(t-1) * sigmoid(W_G norm(e_t))
```

The fused input is normalized and sent through the shared transformer stack.
There is no additive token shortcut and no separate memory writer. The loss is
the standard first-pass language-model loss plus the mean loss over feedback
passes. Gradients from later passes are not detached.

This is a faithful implementation of the paper's feedback transition,
temporal-parallel passes, and loss. It deliberately retains this repository's
small backbone: LayerNorm, learned absolute positions, untied input/output
weights, and full causal attention. It is therefore an architecture ablation,
not an exact reproduction of the paper's 1B-parameter model. Prefix mixin and
state noise are not part of this branch.

## Probabilistic pass schedule

`--train-pass-schedule` selects the total number of passes once per optimizer
step. Each phase has the form `START=PASS:WEIGHT,...`. The scheduler has its own
random-number generator. Its state and cumulative histogram are stored in each
checkpoint, so a resumed run continues the same sampling sequence.

For example:

```bash
python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture latent_feedback \
  --train-pass-schedule \
    "1=1:1" \
    "25001=1:.75,2:.25" \
    "37501=1:.75,2:.22,3:.03" \
  --device mps \
  --run-dir results/bbh/permutation/latent_feedback/seed_1337
```

Here `1`, `2`, and `3` mean total forward passes, or zero, one, and two feedback
passes. Evaluation still uses the fixed `--n-pass` value from the preset.

## Code review

The model adds two square projections and one normalization layer. The shared
multi-pass base now accepts a temporary pass-count override without changing
its configuration. The scheduler is isolated in `experiments/pass_schedule.py`
and both trainers use the same parser, sampler, logging, and checkpoint state.
The existing three main architectures keep their previous defaults.

The change can be merged without restructuring. The model factory gains one
explicit architecture, while the canonical launchers still default to the
three models on `main`. The main decision before merging is empirical: the GLU
model should improve fixed-pass or recurrent evaluation enough to justify
making a fourth architecture part of the maintained model set.
