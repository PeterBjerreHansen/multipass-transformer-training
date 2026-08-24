# Multi-Pass Transformer Training

This project explores a way to train transformers for recurrent-style inference without training them as token time recurrent models. The key idea is to train transformers with multiple passes over the same token-sequence. Earlier passes write per-token memory states; later passes read shifted versions of those memories, giving each token access to deep-layer information from previous token positions while preserving parallel training.

> **Project status (August 2026).** Wang et al.'s
> [*Full-bandwidth transformer*](https://arxiv.org/abs/2608.08888) independently
> develops the central multi-pass latent-feedback idea explored here and
> validates it at much larger scale. Mozer et al.'s
> [*Recirculation*](https://arxiv.org/abs/2608.17981) studies a closely related
> deep-to-shallow feedback path at inference time. These papers supersede any
> novelty claim for the broad idea. This repository is now a compact historical
> reference for the original implementations and controlled state-tracking
> tasks. Active research continues in
> [multipass-transformer-memory](https://github.com/PeterBjerreHansen/multipass-transformer-memory),
> which studies Memory Attention, Recirculation, sparse access, and larger-scale
> language-model experiments.

## A Motivating Problem: State Tracking

Transformers often struggle with algorithmic state tracking (see, for example, [Li25](https://arxiv.org/abs/2503.02854)), which is why related tasks appear in challenging benchmarks such as BBH (see [Suzgun22](https://arxiv.org/abs/2210.09261)). Transformers have difficulties tracking an increasing number of sequential state changes. The $S_5$ permutation task, for example, looks like "[A,B,C,D,E] swap 1 2 [B,A,C,D,E]" with one swap step.

![Curriculum frontier for S5 permutation tracking](figures/s5_permutation_fig.png "S₅ permutation tracking")

In my implementation models predict only the final state and the number of swaps is increased once validation accuracy exceeds 95%. For these experiments, the baseline transformer and multi-pass models use the `small` preset: 4 layers, 4 attention heads, and 128 embedding dimensions. The baseline is intentionally depth-constrained, while the multi-pass models can reuse a shifted memory state across recurrent passes. The baseline's learned number of state changes therefore flattens in a way that multi-pass training alleviates.

## A Theoretical Motivation

The training-time parallelization of decoder-only transformers is one of the main reasons they scale so well. At layer $l$, the hidden state $h_i^l$ at position $i$ can attend to positions $h_{j\leq i}^{l-1}$ from layer $l-1$, and not to hidden states from the same or deeper layers. This *causal* attention pattern permits hidden states for all token positions in a layer $[h_{1}^{l}, \ldots, h_{n}^l]$ to be computed in parallel during training, but it also disallows attention to previous tokens' deeper-layer hidden states at inference time.

This information flow gives the model no learned latent state independent of the token prefix. Without a KV cache, each generation step recomputes the prefix. A KV cache is runtime state that avoids this repeated computation, but it is functionally determined by the token prefix and does not carry information that a full-prefix evaluation would not reconstruct.

![](figures/inference_pattern_fig.png "Inference Patterns")

The tempting 'fix' would be to let the hidden state $h_{i}^{l}$ at token $i$ depend directly on the hidden state $h_{j}^{\ell}$ at token $j<i$ in the same or deeper layers $\ell \geq l$. But that would introduce a token-time recurrence: position $i$ would have to wait for position $i-1$, and the training-time parallelism would be lost.

![](figures/generation_fig.png "Generation")

But here is an idea: what if we run multiple sequential passes over the same teacher-forced sequence instead of making token $i$ wait for token $i-1$ during training? Token positions remain parallel within each pass, while the passes themselves form a recurrence. Pass 1 writes a memory state. Pass 2 reads a shifted version of that memory. Pass 3 can read the shifted memory from pass 2, and so on. The hope is that such multi-pass training can teach the model to emit memories that are useful and stable enough to support cheaper recurrent-style memory use at inference time.

### Mathematical Setup

The goal is to train the model to read and write a memory state for each token, and then test whether those memories can be reused during generation. There are many possible memory designs. This project focuses on one memory vector per token per pass.

For a token sequence $T = [t_0, \ldots, t_{n-1}]$, let $M^{(k)}$ be the length $n$ memory state written after pass $k$. The all-zero memory is the initial state $M^{(0)} = 0$, and the multi-pass recurrence is:

```math
(L^{(k)}, M^{(k)})
= F_\theta\left(T, \mathrm{Shift}(M^{(k-1)})\right),
\qquad k = 1, \ldots, K
```

Here $L^{(k)}$ is the pass $k$ logit tensor. $F_\theta$ is schematic: an architecture may write memory from any internal representation, not necessarily its final hidden state. The causal constraint is that position $t$ may read only memories written at earlier positions. In a parallel batch, that is implemented by a one-position shift:

```math
\mathrm{Shift}(M)[0] = 0
\qquad
\mathrm{Shift}(M)[t] = M[t - 1]
\quad \text{for } 1 \le t < n
```

That keeps the training computation parallel over token positions while giving each pass access to information written by the previous pass.

## Multi-pass Training

Multi-pass training runs the same teacher-forced sequence through the model several times. Crucially, recurrence is over the pass dimension, not over token time; within each forward pass, all token positions are still computed in parallel, as in the forward pass of an ordinary transformer.

Perhaps the easiest way to illustrate this is to imagine using a transformed last-layer hidden state as the memory. That memory can be fed back into the next pass in different ways, for example by concatenating it to the input stream or by reading it through a separate causal cross-attention path.

![](figures/multipass_training_fig.png "Multi-pass Training")

The $K$-pass training loop is:

> $`M^{(0)} = 0`$<br>
> $`\mathcal{L} = 0`$<br>
> $`\textbf{for } k = 1, \ldots, K:`$<br>
> &nbsp;&nbsp; $`R = \mathrm{Shift}(M^{(k-1)})`$<br>
> &nbsp;&nbsp; $`H = \mathrm{ArchitectureDecoder}(T, R)`$<br>
> &nbsp;&nbsp; $`L^{(k)} = \mathrm{LMHead}(H)`$<br>
> &nbsp;&nbsp; $`M^{(k)} = \mathrm{MemoryWriter}(H)`$<br>
> &nbsp;&nbsp; $`\mathcal{L} = \mathcal{L} + w_k\,\mathrm{LMLoss}(L^{(k)}, Y)`$

Most experiments put the heaviest weight on the final pass. The final pass is therefore trained to do the main predictive work, while earlier passes are encouraged to write memories that make later predictions easier.

The training schema:

1. pass `k` reads the shifted memory state written by pass `k - 1`
2. pass `k` predicts the same next-token targets as the other passes
3. pass `k` writes a new memory state for pass `k + 1`

is exact with respect to this `K`-pass model. No approximation has been introduced yet.

## Mismatch and Append-Recurrent Inference

And how do we get stateful inference out of this? Well, the exact inference procedure for this model is expensive. For every new token, we can run all $K$ passes on the full current prefix. That exact `recompute` procedure preserves the same pass-by-pass recurrence used in training, but it is too expensive for the target inference mode. What we want is append-recurrent inference:

1. Run the prompt exactly for $K$ passes.
2. Cache the final prompt memory state $M_{\mathrm{prompt}}^{(K)}$.
3. Generate the first token from the final prompt logits.
4. Run one pass over the extended prefix using the persistent memory cache.
5. Append only the memory written for the newest token.
6. Repeat without rewriting the older cached memories.

![](figures/mismatch_fig.png "Training and generation mismatch")

The first generated token is special. After the $K$ prompt passes, the model already has both the final logits for predicting the next token and the final prompt memory $M_{\mathrm{prompt}}^{(K)}$. So no extra recurrent pass is needed to sample the first token. Once $t_{n+1}$ has been generated, the model runs one pass over the extended prefix while reading the persistent prompt memory. It keeps the old entries fixed and appends the newly written memory for the generated token:

```math
\mathrm{Append}\left(M_{\mathrm{prompt}}^{(K)}, \widetilde M_{\mathrm{new}}\right)
```

The next generated token is then produced from a memory containing both final-pass prompt memories and a memory written by the online recurrent procedure. Each following step appends one more such memory. That is the approximation. Exact recomputation would rerun all $K$ passes on the longer prefix. Causality means that the prompt positions would be reconstructed identically, but the new position would be processed through the full sequence of $K$ pass updates. Append-recurrent inference reuses the final prompt memory and gives the new position only one online recurrent update before appending its memory. The project therefore depends on a stability question:

> Does multi-pass training produce final-pass memories that remain useful when they are frozen and extended recurrently with newly generated memories?

If yes, generation can pay for the $K$-pass computation once on the prompt and then continue with one pass per generated token. If no, the recurrent memory drifts away from the finite-pass model. The real empirical question is the gap between `recompute` and `append_recurrent` generation, especially as the generated suffix becomes longer.

## Experiments

### Long-Range Trace Tasks

Okay, but the state-tracking tasks introduced earlier had only a few tokens to predict. Is this not a cherry-picked set of tasks that avoids the mismatch problem?

Yes, partly. The BBH curriculum tasks isolate whether the model can learn repeated state updates without trace supervision, but final-answer-only supervision does not stress test append-recurrent generation over a long suffix. The mismatch problem only becomes unavoidable when the model has to keep generating after the prompt and repeatedly feed its own recurrent memory cache forward.

That is why the repo also includes longer-range trace tasks. These are fixed-trace generation problems where the model must emit a long legal suffix after the prompt, so `recompute` versus `append_recurrent` evaluation becomes a real test of recurrent stability. One motivation is the world model studied in [OthelloGPT](https://arxiv.org/pdf/2309.00941), an eight-layer GPT-2-style model trained to predict legal sequences of [Othello](https://www.eothello.com/) moves. Because move legality depends on the evolving board state, the model must learn an implicit form of board-state tracking. Most Othello games last about 60 moves, making legal continuation a useful long-range state-tracking task.

![](figures/trace_plot_figs.png "trace")

The plot above comes from an earlier architecture sweep. Some plotted variants are now preserved on the `archived-architectures` branch rather than supported on `main`. The current models still need a matched rerun under both `recompute` and `append_recurrent` before the figure is updated.

## Multi-pass Architectures

`main` supports three aligned-memory multi-pass designs. All use the shared
training and inference methods implemented by `MultiPassTransformer`. A
separate sandwich depth loop is included below, but it deliberately does not
pretend to support the aligned append-recurrent memory contract.

The notation in this section is tensor-level: $X$ is the token-embedding stream, $M^{(k)}$ is the full memory written at pass $k$, and $R = \mathrm{Shift}(M^{(k-1)})$ is the memory read at the next pass. The shared wrapper controls recurrence and shifts the previous memory. Each model owns one complete pass, including its final normalization, language-model head, and memory write.

### Memory Through Attention: The MemoryAttention Architecture

MemoryAttention retains an ordinary causal token decoder. Its decoder is:

> **MemoryAttention decoder**
>
> $`H = X`$<br>
> $`\textbf{for each decoder block:}`$<br>
> &nbsp;&nbsp; $`H = H + \mathrm{CausalSelfAttention}(\mathrm{LN}_{\mathrm{self}}(H))`$<br>
> &nbsp;&nbsp; $`H = H + \mathrm{CausalCrossAttention}\left(Q=\mathrm{LN}_{q}(H),\ KV=\mathrm{LN}_{kv}(R)\right)`$<br>
> &nbsp;&nbsp; $`H = H + \mathrm{MLP}(\mathrm{LN}_{\mathrm{mlp}}(H))`$<br>

Causal cross-attention is applied over $R$ as a separately addressable key/value source; the memory is not concatenated with the token stream. Its inclusive causal mask permits query position $t$ to read memory slots $s\leq t$. Because $R_s=M_{s-1}$, this is strict causality with respect to the unshifted memory: only memories from original positions before $t$ are readable. The reader is an ordinary residual branch with no learned gate. On pass one, $R=0$, so the cross-attention contribution is exactly zero and the model begins as a causal token decoder.

By default, every decoder block reads a memory vector of width `n_embd`, which
preserves the historical implementation. Two small levers cover the useful
variants without defining new architectures:

- `--memory-read-layers 1 3` installs readers only after those zero-based
  decoder layers. Omitting the option reads memory at every layer.
- `--memory-width 64` changes the written memory-vector width while leaving
  the token residual stream at `n_embd`. Omitting it uses `n_embd`.

Unselected readers are not instantiated, so narrower placement reduces both
parameters and compute rather than merely masking unused modules.

### Residual Memory Fusion: The MemoryAdd Architecture

MemoryAdd keeps the ordinary token stream intact and learns a residual correction from the shifted recurrent memory:

> **MemoryAdd decoder**
>
> $`H = X + W_{\mathrm{mem}}\mathrm{LN}_{\mathrm{mem}}(R)`$<br>
> $`\textbf{for each causal decoder block:}`$<br>
> &nbsp;&nbsp; $`H = \mathrm{DecoderBlock}(H)`$<br>

### GLU Latent Feedback

LatentFeedback implements the asymmetric GLU transition from
[Wang et al., *Full-bandwidth transformer*, Section 3.1](https://arxiv.org/html/2608.08888#S3.SS1).
The previous top-layer state supplies the value path. The token embedding
controls the sigmoid gate. During parallel training, the shifted memory $R$
supplies the previous state.

> **LatentFeedback decoder**
>
> $`\textbf{if } k=1: \quad H=X`$<br>
> $`\textbf{otherwise:}`$<br>
> &nbsp;&nbsp; $`H=\mathrm{LN}_{\mathrm{fb}}\left(W_U R\odot\sigma\left(W_G\mathrm{LN}_{\mathrm{fb}}(X)\right)\right)`$<br>
> &nbsp;&nbsp; $`H_0=X_0`$<br>
> $`\textbf{for each causal decoder block:}`$<br>
> &nbsp;&nbsp; $`H=\mathrm{DecoderBlock}(H)`$<br>
> $`H=\mathrm{LN}_{f}(H)`$<br>
> $`M^{(k)}=H`$

Position zero has no earlier memory, so it retains its token embedding. The
decoder writes its final top-layer state to the next memory state. This
implementation uses the repository's LayerNorm, pass-loss controls, and
append-recurrent inference contract. It does not reproduce the paper's full
model or training setup.

### Sandwiched Depth Recurrence

`sandwich_loop` runs the first physical decoder block once as a prelude,
repeats the middle blocks for `max_passes` iterations, and runs the final block
once as a coda:

> **Sandwich loop**
>
> $`H=\mathrm{Prelude}(X)`$<br>
> $`\textbf{repeat } K \textbf{ times:}\quad H=\mathrm{Core}(H)`$<br>
> $`H=\mathrm{Coda}(H)`$

It is a small weight-tied depth-loop reference inspired by the
`sandwiched-recurrence` branch. It is not Mozer et al.'s token-step
Recirculation: it recurs only in depth, has no shifted memory state, and supports
only `recompute` generation. A pass schedule may vary its core iterations, but
only the final coda output receives language-model loss.

## Tasks

The repository has two task families:

- BBH curriculum tasks predict one final answer. Their difficulty increases after validation accuracy reaches 95%.
- Trace tasks generate a full suffix from a preset-defined distribution. They test long continuations under `recompute` and `append_recurrent` inference.

Task generators and task-specific metrics live under `tasks/`. See [tasks/README.md](tasks/README.md) for the task catalogue, distributions, and evaluation definitions.

## Running experiments

`main` supports `transformer`, `memory_attention`, `memory_add`, `latent_feedback`,
and `sandwich_loop`. The canonical launchers keep the original four-model
aligned-memory comparison by default; `sandwich_loop` is opt-in because it has
a different compute and inference contract. This is a controlled architecture
comparison, not a reproduction of LatentFeedback's paper-specific training
protocol. The separate scheduled example below tests its central pass policy.

```bash
bash scripts/bbh/run.sh
bash scripts/trace/run.sh
```

Use environment variables to select tasks, architectures, seeds, the device, and the result root:

```bash
DEVICE=mps \
TASKS=shortest_path \
ARCHITECTURES="transformer memory_attention memory_add latent_feedback" \
SEEDS="1337 2027 4099" \
RESULT_ROOT=results/trace \
bash scripts/trace/run.sh
```

Each run writes `config.json`, `metrics.jsonl`, `best.pt`, and `latest.pt`. A run directory contains one training history. Use `--resume-from` to continue that history.

Use `tests/test_smoke.sh` for a quick end-to-end check. Use `tests/test_shortest_path.sh` for the complete shortest-path workflow check.

## Evaluation and diagnostics

Evaluate a trained trace run with its task-specific metrics:

```bash
RUN_DIR=results/trace/shortest_path/main/memory_attention/seed_1337 \
DEVICE=mps \
bash scripts/trace/eval.sh
```

The launcher reads the task and architecture from `config.json`. It evaluates `best.pt` by default. Set `CHECKPOINT=latest` to evaluate the final training state.

Run memory-use and pass-dynamics diagnostics separately:

```bash
uv run python -m experiments.diagnose_memory \
  --input-run-dir results/trace/shortest_path/main/memory_attention/seed_1337 \
  --extra-passes 6 \
  --schedule-gap-horizon 16
```

The report includes memory interventions, pass dynamics, schedule gap, and effective rank. Training logs also contain global and component gradient norms.

## Plotting notebooks

The tracked, output-free notebooks under `figures/` read the current artifact schemas directly:

- `01_bbh_curricula.ipynb` plots the $S_5$ permutation frontier and overall BBH curriculum coverage. It does not join loss across difficulty changes.
- `02_trace_learning.ipynb` plots conventional fixed-difficulty learning curves and measured throughput for Othello and shortest path.
- `03_deployment_and_othello.ipynb` compares paired `recompute` and `append_recurrent` quality, per-position free-generation drift, teacher-forced schedule gaps, and Othello random-prefix/legal-set metrics.
- `04_ablation_diagnostics.ipynb` mirrors the merge-decision rules with paired seed deltas, quality–efficiency plots, memory interventions, extra-pass dynamics, and schedule-gap comparisons.

The notebooks do not automatically save or overwrite any tracked figures. Select a comparable result root and uncomment an individual `savefig` line only when a plot is worth keeping. Start Jupyter from the repository root:

```bash
uv run jupyter lab figures/
```

Measured throughput should only be compared across matched devices, batch sizes, task difficulty, evaluation counts, and output-length distributions. The notebooks show individual seeds alongside unsmoothed medians and avoid confidence bands that would overstate what a three-seed ablation establishes.

## Architecture Examples

Baseline transformer:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture transformer
```

MemoryAttention:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture memory_attention
```

MemoryAttention with two readers and a narrow memory vector:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture memory_attention \
  --memory-read-layers 1 3 \
  --memory-width 64
```

MemoryAdd:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture memory_add
```

Every architecture uses learned absolute position embeddings by default. RoPE
is an optional backbone-level replacement:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture memory_attention \
  --position-encoding rope \
  --rope-theta 10000
```

RoPE rotates causal self-attention queries and keys in every architecture and
requires an even attention-head dimension. MemoryAttention cross-attention remains
content-addressed without RoPE; assigning positions to its shifted memory keys
is intentionally left as a separate ablation.

LatentFeedback with a checkpointed random training-depth mixture:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture latent_feedback \
  --max-passes 3 \
  --pass-mixture 0.75 0.22 0.03 \
  --pass-loss-weights-by-k 1 1 \
  --pass-loss-weights-by-k 2 0.5 0.5 \
  --pass-loss-weights-by-k 3 0.5 0.25 0.25
```

The mixture entries correspond to one, two, and three passes. They are
normalized automatically, and one pass count is sampled for the whole batch on
each optimizer step. The sampler RNG and histogram are saved in checkpoints.
Each repeated `--pass-loss-weights-by-k` occurrence begins with K and then
provides exactly K weights. This makes every sampled objective explicit: K=1
uses `[1]`, K=2 uses `[0.5, 0.5]`, and K=3 uses
`[0.5, 0.25, 0.25]`. The K=3 objective gives half the loss to the direct pass
and splits the other half across the feedback passes. This is the $\lambda=1$
weighting from the FBT objective. Evaluation always uses `--max-passes`.

Fixed-point training is available for `memory_attention`, `memory_add`, and
`latent_feedback`:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture latent_feedback \
  --train-pass-mode fixed_point \
  --min-passes 2 \
  --max-passes 4 \
  --fixed-point-memory-tol 0.1 \
  --fixed-point-kl-tol 1e-3
```

After the minimum depth, each example stops when both its relative
L-infinity memory change and consecutive-pass logit KL are within tolerance.
The memory check covers real, non-padding token positions; the KL check covers
supervised target positions. Halting decisions are detached, but gradients
flow through every pass that an example executes. Training gives equal weight
to the first-pass loss and the final adaptive-pass loss. `--pass-mixture`
cannot be combined with this mode.

Evaluation and generation remain fixed at `--max-passes`; fixed-point training
does not silently change deployment behavior. Metrics log mean executed
passes, convergence rate, the pass-count histogram, and final residual/KL
summaries. Active-sub-batch shrinking reduces arithmetic, though it may not
improve wall-clock time on every accelerator or compiled execution path.

Sandwich depth loop:

```bash
uv run python -m experiments.train_bbh \
  --preset permutation_main \
  --architecture sandwich_loop \
  --max-passes 4
```

## Requirements

The code is written in Python and PyTorch. The default device is selected automatically: CUDA when available, then MPS, otherwise CPU.

Create the project environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/`, installs the project in editable mode, and installs the
development tools used by the tests and plotting notebooks. Use `uv run` for
commands so they always use this environment; activation is optional. Run the
test suite with:

```bash
uv run pytest
```

To choose a device explicitly, pass `--device cpu`, `--device mps`, or `--device cuda` to the training scripts.

Local reference PDFs can live under `papers/`; that directory is ignored by git.
