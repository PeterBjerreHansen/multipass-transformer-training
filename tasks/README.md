# Tasks

This directory contains task generators, shared batches, and task-specific
evaluation code. Training and checkpoint code lives under `experiments/`.

## Task families

The repository uses two task families:

| Family | Supervision | Difficulty | Tasks |
| --- | --- | --- | --- |
| BBH curriculum | Predict the final answer | Increases after validation accuracy reaches 95% | `permutation`, `tracking`, `pointer_chasing`, `state_machine` |
| Trace generation | Generate the full target suffix | Drawn from the selected preset distribution | `othello`, `shortest_path`, `maze` |

### BBH curriculum tasks

- `permutation` applies successive swaps to five objects. The model predicts
  the final permutation. The curriculum level is the number of swaps.
- `tracking` applies swaps, rotations, and reversals to five objects. The model
  predicts which object occupies a queried position.
- `pointer_chasing` presents a shuffled directed odd cycle. At level `L`, the
  model follows `L` transitions through `2L + 1` labeled nodes.
- `state_machine` presents a shuffled deterministic transition table and an
  action sequence. The model predicts the final state.

These tasks use final-answer-only supervision. They test repeated state updates
without giving the model an intermediate trace.

### Trace tasks

- `othello` generates legal move sequences from the standard initial board.
  Evaluation measures continuation legality and legal-set probability.
- `shortest_path` presents a shuffled directed acyclic graph with one shortest
  route. The model generates the complete route from the given start to goal.
- `maze` uses finite Searchformer-style random-wall datasets. The model
  generates a cell path or an action sequence.

Trace tasks test long autoregressive continuations. Multi-pass checkpoints can
run with exact `recompute` inference or approximate `append_recurrent`
inference.

## Shortest-path distributions

The generator permutes node labels and shuffles graph edges for each example.
It verifies that each graph has exactly one shortest route.

| Preset | Nodes | Route length | Maximum out-degree | Longer alternatives |
| --- | ---: | ---: | ---: | ---: |
| `shortest_path_easy` | 8-12 | 3-4 edges | 2 | 1-2 |
| `shortest_path_main` | 16-26 | 5-10 edges | 2 | 4-6 |

The main distribution samples route length before graph size. This keeps all
route lengths feasible and approximately balanced.

Shortest-path evaluation reports exact optimal-route accuracy. It also reports
accuracy for each generated transition.

## Maze data

The maze task reads finite, pre-tokenized datasets. It does not generate data
when a requested artifact is missing.

Create the canonical mazes with the separate `maze-data-generator` repository.
Then compile them into the directory selected by `--maze-data-dir`:

```bash
maze-data compile-all \
  --input data/searchformer-10.jsonl \
  --output-dir /path/to/multi_pass_transformer/data/maze/searchformer-10
```

The task supports sparse-cell and dense-grid prompts. It supports cell-path and
action targets. Route policies are `astar`, `uniform_shortest`, and `dfs`.

Maze evaluation reports:

- `optimal_route`: the route is legal, reaches the goal, and has shortest
  length.
- `exact_target_route`: the generated suffix matches the selected target.

## Task-specific evaluation

Task metrics remain next to their task implementations:

- `tasks/trace/othello_eval.py` defines legality and legal-set metrics.
- `tasks/trace/shortest_path_eval.py` defines optimal-route and per-step
  metrics.
- `tasks/trace/maze_eval.py` checks route legality and optimality.

`experiments/eval_trace.py` loads checkpoints and runs the shared evaluation
loop. `experiments/eval_othello_prefix.py` adds Othello random-prefix
evaluation.
