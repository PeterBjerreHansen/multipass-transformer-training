# Maze experiment launchers

These launchers use the verified `searchformer-10` corpus and keep the
scientific conditions in the output path. They run the normal `maze_main`
training preset and evaluate every completed run on the deterministic test
selection.

Compile the corpus first from the sibling `maze-data-generator` repository:

```bash
PYTHONPATH=../maze-data-generator/src \
  python -m maze_data compile-all \
  --input ../maze-data-generator/data/searchformer-10.jsonl \
  --output-dir data/maze/searchformer-10
```

Then run one of the frozen matrices:

```bash
bash scripts/trace/maze_baseline.sh
bash scripts/trace/maze_representation_ablation.sh
bash scripts/trace/maze_policy_ablation.sh
```

The default seeds are `1337 2027 4099`. Set `DEVICE`, `SEEDS`,
`ARCHITECTURES`, `MAZE_DATA_DIR`, or `RESULT_ROOT` to select the execution
environment without changing the task conditions. DFS is intentionally not in
the policy ablation until evaluation has a separate valid-target metric.

The baseline and policy matrices include LatentFeedback under the same shared
training protocol as the other multi-pass architectures. This isolates the
architecture. It does not reproduce the paper-specific pass schedule and loss
weights documented in the root README.
