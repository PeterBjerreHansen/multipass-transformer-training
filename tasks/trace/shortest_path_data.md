# Fixed shortest-path data

Shortest-path training reads finite compiled datasets. It does not generate
graphs during training, validation, evaluation, or resume.

Generate and verify the two full artifacts from the repository root:

```bash
python3 -m tasks.trace.shortest_path_data generate \
  --dataset main --output-dir data/shortest_path/main --workers 8
python3 -m tasks.trace.shortest_path_data generate \
  --dataset easy --output-dir data/shortest_path/easy --workers 8
python3 -m tasks.trace.shortest_path_data verify \
  --data-dir data/shortest_path/main
python3 -m tasks.trace.shortest_path_data verify \
  --data-dir data/shortest_path/easy
```

Both datasets use root seed `20260813`. `main` contains 5,000,000 training
examples and 8,192 examples in each held-out split. `easy` contains 1,000,000
training examples and the same held-out counts. Path lengths are balanced to
within one example in every split. Candidate seeds make the bytes independent
of `--workers`. Labeled graph problems are deduplicated across all splits;
edge serialization order is not part of the duplicate identity.

Each artifact contains `manifest.json`, `vocab.json`, `summary.json`, and four
NumPy arrays per split. Tokens, sequence lengths, and prompt lengths use
`uint8`. Example IDs use fixed-width 24-byte strings. Token rows are padded to
the manifest's maximum sequence length. The loader memory-maps these files,
checks the manifest contract, and uses dynamic in-batch padding and suffix-only
target masking.

Training samples the `train` split with replacement through the checkpointed
training RNG. Checkpoint selection and diagnostics use deterministic,
no-replacement examples from `validation`. Post-training `eval_trace` uses
deterministic, no-replacement examples from `test`. Saved runs record the
dataset ID and fail to resume if the artifact identity changes.
