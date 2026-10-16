# Bounded CPU checkpoint reference

This development check compares uninterrupted training with an interrupted/resumed run on deterministic, authored token IDs. It does not download data or models, execute generated programs, use a GPU, contact a tracker or provision resources. It does not establish coding capability, production stability, source approval or durable backup.

Run the tiny preset first:

```sh
uv run --locked python -m tools.cpu_resume_check --output runs/cpu-resume-tiny
```

The explicit nano preset uses width256, four layers/four heads, vocabulary32,768 and full256-token context:11,536,640 parameters. It can take considerably longer on a laptop and saves large local checkpoints. Inspect available RAM/disk and keep the output under an ignored location:

```sh
uv run --locked python -u -m tools.cpu_resume_check \
  --size nano --steps 8 --seeds 0 7 23 \
  --precisions float32 bfloat16 --output runs/cpu-resume-nano
```

Only the two fixed presets are accepted, with2–16 steps and1–8 distinct seeds. Both use CPU, one intra-op thread, batch2 and accumulation2. The synthetic token-ID stream is not a trained/frozen production tokenizer or a corpus compression test. Each precision has its own uninterrupted reference; float32 and bfloat16 trajectories are not expected to be identical to one another.

## What is compared

For each seed/precision, the tool trains a fresh uninterrupted reference; separately trains halfway and saves; then constructs a fresh Trainer, deliberately scrambles Torch RNG and restores the partial checkpoint. Subsequent losses must match exactly. Final model parameters, every optimizer field, per-rank loader/RNG state, step and configuration/input identities must match, and the next random probe must be identical. Resume SHA/source/step metadata must be persisted. The comparison uses zero numerical tolerance and remains active under optimized Python (`-O`). It does not alter training arithmetic or checkpoint formats.

The output directory must not already exist. A local `report.json` is updated after each completed case, with configuration, source/checkpoint hashes, loss sequence, elapsed time and limitations. A caught failure records `failed` and its error type before re-raising; completed earlier cases remain recorded. Abrupt process/machine loss can leave `running`, not a false completed result. Model checkpoints, run records and events remain local. The tool does not upload, delete or verify durable copies of these files; hashes are identity checks, not signatures or durability proof.

`tests/test_cpu_resume_check.py` runs only tiny cases in CI, checks invalid requests/output preservation, failure records and exact-comparison rejection. Actual two-process restart references are separate in `tests/test_distributed_resume.py`; this tool recreates Trainers in one process and is not distributed fault-recovery evidence. The [training walkthrough](walkthroughs/training-loop.md) explains those boundaries and the remaining GPU acceptance work.
