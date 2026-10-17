# Bounded CPU checkpoint reference

This development check compares uninterrupted training with an interrupted/resumed run on deterministic, authored token IDs. It does not download data or models, execute generated programs, use a GPU, contact a tracker or provision resources. It does not establish coding capability, production stability, source approval or durable backup.

Run the tiny preset first:

```sh
uv run --locked python -m tools.cpu_resume_check --output runs/cpu-resume-tiny
```

The explicit nano preset uses width 256, four layers/four heads, vocabulary 32,768 and full 256-token context: 11,536,640 parameters. It can take considerably longer on a laptop and saves large local checkpoints. Inspect available RAM/disk and keep the output under an ignored location:

```sh
uv run --locked python -u -m tools.cpu_resume_check \
  --size nano --steps 8 --seeds 0 7 23 \
  --precisions float32 bfloat16 --output runs/cpu-resume-nano
```

Only the two fixed presets are accepted, with 2–16 steps and 1–8 distinct seeds. Both use CPU, one intra-op thread, batch 2 and accumulation 2. The synthetic token-ID stream is not a trained/frozen production tokenizer or a corpus compression test. Each precision has its own uninterrupted reference; float32 and bfloat16 trajectories are not expected to be identical to one another.

## What is compared

For each seed/precision, the tool trains a fresh uninterrupted reference; separately trains halfway and saves; then constructs a fresh Trainer, deliberately scrambles Torch RNG and restores the partial checkpoint. Subsequent losses must match exactly. Final model parameters, every optimizer field, per-rank loader/RNG state, step and configuration/input identities must match, and the next random probe must be identical. Resume SHA/source/step metadata must be persisted. The comparison uses zero numerical tolerance and remains active under optimized Python (`-O`). It does not alter training arithmetic or checkpoint formats.

The output directory must not already exist. A local `report.json` is updated after each completed case, with configuration, source/checkpoint hashes, loss sequence, elapsed time and limitations. A caught failure records `failed` and its error type before re-raising; completed earlier cases remain recorded. Abrupt process/machine loss can leave `running`, not a false completed result. Model checkpoints, run records and events remain local. The tool does not upload, delete or verify durable copies of these files; hashes are identity checks, not signatures or durability proof.

The deliberately stopped halfway Trainer retains a `running` training record because it has not reached its configured final step. That saved record is an interruption fixture, not a live-process indicator. Use the top-level check report for the overall result; a completed check also verifies the resumed Trainer's completed record.

`tests/test_cpu_resume_check.py` runs only tiny cases in CI, checks invalid requests/output preservation, failure records and exact-comparison rejection. Actual two-process restart references are separate in `tests/test_distributed_resume.py`; this tool recreates Trainers in one process and is not distributed fault-recovery evidence. The [training walkthrough](walkthroughs/training-loop.md) explains those boundaries and the remaining GPU acceptance work.

## Recorded nano check

On 2026-10-16, clean source revision `f7e5aae3dae7dde964afbb4bc462693462553c7d` completed the nano command above on local CPU with seeds 0, 7 and 23 in both float32 and bfloat16. All six cases matched uninterrupted 8-step training against a 4-step checkpoint plus 4 resumed steps exactly, including losses, model/optimizer state, loader/RNG state and the next random probe. All final model parameters were finite. The report and all 18 training records identify that clean revision; raw reports and checkpoints remain ignored and local, not included in Git.

This is a bounded authored-input checkpoint check, not real-corpus training, a benchmark score, GPU validation, fault recovery or durable-backup acceptance. Some other local verification ran concurrently, so elapsed times are not presented as a normalized throughput or hardware comparison. Phase acceptance remains open.
