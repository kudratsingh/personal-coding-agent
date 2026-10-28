# Coding agent offline observability

This first slice is a **read-only, offline** training and validation dashboard built from existing local `runs/<run_id>/record.json` and `events-rank0.jsonl` or `events.jsonl`. Nothing in this change touches the trainer, checkpointing, model, sandbox, RunPod monitoring or paid infrastructure.

Run from the repository root:

```sh
uv run --locked python -m observability.dashboard \
  --runs-root runs \
  --output runs/observability/index.html
# Open the generated HTML in your browser.
```

The dashboard shows line plots for existing scalar training metrics (loss, learning rate, gradient norm, throughput) and domain-specific held-out validation, plus basic run status, truncated commit and recorded cost. Use the legend to compare experiments. It does not assert benchmark coding capability and does not modify log files. A partially appended JSONL row is skipped. Nonfinite and nonnumeric values, sensitive fields, and arbitrary log text are ignored.

The generated HTML is self-contained and does not load CDNs or upload metrics. It is placed in the ignored `runs/` directory by default, and **must not be committed or shared without review**. Only regular files in immediate run directories are read; symlinked run folders and symlinked event files are rejected. Maximum 40 runs, 120 metric types and 2,400 sampled points per metric per run; files over 32 MiB are skipped.

**Not in scope:** W&B or MLflow uploads, automatic tracking hooks in existing training processes, real-time provider billing, tracing untrusted code execution, GPUs, metrics ingestion services, or deployment. Future additions should be separate PRs and remain opt-in.
