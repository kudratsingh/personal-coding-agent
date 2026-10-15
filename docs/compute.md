# Compute estimates

Use `uv run --locked python tools/cost.py --help` for a two-stage estimate. All prices and throughput values must be supplied explicitly and verified before a real run; the tool does not provision hardware.

GPU-hours equal tokens divided by per-GPU tokens/second and 3,600, multiplied by an optional slowdown and layer-parameter ratio. Wall time divides GPU-hours by the GPU count. Cost multiplies GPU-hours by the supplied per-GPU hourly rate. Parallel scaling is an estimate, not a measurement.

Operational budgets, approvals, and the detailed spend ledger remain private. Every paid run still requires specific owner approval, working save/resume, durable artifact storage, and tested shutdown controls.
