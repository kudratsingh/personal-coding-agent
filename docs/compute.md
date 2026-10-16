# Compute estimates

Use `uv run --locked python tools/cost.py --help` for a two-stage estimate. All prices and throughput values must be supplied explicitly and verified before a real run; the tool does not provision hardware.

GPU-hours equal tokens divided by per-GPU tokens/second and 3,600, multiplied by an optional slowdown and layer-parameter ratio. Wall time divides GPU-hours by the GPU count. Cost multiplies GPU-hours by the supplied per-GPU hourly rate. Parallel scaling is an estimate, not a measurement.

Operational budgets, approvals, and the detailed spend ledger remain private. Every paid run still requires specific owner approval, working save/resume, durable artifact storage, and tested shutdown controls.

## Workflow runner review

`make workflow-check` (also in `make check`) reads workflow YAML without executing it. The reviewed runner labels are `ubuntu-latest` and `macos-latest`. Literal labels and simple literal `${{ matrix.os }}`-style runner matrices are supported. Unknown labels, runner groups/arrays, dynamic expressions, reusable jobs and matrices with `include`/`exclude` require review rather than being silently accepted. Authored tests check accepted declarations, rejection cases, symlinks and safe parse-error reporting. The tool does not provision or modify runners.

Run this locally before pushing workflow changes. CI is a second check, not a payment firewall: an unreviewed job may already have dispatched before CI reports the violation. The validator does not audit action internals, arbitrary shell commands, cloud services, artifact/cache storage, account billing or repository visibility. A passing result never authorizes a charge.

As reviewed on 2026-10-16, [GitHub's billing policy](https://docs.github.com/en/billing/concepts/product-billing/github-actions) states that standard hosted runners are free for public repositories, while larger runners are charged even for public repositories. Visibility, storage limits and current terms must still be checked separately before running workflows. Do not expand the runner policy or use paid resources without explicit owner approval for the specific run.
