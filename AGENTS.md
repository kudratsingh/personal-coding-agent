# Agent instructions

These instructions apply to coding agents and human contributors.

## Start each session

1. Read `README.md`, `docs/status.md`, and the relevant section of `docs/roadmap.md`.
2. Check the working tree before editing; preserve unrelated changes.
3. State the narrow task you will complete and its verification method.
4. Read `docs/development.md` before changing tooling, tests, or project structure.
5. In the owner's checkout, read `private/START_HERE.md` and `private/planning/STATUS.md` if present. They route to the detailed work orders and decisions. A public clone does not need private files: use the public milestone instead.

## Working rules

- Keep each change focused and document user-visible behavior.
- Write or update tests with implementation changes. Do not claim a result that has not been reproduced.
- Treat generated code as untrusted. Do not execute it outside a deliberately isolated runner.
- Do not train on benchmark tests or put secrets, private data, weights, or run artifacts in Git.
- Never launch paid compute, provision cloud resources, or incur a charge without the owner’s explicit approval for that run.
- Keep plans, credentials, raw datasets, experiment notes, and other non-public material in ignored locations such as `private/`, `plans/`, or `docs/internal/`.
- Ask before choices that change scope, licensing, public claims, data sources, or cost.
- Implement core tokenizer, model, optimizer, training, sampling, RL, and metric logic here. Use tensor/autograd primitives; ready-made models, trainers, and optimizers belong only in reference tests or baseline tools. Optimized paths require equivalence tests first.
- Ship a reference test and a plain-language walkthrough with every core component.
- Use openly licensed training data with recorded provenance; verify current source terms before downloading. Never infer downstream rights from a dataset label alone.
- Save run records and artifacts in durable storage before tearing down a compute machine.
- Use one focused branch and pull request per work order after the initial repository bootstrap. Do not merge without passing checks.

## Definition of done

A change is done when its relevant checks pass, documentation is accurate, and no private material is staged for commit. Update `docs/status.md` when a public milestone changes.

## Repository map

```text
AGENTS.md             shared contributor and agent rules
docs/                 public project documentation
private/              ignored owner-only source material and notes
plans/                ignored working plans
docs/internal/        ignored internal documentation
.github/              issue and pull-request templates
```
