# Public roadmap

The roadmap describes direction, not a promise of schedule or performance. Detailed implementation plans, experiment operations, and budgets remain private.

| Phase | Outcome |
| --- | --- |
| 0. Foundation | Repository layout, development tooling, tests, configuration, and reproducible run records. |
| 1. Data and tokenizer | Licensed data intake, contamination controls, and a byte-level tokenizer with reference tests. |
| 2. Model and training | A compact causal model, optimizer, training loop, checkpoints, and local smoke tests. |
| 3. Evaluation and safety | Evaluation harness, task format, metrics, and an isolated execution environment. |
| 4. End-to-end rehearsal | A small reproducible pipeline run that validates integration and reporting. |
| 5. Pretraining | Documented base-model training runs and measured language/code trade-offs. |
| 6. Post-training | Supervised tuning and test-based optimization, evaluated with appropriate controls. |
| 7. Agent experience | An iterative code-repair loop and a local demonstration interface. |
| 8. Release | Model card, data and evaluation documentation, reproducibility material, and release decision. |

## Phase gates

Each phase must have passing relevant tests, an updated public status, and a documented review before the next phase begins. Work that needs paid compute, private data, or a license decision is approved separately by the owner.
