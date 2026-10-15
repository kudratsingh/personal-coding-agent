# Architecture overview

The intended system is a compact coding agent built in layers:

```text
licensed data -> tokenizer -> causal language model -> training and checkpoints
                                              |
                                              v
task evaluation <- isolated code runner <- agent repair loop <- generation engine
```

## Design boundaries

- **Core learning path:** tokenizer, model, optimizer, and training loop are owned by this project and validated against reference behavior where possible.
- **Safety boundary:** generated programs are treated as untrusted and run only in a constrained execution environment.
- **Data boundary:** evaluation material is separated from training material; data provenance and licensing are recorded before use.
- **Reporting boundary:** reported numbers come from a documented harness and include enough context to interpret them.

Implementation details will be added as the corresponding components are built and reviewed.
