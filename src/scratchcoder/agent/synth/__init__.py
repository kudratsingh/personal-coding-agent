"""Synthetic agent-training repositories (AGENT-SFT-1 data, plan section 4.2 option a).

A **template** describes a small stdlib-only repository family: two or three source modules,
a public test module, one target function with several bug kinds (the original buggy or stub
state), and a per-function behaviour model written in project code (the *spec function*) that
computes every expected value as a literal. A **seed** picks identifier and module names,
constants and structural variants, so one template yields many distinct repositories.

Nothing in this package executes emitted source. The host parses text with :mod:`ast` and
computes expected values with the spec functions; the emitted repositories, their reference
fixes, the AST mutants used as wrong attempts and any independent oracle run only inside the
sandbox, through :func:`scratchcoder.eval.harness.isolated_execution`.

Modules:

- :mod:`.template`: the template and function model, rendering and bug application.
- :mod:`.mutation`: AST mutation operators and negative controls.
- :mod:`.catalog`: the templates and their families.
- :mod:`.instances`: building one instance (a harness v0 case directory) and its static checks.
- :mod:`.independent`: the independent-oracle file format and its sandbox embedding.
- :mod:`.ledger`: the persisted sandbox-call ledger and its cap.
- :mod:`.rows`: driving the harness loop with a scripted policy and rendering chat-v1 rows.
- :mod:`.contamination`: the status-only agent-case contamination check.
- :mod:`.mix`: mixing agent and replay rows by assistant-loss tokens.
"""

GENERATOR_VERSION = "agent-synth-v1"
