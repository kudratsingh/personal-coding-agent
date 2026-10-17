# ADR 0010: disclosed HumanEval+ task-32 exclusion

Status: accepted. Date: 2026-10-17.

## Decision

Treat HumanEval/32 as a known numerical-tolerance failure and close its investigation. Current project HumanEval+ scoring uses the pinned 164-task OriginFmt release with exactly task 32 excluded, for a denominator of **163**. Preserve all remaining tests, solutions and input bytes. No additional oracle correction or numerical diagnostic is authorized.

An explicit `--exclude-find-zero` switch selects this policy and cannot be combined with the historical correction switch from [ADR 0007](0007-find-zero-oracle-correction.md). Original release intake still verifies all 164 tasks; exclusion precedes both canonical and model execution. The canonical preflight must pass all 163 retained tasks.

Every canonical/checkpoint report and scorecard records the original and scored counts, excluded task ID, reason, release identity and effective task hash. The scope explicitly states that this is **not the full suite**. Existing full-suite results remain unchanged; subset and full-suite scores must not be silently compared.

## Verification and consequences

Authored fixtures verify zero executions/generations of task 32, unchanged retained tests, the 163-task denominator and disclosure in both saved report kinds and scorecards. An isolated free-runner evaluation supplies actual subset scores, recorded in [the evidence page](../evalplus.md). Low synthetic-only nano scores establish integration, not useful coding capability. This decision neither changes the original HumanEval reference protocol nor authorizes paid compute.
