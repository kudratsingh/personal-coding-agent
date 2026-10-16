# Multiple-choice evaluation walkthrough

The checkpoint evaluator uses our tokenizer, transformer, and [answer scorer](kv-cache.md). It does not generate or execute a program. Each task has a question, aligned label/text arrays, one answer label, immutable source revision, license, and held-out split. The correct label is kept outside the model prompt.

The prompt lists the question and all options, ending with `Answer:`. Each candidate is a space-prefixed label, such as ` A`. We sum that continuation's token log probabilities, not the prompt's probabilities, and select the highest score. Multi-token labels are not length-normalized; ties select the first listed option. These choices are recorded in the report and must remain fixed for comparisons.

Accuracy is the fraction of tasks answered correctly. A seeded task-bootstrap interval describes uncertainty across the supplied tasks; it does not establish generalization beyond them. A question that cannot fit in the configured context counts as incorrect with a `context_limit` status. It is never silently shortened or dropped. Invalid/nonfinite scores abort reporting.

`tests/test_generation.py` compares the batched answer scorer against a slow per-answer full-model reference. `tests/test_choices.py` checks hand-computed accuracy, ties, answer-key separation, schema validation, provenance, context limits, and an end-to-end synthetic checkpoint CLI run. Synthetic test scores are not ARC benchmark results.

Run an explicitly prepared local suite:

```sh
uv run --locked scratchcoder eval-choice --checkpoint runs/example/checkpoint.pt --tokenizer runs/example/tokenizer.json --tasks runs/example/choices.jsonl --output runs/example/choice-eval
```

JSONL fields are `id`, `question`, `labels` (array), `choices` (array), `answer`, `source`, `source_revision`, `license`, and `split` (`dev`, `test`, or `eval`). `adapt_arc_row` converts the explicit question/choices/answerKey schema only when given validated evaluation-only provenance. It does not download data or establish source rights. Reports and scorecards remain ignored local artifacts.
