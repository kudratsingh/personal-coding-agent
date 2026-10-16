# Reference-model evaluation

The optional `baseline` dependency extra is used only by `tools/baselines/`; it is not a core training dependency. The HumanEval comparison pins the benchmark revision/content hash and the model revision, loads safetensor weights without remote model code, and executes every candidate through the isolated runner.

The tool uses greedy completion with an explicit token budget and stop strings, records all settings/hashes/outcomes, and compares the resulting rate with the model card's stated value. The card does not fully specify decoding settings, so a mismatch must be investigated rather than described as a reproduced score.

The manually dispatched Reference benchmark workflow uses a standard public-repository CPU runner. It does not provision rented GPUs or upload model caches, weights, or generated programs. Public benchmark provenance and outcome-status records are retained in its console log. Ordinary CI tests use synthetic rows and do not download the reference model.

## Observed comparison

The [initial recorded run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38780757269) completed all 164 tasks with greedy decoding and a 256-token completion budget: **1/164 passed (0.61%)**, versus the card's stated 7.84%. This is not a reproduced score. The card marks its score unverified and does not specify the full decoding protocol. That uncertainty is a limitation, not an established explanation for the gap.

Subsequent runs first require every canonical solution to pass the same hidden tests through isolation. `--canonical-only` runs this check without loading model weights. A failed canonical check stops model benchmarking. Model reports additionally retain allowlisted failure categories, never raw exception text, generated programs, or hidden tests. Token-budget comparisons must be recorded separately; do not tune on the benchmark and present the best result as an independent validation.
