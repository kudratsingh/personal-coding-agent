# Reference-model evaluation

The optional `baseline` dependency extra is used only by `tools/baselines/`; it is not a core training dependency. The HumanEval comparison pins the benchmark revision/content hash and the model revision, loads safetensor weights without remote model code, and executes every candidate through the isolated runner.

The tool uses greedy completion with an explicit token budget and stop strings, records all settings/hashes/outcomes, and compares the resulting rate with the model card's stated value. The card does not fully specify decoding settings, so a mismatch must be investigated rather than described as a reproduced score.

The manually dispatched Reference benchmark workflow uses a standard public-repository CPU runner. It does not provision rented GPUs or upload model caches, weights, or generated programs. Public benchmark provenance and outcome-status records are retained in its console log. Ordinary CI tests use synthetic rows and do not download the reference model.
