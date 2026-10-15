# Evaluation policy

No benchmark result is available yet. Future reports must identify the evaluated checkpoint, source revision, dataset revision, split, decoding settings, seed, execution limits, and metric definition.

Report uncertainty across tasks or seeds as appropriate, and include negative results. Compare checkpoints on the same task set and settings; distinguish development tuning from held-out reporting.

Validate the harness against simple known answers and a reference baseline before trusting model scores. Generated code must run through the isolated runner, with hidden tests kept out of model-visible feedback.

The planned scorecard compares pretraining stages, supervised tuning, test-based optimization, and the repair loop. Scorecard rows will be added when supported by reproducible evidence.
