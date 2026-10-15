# Evaluation metrics

For `n` sampled attempts and `c` successes on one task, unbiased pass@k is `1 - choose(n-c,k)/choose(n,k)`. Compute the failure probability as a product and use log/expm1 arithmetic to avoid cancellation. If fewer than k failures exist, every k-subset contains a success and the answer is one.

For four attempts with one correct answer and k=2, half of the six possible subsets contain the success, so pass@2 is 0.5. Tests exhaustively enumerate all subsets up to n=8 and compare the estimator with exact rational arithmetic.

Mean pass rate gives each task equal weight, even when sample counts differ. Greedy pass@1 is reported only for greedy single-attempt decoding. Multiple-choice accuracy is the fraction of aligned answers that match. Bits per byte divides total negative log likelihood in nats by `log(2)` and the original UTF-8 byte count.

For a paired bootstrap, compute each task's after-minus-before difference, resample the same task indices for both methods, and take percentile bounds over seeded resamples. This preserves task difficulty pairing; it does not replace multi-seed measurements or establish statistical power for tiny samples.

Questions: why unbiased pass@k (avoid treating observed successes as independent success probabilities); why task weighting (prevent oversampled tasks dominating); why pairing (remove task-difficulty variation); why a seed (reproduce resamples); why original bytes (compare across tokenizers).
