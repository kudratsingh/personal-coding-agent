# Generation and the key-value cache

For a prompt, every transformer layer computes keys and values `[B,H,T,D]`. Subsequent tokens need new queries, keys, and values only for their own positions. Cache the earlier keys/values, append the new ones, and rotate the new query/key at the correct absolute offset. Mask against absolute positions, not a fresh triangle starting at zero.

For a length-five prompt followed by one token, the new query is `[B,H,1,D]`; cached keys/values grow to `[B,H,6,D]`. Its score row can see all six positions. Tests compare every cached logit with a full-sequence forward pass.

Temperature zero chooses the largest logit. Positive temperature scales logits before top-k and nucleus filtering; seeded categorical sampling then chooses an allowed token. Batched stop tokens finish each row independently. Generation stops at the context boundary rather than silently dropping positional history.

Sampling requires nonempty dense floating logits shaped `[V]` or `[B,V]`, finite nonnegative temperature, a positive integer top-k when supplied, and finite top-p in `(0,1]`. Invalid controls/logits fail before drawing random numbers. The ordinary float32 scaling path is retained exactly. If extreme finite inputs overflow that scaling, a float64 fallback centers before division for small temperatures, or divides before centering for large ones. This keeps at least one finite zero per row; negative infinity can represent negligible mass. It does not clamp the requested temperature or change the top-k/top-p protocol. The fallback requires device support for float64 tensor arithmetic; GPU precision remains separately unvalidated.

`tests/test_sampling_integrity.py` checks extreme categorical probabilities against hand-computed weights, including subnormal temperatures and large float64 logits. It also checks ordinary seeded draws against the pre-fix expression across supported dtypes, temperatures and filters, plus input preservation and rejection without RNG consumption. These are numerical/component tests, not benchmark parameter tuning or new model scores.

Answer scoring sums only continuation-token log probabilities. A padded batch is safe because causal attention prevents later padding from changing earlier scores. A slow per-prefix reference verifies this indexing.

For base-model code evaluation, prompt whitespace is part of the protocol: `rstrip()` removes trailing whitespace before encoding `[1,T]` prompt IDs. The completion retains its initial newline/indentation, and the scored source is stripped prefix plus completion up to the first original stop string. The reference model and project checkpoint generator share this rule; no extra public-test instruction is appended in base mode. Authored tests observe the actual tokenizer/generation prefix and persisted report/scorecard metadata. Legacy instruction-style generation remains explicitly selectable and separately recorded. See the [evaluation policy](../evaluation.md) for comparability limits.

Questions: why cache keys/values but not earlier queries (new tokens do not use them); why offset rotary positions (preserve absolute relative geometry); why sum answer log probabilities (sequence likelihood); why deterministic greedy batch parity (detect hidden batch dependence); why context limits (cache growth is bounded by the configured model).
