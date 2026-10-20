# Full-build planning status

The P1-6 estimate was prepared on 2026-10-19 as an owner-only experiment record. It covers download/raw/filtered sizes, token counts per source, CPU-machine wall-hours, monthly storage, assumptions and stop conditions for the amounts in the data plan. **It is not approved, changed or deferred by the owner yet. Full builds: 0; paid runs: 0.**

The estimate uses the [measured sample retention](real-samples.md) and [frozen-tokenizer counts](tokenizer-freeze.md), rather than a dataset's advertised token count. Python retained **9.66388% of sample text bytes** under the explicit file-license and content filters. Applying that prefix rate to the planned approximately 64 GB source gives a conditional **1.7405B unique-token forecast**, not a full-source measurement. Larger one-pass data budgets are not demonstrated attainable under this policy; do not assume additional named files, repeat epochs, new sources or relaxed license rules.

Task-source retention, context fit and verification time remain assumptions, with **0 final trained-pool admissions**. Final held-out-pool exclusion, component rights and production host approval are unresolved. CPU verification can dominate the build; pure tokenization time is not the total estimate. Ordered prefixes do not provide representative full-source confidence intervals.

Storage prices were checked against the current [Cloudflare R2 Standard pricing](https://developers.cloudflare.com/r2/pricing/) as a planning reference only. No storage account, bucket, upload, quota change, CPU/GPU machine or paid service was created. No GPU pilot was prepared or launched. The private record retains operational quantities and budget details; public clones need no private files to understand these limitations.

Next owner decision is approve/change/defer the estimate and resolve source capacity before any full build, including a laptop build. See [G1 evidence](gates/g1-data-tokenizer.md); tokenizer freezing alone does not satisfy that owner-decision condition.
