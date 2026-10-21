# GPU pilot approval and readiness

Prepared 2026-10-20; approval status corrected 2026-10-21 from the saved owner
instruction. The owner conditionally approved the proposed single-A40 pilot.
Launch awaits the landed tokenizer, explicit funding confirmation, durable
checkpoint storage and operational readiness. **Actual GPU runs: 0; GPU-hours:
0; paid resources launched: 0.** P1-6 and rehearsal execution remain unapproved.

The owner-only experiment record includes the question, proposed single-GPU
machine/provider, parse-checked configuration, source/config/tokenizer identities,
current primary-source price quote, duration/dollar estimate, artifact destination
and numeric stop rules. Detailed operating plans and budgets remain ignored.

The question is whether the from-scratch small model trains stably on the proposed
hardware, its actual synchronized token throughput and peak memory, and whether
full checkpoint copy/resume and enforced shutdown work. No throughput, loss,
resume, upload or watchdog result is claimed by this plan.

The private proposal now binds the replacement **65,536-entry vocabulary** and
**135,285,504-parameter S shape**, with the [current tokenizer identities](tokenizer-freeze.md).
Earlier 32,768-entry configuration identities remain historical.

The [file-training command](walkthroughs/training-loop.md) now accepts an explicit
CPU/CUDA device and synchronizes the chosen CUDA device at step-timing boundaries.
CPU and mocked-CUDA tests cover selection, timing boundaries and token arithmetic;
they do not establish hardware precision, resume correctness or measured GPU speed.

Before launch, resolve the conditional approval's prerequisites and check the
exact quote against the approved run. Remaining readiness includes a pinned
CUDA-enabled environment (the Linux development lock uses CPU PyTorch), hardware
equivalence/resume checks, periodic checkpointing, an external watchdog, frozen
approved shards and verified durable artifact transfers. No machine, payment, credit purchase,
volume or storage account was created. The provider/machine remains proposed,
not selected for the full build or generated-code execution.

The proposed durable destination is a private folder on the owner's laptop SSD.
The owner must confirm that destination and keep it available for a verified
checkpoint/log/config transfer before teardown. Hash receipts and a readability
check are required; the laptop is a separate copy from the rented machine,
not a remote backup. No storage account or credentials in chat are needed for
that choice. The transfer path still needs review and testing.

The [tokenizer stays frozen](tokenizer-freeze.md), [P1-6 awaits approval](full-build-plan.md),
and [Phase 4 remains planning only](rehearsal-plan.md). No provisioning or launch
occurs while the funding, storage and readiness prerequisites remain unresolved.
