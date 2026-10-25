# GPU pilot approval and readiness

Prepared 2026-10-20; approval status corrected 2026-10-21 from the saved owner
instruction. The owner conditionally approved the proposed single-A40 pilot.
Launch awaits the landed tokenizer, explicit funding confirmation, durable
checkpoint storage and operational readiness. **The pilot ran on 2026-10-24: one A40
Pod for 3,679 seconds, $0.60** (see the measured results below). P1-6 is approved
for the S rehearsal build only, after this pilot report.

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
exact quote against the approved run. The 2026-10-22 readiness audit added the
checked-in [A40 specification](../configs/pilot-a40-s.json), an in-Pod
self-destruct timer, a Pod-identity capture tool, a receipt assembler, a
token-level leakage audit, an analytical feasibility calculator and a
finalization reserve; the complete verdict, blockers and exact commands are in
[PILOT_LAUNCH_READINESS.md](../PILOT_LAUNCH_READINESS.md) and
[PILOT_RUNBOOK.md](../PILOT_RUNBOOK.md). On 2026-10-24 the owner selected the
agreed specification, with `torch.compile` off for training and one separate
20-update compiled measurement probe. Runpod offers no provider-enforced Pod
lifetime, so termination remains layered best-effort and the owner must accept
that residual risk explicitly in the readiness receipt. No machine, payment,
credit purchase, volume or storage account was created by that audit. The
provider/machine remains proposed, not selected for the full build or
generated-code execution.

The proposed durable destination is a private folder on the owner's laptop SSD.
The owner must confirm that destination and keep it available for a verified
checkpoint/log/config transfer before teardown. Hash receipts and a readability
check are required; the laptop is a separate copy from the rented machine,
not a remote backup. No storage account or credentials in chat are needed for
that choice. The transfer path still needs review and testing.

The [tokenizer stays frozen](tokenizer-freeze.md), [P1-6 awaits approval](full-build-plan.md),
and [Phase 4 remains planning only](rehearsal-plan.md). No provisioning or launch
occurs while the funding, storage and readiness prerequisites remain unresolved.

## Measured A40 pilot (2026-10-24)

The owner authorized the pilot as merged and accepted best-effort termination.
One Secure Cloud A40 Pod (`CA-MTL-1`, driver 595.91.07, torch 2.9.1+cu128) ran the
agreed specification from commit `bc223f3`; the preflight fix in that commit
(PyTorch reports the GPU UUID without nvidia-smi's `GPU-` prefix and about
578 MiB less memory) was found and fixed on the Pod before training. Lifetime from
the create request to the confirmed delete was **3,679 seconds**; the account
balance fell **$0.60** (the $5 cap and the $1.08 cost stop were never approached).

| Measurement | Result |
| --- | --- |
| A40 preflight | bf16 precision, exact resume and fused-attention equivalence passed on the device |
| Probes (eager, 2 accumulated microbatches) | B=16 **34,767 tok/s**, peak reserved 31.6 GB (chosen, accumulation 8); B=32 out of memory; B=8 32,718; B=4 29,057 |
| Compile probe (child process, B=16) | 20/20 updates, **55,464 vs 35,325 tok/s = 1.57x**; one-off compile about **48 s**; break-even about **36** training updates; peak reserved 27.8 GB |
| Learning-rate screen, held-out loss at step 100 | 3e-4 **7.0996**, 6e-4 **6.9420** (winner), 1.2e-3 **7.0579** |
| Continuation | winner to step **500**: **91,750,400** training tokens, **35,974 tok/s**, peak reserved 31.5 GB |
| Final held-out loss (same 32 windows, Mac CPU float32) | **5.5789**, **1.6387** bits per byte |
| Quoted GPU cost per billion tokens | **$4.56** |
| Final checkpoint | 1,623,507,391 bytes, SHA-256 `fb2118bc...`; fresh-trainer resume test passed on the Pod; on the Mac with matching hash and weights-only load |
| Transfer | 1.5 GiB test pull 143 s (about 91 Mbit/s); 11 bundles verified on the Mac, records/logs archived to the private repository with read-back |
| Shutdown | supervisor delete 204, then 404; watchdog `absent`; 0 Pods and 0 network volumes afterwards |

The live run also exposed runbook gaps that are now fixed: before the supervisor's
heartbeat exists the watchdog must be re-baselined through setup, it must read
`supervisor/heartbeat.json`, known-hosts files need a path without whitespace, and
detached launches must not hold the SSH session open. This is one pilot, not a
throughput guarantee or model-quality claim.
