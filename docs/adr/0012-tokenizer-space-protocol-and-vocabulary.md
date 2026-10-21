# ADR 0012: leading-space BPE and 65,536-entry vocabulary

Status: accepted by owner, 2026-10-20. Replaces the October 19 tokenizer freeze
after the authorized replacement fit completed. Further retraining requires
fresh approval.

## Cause and decision

The old pre-split isolated all whitespace from words. The diagnostic sentence
`the quick brown fox jumps over the lazy dog` encoded as **18 project tokens**
versus **9 GPT-2 tokens**; eight standalone spaces proved the suspected bug.
Attach a single ASCII leading space to the following nonwhite chunk. Keep
multi-character whitespace runs, tabs and newline runs separate for code.
This borrows the single-space behavior, not GPT-2's entire regex.

The owner selected **65,536 total vocabulary entries for compression**, including
all 256 raw bytes and 16 special slots: **65,264 learned merges**, largest ID
**65,535**. A uint16 shard boundary test covers that ID and every special slot.
Embedding/output weights remain tied: update all [model counts](../model.md)
rather than adding an independent output matrix.

## Reproducibility and compatibility

New training emits tokenizer serialization format **2**, which binds the split
protocol into the identity hash. Format **1** loads with its original split rule,
serialization bytes and hash; old checkpoints are not reinterpreted. Reference
and cached encoders use the artifact's version, not a global replacement regex.
Synthetic slow-recount, independently trained library and fast-path comparisons
exercise the corrected rule before the real fit.

Exactly one replacement production fit is authorized, on the original
**96,223,042 UTF-8 bytes per domain**, identical length-framed corpus hashes and
original held-out English/Python documents. The optional 32,768 fit is omitted.
The completed fit's merge SHA-256 is
`b0297f478dd5172a690752ceac8f997fccbd95e36e5787ca56a2c555083c7b7b`;
the dated owner decision is D-24. Old artifacts are preserved.
[Freeze evidence](../tokenizer-freeze.md) reports
held-out token totals beside pinned GPT-2 and the harmonic 50/50-byte-mixture
BPT, `2 / (1 / English_BPT + 1 / Python_BPT)`.

This replacement does not approve a corpus build or model training. The owner
conditionally approved one A40 pilot, but funding confirmation, landed tokenizer,
durable storage and the pilot's operational prerequisites remain launch gates.
