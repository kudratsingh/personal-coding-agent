# Model and tokenizer configuration

The model is a from-scratch decoder-only transformer: rotary positions,
pre-layer RMS normalization, causal attention, a 4×-width GELU feed-forward
layer, no biases and no pretraining dropout. Input embeddings and the output
projection **share one weight matrix**. See the [model walkthrough](walkthroughs/transformer.md).

## Vocabulary and special tokens

The owner's **2026-10-20 decision selects 65,536 total entries for compression**,
replacing the October 19 32,768 freeze after correcting the single-leading-space
pre-split. The [freeze evidence](tokenizer-freeze.md) records the artifact and
measurements; no subsequent retraining is authorized implicitly.

| Date | Decision | Replaces |
| --- | --- | --- |
| 2026-10-20 | 65,536 total entries chosen for compression after the leading-space fix; one replacement fit on the same sample | October 19 32,768-entry freeze; historical artifacts retained |

| IDs | Tokens | Meaning |
| --- | --- | --- |
| 0–255 | Raw UTF-8 bytes | Lossless initial byte alphabet |
| 256 | `<\|bos\|>` | Document/conversation boundary |
| 257 / 258 | `<\|user_start\|>` / `<\|user_end\|>` | User message |
| 259 / 260 | `<\|assistant_start\|>` / `<\|assistant_end\|>` | Assistant message |
| 261 / 262 | `<\|python_start\|>` / `<\|python_end\|>` | Python tool call |
| 263 / 264 | `<\|output_start\|>` / `<\|output_end\|>` | Tool output |
| 265–271 | `<\|reserved_0\|>` through `<\|reserved_6\|>` | Seven reserved slots |
| 272–65,535 | 65,264 learned merges | Byte sequences |

Specials are **inside**, not in addition to, the 65,536 entries. The largest
ID is **65,535**: all IDs fit unsigned **16 bits**. The shard boundary test
writes/reads that maximum alongside all 16 specials, including document BOS.
Ordinary input cannot inject special IDs without explicit opt-in.

## Exact parameter counts

For vocabulary `V`, width `W` and `L` layers:
`parameters = V × W + L × (12 × W² + 2 × W) + W`.
The first term is the **single shared embedding/output table**; the other
terms include attention, feed-forward matrices and all RMS gains. Rotary
positions add no parameters. Counts below are computed by
`scratchcoder.model.gpt.expected_parameters`, not rounded size labels.

| Planned shape | Layers | Width | Heads | Context | Shared embedding, 65,536 × width | Total parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nano, full-vocabulary shape | 4 | 256 | 4 | 256 | 16,777,216 | 19,925,248 |
| pilot, original outline shape | 8 | 512 | 8 | 1,024 | 33,554,432 | 58,728,960 |
| S / current pilot proposal | 12 | 768 | 12 | 1,024 | 50,331,648 | 135,285,504 |
| M | 16 | 1,024 | 16 | 2,048 | 67,108,864 | 268,469,248 |
| L | 20 | 1,280 | 20 | 2,048 | 83,886,080 | 477,154,560 |

The selected rehearsal remains **S**, 1B stage-1 English tokens followed by
1B mixed tokens (0.9B Python / 0.1B unseen English). Larger shapes are plans,
not approved runs. The current pilot proposal uses S, **not** the older
eight-layer outline pilot. Token budgets are unchanged by this vocabulary
decision; parameter counts and earlier compute assumptions are not unchanged.

The two checked-in runnable YAML files intentionally remain small synthetic
fixtures, **not production-tokenizer configurations**:

| Actual config | Vocabulary | Layers / width / heads / context | Shared embedding | Total parameters |
| --- | ---: | --- | ---: | ---: |
| [nano.yaml](../configs/nano.yaml) | 512 | 4 / 256 / 4 / 256 | 131,072 | 3,279,104 |
| [smoke.yaml](../configs/smoke.yaml) | 512 | 2 / 32 / 4 / 64 | 16,384 | 41,120 |

Do not pair either fixture with the production vocabulary or silently resize
an existing checkpoint. Training already checks tokenizer/config identities.
