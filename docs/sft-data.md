# Rehearsal SFT data

`tools/sft_data.py` builds the supervised fine-tuning data for the S rehearsal from the [task pool](task-pool.md) and a set of language conversations. Every output is bound by SHA-256 to its inputs and the tokenizer. The rows use the [chat-v1 format](walkthroughs/chat-format.md) and are read by the [SFT batch source](walkthroughs/sft-batching.md). Outputs stay in the ignored `runs/` tree, because they hold reference solutions. Nothing here runs code or spends money.

## Status

- **Code rows** can be built now from the 2026-10-25 pool.
- **The language intake is approved.** On 2026-10-26 the owner accepted the dataset card's Apache-2.0 label plus the dataset paper as the [terms of record](#terms-of-record), for human-written conversation paths only. The `download`, `language` and `build` stages run end to end. No build of record has been made yet.
- **AceCode is flagged** and left out of every source list by default. It is used only with `--include-acecode`.

## Build

```sh
uv run --locked python -m tools.sft_data build \
  --pool-spec configs/task-pool-rehearsal.json \
  --pool-dir runs/task-pool-20261025/pool \
  [--language runs/<intake>/language/conversations.jsonl] \
  --output runs/sft-data-<date>
```

Run it from the main checkout, where the ignored `runs/` tree and the frozen tokenizer are. The pool spec supplies the tokenizer path and its expected identity and file SHA-256.

1. **Sources.** By default only OpenCodeInstruct, TACO and MBPP train tasks are used. `--include-acecode` adds AceCode-87K. Tasks from any other source are counted as excluded.
2. **RL candidates.** Before any SFT row is made, a holdout is taken for RL. It holds 1,000 tasks by default and 2,000 with `--include-acecode`; `--rl-candidates` overrides the number.
   - **Strata:** the tasks are split by source in proportion to each source's train tasks, by largest remainder. Floors come first, then the leftover units go to the largest fractional parts, ties to the source name.
   - **Selection:** within each source, the tasks with the lowest `sha256("rl-candidate-v1|" + task ID)` are taken.
   - **Pool counts:** for the 2026-10-25 pool this gives OpenCodeInstruct 540, TACO 391 and MBPP 69. With AceCode it gives AceCode 1,085, OpenCodeInstruct 494, TACO 358 and MBPP 63.
   - **Use:** these tasks never become SFT rows.
3. **Code rows.** Every other selected task's verified reference is rendered in chat-v1. Code is never subsampled: every task with a reference whose row fits becomes a row. A task without a reference, or whose row has more than 1,024 input/target pairs, is dropped and counted.
4. **Language rows.** Each conversation is rendered in chat-v1, and rows over 1,024 pairs are dropped and counted.
   - **Dev set:** the 300 conversations with the lowest `sha256("language-dev-v1|" + ID)` become the language dev set.
   - **Training rows:** language is sized to the code. The share is measured in assistant-plus-context tokens (every token of a row). If all the remaining conversations together keep the language share of the training tokens at or below 30%, all of them are used. Otherwise they are taken in that hash order until the next one would take the share past 30%. That conversation and every later one are dropped and counted as `over_share`, so the training rows are a prefix of the hash order.
   - **Exact cap:** the test is `language / (language + code) <= 0.3`, compared as exact fractions with 0.3 read as 3/10, so a share of exactly 30% is kept.
   - **No count cap:** there is none by default. `BuildConfig(language_train_max=N)` adds one; conversations past it count as `over_train_max`.
   - **Provenance:** the language stage's `record.json` must sit beside `conversations.jsonl`. The build refuses the conversations unless that record is completed, of this schema, lists their SHA-256 and carries the owner's approval. It then copies the record's provenance into the build record.
5. **Code dev rows.** The pool's dev tasks are rendered with their references for the assistant-only dev loss.

### Outputs

| File | Contents |
| --- | --- |
| `train.jsonl` | SFT rows, code and language, sorted by ID: `id`, `source`, `ids`, `spans` |
| `language_dev.jsonl` | the language dev rows |
| `code_dev.jsonl` | the pool dev tasks as rows |
| `rl_candidates.jsonl` | `id` and `source` of each RL candidate |
| `record.json` | see below |

`record.json` holds:

- the input paths and SHA-256 values, the tokenizer identity and the configuration;
- the language input: the conversations' and the language record's paths and SHA-256 values, and the provenance copied from that record:
  - the dataset, the pinned revision and the license label;
  - the terms of record, the terms review date, the owner's ruling and its date;
  - the SHA-256 of the spec and of the download receipts;
  - the dataset card check;
  - each downloaded file's URL, size and SHA-256;
- code: train tasks per source, the excluded sources, the RL strata, the SFT rows in total and by source, and the drops by reason;
- language counts: seen, over length, dev, available for training, training, and the conversations each cap dropped;
- per domain: rows, tokens and assistant tokens;
- the shares:
  - `language`: the measured language share of all training tokens, which the cap bounds;
  - `code`: the code share of the same;
  - `language_assistant_only`: the language share of the assistant (loss-target) tokens, for information;
  - `language_cap`, `cap_bound` (whether the cap dropped any conversation) and `cap_dropped` (how many);
- each output file's SHA-256, rows and bytes.

## Language intake: OpenAssistant oasst2

The language source is OpenAssistant's oasst2, English human-written paths only. It is taken under five conditions:

1. recorded terms;
2. dropping synthetic, model-written, deleted and failed-review messages;
3. PII filtering;
4. contamination screening;
5. a named exception for the download.

### Terms of record

On 2026-10-26 the owner accepted the dataset card's Apache-2.0 label plus the dataset paper as the terms of record. The approval covers SFT language data from human-written conversation paths only: no message marked `synthetic` and no message carrying a `model_name`.

[configs/sft-language-oasst2.json](../configs/sft-language-oasst2.json) records the ruling as `owner_ruling: approved` and `owner_ruling_on: 2026-10-26`. Its `terms_of_record` holds:

- **The card:** `README.md` at the pinned revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`. It is 10,606 bytes, with SHA-256 `52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31` and git blob ID `ce94775b146d09acf5c5ea31992a9f322e12de76`. Its front-matter line 2 is `license: apache-2.0`.
- **The paper:** Köpf et al., "OpenAssistant Conversations -- Democratizing Large Language Model Alignment", arXiv:2304.07327 v2 (2023-10-31). The card links it. Its abstract states: "We release our code and data under a fully permissive licence."
- **The scope:** the human-written paths rule above.

The spec loader checks every one of these fields by exact keys. The card must be read at the pinned revision, and its label must match `license_label`. A spec without them is refused.

**What the terms check found (2026-10-26).** The label and the paper are the only terms there are:

- **No license file.** The Hugging Face repository `OpenAssistant/oasst2` at the pinned revision lists `.gitattributes`, six `.jsonl.gz` exports, two Parquet files and `README.md`.
- **Only a metadata label.** `README.md` states the license only in its front matter, line 2: `license: apache-2.0`. Its text has no license or contributor terms.
- **The contributor terms grant no license.** The README links the project repository, `LAION-AI/Open-Assistant`. At commit `f1e6ed9526f5817531f3ab85441a40b3671ddccb` its `LICENSE` is Apache 2.0 for the code. The collection portal's terms of use are in `website/src/components/ToS.tsx`; sections 2.1 and 2.2 read:
  > 2.1 "The portal serves as a platform for creating data to train an interactive agent for scientific purposes. All text and prompt generated through the service are used for scientific purposes, in particular for the optimization of the AI."
  >
  > 2.2 "The input of texts on the portal and the subsequent generation of text by the artificial intelligence provided by the portal do not give rise to any works protected by copyright. The user who has entered the text for the generation of the text shall have neither the exclusive rights of use nor any rights of an author to the generated text."
- **Not found:** `oasst-data/README.md` in the project repository has no license statement.

Under the project's data rule, a label without a license file is the owner's decision. The owner accepted the label plus the paper's statement as the terms of record.

### Running the intake

1. **The ruling.** `owner_ruling` is `approved`, `rejected` or null, and `owner_ruling_on` is set or left null with it. `download` and `language` run only when the ruling is `approved`; a date alone, or a rejection, downloads nothing.
2. **Download.** `download` makes these checks before it fetches anything:
   - the pinned revision resolves to itself;
   - the card listed at that revision has the terms of record's size and git blob ID, so the card the owner accepted is the one pinned;
   - the file listing matches;
   - the plan is within `max_download_bytes`.

   It then fetches `2023-11-05_oasst2_ready.trees.jsonl.gz` (54,370,156 bytes, SHA-256 `7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b`) over HTTPS. Each file is size- and SHA-256-checked. `download-receipts.json` records the spec's SHA-256, the source, the ruling, the terms of record, the card check and each file's URL, size and SHA-256.

   ```sh
   uv run --locked python -m tools.sft_data download \
     --language-spec configs/sft-language-oasst2.json --directory runs/oasst2-<date>
   ```

3. **Adapt and screen.** `language` adapts and screens the trees into `language/conversations.jsonl` and `language/record.json`. The record's `provenance` carries the receipts' source, ruling and terms, the receipts' SHA-256 and each downloaded file. The stage needs the `evaluation` extra to read ARC-Easy.

   ```sh
   uv run --locked --extra evaluation python -m tools.sft_data language \
     --language-spec configs/sft-language-oasst2.json --directory runs/oasst2-<date> \
     --pool-spec configs/task-pool-rehearsal.json --pool-dir runs/task-pool-20261025/pool
   ```

**Eligible messages.** A message can appear in a conversation only if all of these hold:

- its language is `en`;
- it is human-written: `synthetic` is false (else `synthetic`), and `model_name` is null or empty (else `model_written`);
- `deleted` is false;
- `review_result` is true;
- its `pii` label value, if present, is zero;
- its text is not empty.

A message missing any of these fields counts as `missing_metadata`. That includes `synthetic` and `model_name`, so a message without either is not trusted to be human-written. A message with both marks counts once, as `synthetic`. Every message reason is counted, zero included.

In the pinned ready-trees export, all 135,174 messages carry both fields, and none is marked synthetic or carries a model name; the same holds for its 64,513 English messages. The checks stay, so another export cannot bring model-written text in.

Labels must be the README's mapping form: each label name maps to an object with a numeric `value`. Labels in any other shape, such as lists or columnar `{"name": [...], "value": [...]}` objects, also count as `missing_metadata`. A message with no labels, or no `pii` label, stays eligible, as the PII condition filters on the PII labels. Such messages are counted: `message:eligible_no_pii_label` over all eligible messages, and `kept_messages_no_pii_label` and `tree:kept_with_no_pii_label` over the kept conversations. The manual sample scan should cover them.

**Choosing a conversation.** Only trees in `ready_for_export` state with an eligible English prompter root are used. From each tree the adapter follows the best-ranked eligible reply at every turn: rank 0 first, unranked replies last, message ID breaking ties. A chosen path therefore never contains an ineligible message: if the best-ranked reply is model-written, the walk takes the next eligible reply, and with none it ends. The conversation is the longest prefix of that path that ends on an assistant turn and renders to at most 960 chat-v1 tokens. A tree whose first exchange is over 960 tokens is dropped. One conversation comes from each tree.

**Screens, in order:**

1. any e-mail address in any message;
2. a word n-gram match against HumanEval, HumanEval+ and MBPP+ prompts, solutions and tests, ARC-Easy questions and choices, and the pool's dev and test tasks. This is the pool's 13-gram screen with at least three words per n-gram;
3. a first user turn that near-duplicates a benchmark or holdout prompt, at 5-word-shingle Jaccard of 0.85 or more;
4. a first user turn that near-duplicates, by the same measure, the first user turn of a conversation already kept (`repeated_prompt`). The earliest conversation in input order is kept. As in the pool's own screen, this keeps one copy of a repeated prompt, so the language dev set and the training rows never share an opening prompt.

The record counts every message by reason, every dropped tree by reason, each screen, the exchanges per conversation and the total tokens.

## Limitations

- **Terms.** The terms of record are a metadata label and a sentence in the paper's abstract, accepted by the owner. There is no license file or contributor grant.
- **Fields.** The adapter assumes the README's message fields and mapping-form labels. Missing fields and labels in another shape show up as `missing_metadata` counts, not as silent passes.
- **PII.** The e-mail screen and the PII label do not find every kind of personal data, and messages that no reviewer labelled for PII are kept (and counted). A manual sample scan is still needed before training.
- **Loss-target share.** The 30% cap is on all tokens. Code rows carry long prompts and short programs, so language takes a larger share of the assistant (loss-target) tokens than of all tokens. The record reports both.
- **Code rows.** They inherit the pool's limitations, including its unchecked seed provenance.
