# Rehearsal SFT data

`tools/sft_data.py` builds the supervised fine-tuning data for the S rehearsal from the [task pool](task-pool.md) and a set of language conversations. Every output is bound by SHA-256 to its inputs and the tokenizer. The rows use the [chat-v1 format](walkthroughs/chat-format.md) and are read by the [SFT batch source](walkthroughs/sft-batching.md). Outputs stay in the ignored `runs/` tree, because they hold reference solutions. Nothing here runs code or spends money.

## Status

- **Code rows** can be built now from the 2026-10-25 pool.
- **The language intake is stopped.** The terms check found no license file and no license grant in the contributor terms; see [the terms finding](#language-intake-openassistant-oasst2). Nothing was downloaded. The `download` stage refuses until the owner's ruling is recorded in the spec.
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
3. **Code rows.** Every other selected task's verified reference is rendered in chat-v1. A task without a reference, or whose row has more than 1,024 input/target pairs, is dropped and counted.
4. **Language rows.** Each conversation is rendered in chat-v1, and rows over 1,024 pairs are dropped and counted.
   - **Dev set:** the 300 conversations with the lowest `sha256("language-dev-v1|" + ID)` become the language dev set.
   - **Training rows:** the rest, in that hash order, become training rows until either cap is reached: the language share of assistant-plus-context tokens (every token of a row) would pass 30%, or 3,000 rows. The first row past a cap ends the selection, so the training rows are a prefix of the hash order.
   - **Shortfall:** if the conversations run out first, the measured share is lower, and the record says the target was not met.
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
- train tasks per source, the excluded sources, the RL strata and the drops by reason;
- language counts: seen, dev, train and each cap;
- per domain: rows, tokens and assistant tokens;
- the measured shares, by all tokens and by assistant tokens only;
- each output file's SHA-256, rows and bytes.

## Language intake: OpenAssistant oasst2

The proposed language source is OpenAssistant's oasst2, English human-written paths only. It needs the owner's approval of five conditions:

1. recorded license or contributor terms;
2. dropping synthetic, deleted and failed-review messages;
3. PII filtering;
4. contamination screening;
5. a named exception for the download.

**The terms finding (2026-10-26).** It is recorded in [configs/sft-language-oasst2.json](../configs/sft-language-oasst2.json).

- **No license file.** The Hugging Face repository `OpenAssistant/oasst2` at revision `179dd21fc55192153d94adb0e0ce8f69e222bf75` lists `.gitattributes`, six `.jsonl.gz` exports, two Parquet files and `README.md`.
- **Only a metadata label.** `README.md` (10,606 bytes, SHA-256 `52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31`) states the license only in its front matter, line 2: `license: apache-2.0`. Its text has no license or contributor terms.
- **The contributor terms grant no license.** The README links the project repository, `LAION-AI/Open-Assistant`. At commit `f1e6ed9526f5817531f3ab85441a40b3671ddccb` its `LICENSE` is Apache 2.0 for the code. The collection portal's terms of use are in `website/src/components/ToS.tsx`; sections 2.1 and 2.2 read:
  > 2.1 "The portal serves as a platform for creating data to train an interactive agent for scientific purposes. All text and prompt generated through the service are used for scientific purposes, in particular for the optimization of the AI."
  >
  > 2.2 "The input of texts on the portal and the subsequent generation of text by the artificial intelligence provided by the portal do not give rise to any works protected by copyright. The user who has entered the text for the generation of the text shall have neither the exclusive rights of use nor any rights of an author to the generated text."
- **Not found:** `oasst-data/README.md` in the project repository has no license statement.

No license or contributor grant beyond the metadata label was found, and the portal terms describe the text as collected for scientific purposes. Under the project's data rule, custom terms and an unclear license are the owner's decision. The intake therefore stops before any download.

### Once the owner rules

1. **Record the ruling.** The owner sets `owner_ruling_on` in the spec to the ruling date.
2. **Download.** `download` checks that the pinned revision resolves to itself and that the file listing matches. It refuses a plan above `max_download_bytes`, then fetches `2023-11-05_oasst2_ready.trees.jsonl.gz` (54,370,156 bytes) over HTTPS. Each file is size- and SHA-256-checked, and the receipts go in `download-receipts.json`.

   ```sh
   uv run --locked python -m tools.sft_data download \
     --language-spec configs/sft-language-oasst2.json --directory runs/oasst2-<date>
   ```

3. **Adapt and screen.** `language` adapts and screens the trees into `language/conversations.jsonl` and `language/record.json`. It needs the `evaluation` extra to read ARC-Easy.

   ```sh
   uv run --locked --extra evaluation python -m tools.sft_data language \
     --language-spec configs/sft-language-oasst2.json --directory runs/oasst2-<date> \
     --pool-spec configs/task-pool-rehearsal.json --pool-dir runs/task-pool-20261025/pool
   ```

**Eligible messages.** A message can appear in a conversation only if all of these hold:

- its language is `en`;
- `synthetic` is false and `deleted` is false;
- `review_result` is true;
- its `pii` label value, if present, is zero;
- its text is not empty.

A message missing any of these fields counts as `missing_metadata`. Labels must be the README's mapping form.

**Choosing a conversation.** Only trees in `ready_for_export` state with an eligible English prompter root are used. From each tree the adapter follows the best-ranked eligible reply at every turn: rank 0 first, unranked replies last, message ID breaking ties. A chosen path therefore never contains an ineligible message. The conversation is the longest prefix of that path that ends on an assistant turn and renders to at most 960 chat-v1 tokens. A tree whose first exchange is over 960 tokens is dropped. One conversation comes from each tree.

**Screens, in order:**

1. any e-mail address in any message;
2. a word n-gram match against HumanEval, HumanEval+ and MBPP+ prompts, solutions and tests, ARC-Easy questions and choices, and the pool's dev and test tasks. This is the pool's 13-gram screen with at least three words per n-gram;
3. a first user turn that near-duplicates a benchmark or holdout prompt, at 5-word-shingle Jaccard of 0.85 or more.

The record counts every message by reason, every dropped tree by reason, each screen, the exchanges per conversation and the total tokens.

## Limitations

- **No language data yet.** Until the owner rules, the build can make code rows only.
- **Untested on the real file.** The adapter assumes the README's message fields and mapping-form labels. A different real format shows up as `missing_metadata` counts, not as silent passes.
- **PII.** The e-mail screen and the PII label do not find every kind of personal data. A manual sample scan is still needed before training.
- **Code rows.** They inherit the pool's limitations, including its unchecked seed provenance.
