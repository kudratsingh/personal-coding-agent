# ARC-Easy evaluation intake

The loader selects the complete **ARC-Easy test split: 2,376 questions**, retaining original IDs, question text, choice labels/order, and answer keys. It does not use ARC-Challenge, a sampled subset, or train/validation questions for scoring.

## Source and terms

The [official AI2 landing page](https://allenai.org/data/arc) redirects to [AllenAI's dataset repository](https://huggingface.co/datasets/allenai/ai2_arc). Intake pins revision `210d026faf9955653af8916fad021475a3f00453`. Its [pinned card](https://huggingface.co/datasets/allenai/ai2_arc/blob/210d026faf9955653af8916fad021475a3f00453/README.md) declares CC-BY-SA-4.0, but the prose licensing section says “More Information Needed”; the tag alone is not our rights review. The [original ARC distribution registry](https://registry.opendata.aws/allenai-arc/), explicitly managed by the Allen Institute for AI, independently declares CC BY-SA and links the [4.0 terms](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).

Attribution: Allen Institute for AI; Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord (2018), *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, [paper](https://arxiv.org/abs/1803.05457). Original question sources and release notices remain upstream. Preserve attribution, license links and modification notices if redistributing material; adapted material is subject to ShareAlike and no additional restrictions. This repository does not redistribute raw/converted questions or select a project code/weight license. Evaluation-only intake is not approval to train on ARC or a claim that third-party rights disappear.

## Pinned files

Files live under `ARC-Easy/` at the pinned revision. All manifests have evaluation-only purpose. Train/validation files are solely for schema/key validation and cross-split ID exclusion, never training, tokenizer fitting, prompt examples, or checkpoint selection.

| Source split | Rows | Bytes | SHA-256 |
| --- | --- | --- | --- |
| train | 2,251 | 330,598 | `b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5` |
| validation | 570 | 86,080 | `ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74` |
| test | 2,376 | 346,257 | `4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177` |

## Walkthrough

Install the optional locked parser with `uv sync --locked --extra evaluation`. PyArrow 26.0.0 is file-format tooling only, not a model, trainer, optimizer, or metric implementation. Ordinary tests never download ARC; dedicated CI tests parsing on authored fixtures.

1. Explicit `--download` fetches only the three small files through size/hash-verified atomic downloads. No dataset script runs; the unrelated large science corpus archive is not fetched.
2. Intake verifies each file before parsing. It bounds file size, row count/groups, metadata allocations and expanded table size, disables Arrow extension types, and requires reviewed column/nested-choice schemas.
3. Unique IDs, aligned/unique labels, valid answer keys, exact counts and zero cross-split ID overlap are required. Only test questions leave the loader. ID checks do not establish semantic deduplication against arbitrary external corpora.
4. Scoring uses unchanged [ADR 0005](adr/0005-choice-scoring.md): fixed question/options prompt, summed space-prefixed label log likelihoods, listed-order tie break. The callback receives public text and labels, never the answer key. Oversized prompts count as context limits in the full denominator; no shortening or task dropping.
5. Reports retain manifests, original split counts, task/checkpoint/tokenizer hashes, model context, settings and per-task status/score. Safe logs use a metadata line and separate task lines, excluding question/choice text and raw answer keys. Multiple-choice evaluation executes no generated programs.

Example with an own-project checkpoint:

```sh
uv run --locked --extra evaluation python tools/arc_check.py \
  --data-directory runs/arc-data --download \
  --checkpoint runs/eval-nano/train/checkpoint.pt \
  --tokenizer runs/eval-nano/data/tokenizer.json \
  --output runs/arc-evaluation
```

The manual **ARC-Easy nano validation** workflow prepares a checkpoint from repository-authored synthetic fixtures only, then scores all test questions on a standard public CPU runner. No paid compute or real-corpus training is authorized. Verified local intake found all expected rows and zero cross-split ID overlap. A complete nano run is pending; loader tests are not capability evidence. This label-scoring protocol is not claimed equivalent to published ARC leaderboard settings.
