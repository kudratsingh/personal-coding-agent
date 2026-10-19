# Frozen real-source tokenizer

Frozen on 2026-10-19 by **one** from-scratch `ByteBPE.train` call at source `121fa543aaed4801a5171bf3f7b363c463248be6`. Retraining requires the owner's explicit approval. The source data and production tokenizer artifact remain local and ignored; hashes are public reproducibility identities, not a claim of durable remote backup.

## Identity

| Identity | Value |
| --- | --- |
| Vocabulary / learned merges / reserved byte-special prefix | 32,768 / 32,496 / 272 |
| Tokenizer identity (`ByteBPE.sha256`) | `943afc8f97d58882133825b1fd191d4944712809e42d704d799604addf0ecf14` |
| Merge-table SHA-256 | `435729be21650397918abf42c56fd2aee742b12dbc3644e7095ee7c58fc0c1e7` |
| Saved tokenizer-file SHA-256 | `94161ac2258759ca789bd5340195fc65a58361da29ed030bdbaee0c488e0dc8e` |

The merge-only hash is SHA-256 of the ordered pair array serialized as compact JSON with separators `(',', ':')`. The tokenizer identity additionally binds format and special tokens; the saved-file hash also binds its trailing newline. These intentionally differ. The owner-only decisions file records the frozen identities under **Decided, D-19**. Old fixture-trained checkpoints retain their own tokenizer identity; this table does not silently replace theirs.

## Training selection

From the [reviewed real samples](real-samples.md), training used exactly **96,223,042 UTF-8 bytes web and 96,223,042 bytes Python**—**192,446,084 total**, **50% each by bytes**. This comprises **19,824 web document prefixes and 10,741 Python documents**. Only stage-1 web and Python training splits were eligible; validation and stage-2 English were excluded. Ordered selection preserves earlier document boundaries and permits one valid UTF-8 final prefix. The declared cap was 100,000,000 bytes per domain; the smaller available Python training set limited both. No benchmark text or third-party tokenizer trainer was used.

Training took **505.8937 seconds** on the arm64 macOS laptop, one CPU process. The record journals input/corpus hashes before training and saves the frozen artifact/identities before measurement. The CLI refuses an existing output directory. Neither a later encoding error nor a new output name authorizes another production training run.

## Measured compression and sample token totals

All counts below exclude BOS; a separate column adds one BOS per document. Validation rows were not used to fit merges.

| Source / scope | Documents | UTF-8 bytes | Tokens, no BOS | Tokens with document BOS | Bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web, all retained | 103,863 | 498,123,158 | 182,749,140 | 182,853,003 | 2.72572 |
| Python, all retained | 10,786 | 96,638,864 | 27,195,275 | 27,206,061 | 3.55352 |
| Held-out English | 524 | 2,589,956 | 944,567 | 945,091 | **2.7419505445** |
| Held-out Python | 45 | 415,822 | 123,384 | 123,429 | **3.3701452376** |

Split token totals, no BOS: stage-1 web **163,150,865**, stage-2 web **18,653,708**, Python training **27,071,891**; the remaining validation totals are above. Recounting all retained data took **53.8215 seconds web** and **31.9963 seconds Python**, using four ordered CPU workers. These timings include read/JSON/count overhead and small reference probes; they are not a guaranteed full-build speed or a pure encoder microbenchmark.

Both 256-character real validation probes produced identical reference/fast IDs and exact Unicode round-trips. This is **512 characters checked**, not the roadmap's separate 100 MB reference/fast identity condition. Existing randomized slow/external BPE references, Unicode/code round-trips and fast-path fixtures passed in the **648 passed / 18 explicit runtime skips** local suite; [PR #69](https://github.com/kudratsingh/personal-coding-agent/pull/69) carries source/evidence CI. No production retraining was used to demonstrate determinism; repeat-training identity tests use authored corpora.

The run command was `uv run --locked python -m tools.freeze_tokenizer --samples runs/real-samples-20261019/filtered-complete --output runs/tokenizer-freeze-20261019`. The completed ignored `record.json` and `measurements.json` preserve full identities, selection, timings and splits. No GPU, paid resources, model training or full corpus build ran. G1's tokenizer-freeze condition is met; owner review of the full-build estimate remains a separate gate condition.
