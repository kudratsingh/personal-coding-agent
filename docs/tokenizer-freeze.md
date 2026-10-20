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

## GPT-2 comparison on identical held-out text (2026-10-20)

This is an encoding-only comparison, **not retraining**. Both tokenizers receive
the exact original frozen validation documents, with whitespace unchanged,
per-document encoding and no BOS/EOS injection, padding or truncation. Counts
are total UTF-8 bytes divided by total tokens, not the mean of per-file ratios.
Both tokenizers round-trip every compared document exactly.

| Original held-out text | Files | UTF-8 bytes | Project tokens | GPT-2 tokens | Project bytes/token | GPT-2 bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 524 | 2,589,956 | 944,567 | 554,882 | **2.7419505445** | **4.6675797737** |
| Python | 45 | 415,822 | 123,384 | 189,083 | **3.3701452376** | **2.1991506376** |

The project uses **70.22844% more English tokens** and **34.74612% fewer Python
tokens** on these samples. Higher bytes/token means fewer tokens for this text,
not better model capability. Vocabularies differ (**32,768 project / 50,257
GPT-2**) and the small, license-filtered ordered-prefix Python holdout is not
representative of all Python. No merge-table change is proposed by these numbers.

Reference: [OpenAI GPT-2 / Hugging Face distribution](https://huggingface.co/openai-community/gpt2),
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Only `tokenizer.json`
(**1,355,256 bytes**) was downloaded under the owner's explicit exception;
**0 model-weight bytes / 0 new corpus bytes**. SHA-256:
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`;
Git-blob SHA-1 matched the pinned source metadata. The existing locked baseline
extra supplied `tokenizers 0.22.2`; the local JSON loader made no model/script/API
execution or additional fetch. Receipts/comparison output remain ignored.

Length-framed held-out corpus hashes: English
`a4d6fc336f4ca5b7156352a0197a4716e906b2f365280ab7828e1d5ac9de4f94`,
Python `542248a29eb7f9993a364c6e665b3c06d16d975c3e5650c72b4324836ba1dea5`.
The measurement rechecked the original document-file hashes, reproduced the
project's original token totals and verified the frozen artifact hash before
and after. The owner-only record is under `runs/gpt2-tokenizer-reference-20261020/`.

Separately, the new header-classified sample recount contains **35,009,073 Python
tokens / 122,701,374 bytes** (**35,022,906 with BOS**). Its newer validation
selection is **52 files / 475,430 bytes / 140,646 tokens = 3.3803307595 BPT**.
That changed selection is **not** silently substituted into the comparison table
above. Both sample counts use the same immutable production tokenizer.
