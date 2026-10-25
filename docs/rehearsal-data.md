# S rehearsal pretraining data

`tools/rehearsal_build.py` builds the small-model rehearsal's pretraining tokens from pinned source files on one local machine, with the owner's approval for the **S build only** (P1-6). It produces exact-quota token shards plus the records needed to check them: source receipts, license decisions, exclusions, credits and a verification pass. The spec is [configs/rehearsal-data-s.json](../configs/rehearsal-data-s.json). Raw data, units and shards stay in the ignored `runs/` tree; nothing here is redistributed.

| Stream | Source | Tokens |
| --- | --- | ---: |
| `stage1-web` | FineWeb-Edu `sample-10BT`, files 000 and 001 | exactly 1,000,000,000 |
| `stage2-web` | the same files, documents unseen in stage 1 | exactly 100,000,000 |
| `stage2-python` | StarCoderData Python, files 00000 to 00004 | exactly 900,000,000 |
| `validation-web`, `validation-python` | every validation-split document of those files | all of them |

Token counts are **shard tokens**: each document starts with one BOS token, and BOS tokens count toward the quota.

## The four stages

```bash
uv run --locked --extra evaluation python -m tools.rehearsal_build download --spec configs/rehearsal-data-s.json --build runs/rehearsal-s-YYYYMMDD
uv run --locked --extra evaluation python -m tools.rehearsal_build licenses --spec ... --build ...
uv run --locked --extra evaluation python -m tools.rehearsal_build build    --spec ... --build ...
uv run --locked --extra evaluation python -m tools.rehearsal_build verify   --spec ... --build ...
```

1. **Download.** The tool asks the Hub for each pinned revision's file list and records every file's size and LFS SHA-256. It refuses the plan if it exceeds the spec's `max_download_bytes`. Each file is hashed while it streams and kept only if its size and hash match. A file that an earlier verified receipt already holds is hard-linked instead of fetched again. `HF_TOKEN` is sent only to `huggingface.co`, never across the redirect to the storage CDN, and not at all for the public web dataset.
2. **Licenses.** StarCoderData has no license column, so each Python row is joined, on its exact repository and path, to The Stack dedup v1.2's detected repository licenses (`tools/stack_licenses.py`). Only the key and license columns are read, plus each file's Parquet footer. Requests that would touch the content column are refused before they are sent. A key that appears more than once is joined once.
3. **Build.**
   - *Review.* Each file is split into units of 16 row groups, reviewed in parallel worker processes. A Python file is admitted on an explicit permissive header first. Without a header, it needs repository metadata in which **every** listed ID is on the D-05 allowlist. Each Python unit receives only its own rows' licenses. Admitted documents then pass the [existing quality rules](real-samples.md) and the benchmark contamination index, and are encoded with the [frozen tokenizer](tokenizer-freeze.md).
   - *Assembly.* The units are then walked in spec order. A document whose exact content hash was already seen anywhere in the build is a duplicate. Each kept document's split comes from its content hash, so identical text always lands in the same split. Documents fill their stream's quota in order; the one that crosses the quota keeps only the tokens that fill it exactly, and the cut is recorded. Shards are written in the standard [token shard format](walkthroughs/shards.md).
4. **Verify.** This stage runs inside `build` before anything is published and can be rerun at any time. It reopens every shard and checks its hash, manifest, offsets, BOS positions and vocabulary. It also checks:
   - each document's token hash;
   - the quota totals;
   - that no two streams share a document;
   - the cut rule;
   - that every listed document still meets its admission rule.

   It also re-encodes the source text of the first, last and cut document of each shard. It deletes any earlier `verification.json` first, so a failure never leaves a stale pass behind.

## Reuse and stop rules

A unit is reused only if its record carries the same input fingerprint. The fingerprint covers the tokenizer, the license mapping, the contamination set, the filter settings, the classifier and the review code. Any change redoes the unit, and so does a missing file. The build stops before writing if free disk would fall below the spec's headroom, if any source hash differs, if the license mapping names different files, or if a quota cannot be filled. Output is published only after verification passes; a failure leaves `output.tmp` for inspection.

## Records

`output/report.json` records:
- every reviewed row's primary decision by domain, in the same classes as the [sample review](real-samples.md);
- per-stream credits by license ID;
- the cut document and the unused remainder;
- source terms and obligations;
- all hashes;
- two identities: `tokens_identity` for the token multiset and `build_identity` for the whole record.

`output/documents.jsonl` gives every shard document's source row, content hash, token hash and license provenance. `output/exclusions.jsonl` lists every rejected row with its reason.

## Verification

`uv run --locked --extra evaluation pytest tests/test_rehearsal_build.py` builds authored Parquet fixtures through all four stages offline. It covers:
- exact quotas, cuts (including a cut that keeps only the BOS token), duplicates across files, contamination and every license class;
- token handling, range-resumed downloads, unit reuse and redo;
- a failed verification that publishes nothing;
- tampering that verification catches.

## Result of the 2026-10-25 build

The build ran on the owner's Mac (12 CPUs, 10 review workers) at commit `33f7026` with a clean tree. It took about 28 minutes: 24 for review, 3 for writing shards and the rest for assembly and verification. No paid resource was used. Verification passed on the published output and again on its own. It found **0** shared documents between any two streams and re-encoded **36** documents exactly from source.

| Stream | Shard tokens (BOS included) | Documents | Shards | Cut document |
| --- | ---: | ---: | ---: | --- |
| `stage1-web` | **1,000,000,000** | 1,037,577 | 8 | 7 of 136 tokens kept |
| `stage2-web` | **100,000,000** | 104,472 | 1 | 929 of 6,794 tokens kept |
| `stage2-python` | **900,000,000** | 862,499 | 7 | 18,842 of 21,311 tokens kept |
| `validation-web` | 6,948,391 | 7,341 | 1 | none |
| `validation-python` | 5,334,710 | 4,848 | 1 | none |

`tokens_identity` is `04c51932d17e52e7fe86521c244d2f2098f97c050d6d531049b43d7d70363e74`; `build_identity` is `e91a37eb821fb1e7beb4bd5f447a392fa8ae0ceb2973d53f7db8264434200ed5`.

### Sources

| Item | Value |
| --- | --- |
| Downloads | **6,267,680,658** listed bytes, exactly the approved estimate; 3,715,728,005 transferred and two files reused from the sample review, all hash-verified |
| License lookup | 144 The Stack files read for **738,634,555** bytes. No content column chunk was fetched; the footers carried 21,979,081 bytes of content statistics, held in memory only |
| License join | all **1,090,395** Python files matched on exact repository and path |

### What was kept

| Domain | Files reviewed | UTF-8 bytes reviewed | Kept files | Kept bytes | Byte retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web | 1,455,000 | 6,942,037,359 | 1,445,703 | 6,884,155,866 | 99.16622% |
| Python | 1,090,395 | 5,155,806,150 | 989,107 | 4,291,009,543 | **83.22674%** |

Python retention matches the sample review's 82.80882%, so the license join did not miss files.

| Python primary decision | Files | UTF-8 bytes |
| --- | ---: | ---: |
| Kept | 989,107 | 4,291,009,543 |
| Unrecognized or unreadable inline header | 45,422 | 353,187,827 |
| Off-allowlist repository license | 24,947 | 144,622,879 |
| Contamination | 12,764 | 167,984,272 |
| Generated marker | 7,386 | 118,022,575 |
| Copyleft header | 6,843 | 75,075,273 |
| Too short | 3,199 | 76,944 |
| Restricted header | 714 | 5,817,442 |
| Low alphanumeric share | 7 | 8,821 |
| Exact duplicate | 6 | 574 |

The web drops are 7,766 exact duplicates and 1,531 contamination matches. Kept Python files rest on an explicit header (72,837) or on repository metadata (916,270).

| Kept license IDs | Count |
| --- | ---: |
| MIT | 649,287 |
| Apache-2.0 | 216,573 |
| BSD-3-Clause | 74,946 |
| BSD-2-Clause | 19,365 |
| Unlicense | 16,890 |
| CC0-1.0 | 8,620 |
| CC-BY-4.0 | 3,213 |
| 0BSD | 2,267 |
| WTFPL | 2,166 |
| ISC | 1,085 |
| PSF-2.0 (inline only) | 9 |

There are **131** distinct off-allowlist repository IDs. The largest are ECL-2.0 (2,294 files), CC-BY-3.0, BSD-3-Clause-LBNL, CNRI-Python, bzip2-1.0.6, BSD-2-Clause-FreeBSD, RSA-MD, MIT-0 and PSF-2.0 at repository level. They stay out under D-05.

### What was left over

After the quotas filled, **255,835** stage-1 web documents (247,541,480 tokens), **40,478** stage-2 web documents and **121,760** Python training documents (128,213,656 tokens) were left unused. That is the margin that five Python files and two web files gave over the quotas.

### One fix during the run

The first build attempt stopped on a Python file with a NUL byte after an indented block. CPython 3.12's tokenizer reports that as a `SystemError`, which the header classifier did not expect. The classifier now treats it like any other tokenizer failure. Two files in the build have such a byte. The failed attempt published nothing; the fix changed every unit's fingerprint, so all units were reviewed again.

### Limitations

- The final verified task-pool holdout does not exist yet. Its exclusion must be repeated before training, and the build rebuilt if it removes any document; `final_training_admission` is therefore false.
- Repository license lists are The Stack's detected licenses, not a legal warranty.
- Only exact duplicates are removed; the sources' own near-deduplication is relied on.
- Shards, units and raw files exist only on the owner's Mac (about 17 GB); no storage account exists. They can be rebuilt from the pinned sources and this spec.
