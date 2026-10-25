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
