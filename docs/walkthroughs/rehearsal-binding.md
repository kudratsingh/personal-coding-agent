# Rehearsal data binding

`tools/rehearsal_binding.py` ties the admitted [S rehearsal build](../rehearsal-data.md) to the two rehearsal runs. Run A reads `stage1-web`, then `stage2-python` with `stage2-web`. Run B reads all three training streams mixed. The binding is one JSON file that pins every file the runs read, each by size and SHA-256. For every stage it records the document order the run must follow and the set of documents it must read. Both runs then read from the same pinned record, and either run can be checked against it.

## Worked example

**Order.** Take a stage with two shards whose SHA-256 values are `aaaa…` (64 `a`s, 3 documents) and `bbbb…` (2 documents), and seed 17. Each document gets a sort key: the first 8 bytes of `sha256("17|aaaa…|0")`, `sha256("17|aaaa…|1")` and so on, read as a big-endian integer. Sorting the five keys gives the reading order as (shard ordinal, document index) pairs: `(0,0) (0,1) (1,1) (1,0) (0,2)`. Seed 18 gives `(1,0) (1,1) (0,0) (0,2) (0,1)`. `order_sha256` is the SHA-256 of those pairs packed as a little-endian `uint16` ordinal and `uint32` index, six bytes per document: `fbf104b0…` for seed 17. Anyone can recompute it from the shard hashes, the document counts and the seed (`scratchcoder.data.onepass`).

**Multiset.** Each row of `documents.jsonl` carries `tokens_sha256`: the SHA-256 of that document's `uint16` little-endian tokens, BOS included. A stage's `multiset_sha256` is the sum of its documents' hashes modulo 2^256. With three documents whose hashes ended `…05`, `…03` and `…0a` (the rest zero), the digest ends `…12` whatever order the documents are read in. Sums of disjoint sets add, so the digests of run A's two stages must add up to run B's single stage. The build refuses to write a binding where they do not.

**Counts.** The runs pack 1,024-token windows, with 16 windows per microbatch and 8 accumulation steps, so one update is 128 windows (131,072 tokens). Each shard token counts as one target. Stage 1 of run A holds exactly 1,000,000,000 tokens. That is `ceil(1e9 / 1024) = 976,563` windows and `ceil(976,563 / 128) = 7,630` planned updates; the last update holds 51 windows. Run B's 2,000,000,000 tokens are 1,953,125 windows and 15,259 updates, the last of 101 windows.

## Data flow and shapes

1. **`build` (Mac).** The inputs are the build directory (`output/report.json`, `verification.json`, `documents.jsonl` and `shards/`), the spec, an R2 key prefix and the order seed (17). The command refuses the build unless:
   - the report is `completed`, ran from a clean tree and matches the spec's SHA-256;
   - it records `final_training_admission`, a non-null `admission` and a `holdout` section with the same holdout hash;
   - `verification.json` passed for exactly this `report.json` and records the same identities;
   - `rehearsal_build.identities` recomputes both identities in spec stream order.

   It then re-hashes each shard and manifest and opens it as a `TokenShard` with the S model's vocabulary (65,536) and BOS (256). It hashes every document from the shard bytes and streams `documents.jsonl` once. Each listed row must sit at its shard's next position with that document's length, and each stream's multiset from the listing must equal the one from the bytes.
2. **The binding.** It records:
   - the build's identities and record hashes;
   - `stream_order` in spec order;
   - per stream, its shards: `file`, `sha256`, `manifest_sha256`, `size_bytes`, `manifest_size_bytes`, `documents`, `tokens`, `key` and `manifest_key`;
   - the tokenizer at `tokenizer/<name>`;
   - per run, its stages: streams, ordered shard list, `order_seed`, `order_sha256`, `documents`, `tokens`, `targets`, `windows`, `planned_updates` and `multiset_sha256`;
   - `same_multiset` for A and B;
   - the validation streams and the subset rule (1,000,000 target tokens per domain at length 1,024; the subset itself is chosen on the Pod).

   Each R2 key is the prefix plus the file's path relative to `output/`.
3. **`stage` (Mac).** `staging_files` lists (key, path) pairs for `r2_store.upload`. It includes the shards and manifests, the tokenizer, `documents.jsonl`, `exclusions.jsonl`, `report.json` and `verification.json`, and finally the binding at `<prefix>bindings/<binding sha256>.json`. Every file is re-hashed against its pin first.
4. **`fetch` (Pod).** It downloads the run's stage streams, both validation streams and the tokenizer into a root at the binding's relative paths. Each download goes through `Store.download` with the pinned size and SHA-256. A present file with the pinned bytes is skipped. The command writes a receipt.
5. **`verify` (Pod or Mac).** It checks the binding file's SHA-256 against the expected value and re-hashes every needed file under the root. It opens each shard and checks its document and token counts, loads the tokenizer and checks its identity, and recomputes each stage's order. It returns:
   - `streams`: `{name: [(shard sha256, TokenShard)]}`;
   - `stages`: `{run: [stage]}`, where each stage carries its `(sha256, TokenShard)` list and its `order`, an array of shape `[documents]` with fields `shard` (`<u2`) and `index` (`<u4`);
   - the `ByteBPE` tokenizer.

```sh
uv run --locked python -m tools.rehearsal_binding build --build runs/rehearsal-s-YYYYMMDD \
  --spec configs/rehearsal-data-s.json --output runs/rehearsal-s-YYYYMMDD/binding.json \
  --r2-prefix rehearsal-s-YYYYMMDD/
uv run --locked python -m tools.rehearsal_binding stage --binding .../binding.json \
  --expected-sha256 <sha> --build runs/rehearsal-s-YYYYMMDD --receipt .../stage-receipt.json
# On the Pod, with R2_ENDPOINT, R2_BUCKET, R2_POD_ACCESS_KEY_ID and R2_POD_SECRET_ACCESS_KEY set:
python -m tools.rehearsal_binding fetch --binding binding.json --expected-sha256 <sha> \
  --root /workspace/data --run A --receipt fetch-receipt.json --key-prefix R2_POD_
```

`fetch` runs `verify` on the run's files before it prints its summary. None of these commands is run by the tests: they use a fake bucket.

## Design choices and common mistakes

- **Spec order, not file order.** `report.json` is written with sorted keys, so its streams appear as `stage1-web, stage2-python, stage2-web, …`. `tokens_identity` hashes the shards in spec order (`stage1-web, stage2-web, stage2-python, …`). Iterating the report's keys gives a different hash, and a report carrying that hash is refused even when its verification agrees.
- **Two derivations of each multiset.** The digest from `documents.jsonl` must equal the one hashed from the shard bytes. Otherwise a wrong listing would only show at the end of a paid run, when the loader's own multiset failed to match.
- **The caller's pin wins.** Downloads check the binding's SHA-256, not the object's R2 metadata, which any holder of a write key could set. The binding's own key is its hash, so no later upload can replace it. It is uploaded last, so nobody finds a binding before the files it names.
- **One serialization.** A binding is written as sorted, two-space-indented JSON and read back only in that exact form, so its hash identifies the mapping. The file is never rewritten: `build` refuses an existing output, and `verify` only reads.
- **No surprises on disk.** Every path is relative and made of plain components, never `..`, an empty part or a hidden name. `verify`, `fetch` and `stage` refuse a symlinked root, directory or file. A present file with other bytes stops `fetch`; it is not silently replaced.
- **Credentials.** They come only from the environment. On the Pod, `--key-prefix R2_POD_` selects a bucket-scoped key pair, while the endpoint and bucket stay shared. The tool never prints or writes a key.
- **Limits.** SHA-256 gives content identity, not an authenticated publisher. Files must not change while a run reads them. The binding does not choose the validation subset, pack windows or launch anything.

## Verification

Implementation: [`tools/rehearsal_binding.py`](../../tools/rehearsal_binding.py). Reference tests: [`tests/test_rehearsal_binding.py`](../../tests/test_rehearsal_binding.py), run with:

```sh
uv run --locked --extra evaluation pytest tests/test_rehearsal_binding.py
```

The tests build a small admitted build through all four `rehearsal_build` stages, holdout exclusion included, from authored Parquet files. They then check:

- the pinned files, spec stream order, recomputed orders and multisets, and the count arithmetic at the real S sizes;
- refusal of a build that is not completed, dirty, from another spec, unadmitted, missing its holdout, or unverified or stale;
- that `tokens_identity` must be in spec order;
- tampered shards, manifests, listings, tokenizer and binding fields, and the 65,536 vocabulary;
- that A's multiset equals B's, and that the order is deterministic for one seed and changes with another;
- a fetch from a fake bucket that skips present files, rejects corrupted objects and present files with other bytes, and refuses symlinks and traversal;
- the staging list, and credentials taken from the named key pair and never printed.

## Questions and answers

1. *Why can the two runs share one multiset digest when they read documents in different orders?* The digest is a sum of document hashes modulo 2^256, which does not depend on order. A's two stage digests add up to B's.
2. *Why is `tokens_identity` recomputed instead of copied?* The copy could have been computed over `report.json`'s sorted keys. Only a recomputation in spec order proves which shards it names.
3. *Why name the binding's R2 key by its own hash?* An upload would replace a fixed name. A hash name holds one binding, and the Pod already knows the hash it expects.
4. *What happens if a shard on the Pod differs by one byte?* `verify` reports a size or SHA-256 mismatch before any shard is opened. During `fetch`, an object with other bytes is never published, and a present file with other bytes stops the fetch.
5. *Why is the order hashed from shard hashes and counts, not from the files' names?* Names can repeat across builds. The content hashes and counts fix exactly which documents the order covers.
