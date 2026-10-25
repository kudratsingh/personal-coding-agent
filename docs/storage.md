# Artifact storage

`tools/r2_store.py` keeps durable run artifacts in one Cloudflare R2 bucket, under a monthly cost cap. R2 speaks the S3 API; every request is signed with AWS Signature Version 4. Credentials come only from `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and `R2_BUCKET`. They are never printed or written. Prices are the published R2 Standard rates as checked on 2026-10-25: $0.015 per GB-month after a 10 GB free tier.

## One-off uploads and downloads

```bash
uv run --locked python -m tools.r2_store status
uv run --locked python -m tools.r2_store upload --root runs/x --prefix x/ --receipt runs/x/r2.json runs/x/file.bin
```

- `status` lists the bucket and prices what it holds.
- `upload` refuses the whole batch if the bucket's priced storage after it would exceed `--cap-usd` (default $5 a month). Otherwise it PUTs each file with its SHA-256 as the signed payload hash, so R2 itself rejects bytes that differ. The digest is also stored as object metadata, and a HEAD reads it back. A file already stored under the same key with the same digest is skipped.
- `Store.download` streams one object and keeps it only if its size and SHA-256 match the caller's pinned values, not the object's metadata.

## Checkpoint uploads during training

`CheckpointUploader` copies checkpoints to R2 on one background thread, so the training loop never waits for the network.

```python
uploader = CheckpointUploader(store, prefix="rehearsal-a/", cap_usd=5.0, receipt_dir=run / "r2")
uploader.submit("step-001000", [("model.pt", path, sha256)], milestone=False)
...
if not uploader.drain(timeout=1800):
    ...  # still uploading: record status() before teardown
uploader.close()
```

- **Submit never blocks.** It checks its arguments and queues the label. The caller passes each file's SHA-256, already computed for the checkpoint sidecar; `Store.put_known` signs that digest instead of reading the file again.
- **Newest rolling checkpoint wins.** While an upload runs, a newer rolling submission replaces any older one still waiting. The replaced label is listed as `superseded`. Milestones are never dropped; they wait their turn.
- **Cost cap first.** Before each label, the bucket is listed. The label is refused if the bucket's bytes plus this label's bytes would price above `cap_usd`. Rolling labels due for deletion still count, so the check is conservative. A refusal is recorded, not raised.
- **Retries.** HTTP 5xx answers, timeouts and connection errors retry up to four attempts, with backoff of 2, 4 and 8 seconds. A 4xx is never retried; for example, R2 answers 400 when the bytes do not match the signed digest.
- **Verify, then receipt.** Each file is read back with HEAD. Only when every file of the label verifies is `receipt_dir/<label>.json` written. It lists each key with its bytes, SHA-256, attempts and seconds, plus totals and the priced bucket size. A receipt exists only for a verified label.
- **Retention.** After a rolling label verifies, older rolling labels beyond `keep_rolling` (default 2) are deleted, oldest first. Their receipts gain a `deleted_at` time. Keys that a failed rolling label managed to store are deleted at the same point. Milestones are never deleted. A failed delete is recorded and retried after the next verified label.
- **Errors stay on the worker.** Any exception in the worker is recorded in `status()["failures"]` with its label and stage (`upload` or `retention`). Nothing is raised into the training loop.
- **Status.** `status()` reports `last_verified`, `in_flight`, `pending`, `verified`, `rolling`, `superseded`, `deleted`, `cancelled`, `failures` and `refused`.
- **Teardown.** `drain(timeout)` waits for all queued work and returns False on timeout. `close()` refuses new work, cancels labels that have not started (listed as `cancelled`) and waits for the upload in flight. The worker is a daemon thread, so call `drain` before the process exits.

Keys are `prefix + label + "/" + key`. Labels are letters, digits, `.`, `_` and `-`. Files must stay unchanged until their label leaves the queue; R2 rejects changed bytes because the digest no longer matches.

**Limitations.** Retention only knows labels that this uploader instance verified. Labels from an earlier process, such as one before a resume, are never deleted automatically. The cost cap is checked per label against a fresh listing. Two uploaders writing to one bucket at the same time can each pass the check, so together they can go over the cap by up to one label each.

## Verification

`uv run --locked pytest tests/test_r2_store.py tests/test_r2_uploader.py` runs offline against in-memory fake buckets. The tests cover:

- signing and pricing;
- non-blocking submission while a PUT is held;
- newest-rolling-wins coalescing, with milestones kept;
- retention, including partial failures;
- cost-cap refusal;
- retries for 5xx answers, timeouts and connection errors, and no retries for 4xx;
- the signed digest and receipt contents;
- drain timeouts, close and concurrent submissions.

No test touches the network or a real bucket.
