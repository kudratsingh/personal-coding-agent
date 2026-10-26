# Artifact storage

`tools/r2_store.py` keeps durable run artifacts in one Cloudflare R2 bucket, under a monthly cost cap. R2 speaks the S3 API; every request is signed with AWS Signature Version 4. Credentials come only from `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and `R2_BUCKET`. They are never printed or written. Prices are the published R2 Standard rates as checked on 2026-10-25: $0.015 per GB-month after a 10 GB free tier.

## One-off uploads and downloads

```bash
uv run --locked python -m tools.r2_store status
uv run --locked python -m tools.r2_store upload --root runs/x --prefix x/ --receipt runs/x/r2.json runs/x/file.bin
```

- `status` lists the bucket and prices what it holds.
- `upload` refuses the whole batch if the bucket's priced storage after it would exceed `--cap-usd` (default $5 a month). Otherwise it PUTs each file with its SHA-256 as the signed payload hash, so R2 itself rejects bytes that differ. The digest is also stored as object metadata, and a HEAD reads it back. A file already stored under the same key with the same digest is skipped.
- `Store.download` streams one object and keeps it only if its size and SHA-256 match the caller's pinned values, not the object's metadata. A body that ends early is retried like any transient failure, within the same four attempts: Python's HTTP client returns no more bytes on a clean early close instead of raising, and `IncompleteRead` is retried too. A body longer than the pinned size, or a full-length body with another SHA-256, is refused at once.

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
- **Newest checkpoint wins.** While an upload runs, any newer submission drops the rolling labels still waiting that were first submitted before it: a newer rolling label replaces them, and a milestone, which holds newer weights, drops them so it never waits behind a rolling label that has not started. Resubmitting an older milestone keeps the rolling labels submitted after it. The dropped labels are listed as `superseded`. A rolling label already uploading finishes. Milestones are never dropped. Submitting a label that is already waiting replaces it in place; submitting the label that is uploading adds nothing.
- **Cost cap first.** Before each label, the bucket is listed. The label is refused if the bucket's bytes plus this label's bytes would price above `cap_usd`. Rolling labels due for deletion still count, so the check is conservative. A refusal is recorded, not raised.
- **Retries.** HTTP 5xx answers, timeouts and connection errors retry up to four attempts, with backoff of 2, 4 and 8 seconds. A 4xx is never retried within one upload; for example, R2 answers 400 when the bytes do not match the signed digest.
- **Bounded checks.** A default `Store` waits up to 300 s per socket operation, so one call against an endpoint that accepts connections but never answers can take about 20 minutes. `bounded_store(credentials)` gives 20 s per socket operation and two attempts, so a call ends within about 42 s. The rehearsal worker uses it for its checks after the final drain, and the Mac supervisor for its listings and HEAD checks.
- **Milestones are retried until the uploader closes.** A milestone whose upload fails for any reason, or that the cost cap refuses, is queued again after `retry_seconds` (15 s), doubling to at most `retry_max_seconds` (60 s). Submitting it again makes it due at once and restarts its backoff. When several labels are due, milestones go first. A milestone waiting to retry is queued work: `drain` keeps waiting for it, and `status()["retrying"]` maps it to its failures so far. Rolling labels are not retried; the next one replaces them.
- **Verify, then receipt.** Each file is read back with HEAD. Only when every file of the label verifies is `receipt_dir/<label>.json` written. It lists each key with its bytes, SHA-256, attempts and seconds, plus totals and the priced bucket size. A receipt exists only for a verified label.
- **Retention.** After a rolling label verifies, older rolling labels beyond `keep_rolling` (default 2) are deleted, oldest first. Their receipts gain a `deleted_at` time. Keys that a failed rolling label managed to store are deleted at the same point. Milestones are never deleted. A failed delete is recorded and retried after the next verified label.
- **Errors stay on the worker.** Any exception in the worker is recorded in `status()["failures"]` with its label and stage (`upload` or `retention`). Nothing is raised into the training loop. `failures` and `refused` keep the newest 100 entries, so a milestone failing for hours cannot grow the report without bound; `failure_count` counts every failure.
- **Status.** `status()` reports `last_verified`, `in_flight`, `pending`, `retrying`, `verified`, `rolling`, `superseded`, `deleted`, `cancelled`, `failures`, `failure_count` and `refused`.
- **Teardown.** `drain(timeout)` waits for all queued work, milestones waiting to retry included, and returns False on timeout. `close()` refuses new work, cancels labels that have not started or are waiting to retry (listed as `cancelled`) and waits for the upload in flight. The worker is a daemon thread, so call `drain` before the process exits. The rehearsal worker resubmits every unverified milestone and drains for up to 360 s after its last GPU work before it closes, leaving the end of the transfer window for the Mac's SSH pull (see the [rehearsal run walkthrough](walkthroughs/rehearsal-run.md)).

Keys are `prefix + label + "/" + key`. Labels are letters, digits, `.`, `_` and `-`. Files must stay unchanged until their label leaves the queue; R2 rejects changed bytes because the digest no longer matches.

**Limitations.** Retention only knows labels that this uploader instance verified. Labels from an earlier process, such as one before a resume, are never deleted automatically. The cost cap is checked per label against a fresh listing. Two uploaders writing to one bucket at the same time can each pass the check, so together they can go over the cap by up to one label each.

## Verification

`uv run --locked pytest tests/test_r2_store.py tests/test_r2_uploader.py` runs offline against in-memory fake buckets. The tests cover:

- signing and pricing;
- downloads that retry a truncated body or `IncompleteRead` and refuse a longer body or another digest at once;
- non-blocking submission while a PUT is held;
- newest-rolling-wins coalescing, a milestone dropping waiting rolling labels but not the one in flight, with milestones kept;
- milestones retried with backoff until they verify, after a failure or a cost-cap refusal, due at once when resubmitted, and cancelled by `close`;
- retention, including partial failures;
- cost-cap refusal;
- retries for 5xx answers, timeouts and connection errors, and no retries for 4xx within one upload;
- the signed digest and receipt contents, and the bounded failure record;
- drain timeouts, close and concurrent submissions.

No test touches the network or a real bucket.
