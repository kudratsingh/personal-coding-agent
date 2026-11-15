# Component: holdout integrity scan

Purpose: before new tasks can be trained on, find the ones that belong to the same task family as a sealed test task. The scan reads the sealed test split once, only with the owner's recorded approval, logs that read in the seal ledger, and writes nothing about the test split except counts.

Code: [`scratchcoder.data.integrity_scan`](../../src/scratchcoder/data/integrity_scan.py) holds the rules and [`tools/holdout_integrity_scan.py`](../../tools/holdout_integrity_scan.py) is the command.

## Worked example

Take one sealed test task, `T`: a 20-word prompt and a reference `def f(values): ...` that sums three times each value. The subjects are tasks that may be trained on. The bridge `P` is a pool task that is not judged:

| Subject | What ties it to `T` | Status | Reason (in memory only) |
| --- | --- | --- | --- |
| `A` | the same reference, a different prompt | excluded | `solution_ast`, `canonical_ast` |
| `B` | the same reference after renaming every bound name | excluded | `canonical_ast` |
| `C` | 16 of the 20 prompt words (3-word shingle Jaccard 0.64) | excluded | `prompt_paraphrase` |
| `D` | the function name `f`, prompt Jaccard 0.29 | excluded | `prompt_named_paraphrase` |
| `E` | the function name `f`, prompt Jaccard 0.13 | flagged | `prompt_named_flag` |
| `F` | the same reference as `C`, nothing shared with `T` | excluded | `family_chain` |
| `G` | the same reference as bridge `P`, which paraphrases `T` | excluded | `family_chain` |
| `H` | nothing | clear | none |

`F` and `G` are excluded because families are connected components: `C` joins `F` to `T`, and `P` joins `G` to `T`. A family member is excluded however it is reached.

The output lists only `excluded: A B C D F G`, `flagged: E` and `clear: H`. Which signals fired appears only as totals over all subjects, and not at all below 20 subjects. Nothing says which test task matched, or what it says.

## Data flow and shapes

1. **`request`.** This writes `approval-request.json` and opens nothing sealed: the sealed file is only `os.lstat`-ed. The document holds:
   - `request`, which names:
     - the SHA-256 of every module the scan runs, and the clean git commit;
     - the rules version (`task-clusters-v1`);
     - the sealed file's path, with its SHA-256 and task count pinned by the pool build report;
     - each subject and bridge file's SHA-256 and task count;
     - the seal ledger and its current head;
     - the output directory;
   - `request_sha256`, the SHA-256 of the request's canonical JSON;
   - an empty `approval` block.

   The pool report, subjects, bridges, ledger and ledger head record are each refused if they are a sealed file, checked three ways:
   - by name;
   - by inode, against the test, holdout and RL-only files beside the sealed path;
   - by hash, against every sealed file the report pins.

   The report must pin the test file with a 64-hex SHA-256 and at least one task. A task that is both a subject and a bridge is judged as the subject (RL candidates are pool train tasks); the same ID with different content is refused.
2. **Approval.** The owner fills `approved_by`, `approved_at` and `statement`, and quotes `request_sha256`. The approval covers exactly that request. Editing any part of it (output directory, a path, a hash) changes the hash, so the quote no longer matches.
3. **`run --approval FILE`.** Before anything sealed is opened, the run checks that:
   - the approval file is a regular file with one link, under 1 MB, and not sealed by name;
   - the quoted hash equals the canonical hash;
   - the code, git commit (clean tree), rules, sets, pinned report and output names still match;
   - the ledger is the same ledger, at most extended;
   - that request was never scanned, and the approval wording (`approved_by`, `approved_at`, `statement`) was never used for another scan;
   - no output exists.

   Any failure exits with status 2 or 3 and logs nothing.
4. **Ledger, then read.** `integrity_scan_opened` is appended to the seal ledger. It carries the request SHA-256, the approval block, the code identity and the set hashes. Then the sealed file is opened once, without following links. The opened descriptor must be the file whose metadata was checked (device, inode and size), and its SHA-256 and task count must match the pinned values (exit 4 otherwise).
5. **Scan.** Test, subject and bridge items go through `clusters.find_edges`. Union-signal edges join items with a union-find. A subject or bridge whose ID equals a test ID is scanned under an internal ID and joined to that test task. Each subject gets its status from its direct edges to test items and from whether it shares a family with any test item.
6. **Outputs and close.** The outputs are written under temporary names:
   - `exclusions.json`: subject IDs by status only;
   - `summary.json`, with these counts:
     - by status;
     - by signal, totalled over every subject;
     - test tasks scanned, test tasks with a direct edge, and test tasks in a subject's family;
     - bridge tasks in test families, and bridge tasks judged as subjects;
   - `record.json`: the request, the approval and the output hashes.

   Then `integrity_scan_closed` (`completed`, with the output hashes and counts) is appended, and only then are the files renamed into place. From the completed close on, SIGTERM is ignored and nothing deletes the outputs. If a rename still fails, the `.partial` files whose hashes the close records stay on disk.

Any failure before the close, SIGTERM included, removes the temporary files and is logged as `failed` with the error's class name only, with exit 5. The exceptions are a sealed file that differs from the checked file, its SHA-256 or its task count: these exit 4. So outputs exist only beside a completed close.

## Design choices and common mistakes

- **The frozen family rules, not a new rule.**
  - The signals and thresholds are `task-clusters-v1`'s. Equal reference AST, equal reference AST after renaming, equal upstream ID, and prompt paraphrase at Jaccard 0.5 form families.
  - A shared defined name at Jaccard 0.25 is the pool build's named-paraphrase drop rule, so it excludes a subject. At Jaccard 0.10 it is the rules' flag level.
  - Prompt shingles and function names alone would miss a renamed copy of a reference.
- **Bridges matter.** Families chain through any task. Without the pool's train and dev splits passed as bridges, a subject linked to a test task only through a pool task would come out clear.
- **Flags are not decided here.** Deciding a flag would mean showing someone the test task. The scan lists flagged subjects separately and the owner decides what they mean.
- **Nothing about a test task leaves the process.** Edges carry signal keys (AST hash prefixes) and per-pair Jaccard values, and per-subject signal names say what an unnamed test task looks like. So only statuses are written per subject. Signal counts are totals over all subjects, suppressed below 20 subjects, so a small set label cannot expose one subject's signals. Parse errors name a line number at most. The tests plant marker strings in a synthetic sealed file and check every output, the ledger and the console for them.
- **What an exclusion still reveals.** Listing subject `A` as excluded says that some test task resembles `A`. A subject whose ID equals a test ID is listed as excluded under that ID. No dedup output can avoid this. Keep the outputs private, and never inspect excluded subjects to learn about the test split.
- **The approval is bound to the request, not to the file's bytes.** Single use is keyed by the canonical request hash, so re-serialising the approval file does not buy a second scan, and approval wording that has already been used is refused for a new request. The approval block is unauthenticated text: the binding guards against accidents and edits, not against forgery by someone who can run the tool.
- **Log first, then open.** The ledger event is written before the file is opened, so a crash can never leave an unlogged read.
- **Static only.** The command imports no sandbox, generation, RL or scoring code (a test checks the imports). It runs no task code.
- **Ledger compatibility.** Seal code from before these events refuses a ledger that contains them. Run a real scan only with code that knows the events, on every machine that reads that ledger.

## Verification

```bash
uv run --locked pytest -q tests/test_holdout_integrity_scan.py
```

The tests cover:
- every status on hand-built prompts whose Jaccard values are asserted first;
- chains through subjects, through bridges and through a same-ID subject;
- refusals before opening, each of which logs nothing:
  - unapproved, misquoted or edited requests;
  - edited code, commit, rules, scope, sets or ledger head;
  - a dirty checkout, a missing ledger, an empty set;
  - a link, or a misnamed or sealed path;
  - the approval, report or a subject being the sealed file through a hard link elsewhere, or a byte copy;
  - a bridge linked to or copying the holdout union;
  - a ledger head record that is the sealed file;
  - reused approval wording;
  - a report pinning zero tasks;
- the real layout, where subjects sit inside the train bridge;
- a bridge whose ID matches a subject but whose content differs;
- a sealed file swapped after the check (exit 4);
- a failure after the completed close, which keeps the outputs;
- suppressed and totalled signal counts;
  - existing outputs;
- one scan per request, whatever the approval file's bytes;
- a hash mismatch, a parse failure and a failure during the close, each logged as failed without quoting and leaving no outputs;
- the absence of test markers and signal names in every output;
- the import boundary.

## Questions and answers

1. *Why does `F` get excluded when it shares nothing with `T`?* Its reference equals `C`'s, and `C` paraphrases `T`, so all three are one family. Training on `F` would be training on a member of `T`'s family.
2. *Why write the ledger event before opening the file?* If the process dies after opening but before logging, the read would be invisible. Logging first means every read has a record, at worst with no closing event.
3. *Why are flagged subjects not excluded automatically?* The 0.10 level joins unrelated tasks that share common names such as `Book` or `add`. Treating them as exclusions is a policy choice for the owner, made with the flagged count in view.
4. *Can the same approval be used twice?* No. Single use is keyed by the canonical request hash, which the approval must quote. A re-serialised file has the same hash and is refused. An edited request has a new hash that the old approval does not quote.
