# Component: holdout integrity scan

Purpose: before new tasks can be trained on, find the ones that belong to the same task family as a sealed test task. The scan reads the sealed test split once, only with the owner's recorded approval, logs that read in the seal ledger, and writes nothing about the test split except counts.

Code: [`scratchcoder.data.integrity_scan`](../../src/scratchcoder/data/integrity_scan.py) holds the rules and [`tools/holdout_integrity_scan.py`](../../tools/holdout_integrity_scan.py) is the command.

## Worked example

Take one sealed test task, `T`: a 20-word prompt and a reference `def f(values): ...` that sums three times each value. The subjects are tasks that may be trained on:

| Subject | What it shares with `T` | Status | Signals recorded |
| --- | --- | --- | --- |
| `A` | the same reference, a different prompt | excluded | `solution_ast`, `canonical_ast` |
| `B` | the same reference after renaming every bound name | excluded | `canonical_ast` |
| `C` | 16 of the 20 prompt words (3-word shingle Jaccard 0.64) | excluded | `prompt_paraphrase` |
| `D` | the function name `f`, prompt Jaccard 0.29 | excluded | `prompt_named_paraphrase` |
| `E` | the function name `f`, prompt Jaccard 0.13 | flagged | `prompt_named_flag` |
| `F` | the same reference as `C`, nothing shared with `T` | excluded | `family_chain` |
| `G` | nothing | clear | none |

`F` is excluded because `C` joins it to `T`'s family: families are connected components, so a family member is excluded however it is reached. The output lists `A` to `G` with these statuses and signal names. It never says which test task they matched, or what that task says.

## Data flow and shapes

1. **`request`.** This writes `approval-request.json` and opens nothing sealed (`os.lstat` only). The request names:
   - the SHA-256 of every module the scan runs;
   - the rules version (`task-clusters-v1`);
   - the sealed file's path, with its SHA-256 and task count pinned by the pool build report;
   - each subject file's SHA-256 and task count;
   - the seal ledger and the output directory.

   It carries an empty `approval` block.
2. **Approval.** The owner fills `approved_by`, `approved_at` and `statement`. The SHA-256 of the filled file is the approval's identity.
3. **`run --approval FILE`.** Before anything sealed is opened, the run re-checks that:
   - the code, rules, subjects, pinned report and output names still match;
   - the sealed path is a regular file named `test.jsonl`;
   - no output exists;
   - the approval was never used.

   Any failure exits with status 2 or 3 and logs nothing.
4. **Ledger, then read.** `integrity_scan_opened` is appended to the seal ledger, with the approval SHA-256, code identity and subjects. Then the sealed file is opened once, without following links, and its SHA-256 and task count are checked (exit 4 on a mismatch).
5. **Scan.** The test items and every subject item go through `clusters.find_edges` and `clusters.components` (union signals only). Each subject then gets a status from its direct edges to test items and from its component.
6. **Outputs and close.** The run writes:
   - `exclusions.json`: subject IDs by status, with signal names;
   - `summary.json`: counts by status and signal, the number of test tasks scanned and the number with any subject edge;
   - `record.json`: identities, the approval and output hashes.

   It then appends `integrity_scan_closed`. Any failure after the ledger event is logged as `failed` with the error's class name only (exit 5).

## Design choices and common mistakes

- **The frozen family rules, not a new rule.** The signals and thresholds are `task-clusters-v1`'s: equal reference AST, equal reference AST after renaming, equal upstream ID, and prompt paraphrase at Jaccard 0.5 form families. A shared defined name at Jaccard 0.25 is the pool build's named-paraphrase drop rule, so it excludes a subject. At Jaccard 0.10 it is the rules' flag level. Prompt shingles and function names alone would miss a renamed copy of a reference.
- **Flags are not decided here.** Deciding a flag would mean showing someone the test task. The scan lists flagged subjects separately and the owner decides what they mean.
- **Nothing about a test task leaves the process.** Edges carry signal keys (AST hash prefixes) and per-pair Jaccard values. Both describe the test side, so only signal *names* and counts are written. Parse errors name a line number at most. The tests plant marker strings in a synthetic sealed file and check every output, the ledger and the console for them.
- **What an exclusion still reveals.** Listing subject `A` as excluded says that some test task resembles `A`. No dedup output can avoid this. Keep the outputs private, and never inspect excluded subjects to learn about the test split.
- **Log first, then open.** The ledger event is written before the file is opened, so a crash can never leave an unlogged read. One approval allows one scan, and a second scan needs a new approval.
- **Static only.** The command imports no sandbox, generation, RL or scoring code (a test checks the imports). It runs no task code.

## Verification

```bash
uv run --locked pytest -q tests/test_holdout_integrity_scan.py
```

The tests cover:
- every status and signal on hand-built prompts whose Jaccard values are asserted first;
- the subject-chain rule, and a subject whose ID equals a test ID;
- refusals before opening (unapproved, edited code, rules, scope or subjects, a link, a misnamed or sealed path, existing outputs), each of which logs nothing;
- one scan per approval;
- a hash mismatch and a parse failure, each logged as failed without quoting the file;
- the absence of test markers in every output;
- the import boundary.

## Questions and answers

1. *Why does `F` get excluded when it shares nothing with `T`?* Its reference equals `C`'s, and `C` paraphrases `T`, so all three are one family. Training on `F` would be training on a member of `T`'s family.
2. *Why write the ledger event before opening the file?* If the process dies after opening but before logging, the read would be invisible. Logging first means every read has a record, at worst with no closing event.
3. *Why are flagged subjects not excluded automatically?* The 0.10 level joins unrelated tasks that share common names such as `Book` or `add`. Treating them as exclusions is a policy choice for the owner, made with the flagged count in view.
4. *Can the same approval be used twice?* No. The ledger refuses a second `integrity_scan_opened` for an approval SHA-256 it has already seen.
