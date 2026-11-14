# Component: evaluation pipeline (generate, execute, seal, score)

Post-training evaluation generates on a GPU and executes code somewhere else: pool tasks on
the Mac's Docker sandbox, where their hidden tests stay, and the public benchmarks on a
disposable GitHub Actions runner. The two halves are separate commands joined by a
*completions file* whose SHA-256 is its identity. Every later record (results files, the
seal ledger, benchmark artifacts, the scorecard) names that hash.

```text
checkpoint + suite + protocol + sampling
        │  scratchcoder eval-generate            (GPU; runs no generated code)
        ▼
completions.jsonl  ── SHA-256 ──┬─ pool dev ──► scratchcoder eval-execute ──► results.jsonl
                                ├─ pool test ─► scratchcoder eval-seal (ledger)
                                │                 … all four endpoints frozen …
                                │               unseal record + eval-execute ──► results.jsonl
                                │                                                + test baseline
                                │               (eval-test-baseline retries an unlogged baseline)
                                └─ benchmarks ► tools/benchmark_dispatch.py ──► Actions artifact
pool dev tasks ──► scratchcoder eval-baseline ──► constant-answer baseline
results + artifacts + eval-choice reports + baselines ──► scratchcoder scorecard
```

## Worked example

One pool dev task, `add_one`, with the public test `assert add_one(1) == 2` and the hidden
tests `assert add_one(2) == 3` and `assert add_one(-1) == 0`.

1. **Generate (chat-v1, greedy).** The prompt is `<|bos|><|user_start|>Write add_one(x).\n\nassert add_one(1) == 2<|user_end|><|assistant_start|>`, say 45 tokens. The budget is `min(512, 1,024 − 45) = 512` new tokens, and generation stops at `<|python_end|>` or `<|assistant_end|>`. The attempt's seed is `attempt_seed(0, task_id, 0)`, recorded even though greedy decoding draws nothing. The row keeps the prompt length and the SHA-256 of its token IDs, the generated token IDs, the decoded text, the extracted program and the stop reason (`stop`).
2. **Execute.** On the Mac, `eval-execute` rebuilds the prompt with the tokenizer and checks its hash, decodes the tokens and checks the text, and takes the first closed python block again. The block must equal the stored program, or the file is refused. The program then runs with the hidden tests through `isolated_execution`: one fresh container, 2 s for a function task. Suppose it passes. The results line records `passed`, the program's SHA-256 and the reward channel's static audit, `audit_flags: []`.
3. **A host stall.** Had the container hung past the host watchdog, the runner would have returned `timeout` with `host_stalled` set. That is not a verdict on the program, so the executor waits 2 s and runs it again. The second try passes, and the line records `host_failures: 1`. If all four tries had failed, the line would say `unresolved`, and the next run of the same results file would try again.
4. **The constant-answer baseline.** The reward channel can parse these hidden tests, so `degenerate_baseline` scores them on the host. `add_one(2) == 3` holds for no constant. `add_one(-1) == 0` holds for the constant `False`, since `False == 0` in Python, as the assertion itself would decide. The best constant earns 1/2 = 0.5, and it does not fully pass the task.
5. **Score.** With n = 8 samples and 8, 4, 2 and 0 passes on four tasks, the mean pass rate is (1 + 0.5 + 0.25 + 0) / 4 = 0.4375. pass@8 is (1 + 1 + 1 + 0) / 4 = 0.75. The scorecard row shows these beside the suite's constant baseline.

## Data flow and shapes

**Completions file** (`scratchcoder.eval.completions`): JSON Lines, one header, then `task_count × samples` attempt records in (task ID, sample) order.

- **Header:** suite and split; protocol name, record and the record's SHA-256; checkpoint SHA-256; tokenizer identity and vocabulary size; model context; task-file SHA-256, task count and the SHA-256 of the sorted task IDs; excluded tasks with reasons; sampling (samples, temperature, top-p, top-k, seed); max new tokens; sampler version (sampled files only); the seed rule; precision, device, device name, batch size and generation seconds; git commit, dirty flag and creation time.
- **Attempt:** task ID, sample, seed, prompt token count and prompt SHA-256, generated token IDs and their count, completion text, program (or `null`), stop reason (`stop`, `length` or `context_limit`) and budget.
- **Identity:** the SHA-256 of the file's bytes.
- **Writing:** a temporary file, then a rename; an existing file is never replaced.
- **Reading:** checks every field, the attempt grid, every seed, and that the protocol record matches this code's record for that name. `validate_against_tasks` checks the task file's SHA-256 and IDs.

**Suites and protocols** (`scratchcoder.eval.protocols`):

| Suite | Split | Tasks | Protocols |
| --- | --- | ---: | --- |
| `pool` | `dev`, `test` | 2,000 each | `chat-v1` (prompt + public test), `base-pool-completion-v1` |
| `humaneval` | `eval` | 164 | `chat-v1` (benchmark prompt), `base-completion-rstrip-v1` |
| `humaneval-plus` | `eval` | 163 (task 32 excluded in every file) | as HumanEval |
| `mbpp-plus` | `eval` | 378 | as HumanEval |

Budgets are `min(512, context − prompt)` for chat-v1 and the base pool protocol, and `min(256, context − prompt)` for the base benchmark protocol. A prompt with no room left is a `context_limit` attempt; it is never shortened. The record hashes are pinned in `tests/test_completions.py`.

**Results file** (`scratchcoder.eval.executor`): append-only JSON Lines.

- **Header (schema 2):** names the completions SHA-256, embeds the completions header and lists every task's source and authorship. Its `execution` holds the *execution policy*, its SHA-256, and, apart, the audit policy and its SHA-256 (see [Execution-policy identity](#execution-policy-identity)). Schema 1 headers recorded only the runner and supervisor file hashes, the image, the deadline rule text, retries and the audit name.
- **Records:**
  - each `window` (with the execution- and audit-policy SHA-256s it ran under) and `window_end` (counts and sandbox seconds);
  - one `attempt` per scored attempt, with its status (`passed`, `failed`, `timeout`, `output_limit`, `resource_limit`, `oversized`, `no_program` or `context_limit`);
  - an `unresolved` record for an attempt whose host failures outlasted the retries.
- **Resuming:** a later run skips scored attempts and retries unresolved ones, but only under the same execution policy; any difference is refused before a byte is appended. An interruption can leave a torn last line, which the next run drops before appending. A run holds an exclusive lock on the file, so two runs never append to it at once.
- **Older files:** `read_results` and the scorecard read schema 1 and schema 2 files alike. A schema 1 file is never resumed: its header cannot show which limits and code produced its verdicts, and the executor does not infer one after the fact. Start a new results file instead.

### Execution-policy identity

`scratchcoder.eval.execution_policy` builds the record a resumed run must match. It has two halves.

*By value*, so a changed constant or a function replaced in memory shows:

| Field | Current value |
| --- | --- |
| `limits.function` | 2 s, 128 MiB, 16 processes, 65,536 output bytes, 128,000 input bytes |
| `limits.io_by_cases` for 0, 1, 40, 99, 100 and 1,000 cases | 2, 2.25, 12, 26.75, 27 and 60 s (other limits as above) |
| `deadline_rule` | the same rule in words, built from those values |
| `scored_statuses` | `failed`, `output_limit`, `passed`, `resource_limit`, `timeout` |
| `host_retries`, `host_backoff_seconds` | 3; 2, 4, 8 s |
| `input_limit`, `oversized_status` | 128,000 bytes, the limit the executor passes to the scheduler; `oversized` |
| `unexecuted_statuses` | `context_limit`, `no_program` |
| `image`, `host_watchdog_seconds` | the pinned digest; 10 s |
| `execute` | `scratchcoder.eval.harness.isolated_execution` (a callable object is named by its class) |

*By code*: a SHA-256 of the syntax tree, with docstrings dropped, of each enumerated unit:

| Unit | Why it decides verdicts |
| --- | --- |
| `sandbox/runner.py` (whole) | limits, the hardened `docker run` command, the host watchdog, the status mapping |
| `sandbox/program.py` (whole) | the in-container supervisor: deadline, completion marker, I/O comparison |
| `sandbox/budget.py` (whole) | the I/O deadline rule |
| `eval/harness.py`: `SCORED_STATUSES`, `HOST_RETRIES`, the backoff constants, `execution_limits`, `isolated_execution`, `is_host_failure`, `host_backoff`, and every import | limits per call, and what is a verdict versus a host failure |
| `eval/parallel.py`: `Execute`, `OVERSIZED`, `INPUT_LIMIT`, `Call`, `Outcome`, `_size`, `resolve_call`, `run_calls`, and every import | the oversized rule and the retry loop |
| `eval/executor.py`: `NO_PROGRAM`, `CONTEXT_LIMIT`, `CALL_INPUT_LIMIT`, `attempt_plan`, `scored_status`, `attempt_passed`, `run_verdict_calls`, and every import | which attempts run and with which hidden tests, what an unexecuted attempt scores, the input limit passed on, which outcomes are scored, and which status counts as passed |

Comments never reach the syntax tree, and docstrings are removed from it, so documentation edits leave the identity unchanged. So do edits to anything not listed, such as `harness.evaluate`, the executor's record writing and seal checks, or `docs/`. The executor keeps its verdict decisions in the small listed functions for this reason: `execute_completions` itself only wires them together. A listed definition that is renamed or removed is an error, not a silently narrower identity; a test checks that the listed definitions use scratchcoder code only from other listed units. The Python minor version is recorded with the hashes, because the syntax-tree dump can change between versions. The program extraction is not in the policy: it is already checked by re-deriving every stored program against the completions file, whose hash the header names.

The reward channel's static audit only labels passing attempts, so it never changes a verdict. Its identity (the syntax-tree hash of `sandbox/reward.py`) is recorded in the header and in every window but not compared. `ResultsFile.audit_policies` lists the distinct audit hashes the windows recorded; two or more mean the audit-flag counts mix two audits.

*Example.* A dev run is interrupted after 1,200 of 2,000 attempts. Before it resumes, someone changes `IO_SECONDS_PER_CASE` in `budget.py` to 0.5 s. The resumed run computes `limits.io_by_cases["99"]` as 51.5 s, not 26.75 s, and refuses with `(code: scratchcoder.sandbox.budget; deadline_rule; limits differ)`. Nothing is appended. Reverting the change, or starting a new results file, are the only ways on. Had they only reworded the `io_seconds` docstring, the run would have resumed.

**Seal ledger** (`scratchcoder.eval.seal`): hash-chained JSON Lines events `sealed`, `opened`, `resumed`, `completed`, `baseline` and `withdrawn`. `withdrawn` (`scratchcoder eval-withdraw`) names every sealed file of one endpoint's checkpoint and the reason, and is final: the files are never sealed again, listed in an unseal record or executed, nothing more of that checkpoint is sealed, and it is refused once any read has opened. Run 1's two SFT test files are withdrawn this way as a formatting failure, and run 1b's SFT weights stand for the SFT endpoint. Each event names the SHA-256 of the line before it. A *head record*, `<ledger>.head`, beside the ledger holds the event count and the last line's hash and is rewritten after every append. Each test-read window in a results file also names the ledger's count and head hash when the read opened.

**Benchmark artifact** (`tools/benchmark_score.py`): `benchmark-<suite>-<sha256>.json` in the artifact `benchmark-results-<sha256>`. It holds the completions header, the canonical preflight, one entry per attempt, the per-task constant answers and the sandbox seconds.

**Scorecard** (`scratchcoder.eval.scorecard`): one row per checkpoint and completions file, plus ARC rows, then the paired comparisons, GPU- and sandbox-minute totals per checkpoint and suite, and the leak flags. Each row carries:

- the score;
- the suite's constant-answer baseline beside it;
- pool slices;
- unresolved counts;
- GPU- and sandbox-minutes;
- the checkpoint, tokenizer, task-file and protocol hashes and the precision.

## Design choices and common mistakes

- **Extraction happens where the code runs.** On the Mac, chat-v1 extraction reads the token IDs with the tokenizer, because decoded text cannot tell a real `<|python_end|>` from those bytes inside code. The stored program is a cross-check. The Actions runner has no tokenizer, so for chat-v1 benchmark files it uses the stored program; for the base benchmark protocol it re-derives the program from the text and requires agreement.
- **The I/O deadline is the pool verification's.** Every dev I/O task has 99 hidden cases, and each case starts a fresh interpreter under amd64 emulation. A flat 2 s deadline would fail the verified references. `isolated_execution` therefore gives an I/O call 2 s plus 0.25 s per case, at most 60 s, the rule the references passed under (`scratchcoder.sandbox.budget.io_seconds`). Function calls keep 2 s.
- **A host failure is never a score.** `unavailable`, `infrastructure_error` and any result with `host_stalled` are retried up to three times, waiting 2, 4 and 8 s (doubling, at most 30 s). An attempt still failing is unresolved: the scorecard leaves its task out of that row's metrics, counts it and marks the row incomplete. After 32 unresolved attempts in a row (4 × workers, at least 16), the executor stops starting calls. The file stays resumable. `scratchcoder.eval.harness.evaluate` also retries stalls now, and aborts if one persists.
- **Only a program too large to send is `oversized`.** The executor compares the program plus its tests with the input limit of the limits the calls run under (128,000 bytes by default, 2 MiB for HumanEval+ and MBPP+) and scores a larger one `oversized` without running it. Any other `ValueError` from execution (hidden tests that are not valid JSON, more than 100 I/O cases, tests that alone exceed the limit) is a defect of the task: it stops the run, and is never scored against the candidate.
- **`eval-generate` on MPS is greedy only.** The seeded sampler works in float64, which MPS does not support, so a sampled run on `mps` is refused before the checkpoint loads. Sampled files come from `cpu` or `cuda`.
- **The older `scratchcoder eval` path cannot score the test split.** `evaluate` refuses tasks labelled `test`, because that path writes no ledger entry. The pool test split is scored only by `eval-execute` under an unseal record.
- **The test split is read once.** `eval-execute` refuses test completions unless they are sealed in the ledger and an unseal record lists them. The record must name all four frozen endpoints (`base`, `sft`, `rl`, `more-sft`), each a different checkpoint, and every completions file it lists must be sealed. The file must also sit under the endpoint whose checkpoint generated it. The read is logged (`opened`) before anything runs, and the ledger and the record are checked before the sealed task file is opened. The tokenizer is required, so the one read is never spent on completions whose prompts and tokens were not checked. An interrupted read resumes on the same results file (`resumed`). Once a read completes, or while another results file holds it, a second execution is refused.
- **Regenerating buys nothing.** One unseal record governs a ledger. After any read has opened, no file can be sealed and every later read must present the same record. A record may not list two files of one checkpoint, protocol and decoding (greedy or sampled), so a regenerated greedy file cannot be read as a second look at the same endpoint.
- **A results file keeps one execution policy.** Resuming compares the whole recorded policy, not a few hand-picked fields, and refuses any difference, so a long run resumed after a rebase never mixes verdicts from two policies. Before schema 2, only the image, the deadline-rule text and the runner and supervisor file hashes were compared; a changed limit in the harness, the I/O budget, the scored statuses, the input limit, the retries or the execute function went unnoticed, and so would a different host-failure rule.
- **A refused resume reads no hidden tests.** The policy and a resumed file's header (completions hash, schema, policy, unseal record) are checked before the task file is opened, so on the test split a refused resume, or a renamed listed definition, never parses the sealed hidden tests and writes no ledger event.
- **The verdict code is enumerated, not discovered.** An import scan would pull in modules that never touch a verdict, and a file hash would refuse a resume over a docstring. The enumeration is a deliberate list; changing it bumps `EXECUTION_POLICY_VERSION`, which by itself refuses older schema 2 files.
- **The test baseline is computed inside the read.** It reads hidden tests, so `pool_constant_baseline` refuses the test split unless the unseal step calls it. It runs after the first complete test read and is written beside the ledger. The ledger records it with its SHA-256, and later reads reuse it. A baseline with an unresolved call is not recorded, so a later read computes it again. If the last read leaves it unlogged, `eval-test-baseline` retries it, but only after every file the unseal record lists has a completed read and only on the task file those files name. A baseline file saved without its ledger event (an interruption between the two) is checked against the task file and recorded, not computed again.
- **Constant answers cover every pool task or say why not.**
  - *Parseable tasks:* the reward channel scores them on the host and runs no code.
  - *Other function tasks:* each answer becomes a program: every function the hidden tests call returns the constant, or its first argument for echo. The program runs through the evaluation runner with each hidden `assert` wrapped so it reports its own verdict, which gives the fraction passed.
  - *Other I/O tasks:* the program prints the constant, or echoes the input. One call runs all the cases, and the host compares that fixed output with each expected output.
  - *Uncovered tasks:* a task none of these covers is listed as uncovered with a reason, for example hidden tests that call no candidate function and only inspect objects the candidate must create.
- **Benchmarks use their own scoring paths.** HumanEval runs on the reference gate's stdlib image with the default limits. HumanEval+ and MBPP+ run on the EvalPlus NumPy image with 60 s, 4 GiB and a 2 MiB input cap. All three require a passing canonical preflight before any completion is read. Each benchmark task's constant programs return a fixed value, a doctest or public-assert example value, or the first argument. MBPP+ takes the function to define from the public tests' calls plus any function the reference solution defines that those tests call, so a task whose function shares a builtin's name (`mbpp/126`, `sum`) is covered. They run in the same Actions job, and the suite's baseline is the share of tasks some constant passes.
- **The dispatch link is public.** Workflow inputs appear in the run metadata of a public repository. `tools/benchmark_dispatch.py` (run as `python -m tools.benchmark_dispatch`) therefore takes only public-benchmark completions and checks the R2 object's size and SHA-256. It signs a GET link for that one object that expires within an hour, and the workflow downloads it as its first step. The link holds the access key ID and a signature, never the secret. `gh` runs without R2, AWS or Runpod variables in its environment.
- **Pairs are paired.** A comparison uses only the tasks scored in both rows, from the same task file. It reports the excluded count and uses `paired_bootstrap` with 2,000 resamples and seed 0. SFT versus base on the pool is labelled cross-protocol, because chat-v1 and `base-pool-completion-v1` differ. The leak flag marks any checkpoint above 20% greedy pass@1 on HumanEval; a checkpoint without a complete greedy HumanEval row is listed as not evaluated, not as clean, and an incomplete row already above 20% is still flagged.
- **Base-pool stops.** Since 2026-10-26, an I/O task's base completion stops at neither `\nprint(` nor `\nif __name__`. A stdin/stdout program prints at the top level and is often run from a main guard. The name stays `base-pool-completion-v1`; the record carries the stop table, so its hash changed.

## Verification

- `tests/test_completions.py`: pinned protocol hashes, the stop-table change, prompts, budgets and extraction by hand, round trip and immutability, and every read-time refusal.
- `tests/test_eval_generate.py`: equality with direct `generate_many`, seeds independent of batch size, context limits, stop tokens, the task-32 exclusion, devices and the CLI.
- `tests/test_seal_withdrawal.py`: a withdrawal names every sealed file of the checkpoint; withdrawn files and their checkpoint are refused by every later seal, unseal record and read; the replacement checkpoint seals and executes; nothing is withdrawn once a read opened; malformed withdrawals change nothing; the command.
- `tests/test_eval_executor.py`: scoring, the audit, stall and host-failure retries, resume after an interruption and a torn line, the circuit breaker, refusals, and the whole seal: refusals, one read, resume, malformed unseal records, the ledger chain and its head record, regeneration and second unseal records, the preflight before the task file opens, the tokenizer requirement and the baseline retry. For the execution policy: a changed function timeout, I/O budget, deadline rule text, per-case constant, scored statuses, input limit, image, retry count or execute function, or edited host-status, oversized, supervisor, runner, budget or executor pass, unexecuted-status or input-limit code, refuses a resume and appends nothing; a refused test-split resume never reads the task file; docstring, comment and unlisted-code edits resume; an audit change resumes and shows as two audit policies; a renamed listed definition is an error; an interrupted and twice-resumed run records the same attempts as an uninterrupted one; a schema 1 file stays readable by the scorecard but is never resumed.
- `tests/test_execution_policy.py`: the recorded limits against the deadline rule, docstring and comment insensitivity, code sensitivity of listed units only, the closure of the enumeration under its scratchcoder imports, callable names and named differences.
- `tests/test_parallel_calls.py`, `tests/test_constant_baseline.py`: retry bounds, the oversized rule under the default and a given input limit and task-side errors, the baselines by hand (executed, uncovered, I/O, ARC, benchmark programs, a builtin-named MBPP+ entry) and the test-split refusal.
- `tests/test_benchmarks.py`, `tests/test_benchmark_dispatch.py`: canonical blocking, unresolved benchmark attempts, the scoring tool (an MBPP+ task whose tests exceed 128,000 bytes is scored under the 2 MiB limit), the workflow's inputs and step order, the AWS presigned-URL reference vector and the dispatch tool with mocked `gh` and R2.
- `tests/test_scorecard.py`: metrics, slices, baselines, unresolved counts, the fixed-seed pairs, the leak flag, ARC rows and the CLI.
- `tests/test_harness.py`: `evaluate` retries stalls, gives I/O calls the deadline rule and refuses the sealed test split.

```sh
uv run --locked pytest tests/test_completions.py tests/test_eval_generate.py \
  tests/test_eval_executor.py tests/test_execution_policy.py tests/test_parallel_calls.py tests/test_constant_baseline.py \
  tests/test_benchmarks.py tests/test_benchmark_dispatch.py tests/test_scorecard.py
SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_eval_pipeline_sandbox.py
```

The Docker-backed test runs reference and wrong programs through the real runner on authored dev-shaped tasks: a function task, a class task and a 40-case I/O task. It also runs an executed constant baseline. With `SCRATCHCODER_POOL_DEV` set to the pool's dev file, it also runs one short real dev task per source, including a 99-case TACO task. Every reference passed and every wrong program failed.

**Pool dev constant-answer baseline (measured 2026-10-27).** Mean best-constant reward, with the share of tasks a constant answer fully passes:

| Source | Tasks | Covered | Mean reward | Fully passed |
| --- | ---: | ---: | ---: | ---: |
| OpenCodeInstruct | 1,850 | 1,845 | 0.298 | 4.0% (74) |
| MBPP | 124 | 124 | 0.296 | 12.9% (16) |
| TACO | 26 | 26 | 0.216 | 0.0% |
| All | 2,000 | 1,995 | 0.297 | 4.5% (90) |

The reward channel scored 1,893 tasks on the host. Constant programs covered 102 more, in 2,880 sandbox calls. Five OpenCodeInstruct tasks are uncovered: their tests only inspect objects the candidate must create, and call no function a constant could stand in for.

## Questions and answers

1. *Why does the executor need the tokenizer for chat-v1?* Extraction reads token IDs. Only with the tokenizer can it rebuild the prompt, decode the tokens and take the block again, instead of trusting the stored program.
2. *An attempt is unresolved after four tries. Is it a failure?* No. It is counted and its task leaves that row's metrics, which marks the row incomplete. Running the same results file again retries it.
3. *Why compute the test baseline only inside an unseal read?* It reads the hidden tests: the parse needs them, and the executed constants run against them. Computing it earlier would read sealed tests before the endpoints are frozen.
4. *What stops a second look at the test split?* The ledger. A completed read, or an open read held by another results file, refuses any new execution of those completions. Editing or deleting a ledger line breaks its hash chain. The chain alone cannot show that the last lines were removed, so the head record, kept beside the ledger, names the count and last hash and a shorter ledger is refused. The two files must travel together; this guards against accidents and casual edits, not against someone who rewrites both. Regenerating the completions does not help: after the first read no new file can be sealed, and every read must use the one unseal record.
5. *Why is a HumanEval score above 20% flagged?* The reference model scores 14 of 164 under the protocol of record. A 135M rehearsal model far above that more likely saw the benchmark than solved it, so the flag stays until a contamination check clears it.
6. *I edited a comment in `parallel.py` mid-run. Will the resume be refused?* No. Comments and docstrings never reach the hashed syntax tree. Changing `resolve_call`'s code would be refused, and so would changing a limit in memory, because the limits are also recorded by value.
7. *Why is a schema 1 file not resumed under its old four-field check?* That check missed the limits and the host-failure rule, and nothing in the old header can show which ones its verdicts came from. Inferring them now would label old verdicts with an identity they were never checked against. A complete schema 1 file needs no resume and stays readable for scoring; an incomplete one is finished by executing into a new results file.
