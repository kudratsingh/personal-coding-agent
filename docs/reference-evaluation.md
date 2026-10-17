# Reference-model evaluation

## Stripped-prompt protocol (step 1d; owner acceptance pending)

The explicitly authorized step 1d comparison uses the pinned model and benchmark, original five stop strings (including their unchanged position-zero matching), greedy decoding and a 256-token budget. Before tokenization, the raw benchmark prompt is transformed with Python `rstrip()`. The scored program is that exact stripped prompt plus the retained decoded completion; leading completion whitespace is not removed. Reports and scorecards record `prompt_stripped: true` and protocol `base-completion-rstrip-v1`. The source benchmark/task hash remains based on unmodified task records. Canonical answers/tests are unchanged. This tests the owner's trailing-whitespace hypothesis; no claim about the card's original unpublished recipe or successful reproduction is made before the result.

Project base-model code evaluation uses the same shared stripping, stops and candidate assembly. Explicit `--prompt-mode instruction` retains the previous public-tests/fence-extraction path and is labeled separately; historical instruction-style results are not silently relabeled as base completions. No additional benchmark variant is authorized after this single run. The stated 12–13-pass success branch permits keeping the change and proceeding to the remaining ordered steps; otherwise the experiment stops for owner review.

The [single step 1d run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/39042645498), at source `e16667b012fd5feeb29f40eb54c9c100d6c2b0e4`, finished on 2026-10-17: **14/164 passed (8.54%), 0 empty retained completions, and 157 completions reached the 256-token budget**, all 157 without EOS. Canonical preflight passed **164/164**. Checkpoint, tokenizer, task, model/data revisions and runtime identities matched the previous pinned comparison; the run records `prompt_stripped: true`. Failure categories were **105 AssertionError, 23 SyntaxError, 10 TypeError, 6 RecursionError, 4 NameError, 1 AttributeError and 1 ValueError** (150 failures plus 14 passes).

The result is 1.1424 equivalent tasks from the card's 7.84%, within a mathematical two-task tolerance but outside the owner's explicit “12 to 13 passes” success band. Acceptance is therefore awaiting owner clarification: [PR #66](https://github.com/kudratsingh/personal-coding-agent/pull/66) remains draft and unmerged, reference G3 is not declared closed, and no further variant or ordered step 2–4 was started. The shared base-model default described above is part of that unmerged change. The card's original unpublished recipe is not established by this result.

The optional `baseline` dependency extra is used only by `tools/baselines/`; it is not a core training dependency. The HumanEval comparison pins the benchmark revision/content hash and the model revision, loads safetensor weights without remote model code, and executes every candidate through the isolated runner.

The tool uses greedy completion with an explicit token budget and stop strings, records all settings/hashes/outcomes, and compares the resulting rate with the model card's stated value. The card does not fully specify decoding settings, so a mismatch must be investigated rather than described as a reproduced score.

The manually dispatched Reference benchmark workflow uses a standard public-repository CPU runner. It does not provision rented GPUs or upload model caches, weights, or generated programs. Public benchmark provenance and outcome-status records are retained in its console log. Ordinary CI tests use synthetic rows and do not download the reference model.

An explicitly approved comparison can set `--disable-stops` (workflow input `disable_stops`) and `--max-new-tokens 512`. The report records the effective empty stop list, budget, empty-completion/budget counts and per-task failure categories; the default remains the original stop policy and 256-token budget. This changes only the requested protocol controls, not the pinned model, benchmark, raw prompts or greedy decoding.

For owner review, optional `--failure-recipient` accepts a public CMS certificate and emits only encrypted JSON for the first five failing decoded completions, in task evaluation order. The workflow accepts that certificate as input; the private key stays on the owner's machine. No plaintext completion, prompt or hidden test is added to the public report/log. Decryption preserves the completion verbatim, including empty strings and newlines. This transport is not a signed result or durable private backup. [OpenSSL CMS](https://docs.openssl.org/3.5/man1/openssl-cms/) performs the encryption/decryption; it never executes the source text.

## Observed comparison

The [initial recorded run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38780757269) completed all 164 tasks with greedy decoding and a 256-token completion budget: **1/164 passed (0.61%)**, versus the card's stated 7.84%. This is not a reproduced score. The card marks its score unverified and does not specify the full decoding protocol. That uncertainty is a limitation, not an established explanation for the gap.

Subsequent runs first require every canonical solution to pass the same hidden tests through isolation. `--canonical-only` runs this check without loading model weights. A failed canonical check stops model benchmarking. Model reports additionally retain allowlisted failure categories, never raw exception text, generated programs, or hidden tests. Token-budget comparisons must be recorded separately; do not tune on the benchmark and present the best result as an independent validation.

The [canonical-only validation run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38870605573) passed **164/164**. This establishes correct handling of these known answers under this runner, not published-score reproduction or general sandbox security.

The [completed diagnostic rerun](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38870858985) preserved the original greedy/256-token settings and again passed **1/164**, after all 164 canonical answers passed its preflight. Reported model-output categories were 147 assertion failures, 6 syntax errors, 6 type errors, 2 name errors, 2 other failures, and 1 pass; no infrastructure failure was reported. These categories establish repeatability and help narrow investigation, but do not prove the cause of the gap to the card's score. The reference reproduction gate remains open.

## Observational generation diagnostics

The [completed observational run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38887684383) again passed all 164 canonical checks and scored 1/164. All 164 candidate hashes matched the preceding diagnostic run. It additionally records generated token counts, whether the budget was reached, EOS presence, the first matched existing stop-string index, retained/decoded character counts, and syntactic completeness of the unchanged candidate. AST parsing inspects text only; generated programs still execute exclusively through the sandbox. No source text, token IDs, hidden tests or raw syntax-error messages enter these diagnostic records. Both per-task observations and aggregate counts are retained. Fixture tests verify existing truncation behavior, EOS-on-the-last-budget-token accounting, syntax checks and source-text exclusion.

Of 164 completions, 128 reached the budget without EOS, and 36 contained EOS. Existing stop strings matched 126 outputs. There were 87 empty retained completions: 57 stopped at the existing definition marker, while 30 had no stop-string match and contained EOS. Only eight outputs both exhausted the budget without EOS and had no stop-string match. These overlapping observations must not be summed as disjoint failure categories; parsing or a stop match does not establish semantic correctness.

These observations did not alter prompts, model/tokenizer revisions, greedy decoding, budget, stop strings or benchmark tests. A complete canonical preflight is still required. Truncation/EOS counts by themselves cannot establish the cause of the score discrepancy; the reference reproduction gate remains open.

The [pinned model card](https://huggingface.co/bigcode/tiny_starcoder_py/blob/8547527bef0bc927268c1653cce6948c5c242dd1/README.md) does not supply a complete evaluation protocol. Arithmetically, 7.84% cannot be a single deterministic integer pass count over 164 tasks rounded to two decimals: 12/164 is 7.32%, and 13/164 is 7.93%. This suggests an averaged or otherwise different protocol, but is an inference—not an explanation or reproduction of the reported score. No benchmark-driven search for a better-looking decoding setting is authorized by this diagnostic work.

The [pinned saved generation configuration](https://huggingface.co/bigcode/tiny_starcoder_py/resolve/8547527bef0bc927268c1653cce6948c5c242dd1/generation_config.json) contains only model-origin, token-ID and library-version metadata, not a benchmark recipe specifying sample count, decoding budget or stop strings. It therefore does not resolve the score discrepancy or justify silently changing the comparison protocol.


## Authorized no-stop, 512-token comparison

The [2026-10-17 full run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/39035567130) at source `c4451dd9684264a3f89628f2dc8c786baa92f1e1` disabled all stop strings and used a 512-token completion budget, as explicitly requested. The pinned model revision, raw prompt format, greedy decoding, benchmark bytes/task hash, tokenizer, float32 CPU execution, two threads and locked library versions matched the preceding observational run. All 164 canonical solutions passed the same isolated preflight.

**Model result: 0/164 passed (0.00%); 30 empty/whitespace-only decoded completions; 128 reached 512 tokens, all without EOS.** The other 36 generations contained EOS. Failure totals were 103 SyntaxError, 55 AssertionError, 2 TypeError, and 1 each of AttributeError, IndentationError, NameError and other_failure. These total 164 failures, not infrastructure errors.

The difference from the card's 7.84% is 12.8576 equivalent tasks, outside the two-task criterion. Disabling stops and increasing the budget did not reproduce the card score. This does not explain the unpublished original protocol or establish a different decoding recipe. The conditional stop-at-zero fix was not authorized by this outcome and was not applied. No further diagnostic variant or later ordered step was started; reference acceptance remains open pending owner direction.

All per-task outcomes are below. The first five failing decoded completions in actual evaluation order (HumanEval/0, /1, /10, /100, /101) were returned encrypted to the owner, decrypted locally and checked against the evaluated candidate hashes. The first two completions contain zero characters. Verbatim source text, hidden tests, weights and private run artifacts are not published here.

<details>
<summary>Per-task failure outcomes: all 164 tasks</summary>

| Task | Recorded failure category |
| --- | --- |
| HumanEval/0 | AssertionError |
| HumanEval/1 | AssertionError |
| HumanEval/2 | SyntaxError |
| HumanEval/3 | AssertionError |
| HumanEval/4 | TypeError |
| HumanEval/5 | AssertionError |
| HumanEval/6 | AssertionError |
| HumanEval/7 | AssertionError |
| HumanEval/8 | AssertionError |
| HumanEval/9 | AssertionError |
| HumanEval/10 | SyntaxError |
| HumanEval/11 | AssertionError |
| HumanEval/12 | AssertionError |
| HumanEval/13 | SyntaxError |
| HumanEval/14 | AssertionError |
| HumanEval/15 | SyntaxError |
| HumanEval/16 | SyntaxError |
| HumanEval/17 | AssertionError |
| HumanEval/18 | SyntaxError |
| HumanEval/19 | SyntaxError |
| HumanEval/20 | AssertionError |
| HumanEval/21 | AssertionError |
| HumanEval/22 | AssertionError |
| HumanEval/23 | SyntaxError |
| HumanEval/24 | SyntaxError |
| HumanEval/25 | AssertionError |
| HumanEval/26 | AssertionError |
| HumanEval/27 | SyntaxError |
| HumanEval/28 | AssertionError |
| HumanEval/29 | AssertionError |
| HumanEval/30 | SyntaxError |
| HumanEval/31 | AssertionError |
| HumanEval/32 | SyntaxError |
| HumanEval/33 | SyntaxError |
| HumanEval/34 | SyntaxError |
| HumanEval/35 | SyntaxError |
| HumanEval/36 | AssertionError |
| HumanEval/37 | SyntaxError |
| HumanEval/38 | SyntaxError |
| HumanEval/39 | SyntaxError |
| HumanEval/40 | SyntaxError |
| HumanEval/41 | SyntaxError |
| HumanEval/42 | SyntaxError |
| HumanEval/43 | SyntaxError |
| HumanEval/44 | SyntaxError |
| HumanEval/45 | AssertionError |
| HumanEval/46 | SyntaxError |
| HumanEval/47 | SyntaxError |
| HumanEval/48 | SyntaxError |
| HumanEval/49 | SyntaxError |
| HumanEval/50 | SyntaxError |
| HumanEval/51 | SyntaxError |
| HumanEval/52 | SyntaxError |
| HumanEval/53 | SyntaxError |
| HumanEval/54 | SyntaxError |
| HumanEval/55 | SyntaxError |
| HumanEval/56 | SyntaxError |
| HumanEval/57 | SyntaxError |
| HumanEval/58 | SyntaxError |
| HumanEval/59 | other_failure |
| HumanEval/60 | SyntaxError |
| HumanEval/61 | SyntaxError |
| HumanEval/62 | SyntaxError |
| HumanEval/63 | SyntaxError |
| HumanEval/64 | AssertionError |
| HumanEval/65 | SyntaxError |
| HumanEval/66 | AssertionError |
| HumanEval/67 | AssertionError |
| HumanEval/68 | SyntaxError |
| HumanEval/69 | AssertionError |
| HumanEval/70 | SyntaxError |
| HumanEval/71 | SyntaxError |
| HumanEval/72 | SyntaxError |
| HumanEval/73 | SyntaxError |
| HumanEval/74 | SyntaxError |
| HumanEval/75 | SyntaxError |
| HumanEval/76 | AssertionError |
| HumanEval/77 | SyntaxError |
| HumanEval/78 | SyntaxError |
| HumanEval/79 | AssertionError |
| HumanEval/80 | AssertionError |
| HumanEval/81 | SyntaxError |
| HumanEval/82 | SyntaxError |
| HumanEval/83 | AssertionError |
| HumanEval/84 | AssertionError |
| HumanEval/85 | SyntaxError |
| HumanEval/86 | AssertionError |
| HumanEval/87 | SyntaxError |
| HumanEval/88 | SyntaxError |
| HumanEval/89 | SyntaxError |
| HumanEval/90 | AssertionError |
| HumanEval/91 | SyntaxError |
| HumanEval/92 | SyntaxError |
| HumanEval/93 | SyntaxError |
| HumanEval/94 | AssertionError |
| HumanEval/95 | SyntaxError |
| HumanEval/96 | SyntaxError |
| HumanEval/97 | SyntaxError |
| HumanEval/98 | IndentationError |
| HumanEval/99 | SyntaxError |
| HumanEval/100 | SyntaxError |
| HumanEval/101 | AssertionError |
| HumanEval/102 | SyntaxError |
| HumanEval/103 | SyntaxError |
| HumanEval/104 | SyntaxError |
| HumanEval/105 | SyntaxError |
| HumanEval/106 | SyntaxError |
| HumanEval/107 | AssertionError |
| HumanEval/108 | AssertionError |
| HumanEval/109 | SyntaxError |
| HumanEval/110 | AssertionError |
| HumanEval/111 | NameError |
| HumanEval/112 | SyntaxError |
| HumanEval/113 | SyntaxError |
| HumanEval/114 | AssertionError |
| HumanEval/115 | SyntaxError |
| HumanEval/116 | AssertionError |
| HumanEval/117 | SyntaxError |
| HumanEval/118 | SyntaxError |
| HumanEval/119 | SyntaxError |
| HumanEval/120 | AssertionError |
| HumanEval/121 | AssertionError |
| HumanEval/122 | SyntaxError |
| HumanEval/123 | SyntaxError |
| HumanEval/124 | SyntaxError |
| HumanEval/125 | SyntaxError |
| HumanEval/126 | AssertionError |
| HumanEval/127 | AssertionError |
| HumanEval/128 | AssertionError |
| HumanEval/129 | AttributeError |
| HumanEval/130 | SyntaxError |
| HumanEval/131 | SyntaxError |
| HumanEval/132 | SyntaxError |
| HumanEval/133 | AssertionError |
| HumanEval/134 | AssertionError |
| HumanEval/135 | AssertionError |
| HumanEval/136 | SyntaxError |
| HumanEval/137 | SyntaxError |
| HumanEval/138 | AssertionError |
| HumanEval/139 | SyntaxError |
| HumanEval/140 | AssertionError |
| HumanEval/141 | SyntaxError |
| HumanEval/142 | SyntaxError |
| HumanEval/143 | AssertionError |
| HumanEval/144 | SyntaxError |
| HumanEval/145 | SyntaxError |
| HumanEval/146 | SyntaxError |
| HumanEval/147 | SyntaxError |
| HumanEval/148 | TypeError |
| HumanEval/149 | AssertionError |
| HumanEval/150 | AssertionError |
| HumanEval/151 | AssertionError |
| HumanEval/152 | SyntaxError |
| HumanEval/153 | SyntaxError |
| HumanEval/154 | SyntaxError |
| HumanEval/155 | SyntaxError |
| HumanEval/156 | AssertionError |
| HumanEval/157 | SyntaxError |
| HumanEval/158 | SyntaxError |
| HumanEval/159 | SyntaxError |
| HumanEval/160 | SyntaxError |
| HumanEval/161 | SyntaxError |
| HumanEval/162 | SyntaxError |
| HumanEval/163 | SyntaxError |

</details>
