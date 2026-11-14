# Component: static verifier scan

## Worked example

Take this small task. The prompt says "Return the common elements of two lists, in any order." The reference is:

```python
def common(a, b):
    return list(set(a) & set(b))
```

The hidden tests are `assert common([2, 1], [1, 2, 3]) == [1, 2]` and `assert common([5], [5]) == [5]`.

The scan parses the three texts with `ast` and never runs them. It reports:

- R1 `declared_checks`: 2 (both asserts mention `common`, which the reference defines).
- R7 `set_order`, sub-tag `impl_order_dependent`: the reference builds a list by iterating a set, and one test compares a call with an ordered list literal of two elements. Which order a set yields is an implementation detail, so an honest candidate that returns `[2, 1]` fails.
- R10 `order_free_clause`: the prompt says order does not matter, yet a test compares the call with an ordered list literal.

Had the reference returned `sorted(set(a) & set(b))`, or the test used `sorted(common(...)) == [1, 2]`, neither R7 nor R10 would fire.

## Data flow and shapes

One task record in, one `TaskScan` out. The record is a normalized task row: `id`, `prompt`, `mode`, `hidden_tests`, and the reference text in `solution`. `scan_task` returns the R1 metrics (`declared_checks`, `total_asserts`), an ordered tuple of hits (`rule`, `name`, optional `tag`, short `evidence` such as a line number and a snippet) and notes (`reference_missing`, `reference_unparseable`, `hidden_tests_unparseable`, and `scan_error` when a task's text exhausted the stack or memory; that task gets no hits and the scan continues). Each hit has a `task_parseable` field, set for R11 only (otherwise null). `summarize` turns many scans into counts per rule and per rule/sub-tag, plus, for R11, the same sub-tag counts split by `task_parseable=true` and `task_parseable=false`, the R1 histogram and the number of function tasks with two or fewer declared checks. Output is deterministic: keys are sorted and hits are ordered by rule number then tag.

The command-line tool `tools/verifier_scan.py` reads explicit JSONL paths and writes, under `--out`, one `scan-<label>.jsonl` per input (only tasks with at least one hit), `summary.json` (counts, the rule-set version, the commit passed with `--commit`, and the SHA-256 of every input) and `record.json` (every file it opened). Paths in both files are absolute. If any output already exists the tool exits with code 3 and writes nothing. `--id-join IDS POOL` takes an IDs-only file, such as a candidate list, and joins each task's text from the named pool file. IDs with no match are listed in the summary.

## The rules

- **R1 `declared_checks`.** Count of top-level asserts that mention a name the reference defines at its top level (a `def`, `class` or plain assignment; imports do not count). It is a metric on every function task. A task with few declared checks (two or fewer) leans on little reference-dependent evidence.
- **R2 `candidate_independent_assert`.** A top-level assert that mentions no such name. It passes whatever the candidate does.
- **R3 `none_only`.** Every top-level assert compares with `None`. Any candidate that returns `None` passes.
- **R4 `side_effect_prompt_none_only`.** R3 plus a prompt word that says the function acts through output: print, prints, display, writes, "write to", log, logs, "output to" and close variants (the list is `SIDE_EFFECT_KEYWORDS`). The bare imperative "write" is not a keyword, because "write a function" opens most prompts. Such a test cannot see the side effect.
- **R5 `raise_in_prompt`.** The prompt names a raise or an exception type. The current test forms cannot check one. Applies to function and I/O tasks.
- **R6 `tautological_compare`.** A comparison inside an assert whose operands are all literals.
- **R7 `set_order`.** The reference makes a list or tuple by iterating a set without `sorted`, and a test compares a call with an ordered literal of two or more elements. Names are bound per function scope: a set bound to `s` in one function does not taint `s` in another, and the module-level walk does not enter function bodies. A list sorted in place (`s = list(set(a)); s.sort()`) is skipped. `for x in <set>: out.append(x)` with `return out` is a hit. Covered: `list(set(...))`, `tuple(...)`, list comprehensions and generator expressions over a set, and names bound in the same function to set expressions (`set(...)`, set literals and comprehensions, `.intersection`, `.union`, `.difference`, `.symmetric_difference`, the operators `&`, `|`, `-`, `^`, or `set()` filled in a loop). Known false negatives: other accumulation forms (`out += [x]`, `out.extend(...)`) and sets that arrive as arguments. Anything inside `sorted`, `set`, `len`, `sum` and similar reducers is skipped. Sub-tags: `hash_order_candidate` when the expected elements are strings (their order can change between runs), `impl_order_dependent` otherwise.
- **R8 `reference_global`.** A test reads a module-level name that the reference assigns (not a `def` or `class`). Sub-tags: `reference_only_global` (the prompt never names it, so a candidate cannot know it) and `prompt_named_global` (the prompt names the identifier).
- **R9 `rounding_clause`.** The prompt says "rounded to N decimal places" and an expected float literal has more than N decimals (`4.0` and `1.67` are fine for two places). Floats used only as arguments are ignored. For I/O tasks the expected output text is searched for longer decimals.
- **R10 `order_free_clause`.** The prompt says order does not matter ("in any order", "regardless of order" and similar), yet a test compares an unwrapped call with an ordered literal of two or more elements. A test that wraps the call in `sorted` or `set`, or accepts either order with `or`, is fine.
- **R11 `redefinable_outer_bool_check`.** A check of a shape the reward channel parses (`CALL is True/False/None` in either orientation, `not CALL`, `CALL == <any representable literal>`, so `len(f(4)) == 5` counts and `== 0j` does not) whose outermost call is a builtin the reference does not define, or an attribute of a module the tests import. A candidate that redefines that builtin or patches that module attribute can change what the check sees. Sub-tags: `builtin_outer`, `module_attr_outer`. The module mirrors `_parse_predicate`, `_safe_literal`, `_representable` and `_call_is_safe` from the reward parser (it does not import them, because that module pulls in subprocess), and a test compares the mirror with the real parser on many shapes. Each hit also records `task_parseable`: true only when every top-level statement is an import or a parseable assert and there is at least one check, as `parse_function_checks` requires (its byte-size limits are ignored). A hit on an unparseable task is still reported, but the reward channel never reads that task, so the reachable exposure is the `task_parseable=true` count.

Assert-based rules (R1 to R4, R6 to R8, R10, R11) run only on function tasks. An I/O task's hidden tests are JSON input and output cases, so only R5 and R9 apply to it.

## Design choices and common mistakes

- **Diagnostic only.** Nothing in training, evaluation or the sandbox imports the module. It does not change a score or filter a task.
- **Text only.** It uses `ast` and regular expressions. It never runs, compiles for execution, imports or shells out on task, reference or generated text. A test reads the module's own source and fails if it names any such facility.
- **Flags are provisional.** Each rule is a heuristic. A hit says "look here", with its rationale in the evidence. It is not an executed verdict, and a task with no hit is not shown to be sound.
- **Known limits.** Asserts nested in functions or loops are not seen; a test written as a helper function is invisible to R1 to R3. Names are matched by identifier, not by scope, so a shadowing local could mislead R2 or R8. R5 matches words, so prose that merely mentions an exception also fires. R7 follows set values only within one function, and ignores sets that arrive as arguments. Known noise: R4 and R5 are word matches (R4 skips "log(n)", "log n" and "log in", and R5 ignores "with the exception of", but other prose can still fire), and R10 matches "order ... is not important" within one sentence only.
- **A missing reference** turns off R7 and R8 and falls back to the entry-point name for R1 and R2, with a note.

## The sealed-file guard

The command-line tool refuses to run, with a non-zero exit and before opening anything, if any given path (or its symlink target) is a sealed split: the names `test.jsonl`, `holdout.jsonl` and `rl_only.jsonl`; anything matching `eval-*-test-*`; any name containing `holdout` or `rl_only`; any name with `test` or `tests` as a whole word; or any path under a directory whose name contains `holdout`. The rule is conservative on purpose. Files named `dev.jsonl`, `train.jsonl` and `rl_candidates.jsonl` are allowed. The tool takes explicit paths only and never expands a pattern or walks a directory. A path that is a hard link to a sealed-named file in the same directory (same device and inode, compared with `os.stat` only) is refused as well. `record.json` lists the files it opened, as an after-the-fact check, not as the guard.

## Verification

Reference test: [tests/test_verifier_scan.py](../../tests/test_verifier_scan.py). Every rule has a synthetic hit and an honest non-hit, including `sorted(set(...))`, a single-element list literal, `== [x, y] or == [y, x]`, a function that legitimately returns `None`, and the floats `4.0` and `1.67` against "two decimal places". The R11 parity test runs the mirror against the real parser on 40+ shapes. The sealed refusal is tested with temporary files that are never opened (opening is patched to fail).

```text
env -u VIRTUAL_ENV uv run --locked pytest tests/test_verifier_scan.py
```

## Questions and answers

1. Does a task with no hit have a sound verifier? No. The scan looks for a few known shapes only.
2. Why is the bare word "write" not a side-effect keyword? Because "write a function" starts most prompts and would flag nearly everything.
3. Why does `sorted(f(x)) == [1, 2]` not trigger R7 or R10? The test sorts the output first, so the order of the original result no longer matters.
4. Why can R11 matter when the reference never calls the builtin itself? The check is run beside the candidate's code, and the candidate may rebind that builtin or patch that module attribute.
5. Why does the tool take file paths one by one instead of a directory? So that a sealed file can never be picked up by accident.
