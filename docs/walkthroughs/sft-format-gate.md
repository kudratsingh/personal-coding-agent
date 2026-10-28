# Component: the SFT format gate

Supervised fine-tuning (SFT) teaches the model the chat-v1 format ([chat-v1](chat-format.md)). A code answer is `<|python_start|>{code}<|python_end|><|assistant_end|>`. The evaluation takes the first closed python block as the program, so a model that ends its turn before opening a block scores 0 on every task, however good its code would have been. The token-weighted dev loss cannot show this: the format tokens are three of a few hundred targets per row, so one bad token barely moves the mean.

The format gate measures those few tokens directly, on the CPU, and runs no generated code. It is required after every SFT run, before any paid evaluation or RL starts from its weights. The code is `src/scratchcoder/eval/format_gate.py` and the command is `scratchcoder sft-format-gate`.

## Worked example

Take one code dev row (IDs written as names):

```text
<|bos|> <|user_start|> Task 3: write f. ... <|user_end|> <|assistant_start|>
<|python_start|> def f(): ... return 1 <|python_end|> <|assistant_end|>
```

The gate reads three probabilities from it:

| Check | Input | Position | Wanted next token |
| --- | --- | --- | --- |
| first token | the row up to `<|assistant_start|>` (the chat-v1 generation prompt) | the last | `<|python_start|>` |
| close | the whole row but its last token, teacher-forced | the last code token | `<|python_end|>` |
| close | the same forward pass | `<|python_end|>` | `<|assistant_end|>` |

Run 1's SFT weights gave, as means over the gate's 32 rows: P(`<|python_start|>`) 0.258 and P(`<|assistant_end|>`) 0.347 at the first position, so the argmax was `<|assistant_end|>` in all 32 rows. That is an empty completion. P(`<|python_end|>`) at the end of the code was 0.011. Greedy generation on 16 dev prompts gave 16 completions of the single token `<|assistant_end|>` and no program. The cause shows in the embedding: the rows of `<|user_start|>` through `<|python_end|>` had pairwise cosines from 0.993 to 0.997, and the largest among the three output specials (`<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`) was 0.996. The output layer shares those rows, so the model can hardly score one of these tokens above another. The parent (run B's final weights, before SFT) measured 0.997 among the output specials. SFT had barely moved them.

The tiny model in the reference test learns the format in 80 updates. It passes with P(`<|python_start|>`) about 0.997 and 16 of 16 programs. Its special rows' largest cosine is about 0.73.

## Data flow and shapes

1. **Rows.** The SFT code dev file has one chat-v1 row per pool dev task (`render_code_task`). The gate refuses any row that is not a single code exchange with at least one code token. It drops rows whose `n - 1` inputs exceed the context. Then it sorts by `sha256("sft-format-gate-1|" + id)` and takes the first 32. The choice does not depend on file order, and the record lists the row IDs. Each of the 32 rows must have a pool dev task of the same ID, and the row up to `<|assistant_start|>` must equal that task's chat-v1 evaluation prompt token for token. Otherwise the gate stops without a verdict. The record keeps the count, `rows_matching_task_prompts`, which is always 32.
2. **First token.** One forward pass of the prompt, `[1, P]`, gives logits `[1, P, V]`. The gate keeps the softmax of the last position, a vector of `V` probabilities (65,536 for the real tokenizer). The check passes when the argmax is `<|python_start|>` in every row and the mean probability is at least 0.9.
3. **Close.** One forward pass of the row without its last token, `[1, n - 1]`, gives `[1, n - 1, V]`. Position `t` predicts token `t + 1`. With `<|python_end|>` at index `e = n - 2`, the gate reads position `e - 1` for `<|python_end|>` and position `e` for `<|assistant_end|>`. The mean for `<|python_end|>` must be at least 0.2, and the mean for `<|assistant_end|>` at least 0.5.
4. **Separation.** The gate takes the three rows of the tied embedding for the specials the model must choose between as outputs, `<|assistant_end|>`, `<|python_start|>` and `<|python_end|>`, a `[3, width]` matrix. It normalizes them and forms the `[3, 3]` cosine matrix. The largest off-diagonal entry must be below 0.9. The record also keeps, unjudged, the `[7, 7]` cosine matrix of all seven chat-v1 specials (adding `<|bos|>`, `<|user_start|>`, `<|user_end|>` and `<|assistant_start|>`), its largest pair, and the seven rows' norms and smallest pairwise distance.
5. **Greedy smoke.** The gate takes 16 pool dev tasks by the same key. Every task with a code dev row has the same ID, so in practice these are the first 16 of the 32 rows. The evaluation's `generate_many` runs greedy chat-v1 decoding in one length-sorted, left-padded batch of 16, in float32. It allows at most `min(512, context - prompt)` new tokens and stops at `<|python_end|>` or `<|assistant_end|>`. The protocol's own extraction then reads each completion. At least 12 of 16 must give a nonempty program. Programs are hashed and never run.
6. **Record.** The JSON record holds every per-row and per-prompt value, each check's verdict, the thresholds, the SHA-256 of the checkpoint, tokenizer, rows and tasks files, the tokenizer identity, the selection rule, the thread count, the seconds, and the git commit. The command prints a summary without the cosine matrices. It exits 0 when every check passes and 1 when a check fails. It exits 2, with the error on standard error, when the gate could not measure: a missing or unreadable input, an existing output, a rows file that does not match the tasks, or any other error. Only exit 0 with a written record counts as a pass.

## Design choices and common mistakes

- **Measure the format tokens, not the mean loss.** Run 1's selected SFT weights had a dev loss of 1.21. Their first-token loss was about 1.35 nats a row, buried among a few hundred other targets.
- **The gate does not depend on sampling.** The first three checks read probabilities from fixed inputs. Only the smoke generates, and it is greedy. Selection is by hash. Two runs on the same weights give the same record apart from the time fields.
- **Why these thresholds.** No healthy SFT model at this scale has been measured yet, so each bar is set from the training data or from trained rows, not from such a model. Every training row's first assistant token is `<|python_start|>`, so a model that has learned the format puts nearly all its mass there. A bar of 0.9 leaves room for that, and a split between two specials (the failure above) falls far below it. Where code ends is genuinely uncertain, because example usage often follows a function. Count the points where code could end: the last line, or a line whose next nonempty line starts at indent 0. The code really ends at 26% of these points in the SFT train rows and 34% in the code dev rows. A model that closes at the data's rate would score about 0.3 at the true end, and a good one can put real mass on more code after a line like `return x`. The `<|python_end|>` bar is therefore 0.2, still about 20 times run 1's 0.011. Whether greedy decoding actually closes its block is the smoke's job. After `<|python_end|>` the turn always ends in training, so 0.5 for `<|assistant_end|>` is a loose floor. In run B's trained weights, the median largest cosine among 3 random ordinary rows is 0.57, and 99% of such sets are below 0.9. A bar of 0.9 is therefore one that trained rows almost always clear. 12 of 16 programs allows a few prompts where an imperfect model fails, while a format failure gives none.
- **Judge only the output specials.** Under assistant-only loss, `<|bos|>`, `<|user_start|>`, `<|user_end|>` and `<|assistant_start|>` are never targets. As outputs they get only the small push-down that every non-target row gets, so nothing in SFT pulls them apart. In run 1's SFT weights the closest pair of all seven was `<|user_start|>` and `<|user_end|>`. A model that learns to separate the three tokens it must choose between could leave those two above 0.9, and judging them would block it for a reason that does not affect what it generates. The seven-token matrix stays in the record.
- **Separation is a proxy; the other three checks are behaviour.** With the embedding tied to the output, the gap between two tokens' logits is `h · (e_i - e_j)`. Near-parallel rows of equal norm make `e_i - e_j` short, and the gap stays small. A model could in principle tell two close rows apart. If one ever passes the behavioural checks but fails separation, that is a finding to report. It is not a reason to lower the bar.
- **The vocabulary mean is where the specials already are.** In run B's weights the chat specials' rows have a cosine of about 0.994 with the mean of the ordinary rows. A row placed at the mean plus small noise therefore starts nearly as collinear as these did. Separation then has to come from training, and the gate checks that it did.
- **The prompt is the evaluation's prompt.** The first-token check uses the code dev row's prefix, and the gate confirms, for every one of the 32 rows, that it equals the chat-v1 evaluation prompt of the same task. A check on a slightly different prompt (a missing public test, a different strip) could pass while the evaluation fails. If the rows file ever renders a different prompt, the gate stops with exit 2 rather than measuring it.
- **What the gate covers.** It checks the pool prompt shape only, the one the code dev rows and the pool evaluation share. The benchmark evaluations use their own prompts, which this gate does not read.
- **No generated code runs.** The smoke records whether a block opened, whether it closed and the program's SHA-256. Whether the program is correct is the evaluation's job, in the sandbox.
- **A record is never replaced.** The command refuses an existing output path.

## Verification

`tests/test_sft_format_gate.py` covers:

- a tiny model trained on authored chat-v1 code rows passing all four checks through the library and the command (exit 0);
- the same model with the seven special rows set to their mean failing the first-token and separation checks: no row's argmax is `<|python_start|>`, its probability is below 1/7, the cosine is 1 and no program is extracted. The command exits 1;
- the same model with only `<|user_start|>` and `<|user_end|>` set to their mean still passing separation, with the cosine of 1 kept in the unjudged record;
- selection by key regardless of file order, rows that do not fit the context, too few rows, non-code rows, a code row with an empty block, a row whose prompt differs from its task (first and last of the 32), a row with no task, and a non-dev task file (exit 2);
- each threshold at its edge: a mean of exactly 0.9, 0.2 or 0.5 passes and one just below fails; one wrong argmax fails; a cosine of exactly 0.9 fails; 12 of 16 programs pass and 11 fail;
- an existing output path giving exit 2 and leaving the record in place;
- an empty python block not counting as a program.

```sh
uv run --locked pytest tests/test_sft_format_gate.py
uv run --locked scratchcoder sft-format-gate --checkpoint <weights> --tokenizer <tokenizer.json> \
  --rows <code_dev.jsonl> --tasks <pool dev.jsonl or its export> --output <new record.json>
```

## Questions and answers

1. *Why can a low dev loss hide this failure?* The loss is averaged over every assistant token. Getting the one first token wrong in each row costs about 1.3 nats out of a few hundred targets.
2. *Why read the close probabilities teacher-forced rather than from generation?* A model that never opens a block would never reach the close position in generation. Teacher forcing measures the close even when the first token is wrong.
3. *Why is a cosine of 0.99 between two output rows a problem?* With equal norms, the rows' difference is about `norm × sqrt(2 × 0.01)`, or 14% of a row. The logit gap between the two tokens is the hidden state's product with that short vector, so it stays small.
4. *Why do the smoke prompts coincide with the first rows?* Both are chosen by the same hash of the task ID, so the smoke generates from prompts whose first token and close the gate has already measured.
5. *What does the gate not establish?* It shows nothing about whether the programs are correct. Pass rates come from the evaluation.
