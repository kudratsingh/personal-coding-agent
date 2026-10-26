# chat-v1 conversations, loss masks and code extraction

Post-training reads and writes one conversation format, `chat-v1`. The renderer turns a pool task or a language conversation into token IDs plus a loss mask that says which tokens the model is trained to predict. The same renderer builds the generation prompt, so a model is scored on exactly the context it was trained on. Code extraction takes the program back out of a generation. The code is in `src/scratchcoder/data/chat.py`. The base-model pool protocol, `base-pool-completion-v1`, is in `src/scratchcoder/eval/prompt.py`.

## Worked example

Take a pool task with the prompt `Add one.`, the public test `assert f(1) == 2` and the reference `def f(x):\n    return x + 1`. With the merge-free reference tokenizer every ordinary token is one UTF-8 byte, so the counts below can be checked by hand.

The user turn is the prompt, a blank line and the public test: `Add one.\n\nassert f(1) == 2`, 26 bytes. The assistant turn is one python block holding the code, also 26 bytes.

| Positions | Tokens | Carries loss |
| --- | --- | --- |
| 0 | `<\|bos\|>` (256) | no |
| 1 | `<\|user_start\|>` (257) | no |
| 2–27 | the 26 user bytes | no |
| 28 | `<\|user_end\|>` (258) | no |
| 29 | `<\|assistant_start\|>` (259) | no |
| 30 | `<\|python_start\|>` (261) | **yes** |
| 31–56 | the 26 code bytes | **yes** |
| 57 | `<\|python_end\|>` (262) | **yes** |
| 58 | `<\|assistant_end\|>` (260) | **yes** |

The row has 59 tokens and 29 loss tokens. As a training pair it has 58 inputs (`ids[0:58]`) and 58 targets (`ids[1:59]`). The first 29 targets, `<|user_start|>` through `<|assistant_start|>`, are set to `IGNORE` (−100). The 30th target is `<|python_start|>`, predicted from the whole prompt.

The generation prompt is positions 0–29, ending at `<|assistant_start|>`. It is exactly the start of the training row.

A language conversation uses the same turns without the python block. For `Hi` / `Hello` / `Why?` / `Because`, the tokens carrying loss are `Hello`, the first `<|assistant_end|>`, `Because` and the second `<|assistant_end|>`. Every user turn and both `<|assistant_start|>` tokens carry none.

## Data flow and shapes

**Which tokens carry loss.** `mask[i]` is true when `ids[i]` is a target the model is trained to predict from `ids[:i]`. Exactly these tokens carry loss: those after each `<|assistant_start|>`, up to and including that turn's `<|assistant_end|>`. For a code task that is `<|python_start|>`, the code, `<|python_end|>` and `<|assistant_end|>`. `<|bos|>`, every user-turn token and every `<|assistant_start|>` carry none. `mask[0]` is always false, because the first token is never a target.

**Rendering.** `render_code_task(encoder, task)` and `render_conversation(encoder, messages)` return `Rendered(ids, mask)`, two tuples of length `n`. `inputs_targets()` gives `ids[:-1]` and `ids[1:]` (length `n − 1`), with every target outside the mask replaced by `IGNORE`. Turns must alternate user and assistant, start with user and, for a training row, end with assistant. The code in a block is the reference with CRLF line endings changed to LF, leading blank lines removed and trailing whitespace stripped. Nothing else changes, so indentation and string contents are kept.

**User turns.** A pool task's user turn is `prompt.strip() + "\n\n" + public_tests.strip()`. An assertion task's public test is its assertion text. An I/O task's public test is the stored JSON list of `{"input", "output"}` cases. A HumanEval, HumanEval+ or MBPP+ user turn is the benchmark prompt alone, stripped (`benchmark_prompt`). The extracted function then runs with that suite's tests.

**Generation prompts.** `task_prompt`, `benchmark_prompt` and `generation_prompt(encoder, history)` end at `<|assistant_start|>`. For a code task, generation may stop at `<|python_end|>` or `<|assistant_end|>` (`CODE_STOP_IDS`), since nothing after the first block is scored.

**Extraction.** `extract_program(generated_ids, tokenizer)` looks only at the generated tokens. It finds the first `<|python_start|>` and reads up to the next special token. The text in between is the program only if that token is `<|python_end|>`. With no block, an unclosed block, or another special token first (such as `<|assistant_end|>`), there is no program and the attempt scores 0. An empty closed block is an empty program, which fails its tests.

**base-pool-completion-v1.** Base-model scoring of pool tasks uses a plain completion instead:

- **Prompt:** `pool_prompt(task)` is `"""`, a newline, chat-v1's user turn, a newline, `"""` and a newline. A `"""` inside the text is written with a backslash before each quote, so the docstring cannot close early.
- **Budget:** `pool_budget(prompt_tokens)` is `min(512, 1,024 − prompt_tokens)`. A value below one means no room to generate.
- **Stops:** `truncate_pool_completion` cuts at the first of `\nif __name__`, `\nprint(`, `\nassert`, `\n"""` and `\n# Test`. The retained text is the program. These are prefix matches, so `\n# Testing` also stops. Because the prompt ends with a newline, a completion that begins with `assert` keeps nothing.

`prompt_protocol("chat")`, `prompt_protocol("chat-benchmark")` and `prompt_protocol("base-pool")` give the descriptions that reports record.

## Design choices and common mistakes

- **Text never becomes a special token.** Message text is encoded without special-token injection, so a message containing the literal string `<|user_end|>` stays ordinary bytes. The renderer also rejects an encoder that returns a special ID for text.
- **`<|assistant_start|>` carries no loss.** It follows every `<|user_end|>` by construction, and at generation time it is part of the prompt. Training on it would only teach the template.
- **`<|assistant_end|>` carries loss.** The model must learn when to stop. For code tasks `<|python_end|>` carries loss too, so the model learns to close its block.
- **The prompt is a prefix of the row.** A test checks this for every task shape. If the prompt and the training row disagreed, for example in whitespace, the model would be scored on a context it never saw.
- **Extraction works on token IDs, not text.** Decoded text cannot tell a real `<|python_start|>` from those bytes typed inside code, but the IDs can.
- **Pool I/O tasks and the `\nprint(` stop.** A stdin/stdout program usually prints at the top level, so `base-pool-completion-v1` cuts it at its first top-level `print(`. This is the protocol as declared. It affects only base-model pool scores, which are labelled cross-protocol.

## Verification

`tests/test_chat_format.py` covers:

- the special-token IDs against the tokenizer table;
- the hand-built code-task row above, its mask and its targets;
- the prompt as an exact prefix of the row, and of a multi-turn history;
- a multi-turn mask, token by token;
- special-token injection through text, and an injecting encoder;
- turn-order and validation errors;
- code normalization, and the I/O and benchmark user turns;
- every extraction case, and a round trip through a rendered reference;
- the base-pool prompt, quote escaping, every stop, and the budget.

```sh
uv run --locked pytest tests/test_chat_format.py
```

These checks use the merge-free reference tokenizer. The SFT data builder renders with the frozen 65,536-entry tokenizer through the bulk encoder, which `tests/test_bulk.py` checks against the reference encoder.

## Questions and answers

1. *Why does `<|python_start|>` carry loss when it is always the first assistant token for a code task?* The model has to choose to open a block. A generation without one has no program and scores 0.
2. *A model emits `<|python_start|>x = 1<|assistant_end|>`. What is extracted?* Nothing. The first special token after the start is not `<|python_end|>`, so there is no program.
3. *Why is a training row's last token never an input?* It has nothing to predict. A row of `n` tokens gives `n − 1` input/target pairs.
4. *Why strip the prompt and the public test?* So the user turn has one fixed shape: no stray leading newline from a stored test, and exactly one blank line between the two parts.
5. *Why does base-pool stop at `\n"""`?* A base model often closes one docstring and opens another for the next problem. The program ends at the first new top-level docstring.
