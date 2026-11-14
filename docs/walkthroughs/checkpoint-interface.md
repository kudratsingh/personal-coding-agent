# Component: checkpoint interfaces (which protocol a checkpoint gets)

A pretrained model continues plain text. An SFT model was trained on chat-v1 conversations: it expects `<|bos|><|user_start|>…<|user_end|><|assistant_start|>` and answers with `<|python_start|>…<|python_end|>`. Give the SFT model a plain-text prompt, or decode its answer with the special tokens stripped and then look for a Markdown fence, and a working model can score near zero. That score measures the mismatch, not the model.

So every checkpoint has an **interface**, `chat-v1` or `base`, and every generation protocol fits exactly one. `chat-v1` fits `chat-v1`. `base-pool-completion-v1`, `base-completion-rstrip-v1` and the legacy `scratchcoder eval` modes (`base`, `instruction`) fit `base`. The code is `src/scratchcoder/eval/interface.py`.

## Worked example

Three files, and what each records:

| File | Keys that matter | Rule | Interface |
| --- | --- | --- | --- |
| New SFT weights | `"format": "weights-v1"`, `"interface": "chat-v1"` | declared | `chat-v1` |
| Run 1b's SFT weights | `"format": "weights-v1"`, no `interface` key | weights-v1 | `chat-v1` |
| Run B's `mixed-final` | `"version": 1`, `"optimizer"`, a pretraining `config` (no post-training mark) | trainer-checkpoint | `base` |

Asking for a pool evaluation with no protocol named:

- `eval-generate` on either SFT file resolves `chat-v1`. The prompt is the chat-v1 token IDs, ending in `<|assistant_start|>` (ID 259); generation stops at `<|python_end|>` or `<|assistant_end|>`; the program is the first closed python block of the generated IDs.
- `eval-generate` on `mixed-final` resolves `base-pool-completion-v1`. The prompt is the task text and public test in a module docstring, encoded as plain text with no special tokens; the program is the completion before the first stop string of the task's mode.
- `scratchcoder eval` on run 1b's SFT weights stops before loading any weights: `this checkpoint's interface is chat-v1 (found by the weights-v1 rule), and scratchcoder eval has only plain-text prompt modes; evaluate it with scratchcoder eval-generate …`.
- `scratchcoder eval --prompt-mode instruction` on the new SFT weights also stops, naming `--allow-interface-mismatch`. With that flag it runs, and the report records `{"name": "chat-v1", "source": "declared", "prompt_mode_source": "requested", "interface_mismatch_allowed": true}` under `checkpoint_interface`, plus an `interface_override` record (below) in the report and its scorecard row.
- `eval-generate --protocol base-pool-completion-v1 --allow-interface-mismatch` on run 1b's SFT weights runs under the base pool prompt, and its completions header carries:

  ```json
  "interface_override": {
    "override": "--allow-interface-mismatch",
    "override_used": true,
    "requested": "--protocol base-pool-completion-v1",
    "requested_protocol": "base-pool-completion-v1",
    "requested_interface": "base",
    "checkpoint_interface": "chat-v1",
    "checkpoint_interface_source": "weights-v1",
    "reason": "--protocol base-pool-completion-v1 is written for base checkpoints, but this checkpoint's interface is chat-v1 (found by the weights-v1 rule)"
  }
  ```

- Passing `--allow-interface-mismatch` with a protocol that fits (say `--protocol chat-v1` on SFT weights) stops with `nothing is crossed`, so a recorded override always marks a real mismatch.

## Data flow and shapes

- `read_checkpoint_interface(path)` memory-maps the checkpoint with `torch.load(..., weights_only=True, mmap=True)`, so the tensors are not read, and passes the mapping to `checkpoint_interface`.
- `checkpoint_interface(saved)` returns `CheckpointInterface(name, source)`, where `name` is `chat-v1`, `base` or `None` (undeclared). First match wins:
  1. an `"interface"` key: `declared` (an unknown value is refused);
  2. `"format": "weights-v1"`: `chat-v1`, rule `weights-v1`;
  3. a trainer checkpoint (`"version"`, `"optimizer"`, a `config.train` mapping, no `"format"`): `base`, or `chat-v1` when the configuration carries a post-training mark (`special_rows_init` not `none`, `loss_normalization` of `update_targets`, or an `sft` or `rl-references` source in `data_mix`), rule `trainer-checkpoint`;
  4. anything else: undeclared.
- `resolve_protocol(interface, family, requested, allow_mismatch=False)` returns a `ProtocolChoice(name, requested, interface, override)` for `eval-generate`. With `requested=None` the interface chooses: `chat-v1`, or for `base` the pool or benchmark base protocol by `family`. An undeclared interface needs a request. A request must fit unless `allow_mismatch`; a crossed request comes back with `override`, an `InterfaceOverride` whose `record()` is the `interface_override` object.
- `resolve_legacy_mode(interface, requested, allow_mismatch=False, protocol=None)` returns the same kind of choice for `scratchcoder eval`, with `name` `base` or `instruction`; `protocol` is the protocol name the run records, so a crossed mode's override names it. With no request a base checkpoint gets `base`; chat-v1 and undeclared checkpoints are refused.
- `require_fit(...)` underlies both: `None` when the choice fits, an `InterfaceOverride` when a mismatch is deliberately crossed, an error otherwise, and an error when the override is passed but nothing is crossed.
- `check_override_record(record, protocol)` is what `CompletionsHeader` runs on a recorded override: exactly the eight fields, a used override, the header's protocol and its interface, a checkpoint interface that differs, a known rule and a non-empty reason. A header without an override omits the key, so files written before it existed still read.
- `save_weights(..., interface="chat-v1")` writes the key into every new weights-v1 file. Its only callers are post-training (the SFT stage, RL and the more-SFT control).

## Design choices and common mistakes

- **No new weights format.** Every reader of a weights-v1 file reads only the keys it needs (`config`, `tokenizer_hash`, `model`, `run_id`), so adding one key breaks nothing, while `weights-v2` would have had to be accepted everywhere `weights-v1` is. Files written before the key existed need no sidecar: only post-training has ever written weights-v1, and it all trains on chat-v1, so the format alone is the evidence.
- **Historical files are never rewritten.** The rules only read. Run 1 and run 1b's SFT weights keep their bytes and SHA-256s, which the seal ledger and the scorecard name.
- **Why the post-training marks.** No trainer checkpoint of a post-training run exists today, since the SFT stage and the more-SFT control save weights-v1, but a future resumable SFT stage could save full trainer checkpoints. One mark is not enough on its own: only some SFT runs redraw the special-token rows ([special-token rows](special-token-rows.md)), and run 1's SFT did not. So the rule accepts any of three settings that pretraining never uses: the special-row redraw, loss normalized by assistant targets (`update_targets`), or an `sft` or `rl-references` data source. A post-training writer that sets none of them would still be read as `base`, so such a writer should add the `"interface"` key, which wins.
- **Undeclared is not base.** A hand-made file or the shared random initialization records nothing, so the code does not guess. It requires the protocol to be named, and accepts any protocol named.
- **The override is deliberate, not silent.** Crossing interfaces is a legitimate diagnostic, for example base prompting on SFT weights to separate formatting from capability, and it is how a historical procedure is reproduced. It takes a named flag, and everything written under it says so: the completions header (and so every results file, benchmark artifact and scorecard row built from it) or the legacy report and its scorecard row carry `interface_override`, with the requested protocol, both interfaces and the reason. The scorecard also flags any paired comparison that uses an overridden row, since a shared protocol name does not make such a pair comparable. No new weights format was needed for this; the record lives in the evaluation outputs.
- **Legacy names say what ran.** On pool tasks the legacy base mode is recorded as `legacy-pool-rstrip-v1`, not the benchmark-only `base-completion-rstrip-v1`. Older legacy reports keep the old name, which on pool tasks means the same procedure.

## Verification

`uv run --locked pytest tests/test_checkpoint_interface.py` covers:

- each kind of checkpoint, built by its real writer (`save_weights`, `Trainer.checkpoint`) or in the exact historical weights-v1 layout, resolving to its interface;
- SFT weights resolving to chat-v1 prompt IDs, which end at `<|assistant_start|>`, and python-block extraction, and trainer weights resolving to the base-pool prompt with no special tokens, and stop-string extraction, through `generate_completions`;
- `eval-generate` and `scratchcoder eval` refusing implicit and mismatched choices with exit status 2 and a message naming the way forward, before any output or scorecard row is written;
- the explicit legacy modes reproducing their prompts and recording `legacy-pool-rstrip-v1`, `base-completion-rstrip-v1` or `legacy-instruction-v1`;
- a deliberate mismatch being accepted only with the override and recorded with every override field, in the completions header (`eval-generate`) and in the legacy report and its scorecard row; an override that crosses nothing being refused; a malformed override record being refused;
- historical reproduction: `eval-generate --protocol base-completion-rstrip-v1 --allow-interface-mismatch` and legacy `--prompt-mode base --allow-interface-mismatch` on historical weights-v1 SFT weights giving exactly the historical plain-text prompt IDs (the stripped prompt's bytes), with the override recorded;
- EvalPlus checkpoint scoring refusing chat-v1 weights.

`tests/test_scorecard.py` checks that an override in a completions header reaches the scorecard row, the Markdown and the paired comparison, even when both sides share one protocol.

`tests/test_prompt_protocol.py` keeps the legacy candidate assembly, and `tests/test_benchmark_prompts.py` keeps the pipeline protocols' record hashes unchanged. No model was trained or scored for this change.

## Questions and answers

1. **Why does run 1b's SFT checkpoint resolve to chat-v1 without an `interface` key?** It is a weights-v1 file, and only post-training, which trains on chat-v1, has written that format.
2. **What does `scratchcoder eval` do with SFT weights and no `--prompt-mode`?** It stops before loading the weights and points to `eval-generate`, which resolves `chat-v1`.
3. **Can you still run base prompting on SFT weights?** Yes. Name the protocol or mode and pass `--allow-interface-mismatch`. The output records `interface_override`: the requested protocol, the interface it fits, the checkpoint's interface and rule, and the mismatch.
4. **Why is an undeclared checkpoint not treated as base?** Nothing in the file says what it was trained for, so a guess could pick the wrong prompt silently. Naming the protocol makes the choice explicit.
5. **What does `base-completion-rstrip-v1` mean in an old legacy report on pool tasks?** The legacy base mode: the raw prompt after `rstrip()`, the original stop strings and prompt plus completion. New reports call it `legacy-pool-rstrip-v1`.
