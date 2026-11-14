# Component: run 2 schedule simulator

`tools/run2_schedule_sim.py` answers one question before any Pod exists: what do run 2's stop rules do to a whole run at given speeds? Several rules interact. The [post-training run worker](posttrain-run.md) stops at the lower of the ceiling and 120% of the gate's estimate. It keeps a 900 s finalization reserve, and before every update and every Mac exchange it keeps one full Mac wait free (`wait_margin`, 11,746.5 s). The projected-finish rule runs before every stage, and the arm-stop and same-weights rules apply. Their combined effect is easy to get wrong by hand. [Post-training runs](../posttrain-runs.md#an-arms-stop) has a table of it that no committed tool used to produce. The simulator reproduces that table exactly, and a test holds it to it.

Every output is a **forecast at the given rates**, by default multiples of the planning rates. It is not a measurement of run 2.

## Worked example

The planning rates, 320 kept tasks, a 900 s setup:

1. **The estimate.** The real projection (`pilot_feasibility.posttrain2_report`) prices the run at 18,870 s: the 900 s setup, RL, the control, ARC, evaluation, loads and saves, and the 900 s finalization. At $0.60/h that is $3.145.
2. **The gate.** `posttrain_approved.fill_gate` writes the estimate and derives the cost limit, 120% rounded down to the cent: $3.77. That is 22,620 s of Pod time. The in-Pod timer fires 180 s earlier, so the worker's lifetime is 22,440 s, and its finalization reserve starts at 22,440 − 900 = 21,540 s.
3. **RL.** From 900 s: a 20 s load; dev check 0 (105.5 s of generation, a 300 s Mac wait and a 5 s save); then 100 updates of 7.5 s rollouts, a 48 s reward wait and 4.6 s of training; plus 5 more checks. RL ends at about 9,395 s. Before each update the real `should_stop` asks whether 11,746.5 s more would pass the reserve. It never does: 9,395 + 11,746.5 = 21,142 < 21,540.
4. **The control.** Its budget is RL's GPU-busy time, 1,212 s: about 2,100 updates of 0.577 s. With 320 rows of 32 per update, pass 1 ends at update 10. The dev check after it ends at about 9,832 s. Before update 11, 9,832 + 11,746.5 = 21,578 ≥ 21,540, so `should_stop` returns `wait_margin`. The control has spent 10 of about 2,100 updates, under 1% of its budget.
5. **What follows.** The real stage code records a margin arm stop. The control selects its pass-1 check, so its weights differ from the SFT weights and it is evaluated. ARC and both evaluations run under the projected-finish rule. The run ends `stopped` with `wait_margin` at about 15,121 s, with 7,300 s left for the finalization. `more_sft.within_tolerance` is false: the comparison is not compute-matched.

With a 1,500 s setup, everything moves 600 s later. The control stops before its first update and selects its start, RL's update-0 check. That check holds the SFT weights, so the SFT results stand for it, and there is no control at all.

## Data flow and shapes

```text
(mac x, gpu x, setup s, kept tasks K)
  -> scaled_rates: planning Rates with GPU and Mac seconds scaled
  -> projected_estimate: posttrain2_report(..., measured=those rates)    real code
  -> fill_gate + gate_launch_fit: the launch specification or a refusal  real code
  -> SimulatedRun2Worker(spec, SimClock, rates, K)
       execute -> run_stage -> projected_finish -> run_rl / run_more_sft  real code
                      fake GPU: simulated_rl, simulated_control, generate
                      fake Mac: mac_wait through the worker's wait_sleep
                      fake R2: export_label records a label, no time
  -> summarize: per arm, updates against the declared budget, busy/budget,
     within_tolerance, the margin stop (kind, limit), endpoints, spend
```

GPU time covers generation, training and ARC. Mac time covers the reward and dev-check waits. Loads, saves and the setup are not scaled. `K` is the number of control rows after any length drops. It sets the pass boundaries (`ceil(p x K / 32)`) and the loader's 400-pass cap.

## Design choices and common mistakes

- **Real decision code, fake work.** `SimulatedRun2Worker` subclasses `Run2Worker` and overrides file, publication, GPU, Mac and R2 plumbing, plus two stage bodies that need a model: ARC (`run_arc`) and evaluation (`run_eval`). Those two are simulated: they spend the planned time with the real `tick` and record the measured rates as the real ones do. The simulated `run_arc` repeats the real one's bookkeeping of the checkpoints left out, and a drift test compares the two. A test asserts that none of the decision methods is overridden. While it runs, `patched_worker` swaps the model-bound names in `tools.posttrain_run2` (`run_rl_loop`, `run_more_sft`, `load_weights`, `BridgeRewardSource`, `compare_to_loaded`, `file_sha256`), so the real `run_rl` and `run_more_sft` stage bodies execute. The swap is undone afterwards.
- **The inner loops are checked against the real ones.** `simulated_rl` and `simulated_control` copy the control flow of `scratchcoder.rl.loop.run_rl` and `scratchcoder.rl.control.run_more_sft`: the same `should_stop` calls, reward requests, dev checks, end checks and stop reasons. The conformance tests run both on a tiny CPU model with the same scripts and compare every call.
- **Use the Pod's lifetime, not the cost wall.** `posttrain_run.run` gives the worker `pod_seconds = lifetime`: the first of the wall, the in-Pod timer, and the worker's and the watchdog's cost stops. Using the cost wall instead moves every stop 180 s later and gets the table wrong by a whole pass check.
- **The estimate always prices a 900 s setup.** The template's `startup_seconds` is fixed. A slower setup spends margin, never estimate. That is why every 1,500 s column is worse.
- **"Unscored" means the dev check generated but its Mac wait was refused** (`before_wait`): the weights are saved, but no score exists to select them.
- **A budget-complete control can still be a margin stop.** If the control spends its budget but the margin refuses its end check, the real `stop_kind` still reports a margin arm stop. The control is then selected from its pass-8 check.
- **Below the floor is hypothetical.** For `K` < 320 the real worker refuses the task file. The simulator runs it anyway, flagged `admitted_by_floor: false`. At 161 rows the 400-pass loader plans 2,013 updates, fewer than the about 2,100 the budget buys, so the control ends `passes_exhausted`.
- **Not simulated:** collapse, leakage, non-finite values, dev patience (dev scores rise with every check), retries, slow or lost bridges, upload time and the finalization's own time. The time left for the finalization is reported instead. Generation is priced at the token caps, as the projections price it.

## Verification

`tests/test_run2_schedule_sim.py`:

- `test_the_documented_truncation_is_reproduced` checks every row of the table in [post-training runs](../posttrain-runs.md#an-arms-stop): the estimate, the stop, the first stop for 320 and 1,000 tasks after a 900 s and a 1,500 s setup, and the share of the control's budget spent where the table gives one. `test_the_documented_gate_refusals_are_reproduced` checks its last row, the two cases refused at the gate.
- `test_the_simulated_arc_keeps_the_real_bookkeeping` runs the real `Run2Worker.run_arc` (with the model-bound ARC body replaced by a recorder) next to the simulated one and compares the checkpoints evaluated and left out.
- The conformance tests (`test_the_simulated_rl_loop_makes_the_real_loops_calls`, `test_the_simulated_control_makes_the_real_controls_calls`) run the real loops next to the simulated ones.
- `test_the_simulated_worker_sets_every_attribute_the_real_one_does` compares against a real `Run2Worker` built on the run-2 smoke fixtures.
- `test_a_limit_reached_during_a_wait_is_the_real_wait_limit_stop` shows the real `observe_wait` ending a fake wait once the margin is disabled.

```sh
uv run --locked pytest tests/test_run2_schedule_sim.py
uv run --locked python -m tools.run2_schedule_sim --mac 1 --gpu 1 --setup 900 1500 --tasks 320 1000
```

The command prints a Markdown table and, with `--output`, writes the JSON record (`run2-schedule-simulation-v1`). `--projection` adds each scenario of a saved run-2 projection report at its recorded rates and its own estimate.

## Questions and answers

1. **Why does a faster GPU make the control's truncation worse?** It lowers the estimate, and so the stop, but the margin stays at 11,746.5 s. Less room is left after RL for the control.
2. **Why do 320 and 1,000 tasks give different stops at the same rates?** The control's dev checks sit at pass boundaries. With more rows, pass 1 ends later (update 32 rather than 10), so the first check, and the margin decision after it, comes later.
3. **What makes a control "unmatched"?** Its GPU-busy seconds differ from RL's by more than 5% (`within_tolerance` false), whether it was stopped early or ran out of passes.
4. **Does the simulator change any stop rule?** No. `Scenario.stop_fraction` only gives the gate a different estimate so that its own 120% rule yields another stop. It answers "what would it take", for the owner's decision; the worker is unchanged.
