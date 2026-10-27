# Component: the RL reward bridge between the Pod and the Mac

In the rehearsal's second run the RL loop runs on a rented GPU Pod, but its rewards and its dev-check verdicts can only be computed on the Mac: the hidden tests and the Docker sandbox are there. The Mac reaches the Pod over SSH; the Pod cannot reach the Mac. The bridge connects the two through files. The Pod writes each request into an *outbox* directory and waits for the reply in an *inbox* directory. The Mac polls the outbox over SSH, scores the request and pushes the reply back.

The code:

- [`src/scratchcoder/rl/bridge.py`](../../src/scratchcoder/rl/bridge.py): the schemas and the Pod side;
- [`src/scratchcoder/rl/bridge_mac.py`](../../src/scratchcoder/rl/bridge_mac.py): the Mac side and the SSH channel;
- [`src/scratchcoder/rl/bridge_command.py`](../../src/scratchcoder/rl/bridge_command.py): `scratchcoder rl-bridge`.

The RL loop itself is unchanged: the Pod hands `run_rl` a `BridgeRewardSource` as its reward source and a `BridgeDevEvaluator` as its dev evaluator ([rehearsal RL](../rl.md)).

## Worked example

Update 3 of an RL run, with 2 tasks and 4 samples each (the unit tests' shape; the real run sends 32 × 8 = 256 attempts):

1. **The Pod writes a request.** `run_rl` calls `rewards(3, attempts)`. The bridge client, session `0123456789ab`, gives the request sequence number 1. Its ID is `0123456789ab-000001-reward-000003`. The client writes `outbox/.0123456789ab-000001-reward-000003.json.<random>.tmp`, fsyncs it and renames it to `outbox/0123456789ab-000001-reward-000003.json`. The request, abridged:

   ```json
   {"format": "rl-bridge-request-v1", "request_id": "0123456789ab-000001-reward-000003",
    "seq": 1, "kind": "reward", "index": 3, "count": 8,
    "payload_sha256": "<SHA-256 of the canonical items>", "task_file_sha256": "<the Pod's RL task file>",
    "created_unix": 1800000000.0,
    "items": [{"task_id": "t0", "sample": 0, "code": null, "completion_tokens": 1},
              {"task_id": "t0", "sample": 1, "code": "def f(x):\n    return x + 1", "completion_tokens": 14}, "..."]}
   ```

   Sample 0 had no closed python block, so its `code` is `null`. It still travels, so the reply has one reward per attempt, in order. While the Pod waits, `client.heartbeat()` returns `{"status": "awaiting_rewards", "request_id": ..., "kind": "reward", "index": 3, "count": 8, "age_seconds": ...}`.
2. **The Mac polls.** One SSH call lists the outbox (name and size) and the inbox (names). The request has no reply yet, so the Mac pulls it, bounded by 64 MiB. It then checks:
   - the exact key set, and that the ID matches the file name and the `seq`, `kind` and `index` fields;
   - that `count` equals the number of items;
   - the payload SHA-256, recomputed from the items.

   It has no stored reply for this ID. Sequence number 1 is newer than any it answered for this session. The Pod's task-file SHA-256 is the configured one, and `t0` and `t1` are on the kept screening list. So it scores the attempts with `LocalRewardSource.rewards(3, attempts)` in the Docker sandbox. It stores the reply under `replies/` on the Mac, then pushes it: the remote program writes `inbox/.<id>.json.<random>.tmp`, fsyncs it and renames it.
3. **The reply**, abridged:

   ```json
   {"format": "rl-bridge-reply-v1", "request_id": "0123456789ab-000001-reward-000003",
    "kind": "reward", "index": 3, "count": 8, "payload_sha256": "<echoed>", "task_file_sha256": "<echoed>",
    "outcome": "scored", "reason": null, "detail": "", "seconds": 1.5, "answered_unix": 1800000002.0,
    "results": [{"reward": 0.0, "status": "no_program", "audit_flags": [], "seconds": 0.0, "unfinished": 0},
                {"reward": 1.0, "status": "completed", "audit_flags": [], "seconds": 0.8, "unfinished": 0}, "..."]}
   ```

4. **The Pod checks and returns.** The reply must echo the request ID, kind, index, count and both SHA-256 values. It must parse under the exact schema and fit 4 MiB. Every attempt without a program must score 0 with `no_program`. The Pod moves the request and the reply into `done/` and returns the eight rewards to `run_rl`.

**A Mac restart in the middle.** Suppose the Mac dies after it stored the reply but before the push finished. The request is still in the outbox and the inbox has no reply. The restarted Mac pulls the request again and finds a stored reply with the same ID and payload SHA-256. It pushes those exact bytes and never calls the scorer again. If it had died while still scoring, nothing would be stored, and it would score the request once more.

## Data flow and shapes

```
Pod (run_rl)                         Pod disk (bridge root)              Mac (rl-bridge)
BridgeRewardSource.rewards(u, A) ─► outbox/<id>.json  ◄── list, read ── BridgeServer.poll_once
BridgeDevEvaluator → executor(P)     (atomic rename)       (one SSH call   ├─ checks, refusals
   waits; heartbeat:                                        each)           ├─ LocalRewardSource / runner
   awaiting_rewards                  inbox/<id>.json  ◄── write ────────── ├─ replies/<id>.json (stored first)
   ◄── checks the reply ───────────  (atomic rename)                        └─ progress.json, events.jsonl
   moves both to done/
```

**Request** (`rl-bridge-request-v1`), exactly these keys:

| Key | Value |
| --- | --- |
| `request_id` | `<session>-<seq>-<kind>-<index>`: 12 hex digits, 6 digits, `reward` or `dev`, 6 digits |
| `seq` | this client's request number, from 1, increasing |
| `kind`, `index` | `reward` and the RL update, or `dev` and the check index |
| `count` | number of items, 1 to 4,096 |
| `payload_sha256` | SHA-256 of the items as canonical JSON (sorted keys, no spaces, ASCII escapes) |
| `task_file_sha256` | SHA-256 of the Pod's task file for this kind |
| `created_unix` | Pod time of writing |
| `items` | reward: `{task_id, sample, code, completion_tokens}` per attempt (`code` may be `null`); dev: `{task_id, code}` per program |

**Reply** (`rl-bridge-reply-v1`) echoes `request_id`, `kind`, `index`, `count`, `payload_sha256` and `task_file_sha256`. A refusal may set the last three to `null` when the request could not be read. The reply adds:

- `outcome`: `scored`, `host_failure`, `timeout` or `refused`;
- `reason`: set only for a refusal;
- `detail`: at most 500 characters;
- `results`: for `scored`, one row per item, in order. A reward row is exactly `AttemptReward.to_json()`: `reward`, `status`, `audit_flags`, `seconds`, `unfinished`. A dev row is `true` or `false`. Any other outcome carries none.
- `seconds`: the Mac's scoring time;
- `answered_unix`: the Mac time of the answer.

A reward may be `NaN`, which reaches the RL loop and stops it with `non_finite`.

**Directories** on the Pod: `outbox/`, `inbox/`, `done/` (answered pairs), `withdrawn/` (requests given up after a timeout, leftovers from an earlier client, stray replies), `rejected/` (replies that failed the checks). On the Mac, under `--state`: `replies/` (every reply sent, by request ID), `progress.json`, `events.jsonl` and `server.lock`.

**Sizes.** A reward request is 256 attempts of at most 256 tokens. A dev request is at most 200 tasks × 9 programs of at most 512 tokens, a few MB. The bounds are 64 MiB per request, 4 MiB per reply and 4,096 items.

**Failure modes:**

| Failure | Handling | Test |
| --- | --- | --- |
| No reply in time (Mac down, slow, SSH lost) | The Pod withdraws the request to `withdrawn/` and raises `RewardTimeout`; `run_rl` stops with `reward_wait` and saves the weights | `test_a_timeout_withdraws_the_request_and_raises_reward_timeout`, `test_a_mac_that_never_answers_ends_in_reward_timeout` |
| Reply with another ID, kind, index, count or SHA-256, a wrong schema, partial, corrupt, oversized or a symlink | The Pod moves it to `rejected/` and keeps waiting; the Mac re-pushes its stored reply on the next poll. More than `max_rejected` (3) raise `BridgeReplyError` | `test_the_pod_rejects_wrong_replies_and_accepts_the_right_one`, `test_too_many_rejected_replies_raise_a_bridge_reply_error` |
| Host failure the Mac's retries cannot resolve | `host_failure` reply (stored); the Pod raises `RewardHostFailure`, never a reward of 0 | `test_mac_host_failures_and_timeouts_become_stored_replies`, `test_failure_replies_map_to_the_rl_loop_errors` |
| Docker down during a dev check | After 16 unresolved calls in a row, no further program starts; the check is a host failure | `test_a_dead_sandbox_fails_a_dev_check_without_running_every_program` |
| Request over the bound, or wrong schema or payload SHA-256 | Refused (`oversized`, `schema`, `payload_sha256`) without scoring | `test_the_mac_refuses_oversized_and_wrong_schema_and_defers_a_partial_file` |
| Request that does not parse | Looked at again at the next poll; refused (`corrupt`) only if unchanged, so a slow writer is not refused | same |
| Task ID outside the kept list (rewards) or the dev subset (dev), or a wrong task-file SHA-256 | Refused (`unknown_task`, `task_file`); nothing runs; the Pod raises `BridgeRefused` | `test_the_mac_refuses_unknown_tasks_and_a_wrong_task_file` |
| A request seen again (reply lost, quarantined, or a Mac restart) | Answered from the stored reply, never re-scored | `test_a_repeated_request_is_answered_from_the_store_never_rescored`, `test_a_mac_restart_mid_request_never_scores_twice` |
| A reused request ID with another payload | Refused (`conflict`) | `test_a_repeated_request_is_answered_from_the_store_never_rescored` |
| Out-of-order requests (an older one still pending, or one older than an answered one) | Refused (`stale`); only the newest request of a session is scored | `test_out_of_order_requests_are_refused_as_stale` |
| Stray or leftover replies and requests on the Pod | Moved to `withdrawn/` before a request is written; never read as an answer | `test_leftovers_and_stray_replies_never_answer_a_request` |
| A second caller while a request waits, or a second client on the root | `BridgeError`; nothing is written | `test_the_client_carries_one_request_at_a_time`, `test_one_client_per_bridge_root` |
| Mac crash while scoring | Nothing stored; the restarted Mac scores the request once | `test_a_mac_crash_while_scoring_scores_again_after_the_restart` |
| SSH call fails or times out | Retried 4 times after the first try, waiting 1, 2, 4 and 8 s; 30 failed polls in a row end the server | `test_ssh_failures_are_retried_with_bounded_backoff`, `test_transport_failures_are_errors_never_listings` |
| The Pod has no bridge root yet | An idle poll (`pod_ready: false`), not a failure | `test_a_pod_without_a_bridge_root_yet_is_not_ready` |
| A secret in the Mac's environment | The command refuses to start; ssh children get only an allowlisted environment | `test_no_secret_reaches_an_ssh_child`, `test_the_command_refuses_secrets_sealed_files_and_bad_records` |

## Design choices and common mistakes

- **Files, not a socket.** The Pod cannot open a connection to the Mac, and a listening port on either side would be a new attack surface. Files written by atomic rename survive either side restarting, and a reader never sees half a file under its final name. The Mac does all the moving over SSH.
- **Store before push.** The Mac writes the reply to its own disk before it pushes it, so a crash after scoring never leads to a second scoring. A crash before the store loses nothing either: the request is still waiting, and scoring it again is the only way to answer it.
- **Refusals are not rejections.** The Mac *refuses* a request it will not score, and says why. The Pod then stops the run with `BridgeRefused`, because a wrong task file or an unknown task needs a person. The Pod *rejects* a reply that does not answer its request: it sets the reply aside and waits. The Mac sees the request still pending with no reply and pushes its stored reply again, so a damaged copy heals itself.
- **One request at a time.** The RL loop waits for each reply before it continues, so each session has one live request. The sequence number makes "older" well defined: a leftover request is `stale` and never spends sandbox time ahead of the live one. A new client (a restarted worker) draws a new session and first moves everything left in the boxes into `withdrawn/`.
- **The Mac's diagnostics stay on the Mac.** A host-failure or timeout reply says only what kind of failure it was. The exception text, which could quote the channel's details, goes to the Mac's `events.jsonl`. A refusal names only values the Pod sent itself: its task IDs and its task-file SHA-256.
- **The SSH channel is the rehearsal's.** It uses the same options as the rehearsal supervisor: pinned host key, no config file, the given key only, batch mode, the same timeouts and keep-alives. One fixed remote program takes its inputs as JSON on stdin, so no path or name is ever put into a shell command. It opens every path component without following symlinks, reads only from `outbox/` and writes only to `inbox/`. Opening with `O_NONBLOCK` means a FIFO planted in a box cannot hang either side.
- **Secrets stay out of every child.** An ssh child gets only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE` and `TMPDIR`. The Docker children inherit the bridge's own environment, so the command refuses to start while any `R2_` variable, the Runpod key, or a name the `.env` file defines is set. It also clears `SSH_AUTH_SOCK` and `SSH_AGENT_PID`.
- **Mistake to avoid: a flat timeout.** The Pod's timeout must cover the Mac's worst-case scoring time plus the bridge's own allowance. A fixed number stops a run that is merely slow ([rehearsal RL](../rl.md#the-mac-bridge-run-2) gives the inputs).
- **Mistake to avoid: two clients on one bridge root.** A second client would sweep the first one's live request away. The client takes an exclusive lock on `client.lock`, and the server one on its state directory.
- **Mistake to avoid: reading a refusal as a reward.** A refusal is a configuration problem. Scoring it 0 would train the policy on nothing.

## Verification

- [`tests/test_rl_bridge.py`](../../tests/test_rl_bridge.py): schemas and canonical hashing; every Pod-side rejection and mapping, on a fake clock whose sleeps play the Mac; the Mac's refusals, deferral, idempotent replays, stale and conflicting requests, restarts during scoring and during the push, SSH retries and backoff, host failures and timeouts, the pre-scoring check, the progress file, and the sandbox dev executor.
- [`tests/test_rl_bridge_ssh.py`](../../tests/test_rl_bridge_ssh.py): the SSH options equal the rehearsal supervisor's; the child environment and the refusals; stdin-only inputs; transport errors; and the exact remote program, run by a local interpreter in place of `ssh`. The program is checked for listing, bounded reads, missing files, atomic writes, symlinks, unsafe names and roots, and a Pod that is not ready.
- [`tests/test_rl_bridge_end_to_end.py`](../../tests/test_rl_bridge_end_to_end.py): the Pod and the Mac in separate threads:
  - over the local directory pair, and over the remote program, with fakes;
  - with the real `LocalRewardSource` and evaluation runner in Docker. A reference scores 1.0, a wrong program less, a missing one `no_program`, and no container is left behind;
  - `run_rl` for 2 updates and one dev check through the bridge, and the more-SFT control with its dev checks through it, each equal in records and weights to the same run scored locally;
  - the command's checks.

```sh
uv run --locked pytest tests/test_rl_bridge.py tests/test_rl_bridge_ssh.py tests/test_rl_bridge_end_to_end.py
SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_rl_bridge_end_to_end.py
```

## Questions and answers

1. **Why does an attempt without a program cross the bridge at all?** So the reply has exactly one reward per attempt, in order, and the Pod can check the count. The Mac scores it 0 with `no_program` without running anything, and the Pod refuses a reply that does otherwise.
2. **What happens if the Mac answers after the Pod gave up?** The Pod moved the request to `withdrawn/` when it raised `RewardTimeout`. The late reply lands in the inbox as a stray, and the next request moves it aside unread.
3. **How does a restarted Mac know not to score a request again?** It keeps every reply it sent under `replies/`, by request ID. A request whose ID and payload SHA-256 match a stored reply gets that reply's exact bytes.
4. **Why is a host failure never sent as a reward of 0?** A stall or a dead Docker daemon says nothing about the program. Scoring it 0 would teach the policy to avoid whatever the host happened to fail on. The Mac retries; if that fails, the reply is `host_failure` and the run stops cleanly.
5. **What does a heartbeat show while the Pod waits?** `awaiting_rewards` with the request ID, the kind (`reward` or `dev`), the index and how long the wait has lasted. On the Mac, `progress.json` shows the request it is scoring and the last one it answered, with times.
