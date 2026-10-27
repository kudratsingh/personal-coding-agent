"""Reinforcement learning from test rewards, its screening and its more-SFT control.

- ``rewards``: ``Attempt``, ``AttemptReward``, the ``RewardSource`` protocol and the local
  thread-pool source over the reward channel.
- ``rollout``: chat-v1 prompts, seeds and generated attempts.
- ``screening``: the candidate keep rule and its record.
- ``policy``: group-mean advantages and the token-normalized policy-gradient loss.
- ``dev``: the fixed dev subset, dev checks and checkpoint selection.
- ``loop``: the RL run with its stops.
- ``control``: the compute-matched more-SFT control.
- ``checkpoint``: weights-only checkpoints.
- ``bridge``: the reward bridge's schemas and its Pod side (rewards and dev verdicts from
  the Mac, through files).
- ``bridge_mac``: the bridge's Mac side: polling the Pod over SSH, scoring, replying.
- ``bridge_command``: ``scratchcoder rl-bridge``.
- ``fakes``: deterministic stand-ins for tests and smokes.
"""
