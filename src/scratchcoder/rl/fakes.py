"""Deterministic stand-ins for the reward source and the dev evaluator.

They run no code and touch no sandbox, so CPU tests and end-to-end smokes can drive the RL
loop and the more-SFT control without Docker. Both keep the real contracts: a ``None``-code
attempt scores 0 with ``no_program``, and a dev check is labelled with the index it was
asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from scratchcoder.model.gpt import GPT
from scratchcoder.rl.dev import DevCheck
from scratchcoder.rl.rewards import NO_PROGRAM, Attempt, AttemptReward, RewardTimeout


def code_length_reward(attempt: Attempt) -> float:
    """A toy reward in [0, 1] that varies with the program: its length modulo 4, over 3."""
    assert attempt.code is not None
    return (len(attempt.code) % 4) / 3


@dataclass
class FakeRewardSource:
    """Scores each attempt with ``reward_of(attempt)``; ``flags_of`` adds audit flags.

    ``timeout_at`` raises ``RewardTimeout`` on that update; ``calls`` records each request.
    """

    reward_of: Callable[[Attempt], float] = code_length_reward
    flags_of: Callable[[Attempt], tuple[str, ...]] = lambda attempt: ()
    status: str = "completed"
    timeout_at: int | None = None
    calls: list[tuple[int, int]] = field(default_factory=list)

    def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
        self.calls.append((update, len(attempts)))
        if self.timeout_at is not None and update == self.timeout_at:
            raise RewardTimeout(f"fake timeout at update {update}")
        return [
            AttemptReward(0.0, NO_PROGRAM)
            if attempt.code is None
            else AttemptReward(self.reward_of(attempt), self.status, self.flags_of(attempt))
            for attempt in attempts
        ]


@dataclass
class ScriptedDevEvaluator:
    """Returns ``DevCheck`` values from ``script(check)``: (greedy pass@1, 8-sample mean).

    ``calls`` records the checks requested; ``timeout_at`` raises ``RewardTimeout`` there.
    """

    script: Callable[[int], tuple[float, float]] = lambda check: (0.0, 0.0)
    tasks: int = 200
    constant_baseline: float = 0.1
    timeout_at: int | None = None
    calls: list[int] = field(default_factory=list)

    def evaluate(self, model: GPT, check: int) -> DevCheck:
        self.calls.append(check)
        if self.timeout_at is not None and check == self.timeout_at:
            raise RewardTimeout(f"fake dev timeout at check {check}")
        greedy, mean = self.script(check)
        return DevCheck(
            check=check,
            greedy_pass_at_1=greedy,
            mean_pass_n8=mean,
            pass_at_8=min(1.0, mean * 8),
            constant_baseline=self.constant_baseline,
            tasks=self.tasks,
            generation_seconds=0.0,
            wait_seconds=0.0,
        )
