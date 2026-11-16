"""Mixing agent rows with replay rows at a target ratio of assistant-loss tokens.

"1:1" means **loss tokens**: the agent side's loss tokens (``sum(mask[1:])``, what the SFT
loader counts as targets) about equal the replay side's. Row counts are reported beside
them, never used as the ratio.

The side with more loss tokens than its share is cut down: replay rows in a fixed hash order
(``sha256(salt|id)``); agent rows by taking instances template by template in turn (each
template's instances in hash order), so the kept agent rows keep their template spread. An
instance's rows (edit and repairs) are kept or cut together. Diversity is reported as
distinct templates, families, workflow types and mutation kinds, not row counts.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from scratchcoder.data.sft import SFTRow, loss_tokens

SALT = "agent-sft1-mix-v1"


def _order(identifier: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}|{identifier}".encode()).digest()


def _take_until(rows: Sequence[SFTRow], budget: int) -> list[SFTRow]:
    kept, used = [], 0
    for row in rows:
        if used >= budget:
            break
        kept.append(row)
        used += loss_tokens(row)
    return kept


def mix(
    agent: Sequence[SFTRow],
    meta: Mapping[str, Mapping[str, Any]],
    replay: Sequence[SFTRow],
    *,
    ratio: float = 1.0,
    salt: str = SALT,
) -> tuple[list[SFTRow], dict[str, Any]]:
    """The mixed rows (agent first, then replay, each in its kept order) and the report.
    ``ratio`` is agent loss tokens per replay loss token."""
    if ratio <= 0:
        raise ValueError("the ratio must be positive")
    if len({row.id for row in [*agent, *replay]}) != len(agent) + len(replay):
        raise ValueError("row IDs must be unique across both sides")
    agent_tokens = sum(loss_tokens(row) for row in agent)
    replay_tokens = sum(loss_tokens(row) for row in replay)
    replay_sorted = sorted(replay, key=lambda row: _order(row.id, salt))
    if agent_tokens > ratio * replay_tokens:
        groups: dict[str, dict[str, list[SFTRow]]] = defaultdict(lambda: defaultdict(list))
        for row in agent:
            info = meta[row.id]
            groups[str(info["template_id"])][str(info["instance"])].append(row)
        queues = {
            template: sorted(instances.items(), key=lambda item: _order(item[0], salt))
            for template, instances in sorted(groups.items())
        }
        budget = ratio * replay_tokens
        kept_agent: list[SFTRow] = []
        used = 0
        while used < budget and any(queues.values()):
            for template in sorted(queues):
                if not queues[template] or used >= budget:
                    continue
                _, rows = queues[template].pop(0)
                kept_agent.extend(rows)
                used += sum(loss_tokens(row) for row in rows)
        kept_replay = replay_sorted
    else:
        kept_agent = list(agent)
        kept_replay = _take_until(replay_sorted, int(agent_tokens / ratio))
    mixed = [*kept_agent, *kept_replay]
    return mixed, report(kept_agent, kept_replay, meta, ratio, agent, replay)


def report(
    agent: Sequence[SFTRow],
    replay: Sequence[SFTRow],
    meta: Mapping[str, Mapping[str, Any]],
    ratio: float,
    agent_all: Sequence[SFTRow],
    replay_all: Sequence[SFTRow],
) -> dict[str, Any]:
    agent_tokens = sum(loss_tokens(row) for row in agent)
    replay_tokens = sum(loss_tokens(row) for row in replay)
    total_tokens = agent_tokens + replay_tokens
    rows = len(agent) + len(replay)
    infos = [meta[row.id] for row in agent]
    return {
        "ratio_definition": "agent assistant-loss tokens per replay assistant-loss token",
        "target_ratio": ratio,
        "agent": {
            "rows": len(agent),
            "loss_tokens": agent_tokens,
            "rows_available": len(agent_all),
            "loss_tokens_available": sum(loss_tokens(row) for row in agent_all),
            "by_kind": dict(Counter(str(info["kind"]) for info in infos)),
            "distinct_templates": len({info["template_id"] for info in infos}),
            "distinct_families": len({info["family"] for info in infos}),
            "workflow_types": sorted({str(info["workflow_type"]) for info in infos}),
            "mutation_kinds": sorted(
                {str(info["mutation_kind"]) for info in infos if info["mutation_kind"]}
            ),
            "instances": len({info["instance"] for info in infos}),
        },
        "replay": {
            "rows": len(replay),
            "loss_tokens": replay_tokens,
            "rows_available": len(replay_all),
            "loss_tokens_available": sum(loss_tokens(row) for row in replay_all),
            "by_source": dict(Counter(row.source for row in replay)),
        },
        "loss_token_share_agent": agent_tokens / total_tokens if total_tokens else 0.0,
        "row_share_agent": len(agent) / rows if rows else 0.0,
        "achieved_ratio": agent_tokens / replay_tokens if replay_tokens else None,
    }
