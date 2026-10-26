"""OpenAssistant conversation trees to chat-v1 language conversations.

Each tree is a prompter message with nested replies. A message is eligible only if it
is English, not synthetic, not deleted, accepted by review, carries no nonzero PII label
and has text. Labels, when present, must be the README's mapping form (each label name
maps to an object with a numeric ``value``); any other shape is ``missing_metadata``.
Eligible messages that carry no PII label at all are counted separately. From each exported tree the adapter follows the best-ranked eligible reply
at every turn (rank 0 first, unranked replies last, message ID breaking ties), so the
chosen path never contains an ineligible message. It then keeps the longest prefix
that ends on an assistant turn and renders to at most ``max_tokens`` chat-v1 tokens.
Every drop is counted by reason. See ``docs/sft-data.md``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from scratchcoder.data.chat import Message, TextEncoder

ROLES = {"prompter": "user", "assistant": "assistant"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
REQUIRED = ("message_id", "text", "role", "lang", "synthetic", "deleted", "review_result")


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    text: str
    rank: int | None
    reason: str | None
    """Why the message is ineligible, or None."""
    replies: tuple[Node, ...]
    pii_labelled: bool = True
    """Whether the message carries a PII label (only meaningful when eligible)."""


@dataclass(frozen=True)
class Conversation:
    id: str
    source: str
    messages: tuple[Message, ...]
    message_ids: tuple[str, ...]
    tokens: int


def message_reason(raw: Mapping[str, Any]) -> str | None:
    """The first reason a message cannot appear in a conversation, or None."""
    if any(key not in raw for key in REQUIRED):
        return "missing_metadata"
    if raw["role"] not in ROLES:
        return "unknown_role"
    if raw["lang"] != "en":
        return "not_english"
    if raw["synthetic"] is not False:
        return "synthetic"
    if raw["deleted"] is not False:
        return "deleted"
    if raw["review_result"] is not True:
        return "failed_review"
    labels = raw.get("labels")
    if labels is None:
        labels = {}
    if not isinstance(labels, Mapping) or any(
        not isinstance(entry, Mapping) or not _numeric(entry.get("value"))
        for entry in labels.values()
    ):
        return "missing_metadata"
    pii = labels.get("pii")
    if pii is not None and pii["value"] != 0:
        return "pii_label"
    if not isinstance(raw["text"], str) or not raw["text"].strip():
        return "empty_text"
    return None


def _numeric(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_node(raw: Mapping[str, Any], counts: Counter[str]) -> Node:
    """One message and its replies; ``counts`` gains every message's reason.

    An eligible message with no PII label also counts as ``message:eligible_no_pii_label``.
    """
    reason = message_reason(raw)
    counts["messages_seen"] += 1
    counts[f"message:{reason or 'eligible'}"] += 1
    pii_labelled = reason is not None or "pii" in (raw.get("labels") or {})
    if not pii_labelled:
        counts["message:eligible_no_pii_label"] += 1
    rank = raw.get("rank")
    replies = raw.get("replies") or []
    if not isinstance(replies, list):
        raise ValueError("message replies must be a list")
    return Node(
        id=str(raw.get("message_id", "")),
        role=ROLES.get(str(raw.get("role")), "unknown"),
        text=raw["text"].strip() if reason is None else "",
        rank=rank if type(rank) is int else None,
        reason=reason,
        replies=tuple(parse_node(reply, counts) for reply in replies),
        pii_labelled=pii_labelled,
    )


def best_path(root: Node) -> list[Node]:
    """The best-ranked eligible reply at every turn, from an eligible root."""
    path = [root]
    while True:
        eligible = [reply for reply in path[-1].replies if reply.reason is None]
        if not eligible:
            return path
        eligible.sort(key=lambda reply: (reply.rank is None, reply.rank or 0, reply.id))
        path.append(eligible[0])


def fitted_prefix(path: list[Node], encoder: TextEncoder, max_tokens: int) -> int:
    """The number of leading messages in the longest fitting prefix ending on assistant.

    A chat-v1 conversation costs one ``<|bos|>`` plus, per turn, two marker tokens and
    the encoded text, so prefix lengths add up without re-rendering.
    """
    total, best = 1, 0
    for index, node in enumerate(path):
        total += 2 + len(encoder.encode(node.text))
        if total > max_tokens:
            break
        if node.role == "assistant":
            best = index + 1
    return best


def adapt_trees(
    trees: Iterable[Mapping[str, Any]],
    encoder: TextEncoder,
    *,
    max_tokens: int = 960,
    source: str = "OpenAssistant/oasst2",
) -> tuple[list[Conversation], dict[str, int]]:
    """One conversation per exported tree whose best eligible path fits ``max_tokens``."""
    if type(max_tokens) is not int or max_tokens < 7:
        raise ValueError("max_tokens must fit at least one exchange")
    counts: Counter[str] = Counter()
    kept: list[Conversation] = []
    seen: set[str] = set()
    for raw in trees:
        counts["trees_seen"] += 1
        tree_id = raw.get("message_tree_id")
        if not isinstance(tree_id, str) or not tree_id or tree_id in seen:
            raise ValueError("trees need unique message_tree_id strings")
        seen.add(tree_id)
        if raw.get("tree_state") != "ready_for_export":
            counts["tree:not_ready_for_export"] += 1
            continue
        root = parse_node(raw["prompt"], counts)
        if root.reason is not None:
            counts[f"tree:root_{root.reason}"] += 1
            continue
        if root.role != "user":
            counts["tree:root_not_prompter"] += 1
            continue
        path = best_path(root)
        if any(node.role != ("user", "assistant")[index % 2] for index, node in enumerate(path)):
            counts["tree:roles_do_not_alternate"] += 1
            continue
        if len(path) < 2:
            counts["tree:no_eligible_reply"] += 1
            continue
        length = fitted_prefix(path, encoder, max_tokens)
        if length == 0:
            counts["tree:over_tokens"] += 1
            continue
        chosen = path[:length]
        if length < len(path) - (path[-1].role == "user"):
            counts["tree:truncated_to_fit"] += 1
        tokens = 1 + sum(2 + len(encoder.encode(node.text)) for node in chosen)
        unlabelled = sum(not node.pii_labelled for node in chosen)
        if unlabelled:
            counts["kept_messages_no_pii_label"] += unlabelled
            counts["tree:kept_with_no_pii_label"] += 1
        kept.append(
            Conversation(
                id=tree_id,
                source=source,
                messages=tuple(Message(node.role, node.text) for node in chosen),
                message_ids=tuple(node.id for node in chosen),
                tokens=tokens,
            )
        )
        counts["tree:kept"] += 1
    return kept, dict(sorted(counts.items()))


def has_email(conversation: Conversation) -> bool:
    return any(EMAIL.search(message.text) for message in conversation.messages)


def conversation_json(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "source": conversation.source,
        "messages": [{"role": m.role, "text": m.text} for m in conversation.messages],
        "message_ids": list(conversation.message_ids),
        "tokens": conversation.tokens,
    }


def conversation_from_json(raw: Mapping[str, Any]) -> Conversation:
    if not isinstance(raw, Mapping) or set(raw) != {
        "id",
        "source",
        "messages",
        "message_ids",
        "tokens",
    }:
        raise ValueError("a conversation has exactly id, source, messages, message_ids, tokens")
    messages = raw["messages"]
    if not isinstance(messages, list) or any(
        not isinstance(item, Mapping) or set(item) != {"role", "text"} for item in messages
    ):
        raise ValueError("conversation messages are role/text objects")
    if (
        not isinstance(raw["id"], str)
        or not raw["id"]
        or not isinstance(raw["source"], str)
        or not raw["source"]
        or not isinstance(raw["message_ids"], list)
        or type(raw["tokens"]) is not int
    ):
        raise ValueError("conversation identity, source, message IDs and tokens are required")
    return Conversation(
        id=raw["id"],
        source=raw["source"],
        messages=tuple(Message(item["role"], item["text"]) for item in messages),
        message_ids=tuple(str(item) for item in raw["message_ids"]),
        tokens=raw["tokens"],
    )
