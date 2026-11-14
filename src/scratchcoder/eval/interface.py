"""The prompt interface a checkpoint was trained for, and the protocols that fit it.

A checkpoint has one of two interfaces:

- ``chat-v1``: post-trained on chat-v1 conversations. It expects the chat-v1 token IDs
  (role markers, then ``<|assistant_start|>``) and writes its program in a python block.
- ``base``: pretrained on plain text. It continues plain text.

A generation protocol fits one interface: ``chat-v1`` fits ``chat-v1``; every plain-text
protocol (``base-pool-completion-v1``, ``base-completion-rstrip-v1`` and the legacy
``scratchcoder eval`` modes) fits ``base``.

How a checkpoint's interface is found, first match wins (``checkpoint_interface``):

1. ``declared``: the file has a top-level ``"interface"`` key. ``save_weights`` writes it
   from now on; any future checkpoint writer may add it.
2. ``weights-v1``: the file's ``"format"`` is ``weights-v1``. Only post-training writes this
   format (the SFT stage's dev-loss-selected weights, RL and the more-SFT control), and all
   of it trains on chat-v1, so every such file without the key is ``chat-v1``. This covers
   every historical SFT, RL and control checkpoint without rewriting any of them.
3. ``trainer-checkpoint``: the trainer's full checkpoint (``"version"``, ``"optimizer"``
   and a ``"train"`` configuration, no ``"format"``). Only pretraining writes these today;
   it is ``base`` unless its configuration carries a mark that only post-training sets,
   then ``chat-v1``: the special-token rows were redrawn (``special_rows_init`` other than
   ``none``), the loss was normalized by assistant targets (``loss_normalization`` of
   ``update_targets``), or the data mix names a post-training source (``sft`` or
   ``rl-references``). Pretraining sets none of these.
4. Otherwise the interface is undeclared (``None``): a hand-made or fixture file, the
   shared random initialization, or a format this code does not know. An undeclared
   checkpoint is evaluated only under an explicitly chosen protocol.

Nothing here rewrites a checkpoint: the rules read what the file already records.

**Deliberate overrides.** A protocol that does not fit the checkpoint's interface is
refused unless the caller passes ``--allow-interface-mismatch`` (``allow_mismatch``). The
override is never silent: when it is used, the choice carries an ``InterfaceOverride``
whose ``record()`` is written into the completions header (``eval-generate``) or the
report and its scorecard row (legacy ``scratchcoder eval``) under ``interface_override``.
The record names the requested protocol, the interface that protocol is written for, the
checkpoint's resolved interface and the rule that found it, that the override was used,
and why it was needed. The flag is refused where it would change nothing (the protocol
fits, or the checkpoint's interface is undeclared), so a recorded override always marks
a real mismatch. Historical procedures, such as the legacy command's plain-text base mode
on SFT weights, stay reproducible this way, and their outputs say so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scratchcoder.data import chat
from scratchcoder.eval import prompt as prompts

CHAT_INTERFACE = chat.PROTOCOL
BASE_INTERFACE = "base"
INTERFACES = (CHAT_INTERFACE, BASE_INTERFACE)
INTERFACE_KEY = "interface"

DECLARED = "declared"
FROM_WEIGHTS_V1 = "weights-v1"
FROM_TRAINER = "trainer-checkpoint"
UNDECLARED = "undeclared"

PROTOCOL_INTERFACES: Mapping[str, str] = {
    chat.PROTOCOL: CHAT_INTERFACE,
    prompts.POOL_PROTOCOL: BASE_INTERFACE,
    prompts.BENCHMARK_PROTOCOL: BASE_INTERFACE,
    prompts.LEGACY_POOL_PROTOCOL: BASE_INTERFACE,
    prompts.LEGACY_INSTRUCTION_PROTOCOL: BASE_INTERFACE,
}
"""The interface each named generation protocol is written for."""
LEGACY_MODE_INTERFACES: Mapping[str, str] = {"base": BASE_INTERFACE, "instruction": BASE_INTERFACE}
"""``scratchcoder eval --prompt-mode`` values: both encode plain text, no special tokens."""


OVERRIDE_FLAG = "--allow-interface-mismatch"
OVERRIDE_KEY = "interface_override"
"""The key under which completions headers and legacy reports record a used override."""
OVERRIDE_FIELDS = frozenset(
    {
        "override",
        "override_used",
        "requested",
        "requested_protocol",
        "requested_interface",
        "checkpoint_interface",
        "checkpoint_interface_source",
        "reason",
    }
)


class InterfaceMismatchError(ValueError):
    """A protocol was chosen, or would be chosen, that does not fit the checkpoint."""


@dataclass(frozen=True)
class CheckpointInterface:
    """A checkpoint's interface (``None`` when undeclared) and the rule that found it."""

    name: str | None
    source: str

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source}


@dataclass(frozen=True)
class InterfaceOverride:
    """A deliberate, used ``--allow-interface-mismatch``: what was crossed, and why.

    ``requested`` is the choice as named on the command line (``--protocol chat-v1``,
    ``--prompt-mode base``), ``requested_protocol`` the recorded protocol name it ran
    under, ``requested_interface`` the interface that protocol is written for, and
    ``reason`` the mismatch, in words.
    """

    requested: str
    requested_protocol: str
    requested_interface: str
    checkpoint_interface: str
    checkpoint_interface_source: str
    reason: str

    def record(self) -> dict[str, Any]:
        return {
            "override": OVERRIDE_FLAG,
            "override_used": True,
            "requested": self.requested,
            "requested_protocol": self.requested_protocol,
            "requested_interface": self.requested_interface,
            "checkpoint_interface": self.checkpoint_interface,
            "checkpoint_interface_source": self.checkpoint_interface_source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProtocolChoice:
    """What a checkpoint is evaluated under, and how that was decided.

    ``name`` is the protocol (``eval-generate``) or legacy prompt mode (``scratchcoder
    eval``); ``requested`` is what the caller named (``None``: taken from the interface);
    ``override`` is set only when a mismatch was deliberately crossed.
    """

    name: str
    requested: str | None
    interface: CheckpointInterface
    override: InterfaceOverride | None = None


def check_override_record(record: object, protocol: str) -> None:
    """Refuse a recorded override that is not a used override of ``protocol``'s mismatch."""
    if not isinstance(record, Mapping) or set(record) != OVERRIDE_FIELDS:
        raise ValueError(f"{OVERRIDE_KEY} must name exactly {sorted(OVERRIDE_FIELDS)}")
    if record["override"] != OVERRIDE_FLAG or record["override_used"] is not True:
        raise ValueError(f"{OVERRIDE_KEY} records a used {OVERRIDE_FLAG}")
    if record["requested_protocol"] != protocol:
        raise ValueError(f"{OVERRIDE_KEY} names another protocol than the one recorded")
    fits = PROTOCOL_INTERFACES.get(protocol)
    if record["requested_interface"] != fits:
        raise ValueError(f"{OVERRIDE_KEY} names the wrong interface for {protocol}")
    if record["checkpoint_interface"] not in INTERFACES or record["checkpoint_interface"] == fits:
        raise ValueError(f"{OVERRIDE_KEY} must record a checkpoint interface that differs")
    if record["checkpoint_interface_source"] not in {DECLARED, FROM_WEIGHTS_V1, FROM_TRAINER}:
        raise ValueError(f"{OVERRIDE_KEY} names an unknown interface rule")
    for name in ("requested", "reason"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f"{OVERRIDE_KEY} needs a non-empty {name}")


def checkpoint_interface(saved: Mapping[str, Any]) -> CheckpointInterface:
    """The interface of a loaded checkpoint mapping, by the rules in the module docstring."""
    if INTERFACE_KEY in saved:
        declared = saved[INTERFACE_KEY]
        if declared not in INTERFACES:
            raise ValueError(f"checkpoint declares an unknown interface: {declared!r}")
        return CheckpointInterface(declared, DECLARED)
    if saved.get("format") == FROM_WEIGHTS_V1:
        return CheckpointInterface(CHAT_INTERFACE, FROM_WEIGHTS_V1)
    config = saved.get("config")
    if (
        "format" not in saved
        and "version" in saved
        and "optimizer" in saved
        and isinstance(config, Mapping)
        and isinstance(config.get("train"), Mapping)
    ):
        name = CHAT_INTERFACE if _post_training_marks(config) else BASE_INTERFACE
        return CheckpointInterface(name, FROM_TRAINER)
    return CheckpointInterface(None, UNDECLARED)


POST_TRAINING_SOURCES = frozenset({"sft", "rl-references"})
"""Data-mix sources only post-training (SFT and the more-SFT control) trains on."""


def _post_training_marks(config: Mapping[str, Any]) -> list[str]:
    """The settings in a trainer configuration that only post-training uses, if any."""
    train = config["train"]
    marks = []
    if train.get("special_rows_init", "none") != "none":
        marks.append("special_rows_init")
    if train.get("loss_normalization", "capacity") == "update_targets":
        marks.append("loss_normalization")
    data_mix = config.get("data_mix")
    if isinstance(data_mix, Mapping) and POST_TRAINING_SOURCES & set(data_mix):
        marks.append("data_mix")
    return marks


def read_checkpoint_interface(path: Path) -> CheckpointInterface:
    """A checkpoint file's interface. The file is memory-mapped, so its tensors are not read."""
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, Mapping):
        raise ValueError("not a model checkpoint")
    return checkpoint_interface(saved)


def _mismatch(interface: CheckpointInterface, chosen: str, fits: str) -> str:
    return (
        f"{chosen} is written for {fits} checkpoints, but this checkpoint's interface is "
        f"{interface.name} (found by the {interface.source} rule)"
    )


def require_fit(
    interface: CheckpointInterface,
    chosen: str,
    fits: str,
    *,
    allow_mismatch: bool,
    protocol: str | None = None,
) -> InterfaceOverride | None:
    """Refuse ``chosen`` (a protocol or legacy mode written for ``fits``) unless it fits.

    Returns ``None`` when it fits. An undeclared interface fits any explicitly chosen
    protocol. ``allow_mismatch`` is a deliberate override, for a diagnostic or historical
    reproduction that crosses interfaces on purpose: a crossed choice then returns the
    ``InterfaceOverride`` to record (``protocol`` is the recorded protocol name, default
    ``chosen``). The override is refused where nothing is crossed, so a recorded override
    always marks a real mismatch.
    """
    if interface.name is None or interface.name == fits:
        if allow_mismatch:
            why = (
                "the checkpoint does not record its interface, so any explicitly chosen "
                "protocol is accepted"
                if interface.name is None
                else f"{chosen} fits this checkpoint's {interface.name} interface"
            )
            raise InterfaceMismatchError(
                f"{OVERRIDE_FLAG} was passed but nothing is crossed: {why}; drop the flag"
            )
        return None
    reason = _mismatch(interface, chosen, fits)
    if not allow_mismatch:
        raise InterfaceMismatchError(
            reason + f"; choose a protocol that fits it, or pass {OVERRIDE_FLAG} to cross "
            "interfaces deliberately (the override is recorded)"
        )
    return InterfaceOverride(
        requested=chosen,
        requested_protocol=protocol or chosen,
        requested_interface=fits,
        checkpoint_interface=interface.name,
        checkpoint_interface_source=interface.source,
        reason=reason,
    )


def default_protocol(interface: CheckpointInterface, family: str) -> str:
    """The protocol a checkpoint resolves to for a suite family (``pool`` or ``benchmark``)."""
    if family not in {"pool", "benchmark"}:
        raise ValueError("suite family must be pool or benchmark")
    if interface.name is None:
        raise InterfaceMismatchError(
            "the checkpoint does not record its interface (no 'interface' key, not weights-v1, "
            "not a trainer checkpoint); choose the protocol explicitly with --protocol"
        )
    if interface.name == CHAT_INTERFACE:
        return chat.PROTOCOL
    return prompts.POOL_PROTOCOL if family == "pool" else prompts.BENCHMARK_PROTOCOL


def resolve_protocol(
    interface: CheckpointInterface,
    family: str,
    requested: str | None,
    *,
    allow_mismatch: bool = False,
) -> ProtocolChoice:
    """The protocol to generate under: ``requested`` if it fits, else the checkpoint's own.

    With no request the checkpoint's interface chooses (``default_protocol``). A request is
    checked against the interface (``require_fit``); a deliberately crossed request comes
    back with its ``override`` to record.
    """
    if requested is None:
        if allow_mismatch:
            raise InterfaceMismatchError(f"{OVERRIDE_FLAG} needs an explicit --protocol")
        return ProtocolChoice(default_protocol(interface, family), None, interface)
    if requested not in PROTOCOL_INTERFACES:
        raise ValueError(f"unknown protocol {requested!r}")
    override = require_fit(
        interface,
        f"--protocol {requested}",
        PROTOCOL_INTERFACES[requested],
        allow_mismatch=allow_mismatch,
        protocol=requested,
    )
    return ProtocolChoice(requested, requested, interface, override)


def resolve_legacy_mode(
    interface: CheckpointInterface,
    requested: str | None,
    *,
    allow_mismatch: bool = False,
    protocol: str | None = None,
) -> ProtocolChoice:
    """``scratchcoder eval``'s prompt mode: ``base`` or ``instruction``.

    Both legacy modes are plain-text protocols. With no request a ``base`` checkpoint gets
    ``base`` (the command's historical default). A chat-v1 checkpoint has no legacy mode
    that fits, and an undeclared one needs the mode named, so both fail with a message
    that names the way forward. An explicitly requested mode is checked like any protocol;
    ``protocol`` is the protocol name the run will record, for a crossed mode's override.
    """
    if requested is None:
        if allow_mismatch:
            raise InterfaceMismatchError(f"{OVERRIDE_FLAG} needs an explicit --prompt-mode")
        if interface.name == BASE_INTERFACE:
            return ProtocolChoice("base", None, interface)
        if interface.name == CHAT_INTERFACE:
            raise InterfaceMismatchError(
                f"this checkpoint's interface is chat-v1 (found by the {interface.source} "
                "rule), and scratchcoder eval has only plain-text prompt modes; evaluate it "
                "with scratchcoder eval-generate (it resolves to --protocol chat-v1) and "
                "eval-execute"
            )
        raise InterfaceMismatchError(
            "the checkpoint does not record its interface (no 'interface' key, not weights-v1, "
            "not a trainer checkpoint); choose --prompt-mode base or instruction explicitly"
        )
    if requested not in LEGACY_MODE_INTERFACES:
        raise ValueError("prompt mode must be base or instruction")
    override = require_fit(
        interface,
        f"--prompt-mode {requested}",
        LEGACY_MODE_INTERFACES[requested],
        allow_mismatch=allow_mismatch,
        protocol=protocol,
    )
    return ProtocolChoice(requested, requested, interface, override)
