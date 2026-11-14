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
   and a ``"train"`` configuration, no ``"format"``). Only pretraining writes these; it is
   ``base``, unless its training configuration redrew the special-token rows
   (``special_rows_init`` other than ``none``), which only an SFT run does: then
   ``chat-v1``.
4. Otherwise the interface is undeclared (``None``): a hand-made or fixture file, the
   shared random initialization, or a format this code does not know. An undeclared
   checkpoint is evaluated only under an explicitly chosen protocol.

Nothing here rewrites a checkpoint: the rules read what the file already records.
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


class InterfaceMismatchError(ValueError):
    """A protocol was chosen, or would be chosen, that does not fit the checkpoint."""


@dataclass(frozen=True)
class CheckpointInterface:
    """A checkpoint's interface (``None`` when undeclared) and the rule that found it."""

    name: str | None
    source: str

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source}


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
        special_rows = config["train"].get("special_rows_init", "none")
        name = BASE_INTERFACE if special_rows == "none" else CHAT_INTERFACE
        return CheckpointInterface(name, FROM_TRAINER)
    return CheckpointInterface(None, UNDECLARED)


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
    interface: CheckpointInterface, chosen: str, fits: str, *, allow_mismatch: bool
) -> None:
    """Refuse ``chosen`` (a protocol or legacy mode written for ``fits``) unless it fits.

    An undeclared interface fits any explicitly chosen protocol. ``allow_mismatch`` is a
    deliberate, recorded override, for a diagnostic that crosses interfaces on purpose.
    """
    if interface.name is None or interface.name == fits or allow_mismatch:
        return
    raise InterfaceMismatchError(
        _mismatch(interface, chosen, fits)
        + "; choose a protocol that fits it, or pass --allow-interface-mismatch to cross "
        "interfaces deliberately"
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
) -> str:
    """The protocol to generate under: ``requested`` if it fits, else the checkpoint's own.

    With no request the checkpoint's interface chooses (``default_protocol``). A request is
    checked against the interface (``require_fit``).
    """
    if requested is None:
        if allow_mismatch:
            raise InterfaceMismatchError("--allow-interface-mismatch needs an explicit --protocol")
        return default_protocol(interface, family)
    if requested not in PROTOCOL_INTERFACES:
        raise ValueError(f"unknown protocol {requested!r}")
    require_fit(interface, requested, PROTOCOL_INTERFACES[requested], allow_mismatch=allow_mismatch)
    return requested


def resolve_legacy_mode(
    interface: CheckpointInterface, requested: str | None, *, allow_mismatch: bool = False
) -> str:
    """``scratchcoder eval``'s prompt mode: ``base`` or ``instruction``.

    Both legacy modes are plain-text protocols. With no request a ``base`` checkpoint gets
    ``base`` (the command's historical default). A chat-v1 checkpoint has no legacy mode
    that fits, and an undeclared one needs the mode named, so both fail with a message
    that names the way forward. An explicitly requested mode is checked like any protocol.
    """
    if requested is None:
        if allow_mismatch:
            raise InterfaceMismatchError(
                "--allow-interface-mismatch needs an explicit --prompt-mode"
            )
        if interface.name == BASE_INTERFACE:
            return "base"
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
    require_fit(
        interface,
        f"--prompt-mode {requested}",
        LEGACY_MODE_INTERFACES[requested],
        allow_mismatch=allow_mismatch,
    )
    return requested
