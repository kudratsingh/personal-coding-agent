"""Recorded base-completion prompt/stop protocol, shared with reference tooling."""

from collections.abc import Sequence
from typing import Any

BASE_STOPS = ("\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n#")


def base_prompt(text: str) -> str:
    return text.rstrip()


def truncate_completion(text: str, *, stops: Sequence[str] = BASE_STOPS) -> str:
    offsets = [text.find(stop) for stop in stops if stop in text]
    return text[: min(offsets)] if offsets else text


def prompt_protocol(mode: str = "base", *, stops: Sequence[str] = BASE_STOPS) -> dict[str, Any]:
    if mode == "base":
        return {
            "prompt_stripped": True,
            "protocol": {
                "name": "base-completion-rstrip-v1",
                "prompt_format": "raw-task-prompt.rstrip()",
                "stop_strings": list(stops),
                "candidate_assembly": "stripped-prompt+retained-completion",
            },
        }
    return {
        "prompt_stripped": False,
        "protocol": {
            "name": "legacy-instruction-v1",
            "prompt_format": "public-prompt-with-public-tests",
            "stop_strings": [],
            "candidate_assembly": "legacy-fence-extraction-and-entry-point-heuristic",
        },
    }
