"""Pinned evaluation-only intake of the original MBPP task source."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.manifest import ManifestEntry


@dataclass(frozen=True)
class MbppRelease:
    manifest: ManifestEntry
    task_count: int = 974

    def validate(self) -> None:
        self.manifest.validate()
        if (
            self.manifest.source != "google-research-datasets/mbpp"
            or self.manifest.purpose != "eval"
            or type(self.task_count) is not int
            or not 1 <= self.task_count <= 1000
            or not 0 < self.manifest.size_bytes <= 1024 * 1024
        ):
            raise ValueError("MBPP reference intake is a bounded evaluation-only audit")


def mbpp_release() -> MbppRelease:
    revision = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
    return MbppRelease(
        ManifestEntry(
            source="google-research-datasets/mbpp",
            revision=revision,
            url=f"https://raw.githubusercontent.com/google-research/google-research/{revision}/mbpp/mbpp.jsonl",
            license="CC-BY-4.0",
            attribution=(
                "Google Research; Austin et al. (2021), "
                "Program Synthesis with Large Language Models"
            ),
            sha256="ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f",
            size_bytes=563743,
            purpose="eval",
            license_verified=True,
        )
    )


def read_mbpp(path: Path, specification: MbppRelease) -> list[dict[str, Any]]:
    specification.validate()
    if path.stat().st_size != specification.manifest.size_bytes:
        raise ValueError("MBPP byte size differs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != specification.manifest.sha256:
        raise ValueError("MBPP SHA-256 differs")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != specification.task_count:
        raise ValueError("MBPP complete source count differs")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"task_id", "text", "code", "test_list", "test_setup_code", "challenge_test_list"}
            or type(row["task_id"]) is not int
            or not 1 <= row["task_id"] <= 1000
            or row["task_id"] in seen
            or any(not isinstance(row[key], str) for key in ("text", "code", "test_setup_code"))
            or not row["text"].strip()
            or not row["code"].strip()
            or any(
                not isinstance(row[key], list)
                or any(not isinstance(test, str) or not test.strip() for test in row[key])
                for key in ("test_list", "challenge_test_list")
            )
            or len(row["test_list"]) != 3
            or len(row["challenge_test_list"]) > 32
        ):
            raise ValueError("unsupported MBPP source schema or duplicate ID")
        seen.add(row["task_id"])
    return rows
