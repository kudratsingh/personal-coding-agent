"""Offline tests for the private-repository record archive; no network."""

import base64
from pathlib import Path
from typing import Any

import pytest

from tools import private_archive


class FakeGitHub:
    def __init__(self, *, private: bool = True, corrupt: bool = False) -> None:
        self.private, self.corrupt = private, corrupt
        self.files: dict[str, bytes] = {}
        self.calls: list[list[str]] = []

    def __call__(
        self,
        arguments: list[str],
        *,
        payload: dict[str, Any] | None = None,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(arguments)
        if arguments == ["repos/owner/artifacts"]:
            return {"private": self.private, "full_name": "owner/artifacts"}
        endpoint = arguments[-1]
        path = endpoint.split("/contents/", 1)[1]
        if arguments[0] == "--method":
            assert payload is not None
            self.files[path] = base64.b64decode(payload["content"])
            return {"commit": {"sha": "c" * 40}}
        if path not in self.files:
            assert missing_ok
            return {"missing": True}
        data = self.files[path] + (b"!" if self.corrupt else b"")
        return {"sha": "s" * 40, "encoding": "base64", "content": base64.b64encode(data).decode()}


def test_archive_writes_reads_back_and_refuses_public_or_unsafe_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "report.json"
    record.write_text('{"status": "completed"}\n')
    github = FakeGitHub()
    monkeypatch.setattr(private_archive, "_gh", github)
    receipt = private_archive.archive(
        [("builds/s/report.json", record)], "owner/artifacts", message="Save build records"
    )
    assert receipt["files"][0]["size_bytes"] == record.stat().st_size
    assert github.files["builds/s/report.json"] == record.read_bytes()
    with pytest.raises(ValueError, match="unsafe"):
        private_archive.archive([("../x", record)], "owner/artifacts", message="m")
    monkeypatch.setattr(private_archive, "_gh", FakeGitHub(private=False))
    with pytest.raises(ValueError, match="private"):
        private_archive.archive([("a/report.json", record)], "owner/artifacts", message="m")
    monkeypatch.setattr(private_archive, "_gh", FakeGitHub(corrupt=True))
    with pytest.raises(ValueError, match="readback differs"):
        private_archive.archive([("a/report.json", record)], "owner/artifacts", message="m")
