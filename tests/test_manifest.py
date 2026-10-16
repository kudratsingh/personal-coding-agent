import hashlib
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data import manifest
from scratchcoder.data.manifest import ManifestEntry, download


def fixture_entry() -> ManifestEntry:
    return ManifestEntry(
        source="synthetic-test",
        revision="a" * 40,
        url="https://example.invalid/fixture",
        license="MIT",
        attribution="Synthetic test fixture, not an external dataset",
        sha256=hashlib.sha256(b"fixture").hexdigest(),
        size_bytes=7,
        license_verified=True,
    )


def test_repeat_downloads_match(tmp_path: Path) -> None:
    first = download(fixture_entry(), tmp_path / "first", chunks=[b"fix", b"ture"])
    second = download(fixture_entry(), tmp_path / "second", chunks=[b"fixture"])
    assert first.read_bytes() == second.read_bytes() == b"fixture"


@pytest.mark.parametrize("bad", [b"corrupt", b"oversized-data", b"short"])
def test_bad_downloads_do_not_replace_previous_data(tmp_path: Path, bad: bytes) -> None:
    target = tmp_path / "data"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError):
        download(fixture_entry(), target, chunks=[bad])
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    "entry",
    [
        replace(fixture_entry(), license_verified=False),
        replace(fixture_entry(), license="CC-BY-NC-4.0"),
        replace(fixture_entry(), terms_required=True),
        replace(fixture_entry(), revision="main"),
        replace(fixture_entry(), attribution=""),
    ],
)
def test_unapproved_entries_fail_before_network(entry: ManifestEntry, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        download(entry, tmp_path / "data")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [("license_verified", "false"), ("size_bytes", 7.0)])
def test_malformed_review_fields_do_not_publish(field: str, value: Any, tmp_path: Path) -> None:
    entry = replace(fixture_entry(), **{field: value})
    with pytest.raises(ValueError):
        download(entry, tmp_path / "data", chunks=[b"fixture"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", None),
        ("source", "  "),
        ("source", 1),
        ("revision", None),
        ("revision", 1),
        ("url", None),
        ("url", 1),
        ("url", "https://"),
        ("url", "https:///missing-host"),
        ("url", "http://example.invalid/path"),
        ("url", "https://user:password@example.invalid/path"),
        ("url", "https://example.invalid:invalid/path"),
        ("url", "https://example.invalid:99999/path"),
        ("url", "https://example.invalid/has space"),
        ("url", "https://[invalid/path"),
        ("license", None),
        ("license", "  "),
        ("attribution", False),
        ("attribution", "\n"),
        ("sha256", None),
        ("sha256", 123),
        ("purpose", []),
        ("purpose", "unknown"),
        ("license_verified", "true"),
        ("license_verified", 1),
        ("terms_required", "false"),
        ("terms_required", 0),
        ("size_bytes", True),
        ("size_bytes", False),
        ("size_bytes", -1),
        ("size_bytes", "7"),
        ("size_bytes", float("nan")),
        ("terms_accepted_on", True),
        ("terms_accepted_on", "20261016"),
        ("terms_accepted_on", "2026-02-30"),
        ("terms_accepted_on", (date.today() + timedelta(days=1)).isoformat()),
    ],
)
def test_invalid_fields_fail_before_io(
    field: str, value: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = replace(fixture_entry(), **{field: value})

    def forbidden_network(*args: Any) -> Any:
        raise AssertionError("invalid manifest opened the network")

    monkeypatch.setattr(manifest, "_network_chunks", forbidden_network)
    with pytest.raises(ValueError):
        download(entry, tmp_path / "new-directory/data")
    assert not list(tmp_path.iterdir())


def test_terms_date_validation_and_evaluation_policy_unchanged() -> None:
    accepted = replace(
        fixture_entry(), terms_required=True, terms_accepted_on=date.today().isoformat()
    )
    accepted.validate()
    replace(accepted, terms_accepted_on="2020-01-01").validate()
    # The evaluator may record a non-training license; no allowlist expansion.
    replace(fixture_entry(), purpose="eval", license="synthetic-evaluation-only").validate()
    with pytest.raises(ValueError, match="allowlisted"):
        replace(fixture_entry(), license="synthetic-evaluation-only").validate()
