import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

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
