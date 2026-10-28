"""Offline coding-agent dashboard tests using synthetic fixtures only."""

import json
import re
from pathlib import Path

import pytest

from observability.dashboard import load_runs, write_dashboard


def _run(root: Path, name: str, events: list[dict], **meta: object) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "record.json").write_text(json.dumps({"status": "completed", **meta}))
    (folder / "events-rank0.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    return folder


def _metric(step: int, **values: object) -> dict:
    return {"event": "metrics", "fields": {"step": step, "metrics": values}}


def test_collects_training_and_domain_validation(tmp_path: Path) -> None:
    folder = _run(
        tmp_path, "runA",
        [
            _metric(1, loss=2.5, gradient_norm=0.9),
            {"event": "validation", "fields": {
                "step": 1, "metrics": {"validation/python/bits_per_byte": 1.2}
            }},
            _metric(2, loss=1.8, tokens_per_second=46000),
        ],
        actual_cost_usd=3.25, git_commit="a" * 40, seed=42,
    )
    (folder / "events-rank1.jsonl").write_text(json.dumps(_metric(3, loss=999.0)))
    records = load_runs(tmp_path)
    assert records[0]["series"]["loss"] == [[1.0, 2.5], [2.0, 1.8]]
    assert records[0]["series"]["validation/python/bits_per_byte"] == [[1.0, 1.2]]
    assert records[0]["cost_usd"] == 3.25
    assert records[0]["seed"] == 42
    assert records[0]["commit"] == "aaaaaaaa"


def test_filters_private_text_secrets_and_nonfinite_values(tmp_path: Path) -> None:
    folder = _run(tmp_path, "r1", [
        _metric(0, loss=1.0, prompt="PRIVATE PROMPT", bad=float("inf")),
        {"event": "training_failed", "fields": {"traceback": "PRIVATE TRACE"}},
        _metric(1, loss=0.5, **{"secret_token": 4.0, "bad<script>": 3.0}),
    ], config={"password": "PRIVATE CONFIG"})
    with (folder / "events-rank0.jsonl").open("a") as f:
        f.write("{incomplete\n")
    output = tmp_path / "observability/index.html"
    assert write_dashboard(tmp_path, output) == 1
    html = output.read_text()
    for private in ("PRIVATE PROMPT", "PRIVATE TRACE", "PRIVATE CONFIG",
                    "secret_token", "bad<script>"):
        assert private not in html
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html)
    assert match is not None
    assert '"loss"' in match.group(1)
    assert "Infinity" not in match.group(1)


def test_preserves_training_files_and_blocks_overwrite(tmp_path: Path) -> None:
    folder = _run(tmp_path, "active_run", [_metric(1, loss=2.0)])
    before = (folder / "record.json").read_bytes()
    with pytest.raises(ValueError, match="runs/observability"):
        write_dashboard(tmp_path, folder / "record.json")
    assert (folder / "record.json").read_bytes() == before


def test_symlinks_are_not_followed(tmp_path: Path) -> None:
    source = tmp_path / "real"
    _run(source, "r1", [_metric(1, loss=0.7)])
    root = tmp_path / "runs"
    root.mkdir()
    (root / "link").symlink_to(source / "r1", target_is_directory=True)
    assert load_runs(root) == []


def test_empty_directory_still_produces_offline_html(tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    assert write_dashboard(tmp_path / "missing", output) == 0
    html = output.read_text()
    assert "Coding Agent Observatory" in html
    assert "https://" not in html
