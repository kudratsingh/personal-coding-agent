"""Make an offline coding-agent training dashboard from existing local run records.

No imports from model training or cloud code, no network I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeGuard

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
METRIC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,95}\Z")
SENSITIVE = re.compile(r"password|secret|api.?key|access.?token|authorization|credential", re.I)
METRIC_PREFIXES = ("validation/",)
METRIC_FIELDS = {
    "loss",
    "learning_rate",
    "gradient_norm",
    "tokens",
    "step_seconds",
    "tokens_per_second",
    "z_loss",
}
MAX_RUNS = 40
MAX_METRICS = 120
MAX_SAMPLES = 2400
MAX_LOG_BYTES = 32 * 1024 * 1024


def _numeric(value: object) -> TypeGuard[int | float]:
    return (type(value) is int or type(value) is float) and math.isfinite(value)


def _valid_metric(key: object) -> bool:
    return (
        isinstance(key, str)
        and METRIC_NAME.fullmatch(key) is not None
        and not SENSITIVE.search(key)
        and (key in METRIC_FIELDS or key.startswith(METRIC_PREFIXES))
    )


def _read_events(path: Path, series: dict[str, list[list[float]]]) -> None:
    """Stream existing rank-0 metrics and validation events without retaining private data."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_LOG_BYTES:
        return
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") not in ("metrics", "validation"):
                continue
            fields = row.get("fields")
            if not isinstance(fields, dict):
                continue
            step, values = fields.get("step"), fields.get("metrics")
            if type(step) is not int or step < 0 or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if not _valid_metric(key) or not _numeric(value):
                    continue
                if key not in series and len(series) >= MAX_METRICS:
                    continue
                points = series.setdefault(key, [])
                points.append([float(step), float(value)])
                if len(points) > MAX_SAMPLES:
                    points[:] = points[::2] + ([points[-1]] if len(points) % 2 == 0 else [])


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Read only regular files in immediate run folders, never follow per-run symlinks."""
    if not root.is_dir():
        return []
    collected: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir(), key=lambda x: x.name, reverse=True):
        if len(collected) >= MAX_RUNS:
            break
        if not folder.is_dir() or folder.is_symlink() or not RUN_NAME.fullmatch(folder.name):
            continue
        record = folder / "record.json"
        if not record.is_file() or record.is_symlink():
            continue
        try:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metric_series: dict[str, list[list[float]]] = {}
        # Rank zero is the source of aggregate training metrics; don't duplicate other ranks.
        rank_zero = folder / "events-rank0.jsonl"
        if rank_zero.is_file() and not rank_zero.is_symlink():
            _read_events(rank_zero, metric_series)
        else:
            _read_events(folder / "events.jsonl", metric_series)
        status = metadata.get("status")
        cost = metadata.get("actual_cost_usd")
        commit = metadata.get("git_commit")
        seed = metadata.get("seed")
        collected.append(
            {
                "id": folder.name,
                "status": status if status in ("running", "completed", "failed") else "unknown",
                "commit": (
                    commit[:8]
                    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)
                    else None
                ),
                "seed": seed if type(seed) is int and seed >= 0 else None,
                "cost_usd": float(cost) if _numeric(cost) and cost >= 0 else None,
                "series": metric_series,
            }
        )
    return collected


TEMPLATE_PATH = Path(__file__).with_name("template.html")


def write_dashboard(runs_root: Path, output: Path) -> int:
    """Write only an offline report. Never overwrite a run record or training artifact."""
    root = runs_root.resolve()
    destination = output.resolve()
    if destination.is_relative_to(root) and (
        not destination.relative_to(root).parts
        or destination.relative_to(root).parts[0] != "observability"
    ):
        raise ValueError("output inside runs/ must be in runs/observability/")
    data = load_runs(root)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=destination.parent, encoding="utf-8", delete=False
    ) as file:
        temp = Path(file.name)
        try:
            file.write(TEMPLATE_PATH.read_text(encoding="utf-8").replace("__PAYLOAD__", payload))
            file.flush()
            os.fsync(file.fileno())
            os.chmod(temp, 0o600)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, destination)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, help="default: <runs-root>/observability/index.html")
    args = parser.parse_args()
    output = args.output or args.runs_root / "observability" / "index.html"
    try:
        count = write_dashboard(args.runs_root, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Read {count} local runs; dashboard: {output}")


if __name__ == "__main__":
    main()
