"""Stamp progress for the in-Pod idle self-destruct (standard library only).

``python3 tools/pod_progress.py --file <path> --source <label>`` atomically writes
``{"schema_version": 1, "progress_unix": <now>, "source": <label>}``. The post-training
bootstrap stamps its setup progress file after each step, and the runbook stamps it once
more over SSH before the worker starts, so the timer's idle clock (``pod_self_destruct.py
--idle-seconds --activity <path>``) runs only while nothing happens. It runs before the
project is installed, so it imports nothing outside the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence

SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,63}\Z")


def stamp(path: str, source: str, *, clock: Callable[[], float] = time.time) -> dict[str, object]:
    """Write one progress stamp atomically and return it."""
    if not SOURCE.fullmatch(source):
        raise ValueError("the source label must be a short plain label")
    value: dict[str, object] = {"schema_version": 1, "progress_unix": clock(), "source": source}
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".progress.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="the progress file the timer watches")
    parser.add_argument("--source", required=True, help="what made progress, e.g. bootstrap:fetch")
    args = parser.parse_args(argv)
    try:
        value = stamp(args.file, args.source)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
