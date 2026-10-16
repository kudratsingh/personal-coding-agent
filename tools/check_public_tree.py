"""Reject tracked paths protected by repository .gitignore files; never read contents."""

import subprocess
from pathlib import Path

from tools.publication import ignored_tracked_paths

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        paths = ignored_tracked_paths(ROOT)
    except (OSError, subprocess.CalledProcessError):
        print("Publication check failed: Git inventory unavailable.")
        return 1
    if paths:
        print("Tracked paths match repository ignore rules; review before publication:")
        for path in sorted(paths):
            print(repr(path))
        return 1
    print("No tracked paths match repository ignore rules. Contents were not scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
