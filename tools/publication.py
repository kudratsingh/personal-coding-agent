"""Read-only Git inventories using repository .gitignore rules, not local exclusions."""

import os
import subprocess
from pathlib import Path


def git_paths(root: Path, *options: str) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            f"core.excludesFile={os.devnull}",
            "ls-files",
            "-z",
            "--exclude-per-directory=.gitignore",
            *options,
        ],
        capture_output=True,
        check=True,
    )
    return {os.fsdecode(path) for path in result.stdout.split(b"\0") if path}


def ignored_tracked_paths(root: Path) -> set[str]:
    """Find index entries protected by ignore rules, even if force-added."""
    return git_paths(root, "--cached", "--ignored")


def public_documents(root: Path) -> list[Path]:
    paths = git_paths(root, "--cached", "--others") - ignored_tracked_paths(root)
    return [
        root / path
        for path in sorted(paths)
        if Path(path).suffix == ".md"
        and (len(Path(path).parts) == 1 or Path(path).parts[0] == "docs")
        and not path.startswith("docs/internal/")
        and (root / path).exists()
    ]
