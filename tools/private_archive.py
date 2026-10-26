"""Archive bounded run records to the owner's private GitHub artifacts repository.

Each file is written through the contents API, read back and compared byte for byte.
The repository must report itself private before anything is sent. Shards, checkpoints
and large listings belong in object storage (``tools.r2_store``), not here.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.shards import file_sha256
from tools.pilot_supervisor import _REPOSITORY, _gh

MAX_FILE_BYTES = 20 * 1024 * 1024
PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*")


def archive(files: Sequence[tuple[str, Path]], repository: str, *, message: str) -> dict[str, Any]:
    """Write (repository path, local file) pairs and verify each by reading it back."""
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("invalid repository name")
    for target, path in files:
        if not PATH.fullmatch(target) or ".." in target.split("/"):
            raise ValueError(f"unsafe repository path: {target}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path}: only bounded regular files may be archived")
    repo = _gh([f"repos/{repository}"])
    if repo.get("private") is not True or repo.get("full_name") != repository:
        raise ValueError("artifact repository must be verified private")
    rows = []
    for target, path in files:
        content = path.read_bytes()
        endpoint = f"repos/{repository}/contents/{target}"
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content).decode(),
        }
        previous = _gh([endpoint], missing_ok=True)
        if previous.get("missing") is not True:
            payload["sha"] = previous["sha"]
        response = _gh(["--method", "PUT", endpoint], payload=payload)
        restored = _gh([endpoint])
        encoded = restored.get("content")
        if not isinstance(encoded, str) or restored.get("encoding") != "base64":
            raise ValueError("GitHub readback omitted file contents")
        if base64.b64decode("".join(encoded.split()), validate=True) != content:
            raise ValueError(f"{target}: private GitHub readback differs")
        rows.append(
            {
                "path": target,
                "local": str(path),
                "sha256": file_sha256(path),
                "size_bytes": len(content),
                "commit": response["commit"]["sha"],
            }
        )
    return {
        "schema_version": 1,
        "status": "verified",
        "created_at": utc_now(),
        "repository": repository,
        "private": True,
        "files": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--prefix", required=True, help="repository directory for the files")
    parser.add_argument("--root", type=Path, required=True, help="paths are kept relative to it")
    parser.add_argument("--message", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    pairs = [
        (f"{args.prefix}/{path.resolve().relative_to(args.root.resolve()).as_posix()}", path)
        for path in args.files
    ]
    receipt = archive(pairs, args.repository, message=args.message)
    atomic_json(args.receipt, receipt)
    print(json.dumps({"files": len(receipt["files"]), "repository": receipt["repository"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
