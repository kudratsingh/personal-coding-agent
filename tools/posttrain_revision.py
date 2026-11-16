"""Write the code archive and ``revision.json`` of a run launched from one pushed commit.

``python -m tools.posttrain_revision write --launch-commit <40 hex> --branch <name> --output
<dir>`` (Mac, free) refuses unless the commit is a full 40-hex SHA, the working tree is clean
(no change and no untracked file), ``HEAD`` is that commit and the remote-tracking branch
(``refs/remotes/<remote>/<branch>``, as the last ``git fetch`` left it) points at it too: the
code that runs is the pushed branch's tip, exactly. It then archives that commit with ``git
archive`` (whose header records the commit it was written from), checks the header, and
writes ``repo.tar.gz`` and ``revision.json`` (``git_commit``, ``clean``, ``archive_sha256``,
the record every post-training Pod and receipt already read) into the new output directory's
files, refusing to replace either.

Run 1c launches this way (``posttrain_approved.RUN1C_LAUNCH_BRANCH``); the Pod bootstrap's
``--launch-commit`` then checks the archive's header and the extracted repository against the
same commit before anything is installed. Runs 1, 1b and 2 keep their runbooks' writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def archive_commit(archive: Path) -> str | None:
    """The commit ``git archive`` recorded in the archive's global header, if any."""
    with tarfile.open(archive, "r:gz") as tar:
        comment = tar.pax_headers.get("comment")
    return comment if isinstance(comment, str) else None


def check_launch_tree(repo: Path, *, commit: str, branch: str, remote: str = "origin") -> None:
    """Refuse unless ``commit`` is a full SHA, the tree is clean, ``HEAD`` is ``commit`` and
    the remote-tracking ``branch`` points at it."""
    if not FULL_COMMIT.fullmatch(commit):
        raise ValueError(
            f"the launch commit must be a full commit, 40 lower-case hex digits: {commit!r}"
        )
    if not BRANCH.fullmatch(branch) or ".." in branch:
        raise ValueError(f"not a plain branch name: {branch!r}")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("the working tree is dirty: commit or stash first; no code was archived")
    head = git(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if head != commit:
        raise ValueError(f"HEAD is {head}, not the launch commit {commit}")
    reference = f"refs/remotes/{remote}/{branch}"
    try:
        pushed = git(repo, "rev-parse", "--verify", "--quiet", reference + "^{commit}").strip()
    except subprocess.CalledProcessError:
        pushed = ""
    if pushed != commit:
        found = pushed or "nothing"
        raise ValueError(
            f"{reference} is {found}, not the launch commit {commit}: push the branch and "
            "fetch, or name its pushed tip"
        )


def write_revision(
    repo: Path, output: Path, *, commit: str, branch: str, remote: str = "origin"
) -> dict[str, Any]:
    """Check the tree (``check_launch_tree``), then write ``output/repo.tar.gz`` from the
    commit and ``output/revision.json``; return the record."""
    check_launch_tree(repo, commit=commit, branch=branch, remote=remote)
    archive = output / "repo.tar.gz"
    record_path = output / "revision.json"
    for path in (archive, record_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} exists; this writer never replaces one")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar.gz",
            "-o",
            str(archive.absolute()),
            commit,
        ],
        check=True,
    )
    if archive_commit(archive) != commit:
        archive.unlink()
        raise ValueError("git archive did not record the launch commit in the archive's header")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {"git_commit": commit, "clean": True, "archive_sha256": digest}
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.posttrain_revision", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="archive the launch commit and write revision.json")
    write.add_argument("--launch-commit", required=True)
    write.add_argument("--branch", required=True)
    write.add_argument("--remote", default="origin")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        record = write_revision(
            args.repo,
            args.output,
            commit=args.launch_commit,
            branch=args.branch,
            remote=args.remote,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"posttrain-revision: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"revision": str(args.output / "revision.json"), **record}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
