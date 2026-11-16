"""Run 1c's code revision writer (``tools.posttrain_revision``): it archives exactly the
operator's full launch commit, and only when ``HEAD`` is that commit, the tree is clean and
the pushed branch points at it.

Each test builds a throwaway repository with a local bare remote; nothing touches the network
or this checkout. A Mac-side test (the Pod gets the archive, not a repository).
"""

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools import posttrain_approved, posttrain_revision

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required")
BRANCH = posttrain_approved.RUN1C_LAUNCH_BRANCH


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def pushed(tmp_path: Path) -> tuple[Path, str]:
    """A clone whose branch is pushed to its (local, bare) origin; returns it and the tip."""
    remote = tmp_path / "remote.git"
    repo = tmp_path / "clone"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README.md").write_text("fixture\n")
    (repo / "tools").mkdir()
    (repo / "tools" / "tool.py").write_text("print('tool')\n")
    git(repo, "checkout", "-q", "-b", BRANCH)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "fixture")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "origin", BRANCH)
    git(repo, "fetch", "-q", "origin")
    return repo, git(repo, "rev-parse", "HEAD")


def test_the_writer_archives_exactly_the_pushed_launch_commit(
    pushed: tuple[Path, str], tmp_path: Path
) -> None:
    repo, commit = pushed
    output = tmp_path / "control"
    record = posttrain_revision.write_revision(repo, output, commit=commit, branch=BRANCH)
    archive = output / "repo.tar.gz"
    assert record["git_commit"] == commit and record["clean"] is True
    assert json.loads((output / "revision.json").read_text()) == record
    assert posttrain_revision.archive_commit(archive) == commit  # The Pod checks this header.
    with tarfile.open(archive, "r:gz") as tar:
        assert sorted(member.name for member in tar.getmembers() if member.isfile()) == [
            "README.md",
            "tools/tool.py",
        ]
    with pytest.raises(FileExistsError):  # Never replaces a written revision.
        posttrain_revision.write_revision(repo, output, commit=commit, branch=BRANCH)


@pytest.mark.parametrize(
    "short", ["", "abc1234", "a" * 39, "a" * 41, "A" * 40, "g" * 40, "HEAD", BRANCH]
)
def test_the_writer_refuses_anything_but_a_full_commit(
    pushed: tuple[Path, str], tmp_path: Path, short: str
) -> None:
    repo, commit = pushed
    if short == "abc1234":
        short = commit[:7]  # A real abbreviation of the tip is still refused.
    with pytest.raises(ValueError, match="40 lower-case hex digits"):
        posttrain_revision.write_revision(repo, tmp_path / "out", commit=short, branch=BRANCH)
    assert not (tmp_path / "out").exists()


def test_the_writer_refuses_a_head_other_than_the_launch_commit(
    pushed: tuple[Path, str], tmp_path: Path
) -> None:
    repo, commit = pushed
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "later")
    with pytest.raises(ValueError, match=f"not the launch commit {commit}"):
        posttrain_revision.write_revision(repo, tmp_path / "out", commit=commit, branch=BRANCH)


def test_the_writer_refuses_a_commit_the_branch_has_not_pushed(
    pushed: tuple[Path, str], tmp_path: Path
) -> None:
    repo, _ = pushed
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "not pushed")
    local = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=f"refs/remotes/origin/{BRANCH} is .*not the launch"):
        posttrain_revision.write_revision(repo, tmp_path / "out", commit=local, branch=BRANCH)
    with pytest.raises(ValueError, match="refs/remotes/origin/another is nothing"):
        posttrain_revision.write_revision(repo, tmp_path / "out", commit=local, branch="another")


@pytest.mark.parametrize("dirt", ["modified", "untracked", "staged"])
def test_the_writer_refuses_a_dirty_tree(
    pushed: tuple[Path, str], tmp_path: Path, dirt: str
) -> None:
    repo, commit = pushed
    if dirt == "modified":
        (repo / "README.md").write_text("edited\n")
    elif dirt == "untracked":
        (repo / "tools" / "scratch.py").write_text("x = 1\n")
    else:
        (repo / "new.py").write_text("x = 1\n")
        git(repo, "add", "new.py")
    with pytest.raises(ValueError, match="the working tree is dirty"):
        posttrain_revision.write_revision(repo, tmp_path / "out", commit=commit, branch=BRANCH)
    assert not (tmp_path / "out").exists()


def test_the_command_line_writes_or_refuses(
    pushed: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = pushed
    argv = ["write", "--branch", BRANCH, "--repo", str(repo), "--output", str(tmp_path / "c")]
    assert posttrain_revision.main([*argv, "--launch-commit", commit[:12]]) == 1
    assert "40 lower-case hex digits" in capsys.readouterr().err
    assert posttrain_revision.main([*argv, "--launch-commit", commit]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["git_commit"] == commit and printed["clean"] is True


def test_the_command_line_refuses_any_branch_but_run_1cs(
    pushed: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = pushed
    git(repo, "push", "-q", "origin", "HEAD:refs/heads/another")
    git(repo, "fetch", "-q", "origin")
    argv = ["write", "--repo", str(repo), "--output", str(tmp_path / "c")]
    # The tip of another pushed branch, named with that branch, passes every tree check.
    posttrain_revision.check_launch_tree(repo, commit=commit, branch="another")
    assert posttrain_revision.main([*argv, "--branch", "another", "--launch-commit", commit]) == 1
    assert f"launches from {BRANCH} only" in capsys.readouterr().err
    assert not (tmp_path / "c").exists()
