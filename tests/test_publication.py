import subprocess
from pathlib import Path

import pytest

from tools import check_docs, check_public_tree
from tools.publication import ignored_tracked_paths, public_documents


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def repository(root: Path) -> None:
    git(root, "init", "--quiet")
    (root / ".gitignore").write_text(
        "private/\ndocs/internal/\nAGENTS.local.md\n.env*\n!.env.example\n*.pt\n",
        encoding="utf-8",
    )
    git(root, "add", ".gitignore")


def test_docs_excludes_ignored_owner_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository(tmp_path)
    (tmp_path / "docs/internal").mkdir(parents=True)
    (tmp_path / "docs/internal/notes.md").write_text("[private](missing.md)")
    (tmp_path / "AGENTS.local.md").write_text("[private](missing.md)")
    (tmp_path / "README.md").write_text("Public documentation\n")
    # Override the root only; the old checker demonstrably scans both ignored files.
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.main() == 0


def test_force_added_private_paths_and_allowed_example(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / "private").mkdir()
    paths = ["private/plan.md", ".env", "fixture.pt", "private/tab\tname.md"]
    for path in [*paths, ".env.example"]:
        (tmp_path / path).write_text("dummy fixture only")
    git(tmp_path, "add", "--force", "--", *paths, ".env.example")
    assert ignored_tracked_paths(tmp_path) == set(paths)
    assert not public_documents(tmp_path)


def test_public_inventory_includes_unstaged_docs_not_local_excludes(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("root")
    (tmp_path / "docs/new.md").write_text("new")
    git(tmp_path, "add", "README.md")
    (tmp_path / ".git/info/exclude").write_text("README.md\ndocs/new.md\n")
    global_ignore = tmp_path / "global-ignore"
    global_ignore.write_text("README.md\ndocs/new.md\n")
    git(tmp_path, "config", "core.excludesFile", str(global_ignore))
    assert ignored_tracked_paths(tmp_path) == set()
    assert public_documents(tmp_path) == [tmp_path / "README.md", tmp_path / "docs/new.md"]


@pytest.mark.parametrize("target", ["missing.md", "../private/plan.md", "../../outside.md"])
def test_untracked_public_docs_still_validate_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    repository(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/new.md").write_text(f"[bad]({target})")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.main() == 1


def test_symlink_document_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository(tmp_path)
    (tmp_path / "private").mkdir()
    (tmp_path / "private/notes").write_bytes(b"\xff")
    (tmp_path / "README.md").symlink_to("private/notes")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.main() == 1


def test_check_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository(tmp_path)
    (tmp_path / ".env").write_text("dummy, not a credential")
    git(tmp_path, "add", "--force", ".env")
    before = (tmp_path / ".git/index").read_bytes()
    monkeypatch.setattr(check_public_tree, "ROOT", tmp_path)
    assert check_public_tree.main() == 1
    assert (tmp_path / ".git/index").read_bytes() == before
    assert (tmp_path / ".env").exists()
    git(tmp_path, "rm", "--cached", ".env")
    assert check_public_tree.main() == 0


def test_symlink_ancestor_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("fixture")
    git(tmp_path, "add", "docs/guide.md")
    (tmp_path / "docs").rename(tmp_path / "private")
    (tmp_path / "private/guide.md").write_bytes(b"\xff")
    (tmp_path / "docs").symlink_to("private", target_is_directory=True)
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.main() == 1


def test_nested_ignore_negation_and_valid_root_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/.gitignore").write_text("*.md\n!public.md\n")
    (tmp_path / "docs/local.md").write_text("[ignored](missing)")
    (tmp_path / "docs/public.md").write_text("[root](../)\n")
    git(tmp_path, "add", "docs/.gitignore", "docs/public.md")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.main() == 0
    git(tmp_path, "add", "--force", "docs/local.md")
    assert ignored_tracked_paths(tmp_path) == {"docs/local.md"}


def test_checks_fail_outside_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    monkeypatch.setattr(check_public_tree, "ROOT", tmp_path)
    assert check_docs.main() == check_public_tree.main() == 1
