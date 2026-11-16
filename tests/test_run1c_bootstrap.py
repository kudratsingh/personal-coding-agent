"""The Pod bootstrap for run 1c: its timer lifetime, its R2 prefix, its CPU test list, its
launch-commit check (the archive's header, the shipped revision and the extracted repository
against the operator's full commit, before anything is installed) and its runtime check (the
runtime the repository's run 1c specification pins, before the install).

The runtime check imports ``torch``; each test puts a stand-in module first on the path, so
the comparison runs without a GPU.

Run against the rehearsal bootstrap test's stub programs (``test_posttrain_bootstrap``); the
lists of runs 1, 1b and 2 are unchanged. A Mac-side test: it names the patterns the Pod's
list must not need, so it does not run there itself.
"""

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from test_posttrain_bootstrap import arguments, posttrain_setup
from test_rehearsal_bootstrap import (
    BASH,
    NEEDS_DOCKER_OR_NETWORK,
    ROOT,
    calls,
    index,
    revision_for,
    run,
    stop_stub_timers,  # noqa: F401 - the autouse fixture that stops the stub timers.
)

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")
RUN1C_TESTS = (
    "tests/test_run1c_posttrain.py",
    "tests/test_run1c_train_rows.py",
    "tests/test_run1c_integration.py",
    "tests/test_executed_specifications.py",
)
MAC_ONLY = {
    "tests/test_posttrain_runbook.py",
    "tests/test_posttrain_run1b_runbook.py",
    "tests/test_run1c_runbook.py",
    "tests/test_run1c_bootstrap.py",
}


LAUNCH = "7" * 40
SPEC_1C = "configs/rehearsal-a40-s-post-1c.json"
TREE = {
    "README.md": b"a readme\n",
    "tools/tool.py": b"print('tool')\n",
    "docs/a.md": b"doc\n",
    SPEC_1C: (ROOT / SPEC_1C).read_bytes(),  # The runtime check reads runtime.expected here.
}
A40 = {
    "python": "3.12.3",
    "torch": "2.9.1+cu128",
    "torch_cuda": "12.8",
    "cuda_available": True,
    "device_count": 1,
    "device_name": "NVIDIA A40",
}
"""What the stand-ins report by default: run 1c's pinned runtime on one A40 (``python`` is what
``platform.python_version`` returns under the stand-in ``sitecustomize``)."""


@pytest.fixture(autouse=True)
def stand_in_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``torch`` module and a ``sitecustomize`` reporting ``A40`` (``torch.json`` beside them
    holds the values), first on the path of every Python the bootstrap starts."""
    root = tmp_path / "stand-in"
    (root / "torch").mkdir(parents=True)
    (root / "torch.json").write_text(json.dumps(A40))
    (root / "sitecustomize.py").write_text(
        "import json, os, platform\n"
        "_v = json.load(open(os.path.join(os.path.dirname(__file__), 'torch.json')))\n"
        "platform.python_version = lambda: _v['python']\n"
    )
    (root / "torch" / "__init__.py").write_text(
        "import json, os, types\n"
        "_v = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'torch.json')))\n"
        "__version__ = _v['torch']\n"
        "version = types.SimpleNamespace(cuda=_v['torch_cuda'])\n"
        "cuda = types.SimpleNamespace(\n"
        "    is_available=lambda: _v['cuda_available'],\n"
        "    device_count=lambda: _v['device_count'],\n"
        "    get_device_name=lambda index: _v['device_name'],\n"
        ")\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(root))
    return root


def listed_tests(lines: list[str]) -> list[str]:
    return [name for name in lines[index(lines, "-m pytest")].split() if name.startswith("tests/")]


def launch_tree(paths: dict[str, Path], *, header: str | None = LAUNCH) -> Path:
    """An extracted repository and its shipped archive as ``git archive`` writes one: a global
    header whose ``comment`` is the commit, then the files; the revision names ``LAUNCH``."""
    repo = paths["workspace"].parent / "repo"
    buffer = io.BytesIO()
    pax = {} if header is None else {"comment": header}
    with tarfile.open(
        fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax
    ) as tar:
        for name, data in TREE.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_bytes(data)
    paths["archive"].write_bytes(buffer.getvalue())
    paths["revision"].write_bytes(revision_for(paths["archive"], git_commit=LAUNCH))
    return repo


def run_1c(paths: dict[str, Path], repo: Path, **change: str) -> list[str]:
    values = {
        "--run": "1c",
        "--deadline-unix": "1800003420",
        "--lifetime-seconds": "3420",
        "--repo": str(repo),
        "--launch-commit": LAUNCH,
        **change,
    }
    return arguments(paths, **values)


def test_run_1c_arms_its_lifetime_and_runs_its_own_tests(tmp_path: Path) -> None:
    paths = posttrain_setup(tmp_path)
    repo = launch_tree(paths)
    (repo / "tools" / "__pycache__").mkdir()  # The timer's own bytecode is not the tree's.
    (repo / "tools" / "__pycache__" / "tool.cpython-312.pyc").write_bytes(b"cache")
    result = run(paths, run_1c(paths, repo))
    assert result.returncode == 0, result.stderr
    lines = calls(paths)
    timer = lines[index(lines, "pod_self_destruct.py")]
    assert index(lines, "pod_self_destruct.py") == 0
    assert "--max-lifetime-seconds 3420" in timer and "--deadline-unix 1800003420" in timer
    assert "--run 1c" in lines[index(lines, "tools.posttrain_ops round-trip")]
    listed = listed_tests(lines)
    for name in RUN1C_TESTS:
        assert name in listed, name
    # Every post-training test runs there but run 2's and the Mac's own.
    expected = {
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_posttrain_*.py")
        if not path.name.startswith("test_posttrain_run2")
    }
    assert expected - MAC_ONLY <= set(listed) and not MAC_ONLY & set(listed)
    assert len(set(listed)) == len(listed)
    for name in listed:
        path = ROOT / name
        assert path.is_file(), name
        if path.name != "test_posttrain_bootstrap.py":  # It names the patterns it looks for.
            assert not NEEDS_DOCKER_OR_NETWORK.search(path.read_text(encoding="utf-8")), name
    assert "R2 environment loaded for run 1c (names only)" in result.stdout
    assert "run 1c inputs are pinned and verified" in result.stdout
    # The launch check ran after the revision check and before the install, and wrote its record.
    assert f"the repository is exactly its tree ({len(TREE)} files)" in result.stdout
    check = index(lines, f"{paths['workspace']}/launch-check.json")
    assert index(lines, "--source bootstrap:armed") < check < index(lines, "pip install")
    record = json.loads((paths["workspace"] / "launch-check.json").read_text())
    revision = json.loads(paths["revision"].read_text())
    assert record == {
        "kind": "posttrain-launch-check",
        "launch_commit": LAUNCH,
        "revision_commit": LAUNCH,
        "archive_commit": LAUNCH,
        "archive_sha256": revision["archive_sha256"],
        "files": len(TREE),
        "tree_clean": True,
    }
    # The runtime check ran after the environment record and before the install.
    runtime = index(lines, f"{paths['workspace']}/runtime-check.json")
    environment = lines.index(f"- {paths['workspace']}")  # The environment record's step.
    assert check < environment < runtime < index(lines, "pip install")
    assert index(lines, "--source bootstrap:environment") > runtime
    assert "runtime check passed" in result.stdout
    measured = json.loads((paths["workspace"] / "runtime-check.json").read_text())
    expected = json.loads(TREE[SPEC_1C])["runtime"]["expected"]
    assert measured["status"] == "passed" and measured["mismatched"] == []
    assert measured["expected"] == expected
    assert {key: measured["measured"][key] for key in A40} == A40


def runtime_differs(paths: dict[str, Path], root: Path, **change: Any) -> list[str]:
    """Run run 1c's bootstrap with the stand-ins reporting ``change``; it must stop with exit 4
    after the environment record and before the install. Returns the fields recorded as
    differing."""
    (root / "torch.json").write_text(json.dumps({**A40, **change}))
    result = run(paths, run_1c(paths, launch_tree(paths)))
    assert result.returncode == 4, result.stderr
    assert "the Pod's runtime is not the one run 1c's specification pins" in result.stderr
    lines = calls(paths)
    assert f"{paths['workspace']}/launch-check.json" in "\n".join(lines)
    assert not [line for line in lines if "pip install" in line or "round-trip" in line]
    assert "--source bootstrap:environment" not in "\n".join(lines)
    record = json.loads((paths["workspace"] / "runtime-check.json").read_text())
    assert record["status"] == "failed"
    return list(record["mismatched"])


@pytest.mark.parametrize(
    ("change", "fields"),
    [
        ({"python": "3.12.4"}, ["python"]),
        ({"torch": "2.9.0+cu128"}, ["torch"]),
        ({"torch_cuda": "12.6"}, ["torch_cuda"]),
        ({"device_name": "NVIDIA RTX A6000"}, ["device_name"]),
        ({"device_count": 2}, ["device_count"]),
        ({"cuda_available": False}, ["cuda_available", "device_count", "device_name"]),
    ],
)
def test_run_1c_stops_before_the_install_on_another_runtime(
    tmp_path: Path, stand_in_torch: Path, change: dict[str, Any], fields: list[str]
) -> None:
    paths = posttrain_setup(tmp_path)
    assert runtime_differs(paths, stand_in_torch, **change) == fields


def test_run_1c_stops_when_torch_cannot_be_imported(tmp_path: Path, stand_in_torch: Path) -> None:
    paths = posttrain_setup(tmp_path)
    (stand_in_torch / "torch" / "__init__.py").write_text("raise ImportError('no torch')\n")
    fields = runtime_differs(paths, stand_in_torch)
    assert fields == ["cuda_available", "device_count", "device_name", "torch", "torch_cuda"]


@pytest.mark.parametrize("run_name", ["1", "1b", "2"])
def test_only_run_1c_checks_its_runtime(
    tmp_path: Path, stand_in_torch: Path, run_name: str
) -> None:
    paths = posttrain_setup(tmp_path)
    (stand_in_torch / "torch.json").write_text(json.dumps({**A40, "torch": "other"}))
    result = run(paths, arguments(paths, **{"--run": run_name}))
    assert result.returncode == 0, result.stderr
    assert not (paths["workspace"] / "runtime-check.json").exists()


@pytest.mark.parametrize("commit", ["", "7" * 12, "7" * 39, "7" * 41, "G" * 40, "A" * 40])
def test_run_1c_refuses_anything_but_a_full_launch_commit(tmp_path: Path, commit: str) -> None:
    paths = posttrain_setup(tmp_path)
    repo = launch_tree(paths)
    result = run(paths, run_1c(paths, repo, **{"--launch-commit": commit}))
    assert result.returncode == 2
    assert "run 1c needs --launch-commit <40 hex>" in result.stderr
    assert calls(paths) == []  # Refused before the timer: nothing ran.


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("revision", "not the launch commit"),
        ("header", "the archive was written from 8888888888888888888888888888888888888888"),
        ("no-header", "the archive was written from None"),
        ("extra", "1 extra, 0 missing, 0 changed"),
        ("missing", "0 extra, 1 missing, 0 changed"),
        ("changed", "0 extra, 0 missing, 1 changed"),
        ("linked-directory", "pkg2 is a symbolic link, not the tree's directory"),
        ("empty-directory", "emptydir is a directory the launch commit's tree does not have"),
        ("nested-directory", "docs/new is a directory the launch commit's tree does not have"),
        ("foreign-bytecode", "tools/__pycache__/b.cpython-312.pyc is not bytecode of a tracked"),
        ("cache-other-file", "tools/__pycache__/notes.txt is not bytecode of a tracked module"),
        ("cache-directory", "tools/__pycache__/inner is not bytecode of a tracked module"),
        ("linked-cache", "tools/__pycache__ is a symbolic link"),
    ],
)
def test_run_1c_stops_on_a_mismatched_commit_or_a_dirty_tree(
    tmp_path: Path, change: str, message: str
) -> None:
    paths = posttrain_setup(tmp_path)
    header = {"header": "8" * 40, "no-header": None}.get(change, LAUNCH)
    repo = launch_tree(paths, header=header)
    if change == "revision":  # The shipped revision names another commit than the operator.
        paths["revision"].write_bytes(revision_for(paths["archive"], git_commit="9" * 40))
    elif change == "extra":
        (repo / "untracked.py").write_text("x = 1\n")
    elif change == "missing":
        (repo / "docs" / "a.md").unlink()
    elif change == "changed":
        (repo / "tools" / "tool.py").write_text("print('edited')\n")
    elif change == "linked-directory":  # A planted package the walk would not descend into.
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "evil.py").write_text("x = 1\n")
        (repo / "pkg2").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    elif change == "empty-directory":
        (repo / "emptydir").mkdir()
    elif change == "nested-directory":
        (repo / "docs" / "new").mkdir()
    elif change in ("foreign-bytecode", "cache-other-file", "cache-directory"):
        cache = repo / "tools" / "__pycache__"
        cache.mkdir()
        (cache / "tool.cpython-312.pyc").write_bytes(b"cache")  # Allowed: tools/tool.py's.
        if change == "foreign-bytecode":
            (cache / "b.cpython-312.pyc").write_bytes(b"junk")  # No tools/b.py in the tree.
        elif change == "cache-other-file":
            (cache / "notes.txt").write_text("x\n")
        else:
            (cache / "inner").mkdir()
    elif change == "linked-cache":
        (tmp_path / "cache").mkdir()
        (repo / "tools" / "__pycache__").symlink_to(tmp_path / "cache", target_is_directory=True)
    result = run(paths, run_1c(paths, repo))
    assert result.returncode == 6
    assert message in result.stderr
    assert not (paths["workspace"] / "launch-check.json").exists()
    lines = calls(paths)
    assert not [line for line in lines if "pip install" in line or "round-trip" in line]
    assert "--source bootstrap:revision" not in "\n".join(lines)


@pytest.mark.parametrize("run_name", ["1", "1b", "2"])
def test_only_run_1c_takes_a_launch_commit(tmp_path: Path, run_name: str) -> None:
    paths = posttrain_setup(tmp_path)
    result = run(paths, arguments(paths, **{"--run": run_name, "--launch-commit": LAUNCH}))
    assert result.returncode == 2 and "--launch-commit is run 1c's" in result.stderr


@pytest.mark.parametrize(("run_name", "lifetime"), [("1", "17820"), ("1b", "8820"), ("2", "34620")])
def test_the_other_runs_lists_do_not_change(tmp_path: Path, run_name: str, lifetime: str) -> None:
    paths = posttrain_setup(tmp_path)
    deadline = str(1_800_000_000 + int(lifetime))
    change = {"--run": run_name, "--deadline-unix": deadline, "--lifetime-seconds": lifetime}
    result = run(paths, arguments(paths, **change))
    assert result.returncode == 0, result.stderr
    listed = listed_tests(calls(paths))
    assert not [name for name in listed if "run1c" in name]
    assert "tests/test_executed_specifications.py" not in listed
