"""The Mac-side benchmark dispatch tool with mocked gh and R2 (nothing leaves the machine)."""

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from eval_fixtures import (
    benchmark_tasks,
    pool_task,
    save_task_file,
    write_benchmark_completions,
    write_fixture_completions,
)

from scratchcoder.eval import protocols
from scratchcoder.eval.benchmarks import results_filename
from tools import benchmark_dispatch
from tools.benchmark_dispatch import dispatch, gh_environment
from tools.r2_store import Credentials, presigned_get, presigned_url

CREDENTIALS = Credentials(
    endpoint="https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
    access_key_id="AKIDPRESIGN",
    secret_access_key="never-in-a-url",
    bucket="scratch-coder",
)
NOW = dt.datetime(2026, 10, 27, 12, 0, tzinfo=dt.UTC)


def test_presigned_urls_match_the_aws_reference_vector() -> None:
    # AWS's documented Signature Version 4 query-auth example.
    url = presigned_url(
        host="examplebucket.s3.amazonaws.com",
        path="/test.txt",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        now=dt.datetime(2013, 5, 24, tzinfo=dt.UTC),
        expires_seconds=86400,
        region="us-east-1",
    )
    assert url.endswith(
        "&X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
    )


def test_an_r2_link_names_one_object_and_carries_no_secret() -> None:
    url = presigned_get(CREDENTIALS, "post/base/he greedy.jsonl", expires_seconds=3600, now=NOW)
    parts = urlsplit(url)
    assert parts.scheme == "https" and parts.hostname == urlsplit(CREDENTIALS.endpoint).hostname
    assert parts.path == "/scratch-coder/post/base/he%20greedy.jsonl"
    query = parse_qs(parts.query)
    assert query["X-Amz-Expires"] == ["3600"] and query["X-Amz-SignedHeaders"] == ["host"]
    assert query["X-Amz-Credential"][0].startswith("AKIDPRESIGN/20261027/auto/s3/")
    assert "never-in-a-url" not in url
    for key in ("", "post/", "a?b", "a#b"):
        with pytest.raises(ValueError):
            presigned_get(CREDENTIALS, key, expires_seconds=60, now=NOW)
    with pytest.raises(ValueError):
        presigned_get(CREDENTIALS, "k", expires_seconds=0, now=NOW)


class FakeStore:
    def __init__(self, headers: dict[str, str] | None) -> None:
        self.headers = headers
        self.keys: list[str] = []

    def head(self, key: str) -> dict[str, str] | None:
        self.keys.append(key)
        return self.headers


class FakeGitHub:
    """Answers the gh commands the tool runs; the run appears after one empty listing."""

    def __init__(self, *, conclusion: str = "success", report: dict[str, Any] | None = None):
        self.commands: list[list[str]] = []
        self.environments: list[Mapping[str, str]] = []
        self.inputs: dict[str, str] = {}
        self.listings = 0
        self.views = 0
        self.conclusion = conclusion
        self.report = report

    def __call__(
        self, arguments: Sequence[str], environment: Mapping[str, str]
    ) -> subprocess.CompletedProcess[str]:
        command = list(arguments)
        self.commands.append(command)
        self.environments.append(environment)
        output = ""
        if command[:3] == ["gh", "workflow", "run"]:
            fields = [command[index + 1] for index, item in enumerate(command) if item == "-f"]
            self.inputs = dict(field.split("=", 1) for field in fields)
        elif command[:3] == ["gh", "run", "list"]:
            self.listings += 1
            title = f"Benchmark {self.inputs['suite']} {self.inputs['request_id']}"
            runs = [{"databaseId": 7, "displayTitle": "Benchmark humaneval other"}]
            if self.listings > 1:
                runs.append({"databaseId": 42, "displayTitle": title})
            output = json.dumps(runs)
        elif command[:3] == ["gh", "run", "view"]:
            self.views += 1
            done = self.views > 2
            output = json.dumps(
                {"status": "completed" if done else "in_progress", "conclusion": self.conclusion}
            )
        elif command[:3] == ["gh", "run", "download"]:
            directory = Path(command[command.index("--dir") + 1])
            digest = self.inputs["completions_sha256"]
            report = self.report or {
                "kind": "benchmark-results",
                "status": "completed",
                "suite": self.inputs["suite"],
                "completions_sha256": digest,
                "request_id": self.inputs["request_id"],
                "summary": {"passed_attempts": 1},
            }
            (directory / results_filename(self.inputs["suite"], digest)).write_text(
                json.dumps(report)
            )
        return subprocess.CompletedProcess(command, 0, output, "")


def benchmark_file(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, int]:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, _ = write_benchmark_completions(
        root, tasks, {task.id: "\n    return x\n" for task in tasks}
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def run_dispatch(
    root: Path, path: Path, store: FakeStore, github: FakeGitHub, **changes: Any
) -> Any:
    options: dict[str, Any] = {
        "completions_path": path,
        "suite": "humaneval",
        "key": "post/base/humaneval.jsonl",
        "repo": "owner/project",
        "output": root / "artifact",
        "credentials": CREDENTIALS,
        "store": store,
        "runner": github,
        "environ": {
            "PATH": "/usr/bin",
            "GH_TOKEN": "gh-auth",
            "R2_SECRET_ACCESS_KEY": "x",
            "R2_ACCESS_KEY_ID": "y",
            "RUNPOD_API_KEY": "z",
            "MY_SECRET": "w",
        },
        "now": lambda: NOW,
        "sleep": lambda _: None,
        "clock": iter(range(10_000)).__next__,
        "poll_seconds": 1.0,
    }
    options.update(changes)
    return dispatch(**options)


def test_dispatch_signs_one_short_link_waits_and_checks_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, size = benchmark_file(tmp_path, monkeypatch)
    store = FakeStore({"content-length": str(size), "x-amz-meta-sha256": digest})
    github = FakeGitHub()
    summary = run_dispatch(tmp_path, path, store, github)
    assert summary["run_id"] == 42 and summary["completions_sha256"] == digest
    assert summary["conclusion"] == "success" and summary["summary"] == {"passed_attempts": 1}
    assert "completions_url" not in json.dumps(summary) and "X-Amz" not in json.dumps(summary)
    assert store.keys == ["post/base/humaneval.jsonl"]
    url = urlsplit(github.inputs["completions_url"])
    assert url.path == "/scratch-coder/post/base/humaneval.jsonl"
    assert parse_qs(url.query)["X-Amz-Expires"] == ["3600"]
    assert github.inputs["completions_sha256"] == digest and github.inputs["suite"] == "humaneval"
    assert len(github.inputs["request_id"]) == 32
    dispatched = github.commands[0]
    assert dispatched[:4] == ["gh", "workflow", "run", "benchmark-score.yml"]
    assert ["--repo", "owner/project"] == dispatched[4:6] and ["--ref", "main"] == dispatched[6:8]
    downloaded = github.commands[-1]
    assert downloaded[downloaded.index("--name") + 1] == f"benchmark-results-{digest}"
    for environment in github.environments:
        assert set(environment) == {"PATH", "GH_TOKEN"}


def test_dispatch_refuses_unsafe_or_mismatched_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, size = benchmark_file(tmp_path, monkeypatch)
    good = FakeStore({"content-length": str(size), "x-amz-meta-sha256": digest})
    with pytest.raises(ValueError, match="3600"):
        run_dispatch(tmp_path, path, good, FakeGitHub(), expires_seconds=3601)
    with pytest.raises(ValueError, match="this suite"):
        run_dispatch(tmp_path, path, good, FakeGitHub(), suite="mbpp-plus")
    with pytest.raises(ValueError, match="only HumanEval"):
        run_dispatch(tmp_path, path, good, FakeGitHub(), suite="pool")
    for headers, message in (
        (None, "not in R2"),
        ({"content-length": "1", "x-amz-meta-sha256": digest}, "size"),
        ({"content-length": str(size)}, "SHA-256"),
    ):
        github = FakeGitHub()
        with pytest.raises(ValueError, match=message):
            run_dispatch(tmp_path, path, FakeStore(headers), github)
        assert github.commands == []  # Nothing was dispatched.
    tasks = [pool_task("t/1")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    pool = tmp_path / "pool.jsonl"
    write_fixture_completions(pool, tasks_path, tasks, lambda *_: "x = 1")
    github = FakeGitHub()
    with pytest.raises(ValueError, match="public-benchmark"):
        run_dispatch(tmp_path, pool, good, github)
    assert github.commands == []


def test_dispatch_checks_what_comes_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, digest, size = benchmark_file(tmp_path, monkeypatch)
    store = FakeStore({"content-length": str(size), "x-amz-meta-sha256": digest})
    wrong = {"kind": "benchmark-results", "suite": "humaneval", "completions_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="do not name"):
        run_dispatch(tmp_path, path, store, FakeGitHub(report=wrong), output=tmp_path / "a")
    with pytest.raises(RuntimeError, match="failure"):
        run_dispatch(tmp_path, path, store, FakeGitHub(conclusion="failure"), output=tmp_path / "b")

    class NeverAppears(FakeGitHub):
        def __call__(
            self, arguments: Sequence[str], environment: Mapping[str, str]
        ) -> subprocess.CompletedProcess[str]:
            if list(arguments)[:3] == ["gh", "run", "list"]:
                return subprocess.CompletedProcess(list(arguments), 0, "[]", "")
            return super().__call__(arguments, environment)

    with pytest.raises(TimeoutError):
        run_dispatch(
            tmp_path, path, store, NeverAppears(), output=tmp_path / "c", find_timeout_seconds=5
        )


def test_gh_never_sees_storage_or_cloud_credentials() -> None:
    environment = gh_environment(
        {
            "HOME": "/h",
            "GH_TOKEN": "t",
            "R2_BUCKET": "b",
            "AWS_SECRET_ACCESS_KEY": "s",
            "RUNPOD_API_KEY": "r",
            "R2_POD_SECRET_ACCESS_KEY": "p",
            "SOME_SECRET_VALUE": "v",
        }
    )
    assert environment == {"HOME": "/h", "GH_TOKEN": "t"}
    assert benchmark_dispatch.MAX_EXPIRES_SECONDS == 3600
