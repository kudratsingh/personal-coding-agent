"""The Mac execution-window checks, against a fake ``docker``: no container runs here."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scratchcoder.sandbox.runner import IMAGE
from tools import mac_window_check
from tools.mac_window_check import (
    RESIDUAL_SHARES,
    SHARES,
    check,
    clean_environment,
    dotenv_names,
)

IMAGE_ID = "sha256:" + "d" * 64


class FakeDocker:
    """Answers ``image inspect`` and the mount probes; records every call."""

    def __init__(
        self,
        *,
        image_id: str = IMAGE_ID,
        mount_succeeds: bool = False,
        shared: tuple[str, ...] = (),
    ) -> None:
        self.image_id = image_id
        self.shared = SHARES if mount_succeeds else shared
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[3:5] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, self.image_id + "\n", "")
        if argv[3] == "run":
            share = argv[argv.index("-v") + 1].split(":")[0]
            if share in self.shared:
                return subprocess.CompletedProcess(argv, 0, "mounted\n", "")
            return subprocess.CompletedProcess(
                argv, 125, "", f"Mounts denied: The path {share} is not shared from the host"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")


class FakeAgent:
    """``ssh-add -l`` against each socket: one answer per socket, or an empty agent."""

    def __init__(self, answers: dict[str, tuple[int, str]] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[str] = []

    def __call__(self, socket: str) -> subprocess.CompletedProcess[str]:
        self.asked.append(socket)
        code, text = self.answers.get(socket, (1, "The agent has no identities.\n"))
        if code < 0:
            raise FileNotFoundError("ssh-add")
        return subprocess.CompletedProcess(["ssh-add", "-l"], code, text, "")


def test_a_clean_window_passes_and_leaves_no_container() -> None:
    docker = FakeDocker()
    record = check(image_id=IMAGE_ID, environ={"PATH": "/usr/bin", "HOME": "/x"}, run=docker)
    assert record["status"] == "passed"
    runs = [call for call in docker.calls if call[3] == "run"]
    mounts = [run[run.index("-v") + 1] for run in runs]
    probed = SHARES + RESIDUAL_SHARES
    assert mounts == [f"{share}:/x:ro" for share in probed]  # Every default share is probed.
    for run in runs:
        assert "--pull=never" in run and "--network=none" in run and IMAGE in run
        name = run[run.index("--name") + 1]
        assert name.startswith("scratchcoder-window-check-")
        following = docker.calls[docker.calls.index(run) + 1]
        assert following[-3:] == ["rm", "-f", name]  # Removed whatever happened.


def test_a_single_default_share_left_in_place_fails_the_window() -> None:
    """Removing only /Users is not enough: /Volumes, /private and /tmp count."""
    for share in SHARES:
        record = check(image_id=IMAGE_ID, environ={}, run=FakeDocker(shared=(share,)))
        assert record["status"] == "failed"
        probes = {
            row["share"]: row["refused"] for row in record["checks"]["file_sharing"]["shares"]
        }
        assert [name for name, refused in probes.items() if not refused] == [share]


def test_the_built_in_share_is_recorded_as_a_residual_without_failing_the_window() -> None:
    """Docker Desktop cannot unshare /var/folders: it mounts on a Mac with file sharing
    emptied, so the window records it and passes; any removable share still fails it."""
    record = check(image_id=IMAGE_ID, environ={}, run=FakeDocker(shared=RESIDUAL_SHARES))
    assert record["status"] == "passed"
    rows = {row["share"]: row for row in record["checks"]["file_sharing"]["shares"]}
    assert rows["/var/folders"]["refused"] is False and rows["/var/folders"]["residual"] is True
    assert all(rows[share]["refused"] and not rows[share]["residual"] for share in SHARES)
    both = check(image_id=IMAGE_ID, environ={}, run=FakeDocker(shared=("/tmp", "/var/folders")))
    assert both["status"] == "failed" and both["checks"]["file_sharing"]["passed"] is False


@pytest.mark.parametrize(
    "answers, passed",
    [
        ({"/agent": (1, "The agent has no identities.\n")}, True),
        ({"/agent": (2, "Could not open a connection to your authentication agent.\n")}, True),
        ({"/agent": (0, "256 SHA256:abc owner@mac (ED25519)\n")}, False),
        ({"/agent": (1, "error fetching identities: communication with agent failed\n")}, False),
        ({"/agent": (-1, "")}, False),  # ssh-add missing: unverifiable, so the window fails.
    ],
)
def test_the_agent_check_asks_every_socket_and_fails_closed(
    answers: dict[str, tuple[int, str]], passed: bool
) -> None:
    agent = FakeAgent(answers)
    record = check(
        image_id=IMAGE_ID,
        environ={},
        run=FakeDocker(),
        sockets=["/agent", "/launchd/Listeners"],
        run_agent=agent,
    )
    assert agent.asked == ["/agent", "/launchd/Listeners"]
    assert record["checks"]["ssh_agent"]["passed"] is passed
    assert record["status"] == ("passed" if passed else "failed")
    assert "SHA256:abc" not in json.dumps(record)  # Exit codes only, never a key.


def test_exec_asks_the_parent_and_session_agents_not_the_cleaned_child(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The executor's environment has no agent, but Docker Desktop forwards the session's."""
    agent = FakeAgent({"/session/Listeners": (0, "256 SHA256:k key (ED25519)\n")})
    status = mac_window_check.main(
        ["exec", "--image-id", IMAGE_ID, "--", "true"],
        run=FakeDocker(),
        environ={"PATH": "/usr/bin:/bin", "SSH_AUTH_SOCK": "/shell/agent"},
        run_agent=agent,
        session_socket=lambda: "/session/Listeners",
    )
    assert status == 3 and agent.asked == ["/shell/agent", "/session/Listeners"]
    assert "a check failed" in capfd.readouterr().err


@pytest.mark.parametrize(
    "docker, environ, failed",
    [
        (FakeDocker(mount_succeeds=True), {}, "file_sharing"),
        (FakeDocker(image_id="sha256:" + "e" * 64), {}, "image"),
        (FakeDocker(), {"R2_POD_SECRET_ACCESS_KEY": "s"}, "environment"),
        (FakeDocker(), {"SSH_AUTH_SOCK": "/tmp/agent"}, "environment"),
        (FakeDocker(), {"RUNPOD_API_KEY": "k"}, "environment"),
    ],
)
def test_each_check_fails_on_its_own(
    docker: FakeDocker, environ: dict[str, str], failed: str
) -> None:
    record = check(image_id=IMAGE_ID, environ=environ, run=docker)
    assert record["status"] == "failed"
    assert [name for name, item in record["checks"].items() if not item["passed"]] == [failed]
    assert "s" not in json.dumps(record["checks"]["environment"]["set"]) or failed != "environment"


def test_dotenv_names_are_read_and_withheld_without_values(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("export R2_BUCKET=bucket\nMY_OWN_TOKEN=abc\n# OLD=1\nnot a line\n")
    names = dotenv_names(dotenv)
    assert names == ["MY_OWN_TOKEN", "R2_BUCKET"]
    cleaned = clean_environment(
        {"MY_OWN_TOKEN": "abc", "PATH": "/bin", "GH_TOKEN": "g", "SSH_AGENT_PID": "1"}, names
    )
    assert cleaned == {"PATH": "/bin"}


def test_exec_runs_the_executor_without_any_credential(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("MY_OWN_TOKEN=abc\n")
    environ = {"PATH": "/usr/bin:/bin", "MY_OWN_TOKEN": "abc", "R2_SECRET_ACCESS_KEY": "s"}
    script = "import json, os; print('CHILD', json.dumps(sorted(os.environ)))"
    status = mac_window_check.main(
        [
            "exec",
            "--image-id",
            IMAGE_ID,
            "--dotenv",
            str(dotenv),
            "--",
            sys.executable,
            "-c",
            script,
        ],
        run=FakeDocker(),
        environ=environ,
        run_agent=FakeAgent(),
        session_socket=lambda: "/session/Listeners",
    )
    assert status == 0
    child = next(line for line in capfd.readouterr().out.splitlines() if line.startswith("CHILD"))
    names = json.loads(child.split(" ", 1)[1])
    assert "MY_OWN_TOKEN" not in names and "R2_SECRET_ACCESS_KEY" not in names


def test_exec_runs_nothing_when_a_check_fails(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    marker = tmp_path / "ran"
    status = mac_window_check.main(
        [
            "exec",
            "--image-id",
            IMAGE_ID,
            "--receipt",
            str(tmp_path / "r.json"),
            "--",
            "touch",
            str(marker),
        ],
        run=FakeDocker(mount_succeeds=True),
        environ={"PATH": "/usr/bin:/bin"},
        run_agent=FakeAgent(),
        session_socket=lambda: None,
    )
    assert status == 3 and not marker.exists()
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "failed"
    with pytest.raises(SystemExit):
        mac_window_check.main(
            ["check", "--image-id", "latest"],
            run=FakeDocker(),
            environ={},
            run_agent=FakeAgent(),
            session_socket=lambda: None,
        )
