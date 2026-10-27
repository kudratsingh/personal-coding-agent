"""The Mac execution-window checks, against a fake ``docker``: no container runs here."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scratchcoder.sandbox.runner import IMAGE
from tools import mac_window_check
from tools.mac_window_check import check, clean_environment, dotenv_names

IMAGE_ID = "sha256:" + "d" * 64


class FakeDocker:
    """Answers ``image inspect`` and the mount probe; records every call."""

    def __init__(self, *, image_id: str = IMAGE_ID, mount_succeeds: bool = False) -> None:
        self.image_id, self.mount_succeeds = image_id, mount_succeeds
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[3:5] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, self.image_id + "\n", "")
        if argv[3] == "run":
            if self.mount_succeeds:
                return subprocess.CompletedProcess(argv, 0, "mounted\n", "")
            return subprocess.CompletedProcess(
                argv, 125, "", "Mounts denied: The path /Users is not shared from the host"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_a_clean_window_passes_and_leaves_no_container() -> None:
    docker = FakeDocker()
    record = check(image_id=IMAGE_ID, environ={"PATH": "/usr/bin", "HOME": "/x"}, run=docker)
    assert record["status"] == "passed"
    run = next(call for call in docker.calls if call[3] == "run")
    assert ["-v", "/Users:/x:ro"] == run[run.index("-v") : run.index("-v") + 2]
    assert "--pull=never" in run and "--network=none" in run and IMAGE in run
    name = run[run.index("--name") + 1]
    assert name.startswith("scratchcoder-window-check-")
    assert docker.calls[-1][-3:] == ["rm", "-f", name]  # Removed whatever happened.


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
    )
    assert status == 3 and not marker.exists()
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "failed"
    with pytest.raises(SystemExit):
        mac_window_check.main(["check", "--image-id", "latest"], run=FakeDocker(), environ={})
