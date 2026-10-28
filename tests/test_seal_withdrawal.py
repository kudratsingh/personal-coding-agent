"""Withdrawing an endpoint's sealed test files: run 1's SFT weights, replaced by run 1b's.

Run 1 sealed four pool-test completions files, two of the base checkpoint and two of the
SFT weights. The SFT weights failed the format gate (a formatting failure), so their files
are withdrawn by an append-only ledger event, and run 1b's SFT weights stand for the SFT
endpoint. A withdrawn file is never sealed again, listed or executed; nothing more of the
withdrawn checkpoint can be sealed; and no read has to have opened before.
"""

import json
from pathlib import Path

import pytest
from test_eval_executor import (
    FakeSandbox,
    four_endpoints,
    options,
    regenerated,
    sealed_fixture,
    unseal_record,
)

from scratchcoder.cli import main
from scratchcoder.eval.completions import Sampling
from scratchcoder.eval.executor import execute_completions
from scratchcoder.eval.seal import SealError, SealLedger, seal_completions, withdraw_endpoint

OLD_SFT, NEW_SFT = "b" * 64, "e" * 64
REASON = "formatting failure: the SFT format gate failed every check"


def run1_ledger(root: Path) -> tuple[Path, dict[str, Path]]:
    """Base and old-SFT files sealed (two of the SFT checkpoint: greedy and sampled)."""
    ledger = root / "ledger.jsonl"
    base, _, _ = sealed_fixture(root, "a" * 64)
    old_greedy, _, _ = sealed_fixture(root, OLD_SFT)
    old_sampled = regenerated(
        root, "old-sampled.jsonl", OLD_SFT, sampling=Sampling(samples=2, temperature=0.8)
    )
    for path in (base, old_greedy, old_sampled):
        seal_completions(path, ledger)
    return ledger, {"base": base, "old_greedy": old_greedy, "old_sampled": old_sampled}


def test_withdrawal_names_every_sealed_file_of_the_checkpoint(tmp_path: Path) -> None:
    ledger, files = run1_ledger(tmp_path)
    event = withdraw_endpoint(ledger, endpoint="sft", checkpoint_sha256=OLD_SFT, reason=REASON)
    sealed = {
        item["completions_sha256"]
        for item in SealLedger(ledger).events()
        if item["event"] == "sealed" and item["checkpoint_sha256"] == OLD_SFT
    }
    assert event["event"] == "withdrawn" and event["endpoint"] == "sft"
    assert set(event["completions_sha256"]) == sealed and len(sealed) == 2
    assert event["reason"] == REASON
    # Appended to the hash chain: the earlier events are untouched and still verify.
    events = SealLedger(ledger).events()
    assert [item["event"] for item in events] == ["sealed", "sealed", "sealed", "withdrawn"]


def test_withdrawn_files_and_their_checkpoint_can_never_be_sealed_again(tmp_path: Path) -> None:
    ledger, files = run1_ledger(tmp_path)
    withdraw_endpoint(ledger, endpoint="sft", checkpoint_sha256=OLD_SFT, reason=REASON)
    with pytest.raises(SealError, match="withdrawn"):
        seal_completions(files["old_greedy"], ledger)
    another = regenerated(tmp_path, "old-again.jsonl", OLD_SFT)
    with pytest.raises(SealError, match="withdrawn"):
        seal_completions(another, ledger)
    with pytest.raises(SealError, match="withdrawn already"):
        withdraw_endpoint(ledger, endpoint="sft", checkpoint_sha256=OLD_SFT, reason=REASON)
    # The replacement checkpoint seals normally, and the base files stay sealed.
    new, _, _ = sealed_fixture(tmp_path, NEW_SFT)
    assert seal_completions(new, ledger)["checkpoint_sha256"] == NEW_SFT
    assert seal_completions(files["base"], ledger)["event"] == "sealed"


def replaced(root: Path) -> tuple[Path, dict[str, object], dict[str, Path], Path]:
    """Four endpoints sealed, the SFT one withdrawn and replaced by ``NEW_SFT``'s file."""
    ledger = root / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(root, ledger)
    withdraw_endpoint(ledger, endpoint="sft", checkpoint_sha256=OLD_SFT, reason=REASON)
    new, _, _ = sealed_fixture(root, NEW_SFT)
    digest = seal_completions(new, ledger)["completions_sha256"]
    endpoints["sft"] = {"checkpoint_sha256": NEW_SFT, "completions_sha256": [digest]}
    files["old-sft"], files["sft"] = files["sft"], new
    return ledger, endpoints, files, tasks_path


def test_the_replacement_endpoint_executes_and_the_withdrawn_file_never_does(
    tmp_path: Path,
) -> None:
    ledger, endpoints, files, tasks_path = replaced(tmp_path)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    with pytest.raises(SealError, match="withdrawn"):
        execute_completions(
            files["old-sft"],
            tasks_path,
            tmp_path / "old.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    assert not (tmp_path / "old.jsonl").exists()
    summary = execute_completions(
        files["sft"],
        tasks_path,
        tmp_path / "sft.jsonl",
        execute=FakeSandbox(),
        **options(ledger, record),
    )
    assert summary["pending"] == 0 and summary["unresolved"] == 0


@pytest.mark.parametrize("listing", ["checkpoint", "file"])
def test_an_unseal_record_naming_withdrawn_material_is_refused(
    tmp_path: Path, listing: str
) -> None:
    ledger, endpoints, files, tasks_path = replaced(tmp_path)
    old_digest = next(
        item["completions_sha256"]
        for item in SealLedger(ledger).events()
        if item["event"] == "sealed" and item["checkpoint_sha256"] == OLD_SFT
    )
    changed = json.loads(json.dumps(endpoints))
    if listing == "checkpoint":
        changed["sft"] = {"checkpoint_sha256": OLD_SFT, "completions_sha256": [old_digest]}
    else:
        changed["sft"]["completions_sha256"].append(old_digest)
    record = unseal_record(tmp_path / "unseal.json", changed)
    with pytest.raises(SealError, match="withdrawn"):
        execute_completions(
            files["base"],
            tasks_path,
            tmp_path / "b.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    assert not any(item["event"] == "opened" for item in SealLedger(ledger).events())


def test_nothing_is_withdrawn_once_a_read_opened(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    execute_completions(
        files["base"],
        tasks_path,
        tmp_path / "b.jsonl",
        execute=FakeSandbox(),
        **options(ledger, record),
    )
    with pytest.raises(SealError, match="already opened"):
        withdraw_endpoint(ledger, endpoint="sft", checkpoint_sha256=OLD_SFT, reason=REASON)


@pytest.mark.parametrize(
    ("endpoint", "checkpoint", "reason", "match"),
    [
        ("judge", OLD_SFT, REASON, "endpoint"),
        ("sft", "B" * 64, REASON, "SHA-256"),
        ("sft", OLD_SFT, "  ", "reason"),
        ("sft", "f" * 64, REASON, "no sealed completions"),
    ],
)
def test_malformed_withdrawals_are_refused(
    tmp_path: Path, endpoint: str, checkpoint: str, reason: str, match: str
) -> None:
    ledger, _ = run1_ledger(tmp_path)
    before = SealLedger(ledger).head()
    with pytest.raises(SealError, match=match):
        withdraw_endpoint(ledger, endpoint=endpoint, checkpoint_sha256=checkpoint, reason=reason)
    assert SealLedger(ledger).head() == before


def test_cli_withdraw(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger, _ = run1_ledger(tmp_path)
    arguments = [
        "eval-withdraw",
        "--ledger",
        str(ledger),
        "--endpoint",
        "sft",
        "--checkpoint-sha256",
        OLD_SFT,
        "--reason",
        "formatting failure",
    ]
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["event"] == "withdrawn" and printed["reason"] == "formatting failure"
    with pytest.raises(SealError):
        main(arguments)
