"""Contamination, mixing and the ``tools.agent_sft_data`` subcommands on synthetic fixtures.
Every sandbox verdict comes from a scripted executor; nothing generated is executed."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import ARITH, write_case, write_registry
from dev_family_fixtures import (
    dev_family_argv,
    pool_task,
    register,
    words,
    write_build,
    write_instance,
    write_pool,
    write_specs,
)
from synth_fixtures import Chunks, SynthExecutor, claim_errors, families_json

from scratchcoder.agent import cases, workspace
from scratchcoder.agent.synth import contamination, independent, instances, mix, mutation
from scratchcoder.agent.synth import ledger as call_ledger
from scratchcoder.data.chat import Rendered
from scratchcoder.data.sft import SFTRow, read_rows
from scratchcoder.sandbox import runner
from tools import agent_sft_data

TOOL = "tools.agent_sft_data"


def _index(
    tmp_path: Path, benchmarks: Sequence[Path] = (), dev: Path | None = None
) -> contamination.Index:
    write_case(tmp_path / "evalcases")
    write_registry(tmp_path / "evalcases")
    return contamination.build_index(
        [tmp_path / "evalcases"],
        families=families_json(tmp_path / "families.json"),
        benchmarks=benchmarks,
        dev=dev,
    )


def _row(identifier: str, *code: str, text: str = "") -> contamination.RowText:
    return contamination.RowText(identifier, (text, *code), tuple(code))


def test_contamination_rejects_each_signal_and_reports_ids_only(tmp_path: Path) -> None:
    index = _index(tmp_path)
    renamed = 'def twice(n):\n    """Other words."""\n    return n + n'  # The reference, renamed.
    rows = [
        _row("clean", "def area(w, h):\n    return w * h"),
        _row("ngram", text=ARITH),
        _row("tree", renamed),
        _row("names", "def double(v):\n    return v * 7\n\n\ndef other(v):\n    return v"),
        _row("blocked", "def blocked_name(v):\n    return v"),
        contamination.RowText(
            "module", ("x",), ("def fine(v):\n    return v",), ("blockedmod.py",)
        ),
        _row("unparsable", "def broken(:"),
    ]
    result = contamination.check(index, rows)
    assert result == {
        "schema": contamination.SCHEMA,
        "checked": 7,
        "rejected": ["blocked", "module", "names", "ngram", "tree"],
    }


def test_contamination_external_ngrams_need_three_word_tokens(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl.gz"
    words = "compute the running median of a stream of integers using two heaps quickly"
    with gzip.open(bench, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"prompt": words, "canonical_solution": "return 1"}) + "\n")
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"prompt": "1 2 3 4 5 6 7 8 9 10 11 12 13", "solution": ""}) + "\n")
    index = _index(tmp_path, benchmarks=[bench], dev=dev)
    result = contamination.check(
        index,
        [
            _row("bench", text="please " + words),
            _row("numbers", text="1 2 3 4 5 6 7 8 9 10 11 12 13"),
        ],
    )
    assert result["rejected"] == ["bench"]


def test_contamination_reads_instances_rows_and_the_decoded_replay_index(tmp_path: Path) -> None:
    decoded = tmp_path / "decoded-index.jsonl"
    decoded.write_text(
        json.dumps({"id": "r1", "user": ["q"], "assistant": ["def double(x):\n    return 2"]})
        + "\n"
        + json.dumps({"id": "r2", "user": ["q"], "assistant": ["def area(w):\n    return w"]})
        + "\n"
    )
    found = contamination.decoded_rows(decoded)
    assert [row.id for row in found] == ["r1", "r2"]
    assert contamination.check(_index(tmp_path), found)["rejected"] == ["r1"]


def _sft(identifier: str, source: str, loss: int) -> SFTRow:
    ids = (1, 2, *range(10, 10 + loss))
    return SFTRow.of(identifier, source, Rendered(ids, (False, False, *([True] * loss))))


def test_mix_matches_loss_tokens_and_keeps_template_spread() -> None:
    agent = [_sft(f"a{t}-{i}", "agent", 10) for t in range(3) for i in range(4)]
    meta = {
        row.id: {
            "template_id": row.id.split("-")[0],
            "instance": row.id,
            "family": "f",
            "kind": "edit",
            "workflow_type": "W1",
            "mutation_kind": None,
            "reply": f"def f{row.id}(): pass",
            "prompt_sha256": row.id,
        }
        for row in agent
    }
    replay = [_sft(f"r{i}", "replay", 30) for i in range(2)]
    mixed, report = mix.mix(agent, meta, replay, ratio=1.0)
    assert report["agent"]["loss_tokens"] == 60 and report["replay"]["loss_tokens"] == 60
    assert report["agent"]["distinct_templates"] == 3
    assert report["achieved_ratio"] == 1.0 and report["row_share_agent"] == 6 / 8
    small = [_sft("a0-0", "agent", 10)]
    _, report = mix.mix(small, meta, replay, ratio=1.0)
    assert report["replay"]["rows"] == 1 and report["agent"]["rows"] == 1
    with pytest.raises(ValueError, match="unique"):
        mix.mix(small, meta, [_sft("a0-0", "replay", 3)])


def test_mix_caps_instances_that_share_one_reply_and_reports_reply_diversity() -> None:
    # Two templates; template a has 6 instances (2 rows each) sharing 2 replies, b has 2
    # instances with distinct replies.
    agent = [
        _sft(f"{t}{i}-{k}", "agent", 5)
        for t, count in (("a", 6), ("b", 2))
        for i in range(count)
        for k in ("edit", "repair")
    ]
    meta = {
        row.id: {
            "template_id": row.id[0],
            "instance": row.id.split("-")[0],
            "family": row.id[0],
            "kind": row.id.split("-")[1],
            "workflow_type": "W1",
            "mutation_kind": None if row.id.endswith("edit") else "off_by_one",
            "reply": f"def g{int(row.id[1]) % 2}(): pass" if row.id[0] == "a" else row.id[:2],
            "prompt_sha256": row.id,
        }
        for row in agent
    }
    replay = [_sft(f"r{i}", "replay", 100) for i in range(3)]
    _, uncapped = mix.mix(agent, meta, replay, ratio=1.0)
    assert uncapped["agent"]["kept_distinct_replies"] == 4
    assert uncapped["agent"]["kept_max_instances_per_reply"] == 3
    assert uncapped["agent"]["kept_max_rows_per_reply"] == 6
    assert uncapped["agent"]["kept_distinct_prompt_reply_pairs"] == 16
    mixed, capped = mix.mix(agent, meta, replay, ratio=1.0, max_per_reply=1)
    assert capped["agent"]["instances"] == 4 and capped["agent"]["rows"] == 8
    assert capped["agent"]["instances_cut_by_reply_cap"] == 4
    assert capped["agent"]["kept_max_instances_per_reply"] == 1
    assert capped["agent"]["reply_cap_instances"] == 1
    assert capped["agent"]["rows_available"] == 16
    kept = {meta[row.id]["instance"] for row in mixed if row.id in meta}
    assert len(kept) == 4 and {"b0", "b1"} <= kept


def _receipt(path: Path, argv: Sequence[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "mac-execution-window-check",
                "status": "passed",
                "checked_at": datetime.now(UTC).isoformat(),
                "command": ["uv", "run", "python", "-m", TOOL, *argv],
                "checks": {
                    "image": {
                        "passed": True,
                        "image": runner.IMAGE,
                        "expected_id": "sha256:" + "3" * 64,
                        "found_id": "sha256:" + "3" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _loader(path: Path) -> tuple[Chunks, Chunks]:
    encoder = Chunks()
    return encoder, encoder


def _generate(tmp_path: Path, templates: Sequence[str]) -> Path:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    out = tmp_path / "instances"
    argv = [
        "generate",
        "--out",
        str(out),
        "--tokenizer",
        str(tokenizer),
        "--families",
        str(families_json(tmp_path / "families.json")),
        "--templates",
        *templates,
    ]
    assert agent_sft_data.main(argv, loader=_loader) == 0
    return out


def test_generate_and_specs_write_registered_instances_and_text_only_specs(tmp_path: Path) -> None:
    out = _generate(tmp_path, ["stock1", "grid3"])
    loaded = cases.load_cases(out, synthetic=True)
    assert [case.id for case in loaded] == [
        "grid3-stub_none-000",
        "grid3-stub_raise-000",
        "stock1-cap_minus-000",
        "stock1-keyerror-000",
        "stock1-max_for_min-000",
    ]
    record = json.loads((out / "generate.json").read_text())
    assert record["counts"]["kept"] == 5 and record["dropped"] == []
    assert all(len(row["mutation_kinds"]) == 2 for row in record["kept"])
    specs_path = tmp_path / "specs" / "specs.json"
    assert agent_sft_data.main(["specs", "--instances", str(out), "--out", str(specs_path)]) == 0
    document = json.loads(specs_path.read_text())
    stock = next(item for item in document["templates"] if item["template_id"] == "stock1")
    assert stock["param_keys"] == ["cap"] and len(stock["instances"]) == 3
    assert set(stock["instances"][0]) == {"instance", "module", "function", "signature", "params"}
    text = specs_path.read_text()
    # Signatures only: no function bodies, no checks and no expected values.
    assert "assert " not in text and "    return" not in text and "\\n    " not in text
    assert (
        agent_sft_data.main(
            ["generate", "--out", str(out), "--tokenizer", "x", "--families", "y"], loader=_loader
        )
        == 2
    )


def test_the_tool_refuses_sealed_and_held_out_paths(tmp_path: Path) -> None:
    sealed = tmp_path / "test.jsonl"
    assert (
        agent_sft_data.main(
            [
                "contamination",
                "--cases",
                str(tmp_path),
                "--families",
                str(tmp_path / "f.json"),
                "--decoded",
                str(sealed),
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
        == 2
    )
    held = tmp_path / "heldout"
    assert (
        agent_sft_data.main(["specs", "--instances", str(held), "--out", str(tmp_path / "s")]) == 2
    )


STOCK1_ORACLE = (
    "def inputs(params, rng):\n    return [({}, 'a', 1)]\n\n\n"
    "def oracle(params, levels, item, amount):\n"
    "    return min(levels.get(item, 0) + amount, params['cap'])\n"
)


def _independent(tmp_path: Path, files: dict[str, str] | None = None) -> str:
    """An independent directory with its SHA256SUMS; returns the SHA-256 of SHA256SUMS."""
    directory = tmp_path / "independent"
    directory.mkdir(exist_ok=True)
    for name, text in (files if files is not None else {"stock1.py": STOCK1_ORACLE}).items():
        (directory / name).write_text(text, encoding="utf-8")
    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(directory.iterdir())
        if path.name != "SHA256SUMS"
    )
    (directory / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return hashlib.sha256(sums.encode("utf-8")).hexdigest()


PRIOR_CALLS = 7


def _seed_ledger(tmp_path: Path) -> Path:
    """The shared ledger, seeded from one earlier run record (as the ledger command does)."""
    path = tmp_path / "ledger.jsonl"
    if not path.exists():
        run = tmp_path / "earlier-validation"
        run.mkdir()
        (run / "controls.json").write_text(json.dumps({"sandbox_calls": PRIOR_CALLS, "rows": []}))
        argv = ["ledger", "--ledger", str(path), "--seed-from", str(run)]
        assert agent_sft_data.main(argv) == 0
    return path


def _build_argv(
    tmp_path: Path, out: Path, instances: Path, receipt: Path, pin: str, *extra: str
) -> list[str]:
    tokenizer = tmp_path / "tokenizer.json"
    _seed_ledger(tmp_path)
    return [
        "validate-and-build",
        "--instances",
        str(instances),
        "--independent",
        str(tmp_path / "independent"),
        "--independent-sha256",
        pin,
        "--tokenizer",
        str(tokenizer),
        "--out",
        str(out),
        "--receipt",
        str(receipt),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        *extra,
    ]


def _without_target(sources: dict[str, str], module: str, name: str) -> tuple[tuple[str, str], ...]:
    """The packed modules with the target function cut out: what identifies an instance's
    repository whatever version of the target it holds."""
    from synth_fixtures import target_text

    found = []
    for key, text in sorted(sources.items()):
        if workspace.is_test_file(key + ".py"):
            continue
        current = target_text(sources, module, name) if key == module else None
        found.append((key, text.replace(current, "") if current else text))
    return tuple(found)


class CaseExecutor:
    """Routes each call to the scripted executor of the instance whose repository it packs
    (compared with the target function cut out)."""

    def __init__(self, instances: Path) -> None:
        self.routes: list[tuple[tuple[tuple[str, str], ...], SynthExecutor]] = []
        for directory in sorted(instances.iterdir()):
            if not (directory / "synth.json").is_file():
                continue
            synth = json.loads((directory / "synth.json").read_text())
            module, name = synth["target"].split("::")
            spec = json.loads((directory / "case.json").read_text())
            sources = {
                path.stem: path.read_text() for path in sorted((directory / "repo").glob("*.py"))
            }
            executor = SynthExecutor(
                module.removesuffix(".py"),
                name,
                synth["reference_target"],
                synth["designed_failing_tests"],
                test_module=spec["public_tests"][0].removesuffix(".py"),
                errors=claim_errors(synth.get("task_claim")),
            )
            key = _without_target(sources, executor.module, name)
            self.routes.append((key, executor))
        assert len({key for key, _ in self.routes}) == len(self.routes)
        self.calls = 0

    def __call__(self, program: str, tests: str, mode: str):  # type: ignore[no-untyped-def]
        from agent_fixtures import packed_sources

        self.calls += 1
        sources = packed_sources(program)
        matched = [
            executor
            for key, executor in self.routes
            if _without_target(sources, executor.module, executor.name) == key
        ]
        assert len(matched) == 1
        return matched[0](program, tests, mode)


def test_validate_and_build_needs_a_receipt_and_respects_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path)
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, pin, "--max-calls", "10", "--controls")
    _receipt(receipt, ["something", "else"])
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 2
    _receipt(receipt, argv)
    # Three instances x 6 calls + 4 controls > 10: refused before any call.
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 2
    assert executor.calls == 0 and not out.exists()


def test_validate_and_build_writes_rows_records_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path, {"stock1.py": STOCK1_ORACLE, "README.md": "notes\n"})
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, pin, "--controls", "--per-template", "2")
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 0
    record = json.loads((out / "record.json").read_text())
    oracle_sha = hashlib.sha256(STOCK1_ORACLE.encode("utf-8")).hexdigest()
    assert record["independent"]["sums_sha256"] == pin
    assert record["independent"]["files"]["stock1.py"] == oracle_sha
    assert record["independent"]["missing_templates"] == []
    for name in ("instances.jsonl", "controls.jsonl", "rows-meta.jsonl"):
        lines = [json.loads(line) for line in (out / name).read_text().splitlines()]
        assert lines and all(line["independent_sums_sha256"] == pin for line in lines), name
        assert all(line["oracle_sha256"] == oracle_sha for line in lines), name
    assert record["totals"] == {"accepted": 2, "rows": 6}
    assert record["planned_calls"] == 2 * 6 + 4 == executor.calls
    assert record["controls"]["rows"] == 4 and record["controls"]["all_failed"] is True
    assert record["controls"]["templates_passed"] == ["stock1"]
    assert record["controls"]["templates_without_controls"] == []
    built = list(read_rows(out / "rows.jsonl"))
    meta = [json.loads(line) for line in (out / "rows-meta.jsonl").read_text().splitlines()]
    assert [row.id for row in built] == [row["id"] for row in meta]
    assert {row.source for row in built} == {"agent-sft1-edit", "agent-sft1-repair"}
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1 + executor.calls  # The prior entry, then one line per call.
    assert record["ledger"]["total"] == PRIOR_CALLS + executor.calls
    assert hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    assert (
        agent_sft_data.main(
            [
                "manifest",
                "--instances",
                str(instances),
                "--build",
                str(out),
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--out",
                str(manifest),
            ]
        )
        == 0
    )
    document = json.loads(manifest.read_text())
    assert document["builds"][0]["rows_by_kind"] == {"edit": 2, "repair": 4}
    assert document["builds"][0]["distinct_prompt_reply_pairs"] == 6
    assert document["builds"][0]["max_rows_per_reply"] <= 6
    assert document["instances"]["distinct_replies"] >= 1
    assert "task-clusters-v1" in document["contamination_scope"]["development_tasks"]
    assert "not applied" not in document["contamination_scope"]["development_tasks"]
    assert document["ledger"]["total"] == PRIOR_CALLS + executor.calls
    assert len(document["templates"]) >= 40


def test_validate_and_build_refuses_an_unpinned_or_changed_independent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path)
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    wrong = _build_argv(tmp_path, out, instances, receipt, "0" * 64)
    _receipt(receipt, wrong)
    assert agent_sft_data.main(wrong, execute=executor, loader=_loader) == 2
    assert "not the pinned" in capsys.readouterr().err
    # The pinned hash, but a file edited after SHA256SUMS was written.
    argv = _build_argv(tmp_path, out, instances, receipt, pin)
    _receipt(receipt, argv)
    (tmp_path / "independent" / "stock1.py").write_text(STOCK1_ORACLE + "# edited\n")
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 2
    assert "does not match its SHA256SUMS" in capsys.readouterr().err
    # A file SHA256SUMS does not list.
    _independent(tmp_path)
    (tmp_path / "independent" / "stock2.py").write_text(STOCK1_ORACLE)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 2
    assert "does not list stock2.py" in capsys.readouterr().err
    assert executor.calls == 0 and not out.exists()


def test_validate_and_build_refuses_an_unseeded_ledger_and_a_raised_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path)
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, pin, "--controls")
    # A mistyped ledger path would start a fresh count at zero: refused.
    typo = [*argv]
    typo[typo.index("--ledger") + 1] = str(tmp_path / "ledger.json")
    _receipt(receipt, typo)
    assert agent_sft_data.main(typo, execute=executor, loader=_loader) == 2
    assert "does not exist" in capsys.readouterr().err
    raised = [*argv, "--max-calls", str(call_ledger.DEFAULT_CAP + 1)]
    _receipt(receipt, raised)
    assert agent_sft_data.main(raised, execute=executor, loader=_loader) == 2
    assert "approved" in capsys.readouterr().err
    assert executor.calls == 0 and not out.exists()


def test_a_template_whose_control_passes_gives_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path)
    receipt = tmp_path / "receipt.json"

    class WeakChecks(CaseExecutor):
        """The constant control (the fourth call: controls run first) passes the grading
        script, as it would against weak checks."""

        def __call__(self, program: str, tests: str, mode: str):  # type: ignore[no-untyped-def]
            result = super().__call__(program, tests, mode)
            if self.calls == 4:
                return type(result)("passed")
            return result

    executor = WeakChecks(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, pin, "--controls")
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 0
    record = json.loads((out / "record.json").read_text())
    assert record["controls"]["all_failed"] is False
    assert record["controls"]["templates_failed"] == ["stock1"]
    assert record["totals"] == {"controls_failed": 3}
    assert executor.calls == 4  # The controls only; no instance was run.
    assert (out / "rows.jsonl").read_text() == ""
    reasons = [json.loads(line)["reasons"] for line in (out / "instances.jsonl").open()]
    assert reasons == [["controls_not_all_failed"]] * 3
    # Without controls (this run or an earlier one) a template gives no rows either.
    first = out
    out = tmp_path / "build-2"
    argv = _build_argv(tmp_path, out, instances, receipt, pin)
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=CaseExecutor(instances), loader=_loader) == 0
    record = json.loads((out / "record.json").read_text())
    assert record["controls"]["all_failed"] is None
    assert record["controls"]["templates_without_controls"] == ["stock1"]
    assert record["totals"] == {"no_controls": 3}
    # Controls recorded by an earlier build against the same oracle count, and gate.
    out = tmp_path / "build-3"
    argv = _build_argv(
        tmp_path, out, instances, receipt, pin, "--controls-from", str(first / "controls.jsonl")
    )
    _receipt(receipt, argv)
    again = CaseExecutor(instances)
    assert agent_sft_data.main(argv, execute=again, loader=_loader) == 0
    assert json.loads((out / "record.json").read_text())["totals"] == {"controls_failed": 3}
    assert again.calls == 0


def _mix_inputs(tmp_path: Path) -> dict[str, Path]:
    from scratchcoder.data.sft import write_rows

    write_case(tmp_path / "evalcases")
    write_registry(tmp_path / "evalcases")
    agent = [_sft(f"agent-sft1/i{n}/edit", "agent-sft1-edit", 8) for n in range(3)]
    write_rows(tmp_path / "rows.jsonl", agent)
    meta = tmp_path / "rows-meta.jsonl"
    meta.write_text(
        "".join(
            json.dumps(
                {
                    "id": row.id,
                    "kind": "edit",
                    "instance": f"i{n}",
                    "template_id": f"t{n}",
                    "family": "f",
                    "workflow_type": "W1",
                    "mutation_kind": None,
                    "prompt_sha256": row.id,
                    "user_text": words(f"fix{n}area", 8),
                    "reply": f"def area{n}(w, h):\n    return w * h",
                }
            )
            + "\n"
            for n, row in enumerate(agent)
        )
    )
    replay = [_sft(f"r{n}", "replay", 8) for n in range(3)]
    write_rows(tmp_path / "replay.jsonl", replay)
    index = tmp_path / "decoded-index.jsonl"
    index.write_text(
        "".join(
            json.dumps(
                {"id": row.id, "kind": "code", "user": ["q"], "assistant": [f"def v{n}(): pass"]}
            )
            + "\n"
            for n, row in enumerate(replay)
        )
    )
    families = families_json(tmp_path / "families.json")
    results = {}
    for side, flag, source in (("agent", "--rows-meta", meta), ("replay", "--decoded", index)):
        out = tmp_path / f"contamination-{side}.json"
        argv = [
            "contamination",
            "--cases",
            str(tmp_path / "evalcases"),
            "--families",
            str(families),
            flag,
            str(source),
            "--out",
            str(out),
        ]
        assert agent_sft_data.main(argv) == 0
        results[side] = out
    # The dev-family result: instance i2 shares its target program with a dev task.
    root = tmp_path / "synth-instances"
    for n in range(3):
        write_instance(
            root,
            f"i{n}",
            template=f"t{n}",
            task=words(f"task{n}", 12),
            reference=f"def area{n}(w, h):\n    return w * h + {n}\n",
        )
    register(root)
    dev = write_pool(
        tmp_path / "pool" / "dev.jsonl",
        [pool_task("d0", words("devp", 20), "def other(w, h):\n    return w * h + 2\n", "dev")],
    )
    train = write_pool(
        tmp_path / "pool" / "train.jsonl",
        [pool_task(f"r{n}", words(f"trainp{n}", 20), None, "train") for n in range(3)],
    )
    build = write_build(tmp_path / "synth-build", {f"i{n}": True for n in range(3)})
    specs = write_specs(tmp_path / "specs.json", {f"t{n}": words(f"spec{n}", 12) for n in range(3)})
    family = tmp_path / "dev-family.json"
    argv = dev_family_argv(dev, train, [(root, build)], [specs], meta, index, family)
    assert agent_sft_data.main(argv) == 0
    return {"meta": meta, "index": index, "family": family, **results}


def _mix_argv(tmp_path: Path, paths: dict[str, Path], out: str, *registries: str) -> list[str]:
    return [
        "mix",
        "--agent-rows",
        str(tmp_path / "rows.jsonl"),
        "--agent-meta",
        str(paths["meta"]),
        "--replay",
        str(tmp_path / "replay.jsonl"),
        "--replay-index",
        str(paths["index"]),
        "--agent-rejected",
        str(paths["agent"]),
        "--replay-rejected",
        str(paths["replay"]),
        "--dev-family",
        str(paths["family"]),
        "--case-registry-sha256",
        *registries,
        "--out",
        str(tmp_path / out),
    ]


def test_mix_needs_both_contamination_results_for_these_inputs_and_case_sets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _mix_inputs(tmp_path)
    result = json.loads(paths["agent"].read_text())
    assert result["input_kind"] == "rows_meta"
    assert result["input_sha256"] == hashlib.sha256(paths["meta"].read_bytes()).hexdigest()
    registry = cases.registry_sha256(tmp_path / "evalcases")
    assert contamination.registries(result) == {registry}
    assert agent_sft_data.main(_mix_argv(tmp_path, paths, "mix-ok", registry)) == 0
    record = json.loads((tmp_path / "mix-ok" / "mix-record.json").read_text())
    assert record["case_registries"] == [registry]
    assert record["contamination"]["agent"]["input_sha256"] == result["input_sha256"]
    # The dev-family result drops every row of i2; the record states the family rule.
    assert record["dev_family"]["rule"] == "task-clusters-v1"
    assert record["dev_family"]["agent_rejected"] == 1 and record["agent"]["rows"] == 2
    assert "task-clusters-v1" in record["dev_method"]
    assert "not applied" not in json.dumps(record)
    mixed = {row.id for row in read_rows(tmp_path / "mix-ok" / "mix.jsonl")}
    assert "agent-sft1/i2/edit" not in mixed
    # A case set the results were not checked against (e.g. the second evaluation set).
    argv = _mix_argv(tmp_path, paths, "mix-more", registry, "e" * 64)
    assert agent_sft_data.main(argv) == 2
    assert "exactly the named case sets" in capsys.readouterr().err
    # The results swapped: each was computed on the other side's input.
    swapped = {**paths, "agent": paths["replay"], "replay": paths["agent"]}
    assert agent_sft_data.main(_mix_argv(tmp_path, swapped, "mix-swap", registry)) == 2
    assert "checked decoded, not rows_meta" in capsys.readouterr().err
    # Rows metadata changed after the check (a later build): the result is stale.
    paths["meta"].write_text(paths["meta"].read_text() + "\n")
    assert agent_sft_data.main(_mix_argv(tmp_path, paths, "mix-stale", registry)) == 2
    assert "not computed on" in capsys.readouterr().err
    # Both results are required.
    argv = _mix_argv(tmp_path, paths, "mix-none", registry)
    del argv[argv.index("--agent-rejected") : argv.index("--agent-rejected") + 2]
    with pytest.raises(SystemExit):
        agent_sft_data.main(argv)
    assert not any((tmp_path / name).exists() for name in ("mix-more", "mix-swap", "mix-stale"))


def test_a_held_out_prefix_and_generated_instances_are_refused_as_evaluation_cases(
    tmp_path: Path,
) -> None:
    for name in ("heldout-r4", "HeldOut2", "heldout"):
        sealed = tmp_path / "runs" / "agent-cases-x" / name
        argv = ["specs", "--instances", str(sealed), "--out", str(tmp_path / "s.json")]
        assert agent_sft_data.main(argv) == 2
        with pytest.raises(cases.CaseError, match="held-out"):
            cases.load_cases(sealed)
    assert not cases.forbidden_part("held") and cases.forbidden_part("heldout-results")
    # Generated training instances load only when asked for; evaluation loading refuses them.
    out = _generate(tmp_path, ["stock1"])
    with pytest.raises(cases.CaseError, match="generated training instance"):
        cases.load_cases(out)
    assert cases.load_cases(out, synthetic=True)
    write_case(tmp_path / "evalcases")
    write_registry(tmp_path / "evalcases")
    with pytest.raises(cases.CaseError, match="not a generated training instance"):
        cases.load_cases(tmp_path / "evalcases", synthetic=True)


def test_generate_caps_instances_per_reply_and_reports_reply_reach(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    families = families_json(tmp_path / "families.json")
    records = {}
    for cap in ("0", "1"):
        out = tmp_path / f"instances-{cap}"
        argv = [
            "generate",
            "--out",
            str(out),
            "--tokenizer",
            str(tokenizer),
            "--families",
            str(families),
            "--templates",
            "dice2",
            "--per-bug",
            "3",
            "--max-per-reply",
            cap,
            "--capacity-draws",
            "30",
        ]
        assert agent_sft_data.main(argv, loader=_loader) == 0
        records[cap] = json.loads((out / "generate.json").read_text())
    uncapped, capped = records["0"]["replies"], records["1"]["replies"]
    reach = capped["per_template"]["dice2"]["capacity"]
    assert reach == capped["capacity_total"] and 1 <= reach < 9
    assert uncapped["per_template"]["dice2"]["instances"] == 9
    assert uncapped["max_instances_per_reply"] > 1
    # With a cap of one, every kept instance has its own reply.
    assert capped["max_instances_per_reply"] == 1
    assert capped["distinct"] == capped["per_template"]["dice2"]["instances"] <= reach
    assert any("reply cap" in row["problems"][0] for row in records["1"]["dropped"])


def test_generate_passes_over_a_kind_whose_only_mutant_an_earlier_kind_took(
    tmp_path: Path,
) -> None:
    # digits1's div_ten instance: after negate_condition (not n == 0, the same tree as
    # n != 0), flip_comparison has no mutant left; the next kind is used instead.
    out = _generate(tmp_path, ["digits1"])
    record = json.loads((out / "generate.json").read_text())
    assert record["dropped"] == []
    for row in record["kept"]:
        assert len(row["mutation_kinds"]) == 2 == len(set(row["mutation_kinds"]))
        assert set(row["mutation_kinds"]) <= set(row["applicable_mutation_kinds"])


def test_generate_with_another_seed_base_gets_new_ids_and_counts_prior_replies(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    families = families_json(tmp_path / "families.json")

    def generate(out: Path, *extra: str) -> int:
        argv = [
            "generate",
            "--out",
            str(out),
            "--tokenizer",
            str(tokenizer),
            "--families",
            str(families),
            "--templates",
            "stock1",
            "--capacity-draws",
            "30",
            *extra,
        ]
        return agent_sft_data.main(argv, loader=_loader)

    first = tmp_path / "first"
    assert generate(first, "--per-bug", "1", "--max-per-reply", "1") == 0
    before = json.loads((first / "generate.json").read_text())
    used = {row["reference_sha256"] for row in before["kept"]}
    # The same seed base again would reuse the prior IDs: refused.
    assert generate(tmp_path / "again", "--prior-instances", str(first)) == 2
    second = tmp_path / "second"
    argv = ["--seed-base", "7", "--per-bug", "3", "--max-per-reply", "1"]
    assert generate(second, *argv, "--prior-instances", str(first)) == 0
    after = json.loads((second / "generate.json").read_text())
    assert all("-s7-" in row["id"] for row in after["kept"])
    # With a cap of one, a reply the prior set used is never used again.
    assert not used & {row["reference_sha256"] for row in after["kept"]}
    assert after["prior_instances"][0]["instances"] == len(before["kept"])
    # A prior set whose registry changed after generation is refused.
    (first / "registry.json").write_text("{}", encoding="utf-8")
    assert generate(tmp_path / "third", *argv, "--prior-instances", str(first)) == 2


def test_instance_ids_name_a_non_zero_seed_base() -> None:
    assert instances.instance_id("dice2", "ascending", 4) == "dice2-ascending-004"
    assert instances.instance_id("dice2", "ascending", 4, 7) == "dice2-ascending-s7-004"


def test_oracle_only_controls_measure_the_oracle_and_never_gate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    instances = _generate(tmp_path, ["stock1"])
    pin = _independent(tmp_path, {"stock1.py": STOCK1_ORACLE, "README.md": "notes\n"})
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "oracle-controls"
    argv = _build_argv(tmp_path, out, instances, receipt, pin, "--oracle-controls-only")
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 0
    # One template: the reference, then its four controls, against the oracle's checks alone.
    assert executor.calls == 5
    found = [json.loads(line) for line in (out / "oracle-controls.jsonl").read_text().splitlines()]
    assert len(found) == 5 and {row["checks"] for row in found} == {"independent"}
    assert found[0]["control"] == "reference" and found[0]["ok"]
    assert not (out / "rows.jsonl").exists() and not (out / "controls.jsonl").exists()
    record = json.loads((out / "record.json").read_text())
    assert record["rows"] == 5 and record["planned_calls"] == 5 and record["controls"] == 4
    assert record["reference_passed"] == ["stock1"] and record["reference_not_passed"] == []
    assert record["rejected_by_oracle_alone"] == sum(row["ok"] for row in found[1:])
    # These rows never stand in for the full-grading controls that gate rows.
    later = tmp_path / "build"
    argv = _build_argv(
        tmp_path,
        later,
        instances,
        receipt,
        pin,
        "--controls-from",
        str(out / "oracle-controls.jsonl"),
    )
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 0
    built = json.loads((later / "record.json").read_text())
    assert built["controls"]["templates_without_controls"] == ["stock1"]
    assert built["totals"].get("accepted", 0) == 0
    assert (later / "rows.jsonl").read_text() == ""


def test_older_control_rows_without_a_checks_key_still_gate(tmp_path: Path) -> None:
    from tools.agent_sft_data import _prior_controls

    oracle = independent.Oracle("t", "", "abc", True)
    rows = [
        {"template_id": "t", "control": kind, "ok": True, "oracle_sha256": "abc"}
        for kind in mutation.CONTROL_KINDS
    ]
    target = tmp_path / "controls.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _prior_controls([target], {"t": oracle}) == {"t": True}
    marked = [{**row, "checks": "independent"} for row in rows]
    target.write_text("".join(json.dumps(row) + "\n" for row in marked))
    assert _prior_controls([target], {"t": oracle}) == {}


def test_mix_requires_the_dev_family_result_for_these_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression (e): mix refuses without the dev-family result, or with one for another
    input, another rule or thresholds, or fewer replay kinds."""
    paths = _mix_inputs(tmp_path)
    registry = cases.registry_sha256(tmp_path / "evalcases")
    argv = _mix_argv(tmp_path, paths, "mix-none", registry)
    del argv[argv.index("--dev-family") : argv.index("--dev-family") + 2]
    with pytest.raises(SystemExit):
        agent_sft_data.main(argv)
    # A contamination result where the dev-family result belongs.
    wrong = {**paths, "family": paths["agent"]}
    assert agent_sft_data.main(_mix_argv(tmp_path, wrong, "mix-wrong", registry)) == 2
    assert "not a dev-family result" in capsys.readouterr().err
    original = json.loads(paths["family"].read_text())

    def variant(name: str, change: dict[str, Any]) -> int:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({**original, **change}))
        return agent_sft_data.main(_mix_argv(tmp_path, {**paths, "family": path}, name, registry))

    other_input = {**original["agent"], "input_sha256": "0" * 64}
    assert variant("mix-other-input", {"agent": other_input}) == 2
    assert "was not computed on" in capsys.readouterr().err
    swapped = {"agent": original["replay"], "replay": original["agent"]}
    assert variant("mix-swapped", swapped) == 2
    assert "another kind of input" in capsys.readouterr().err
    assert variant("mix-rule", {"dev_family": "task-clusters-v0"}) == 2
    assert variant("mix-rules", {"rules": {**original["rules"], "named_flag": 0.2}}) == 2
    assert variant("mix-kinds", {"replay": {**original["replay"], "kinds": []}}) == 2
    assert "did not judge replay kinds" in capsys.readouterr().err
    pool = {**original["pool"], "dev_sha256": "0" * 64}
    agent_result = json.loads(paths["agent"].read_text())
    agent_result["index"]["dev_sha256"] = "1" * 64
    paths["agent"].write_text(json.dumps(agent_result))
    assert variant("mix-dev", {"pool": pool}) == 2
    assert "another development file" in capsys.readouterr().err
    # The rows metadata changed after the check.
    paths["meta"].write_text(paths["meta"].read_text() + "\n")
    assert agent_sft_data.main(_mix_argv(tmp_path, paths, "mix-stale", registry)) == 2
    assert not any(path.name.startswith("mix-") and path.is_dir() for path in tmp_path.iterdir())
