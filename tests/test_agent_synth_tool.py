"""Contamination, mixing and the ``tools.agent_sft_data`` subcommands on synthetic fixtures.
Every sandbox verdict comes from a scripted executor; nothing generated is executed."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agent_fixtures import ARITH, write_case, write_registry
from synth_fixtures import Chunks, SynthExecutor, families_json

from scratchcoder.agent import cases, workspace
from scratchcoder.agent.synth import contamination, mix
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
    loaded = cases.load_cases(out)
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


def _build_argv(
    tmp_path: Path, out: Path, instances: Path, receipt: Path, *extra: str
) -> list[str]:
    tokenizer = tmp_path / "tokenizer.json"
    return [
        "validate-and-build",
        "--instances",
        str(instances),
        "--independent",
        str(tmp_path / "independent"),
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
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, "--max-calls", "10", "--controls")
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
    (tmp_path / "independent").mkdir()
    (tmp_path / "independent" / "stock1.py").write_text(
        "def inputs(params, rng):\n    return [({}, 'a', 1)]\n\n\n"
        "def oracle(params, levels, item, amount):\n"
        "    return min(levels.get(item, 0) + amount, params['cap'])\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    executor = CaseExecutor(instances)
    out = tmp_path / "build"
    argv = _build_argv(tmp_path, out, instances, receipt, "--controls", "--per-template", "2")
    _receipt(receipt, argv)
    assert agent_sft_data.main(argv, execute=executor, loader=_loader) == 0
    record = json.loads((out / "record.json").read_text())
    assert record["totals"] == {"accepted": 2, "rows": 6}
    assert record["planned_calls"] == 2 * 6 + 4 == executor.calls
    assert record["controls"] == {"rows": 4, "all_failed": True}
    built = list(read_rows(out / "rows.jsonl"))
    meta = [json.loads(line) for line in (out / "rows-meta.jsonl").read_text().splitlines()]
    assert [row.id for row in built] == [row["id"] for row in meta]
    assert {row.source for row in built} == {"agent-sft1-edit", "agent-sft1-repair"}
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == executor.calls
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
    assert document["ledger"]["total"] == executor.calls
    assert len(document["templates"]) >= 40
