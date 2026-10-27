"""SFT data builder, OpenAssistant adapter and language intake, on authored fixtures."""

import gzip
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.chat import Message, render_code_task, render_conversation
from scratchcoder.data.oasst import (
    Conversation,
    adapt_trees,
    conversation_from_json,
    conversation_json,
    message_reason,
)
from scratchcoder.data.onepass import IGNORE
from scratchcoder.data.sft import SFTLoader, SFTRow, read_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import sft_data
from tools.rehearsal_build import Hub
from tools.sft_data import (
    ACECODE,
    BuildConfig,
    build,
    quotas,
    screen_conversations,
    stratified_holdout,
)

TOKENIZER = ByteBPE()
IDENTITY = TOKENIZER.sha256
OCI, TACO, MBPP = "nvidia/OpenCodeInstruct", "BAAI/TACO", "google-research-datasets/mbpp"
ROOT = Path(__file__).resolve().parents[1]


def task(source: str, index: int, *, solution: str | None = None, **fields: Any) -> Task:
    mode = "io" if source == TACO else "function"
    cases = json.dumps([{"input": f"{index}\n", "output": f"{index}\n"}])
    values: dict[str, Any] = {
        "id": f"{source}/{index}",
        "prompt": f"Task {index} from {source.split('/')[-1]}.",
        "public_tests": cases if mode == "io" else f"assert f({index}) == {index}",
        "hidden_tests": cases if mode == "io" else f"assert f({index + 1}) == {index + 1}",
        "source": source,
        "license": "fixture",
        "written_by": "human" if source in {TACO, MBPP} else "model",
        "split": "train",
        "mode": mode,
        "solution": solution or ("print(input())" if mode == "io" else "def f(x):\n    return x"),
    }
    values.update(fields)
    return Task(**values)


def pool(tmp_path: Path) -> Path:
    tasks = [task(OCI, index) for index in range(12)]
    tasks += [task(TACO, index) for index in range(6)]
    tasks += [task(MBPP, index) for index in range(2)]
    tasks += [task(ACECODE, index) for index in range(8)]
    tasks.append(task(OCI, 99, solution="x = 1\n" * 100))  # Over the 400-token test limit.
    save_tasks(tmp_path / "train.jsonl", tasks)
    return tmp_path / "train.jsonl"


PROVENANCE = {"dataset": "OpenAssistant/oasst2", "revision": "a" * 40, "owner_ruling": "approved"}
SCREENED = {"benchmarks": {}, "holdouts": [{"path": "dev.jsonl", "sha256": "b" * 64, "tasks": 1}]}


def language_record(path: Path, **changes: Any) -> Path:
    """A minimal completed language-stage record beside ``path``."""
    record: dict[str, Any] = {
        "schema_version": sft_data.SCHEMA_VERSION,
        "status": "completed",
        "provenance": dict(PROVENANCE),
        "screened_against": dict(SCREENED),
        "files": {"conversations.jsonl": {"sha256": file_sha256(path)}},
    }
    record.update(changes)
    (path.parent / "record.json").write_text(json.dumps(record))
    return path.parent / "record.json"


def conversations(
    tmp_path: Path, count: int = 10, *, texts: list[tuple[str, str]] | None = None
) -> Path:
    path = tmp_path / "conversations.jsonl"
    pairs = texts or [(f"Question {i}?", "An answer. " * (i + 3)) for i in range(count)]
    with path.open("w") as stream:
        for index, (question, answer) in enumerate(pairs):
            item = Conversation(
                id=f"tree-{index}",
                source="OpenAssistant/oasst2",
                messages=(Message("user", question), Message("assistant", answer)),
                message_ids=(f"m{index}a", f"m{index}b"),
                tokens=0,
            )
            stream.write(json.dumps(conversation_json(item)) + "\n")
    language_record(path)
    return path


def test_quotas_reproduce_the_approved_candidate_strata() -> None:
    assert quotas({OCI: 2091, TACO: 1514, MBPP: 266}, 1000) == {OCI: 540, TACO: 391, MBPP: 69}
    assert quotas({ACECODE: 4588, OCI: 2091, TACO: 1514, MBPP: 266}, 2000) == {
        ACECODE: 1085,
        OCI: 494,
        TACO: 358,
        MBPP: 63,
    }
    assert quotas({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}  # Ties go to the sorted name.
    assert quotas({"a": 3}, 0) == {"a": 0}
    with pytest.raises(ValueError):
        quotas({"a": 3}, 4)


def test_stratified_holdout_takes_each_sources_lowest_salted_hashes() -> None:
    tasks = [task(OCI, index) for index in range(12)] + [task(TACO, index) for index in range(6)]
    chosen, shares = stratified_holdout(tasks, 6)
    assert shares == {TACO: 2, OCI: 4}

    def key(item: Task) -> bytes:
        return hashlib.sha256(f"rl-candidate-v1|{item.id}".encode()).digest()

    expected = sorted([t for t in tasks if t.source == TACO], key=key)[:2]
    expected += sorted([t for t in tasks if t.source == OCI], key=key)[:4]
    assert chosen == expected
    assert stratified_holdout(list(reversed(tasks)), 6) == (chosen, shares)


def run_build(tmp_path: Path, **options: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {"sequence_length": 400, "rl_candidates": 5, "language_dev": 3}
    settings.update(options)
    return build(
        pool_train=pool(tmp_path),
        language=conversations(tmp_path),
        output=tmp_path / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(**settings),
    )


def test_build_holds_out_candidates_excludes_acecode_and_records_hashes(tmp_path: Path) -> None:
    record = run_build(tmp_path)
    output = tmp_path / "sft"
    candidates = [
        json.loads(line)
        for line in (output / "rl_candidates.jsonl").read_text().split("\n")
        if line
    ]
    # 13 OpenCodeInstruct, 6 TACO and 2 MBPP tasks: 5 x (13, 6, 2) / 21 = 3.10, 1.43, 0.48,
    # so floors 3, 1, 0 and the last unit to MBPP's larger remainder.
    assert record["code"]["rl_candidates"]["by_source"] == {TACO: 1, OCI: 3, MBPP: 1}
    assert len(candidates) == 5 and all(item["source"] != ACECODE for item in candidates)
    train = list(read_rows(output / "train.jsonl"))
    ids = {row.id for row in train}
    held = {item["id"] for item in candidates}
    assert not ids & held
    assert not any(row.source == ACECODE for row in train)
    assert record["code"]["excluded_sources"] == {ACECODE: 8}
    long_held = f"{OCI}/99" in held  # The over-length task may itself be a candidate.
    assert record["code"]["dropped"] == ({} if long_held else {"over_length": 1})
    assert record["code"]["sft_rows_by_source"] == {TACO: 5, MBPP: 1, OCI: 10 - (not long_held)}
    reference = next(t for t in [task(OCI, index) for index in range(12)] if t.id in ids)
    assert next(row for row in train if row.id == reference.id) == SFTRow.of(
        reference.id, OCI, render_code_task(TOKENIZER, reference)
    )
    for name, entry in record["files"].items():
        assert entry["sha256"] == file_sha256(output / name)
    assert record["inputs"]["pool_train"]["sha256"] == file_sha256(tmp_path / "train.jsonl")
    assert record["tokenizer_identity"] == IDENTITY and record["paid_runs"] == 0
    assert record["config"]["sources"] == [OCI, TACO, MBPP]
    assert json.loads((output / "record.json").read_text()) == record
    with pytest.raises(FileExistsError):
        run_build(tmp_path)


def targets_carrying_loss(row: SFTRow) -> int:
    """Count, by hand, a row's targets ``ids[1:]`` that are not ``IGNORE`` in the loader."""
    loader = SFTLoader([row], sequence_length=len(row.ids) - 1, batch_size=1, accumulation_steps=1)
    _, targets, _ = loader.next_batch()
    return sum(target != IGNORE for target in targets[0])


def row_length(row: SFTRow) -> int:
    return len(row.ids)


MEASURES: dict[str, Callable[[SFTRow], int]] = {
    "loss_tokens": targets_carrying_loss,
    "all_tokens": row_length,
}


def language_row(index: int) -> SFTRow:
    """The fixture conversation ``tree-index`` as its chat-v1 row."""
    messages = [
        Message("user", f"Question {index}?"),
        Message("assistant", "An answer. " * (index + 3)),
    ]
    return SFTRow.of(
        f"tree-{index}", "OpenAssistant/oasst2", render_conversation(TOKENIZER, messages)
    )


@pytest.mark.parametrize("basis", [None, "all_tokens"])
def test_language_dev_is_held_out_by_hash_and_train_is_capped_at_the_share(
    tmp_path: Path, basis: str | None
) -> None:
    # The default basis is loss tokens; the all-token basis is kept as an option.
    record = run_build(tmp_path) if basis is None else run_build(tmp_path, language_cap_basis=basis)
    basis = basis or "loss_tokens"
    other = "all_tokens" if basis == "loss_tokens" else "loss_tokens"
    output = tmp_path / "sft"
    dev = list(read_rows(output / "language_dev.jsonl"))
    rows = list(read_rows(output / "train.jsonl"))
    train = [row for row in rows if row.source == "OpenAssistant/oasst2"]
    code_rows = [row for row in rows if row.source != "OpenAssistant/oasst2"]

    def key(name: str) -> bytes:
        return hashlib.sha256(f"language-dev-v1|{name}".encode()).digest()

    ordered = sorted((f"tree-{index}" for index in range(10)), key=key)
    assert [row.id for row in dev] == ordered[:3]
    assert {row.id for row in train} <= set(ordered[3:])
    # Both bases, recounted from the written rows: loss tokens as the loader's targets
    # that are not IGNORE, all tokens as the row lengths.
    counted = {
        name: {"code": sum(map(measure, code_rows)), "language": sum(map(measure, train))}
        for name, measure in MEASURES.items()
    }
    code_size, language_size = counted[basis]["code"], counted[basis]["language"]
    assert language_size / (language_size + code_size) <= 0.3
    accepted = [name for name in ordered[3:] if name in {row.id for row in train}]
    assert accepted == ordered[3 : 3 + len(accepted)]  # A prefix of the hash order.
    # The first conversation left out would have taken the share past 30%.
    refused = language_row(int(ordered[3 + len(accepted)].split("-")[1]))
    more = language_size + MEASURES[basis](refused)
    assert more / (more + code_size) > 0.3
    language = record["language"]
    assert language["dev"] == 3 and language["train"] == len(train)
    assert language["train_available"] == 7 and "over_train_max" not in language
    assert language["train"] + language["over_share"] == 7
    # 360 code loss tokens and 1,205 code tokens in all: the loss-token cap keeps two
    # conversations here and the all-token cap five.
    assert language["train"] == {"loss_tokens": 2, "all_tokens": 5}[basis]
    shares, tokens = record["shares"], record["tokens"]
    assert record["config"]["language_cap_basis"] == shares["language_cap_basis"] == basis
    assert shares["language_cap"] == 0.3 and shares["cap_bound"]
    assert shares["cap_dropped"] == language["over_share"] > 0
    assert set(shares["bases"]) == {"loss_tokens", "all_tokens"}
    for name, counts in counted.items():
        total = counts["code"] + counts["language"]
        assert shares[name] == {
            **counts,
            "language_share": pytest.approx(counts["language"] / total),
            "code_share": pytest.approx(counts["code"] / total),
        }
        assert tokens["code"][name] == counts["code"]
        assert tokens["language"][name] == counts["language"]
    # Every chat-v1 row's loss tokens are also sum(mask): its first token is never a target.
    assert all(MEASURES["loss_tokens"](row) == sum(row.mask) for row in rows)
    # Code carries long prompts and short programs, so the two bases differ.
    assert shares[other]["language_share"] != pytest.approx(shares[basis]["language_share"])
    # Code is never subsampled: every non-candidate task that fits is a row.
    assert record["code"]["sft_rows"] == sum(record["code"]["sft_rows_by_source"].values())
    first = next(row for row in train if row.id == accepted[0])
    assert first == language_row(int(accepted[0].split("-")[1]))


def test_language_under_the_cap_is_kept_whole(tmp_path: Path) -> None:
    # At a 90% cap all seven training conversations fit: nothing is subsampled.
    record = run_build(tmp_path, language_cap=0.9)
    language, shares = record["language"], record["shares"]
    assert language["train"] == language["train_available"] == 7
    assert "over_share" not in language and "over_train_max" not in language
    assert not shares["cap_bound"] and shares["cap_dropped"] == 0
    assert 0 < shares["loss_tokens"]["language_share"] < 0.9


@pytest.mark.parametrize("basis", ["loss_tokens", "all_tokens"])
@pytest.mark.parametrize(("extra", "kept"), [(0, True), (1, False)])
def test_the_share_cap_is_exact_at_its_boundary(
    tmp_path: Path, basis: str, extra: int, kept: bool
) -> None:
    # A 50% cap lets language equal the code exactly in the cap's basis, not one more.
    tasks = [task(OCI, index) for index in range(3)]
    save_tasks(tmp_path / "train.jsonl", tasks)
    rendered = [render_code_task(TOKENIZER, item) for item in tasks]
    if basis == "loss_tokens":
        code_size = sum(sum(item.mask[1:]) for item in rendered)
        # A one-exchange chat-v1 row's loss tokens are its answer bytes and
        # <|assistant_end|>.
        answer = "x" * (code_size - 1 + extra)
    else:
        code_size = sum(len(item.ids) for item in rendered)
        # A one-exchange chat-v1 row costs five marker tokens plus its text bytes.
        answer = "x" * (code_size - 5 - 1 + extra)
    record = build(
        pool_train=tmp_path / "train.jsonl",
        language=conversations(tmp_path, texts=[("Q", answer)]),
        output=tmp_path / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(
            rl_candidates=0, language_dev=0, language_cap=0.5, language_cap_basis=basis
        ),
    )
    shares = record["shares"]
    assert shares["language_cap_basis"] == basis
    assert record["tokens"]["code"][basis] == shares[basis]["code"] == code_size
    assert record["language"]["train"] == int(kept)
    assert shares["cap_bound"] is not kept and shares["cap_dropped"] == int(not kept)
    if kept:
        assert shares[basis]["language"] == code_size
        assert shares[basis]["language_share"] == 0.5
    else:
        assert shares[basis]["language"] == 0 and shares[basis]["language_share"] == 0.0


def test_language_train_max_and_without_language(tmp_path: Path) -> None:
    assert BuildConfig().language_train_max is None  # No count cap by default.
    with pytest.raises(ValueError):
        BuildConfig(language_train_max=0)
    record = run_build(tmp_path / "a", language_train_max=2, language_cap=0.9)
    assert record["language"]["train"] == 2 and record["language"]["over_train_max"] == 5
    assert not record["shares"]["cap_bound"]
    alone = build(
        pool_train=pool(tmp_path / "b"),
        output=tmp_path / "b" / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(sequence_length=400, rl_candidates=5),
    )
    assert alone["inputs"]["language"] is None and alone["tokens"]["language"]["rows"] == 0
    assert alone["shares"]["loss_tokens"]["language_share"] == 0.0
    assert (tmp_path / "b" / "sft" / "language_dev.jsonl").read_text() == ""


def test_acecode_only_by_explicit_option(tmp_path: Path) -> None:
    assert BuildConfig().candidates == 1000 and BuildConfig().sources == (OCI, TACO, MBPP)
    assert BuildConfig(include_acecode=True).candidates == 2000
    record = run_build(tmp_path, include_acecode=True, rl_candidates=10)
    assert record["code"]["rl_candidates"]["by_source"] == {ACECODE: 3, TACO: 2, OCI: 4, MBPP: 1}
    assert record["code"]["excluded_sources"] == {}
    assert ACECODE in record["config"]["sources"]


def test_code_dev_rows_and_input_validation(tmp_path: Path) -> None:
    dev = [task(OCI, index, split="dev") for index in range(100, 103)]
    save_tasks(tmp_path / "dev.jsonl", dev)
    record = build(
        pool_train=pool(tmp_path),
        pool_dev=tmp_path / "dev.jsonl",
        output=tmp_path / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(sequence_length=400, rl_candidates=0),
    )
    rows = list(read_rows(tmp_path / "sft" / "code_dev.jsonl"))
    assert [row.id for row in rows] == sorted(t.id for t in dev) and record["code"]["dev_rows"] == 3
    save_tasks(tmp_path / "mixed.jsonl", [task(OCI, 1), task(OCI, 2, split="dev")])
    with pytest.raises(ValueError, match="train-split"):
        build(
            pool_train=tmp_path / "mixed.jsonl",
            output=tmp_path / "other",
            encoder=TOKENIZER,
            tokenizer_identity=IDENTITY,
            config=BuildConfig(),
        )
    assert BuildConfig().language_cap == 0.3
    assert BuildConfig().language_cap_basis == "loss_tokens"  # The owner's ruling.
    with pytest.raises(ValueError):
        BuildConfig(language_cap=1.0)
    for bad in ("assistant_tokens", ["loss_tokens"], None, 1):
        with pytest.raises(ValueError, match="language_cap_basis"):
            BuildConfig(language_cap_basis=bad)  # type: ignore[arg-type]


def test_build_record_carries_the_language_records_provenance(tmp_path: Path) -> None:
    record = run_build(tmp_path)
    entry = record["inputs"]["language"]
    assert entry["provenance"] == PROVENANCE
    assert entry["screened_against"] == SCREENED
    assert entry["record"] == {
        "path": str(tmp_path / "record.json"),
        "sha256": file_sha256(tmp_path / "record.json"),
    }
    assert entry["sha256"] == file_sha256(tmp_path / "conversations.jsonl")


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"schema_version": 1}, ValueError),
        ({"status": "partial"}, ValueError),
        ({"provenance": None}, ValueError),
        ({"screened_against": None}, ValueError),
        ({"files": {"conversations.jsonl": {"sha256": "0" * 64}}}, ValueError),
        ({"files": {}}, ValueError),
        ({"provenance": {**PROVENANCE, "owner_ruling": "rejected"}}, PermissionError),
        ({"provenance": {"dataset": "OpenAssistant/oasst2"}}, PermissionError),
    ],
)
def test_build_refuses_language_without_a_matching_approved_record(
    tmp_path: Path, changes: dict[str, Any], error: type[Exception]
) -> None:
    path = conversations(tmp_path)
    language_record(path, **changes)
    train = pool(tmp_path)

    def attempt() -> None:
        build(
            pool_train=train,
            language=path,
            output=tmp_path / "sft",
            encoder=TOKENIZER,
            tokenizer_identity=IDENTITY,
            config=BuildConfig(sequence_length=400, rl_candidates=5, language_dev=3),
        )

    with pytest.raises(error):
        attempt()
    (tmp_path / "record.json").unlink()
    with pytest.raises(FileNotFoundError, match="language stage record"):
        attempt()
    assert not (tmp_path / "sft").exists()


# ----------------------------------------------------------------- OpenAssistant


def message(identifier: str, role: str, text: str, **fields: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "message_id": identifier,
        "role": role,
        "text": text,
        "lang": "en",
        "synthetic": False,
        "model_name": None,
        "deleted": False,
        "review_result": True,
        "rank": None,
        "labels": {"pii": {"value": 0.0, "count": 3}, "spam": {"value": 0.0, "count": 3}},
        "replies": [],
    }
    values.update(fields)
    return values


def tree(
    identifier: str, prompt: Mapping[str, Any], state: str = "ready_for_export"
) -> dict[str, Any]:
    return {"message_tree_id": identifier, "tree_state": state, "prompt": prompt}


def authored_trees() -> list[dict[str, Any]]:
    why = message(
        "p2",
        "prompter",
        "Why?",
        replies=[
            message("a3", "assistant", "Deleted.", rank=0, deleted=True),
            message("a4", "assistant", "Because.", rank=None),
        ],
    )
    return [
        tree(
            "t1",
            message(
                "p1",
                "prompter",
                "What is two plus two?",
                replies=[
                    message("a1", "assistant", "Synthetic four.", rank=0, synthetic=True),
                    message("a2", "assistant", "It is four.", rank=1, replies=[why]),
                ],
            ),
        ),
        tree("t2", message("p", "prompter", "Waiting."), state="growing"),
        tree("t3", message("p3", "prompter", "Hola", lang="es")),
        tree(
            "t4",
            message(
                "p4",
                "prompter",
                "Hi",
                replies=[message("a5", "assistant", "No.", review_result=False)],
            ),
        ),
        tree(
            "t5",
            message(
                "p5",
                "prompter",
                "Hi",
                replies=[
                    message(
                        "a6", "assistant", "Call me.", labels={"pii": {"value": 0.5, "count": 2}}
                    )
                ],
            ),
        ),
        tree(
            "t6",
            message(
                "p6",
                "prompter",
                "Short?",
                replies=[
                    message(
                        "a7",
                        "assistant",
                        "Yes.",
                        rank=0,
                        replies=[
                            message(
                                "p7",
                                "prompter",
                                "More?",
                                replies=[message("a8", "assistant", "x" * 80, rank=0)],
                            )
                        ],
                    )
                ],
            ),
        ),
        tree("t7", message("p8", "prompter", "y" * 80, replies=[message("a9", "assistant", "z")])),
        tree(
            "t8",
            message(
                "p9",
                "prompter",
                "Pick?",
                replies=[
                    message("b2", "assistant", "Second.", rank=1),
                    message("b0", "assistant", "Unranked."),
                    message("b1", "assistant", "First.", rank=0),
                ],
            ),
        ),
    ]


def test_adapter_follows_the_best_eligible_path_and_counts_every_filter() -> None:
    kept, counts = adapt_trees(authored_trees(), TOKENIZER, max_tokens=60)
    by_id = {item.id: item for item in kept}
    assert sorted(by_id) == ["t1", "t6", "t8"]
    assert [m.text for m in by_id["t1"].messages] == [
        "What is two plus two?",
        "It is four.",
        "Why?",
        "Because.",
    ]
    assert by_id["t1"].message_ids == ("p1", "a2", "p2", "a4")
    assert [m.role for m in by_id["t1"].messages] == ["user", "assistant", "user", "assistant"]
    assert [m.text for m in by_id["t6"].messages] == ["Short?", "Yes."]  # Truncated to fit.
    assert [m.text for m in by_id["t8"].messages] == ["Pick?", "First."]
    rendered = render_conversation(TOKENIZER, by_id["t1"].messages)
    assert by_id["t1"].tokens == len(rendered.ids) <= 60
    assert counts == {
        "message:deleted": 1,
        "message:eligible": 16,
        "message:empty_text": 0,  # Every message reason is listed, zero included.
        "message:failed_review": 1,
        "message:missing_metadata": 0,
        "message:model_written": 0,
        "message:not_english": 1,
        "message:pii_label": 1,
        "message:synthetic": 1,
        "message:unknown_role": 0,
        "messages_seen": 21,
        "tree:kept": 3,
        "tree:no_eligible_reply": 2,
        "tree:not_ready_for_export": 1,
        "tree:over_tokens": 1,
        "tree:root_not_english": 1,
        "tree:truncated_to_fit": 1,
        "trees_seen": 8,
    }
    assert conversation_from_json(json.loads(json.dumps(conversation_json(kept[0])))) == kept[0]


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"model_name": "oasst-sft-0_3000,max_new_tokens=400 (..)"}, "model_written"),
        ({"model_name": " "}, "model_written"),
        ({"model_name": 7}, "model_written"),
        ({"model_name": ""}, None),
        ({"model_name": None}, None),
        # Both marks: the first reason is counted, so the message is counted once.
        ({"synthetic": True, "model_name": "oasst-sft-0"}, "synthetic"),
        ({"synthetic": None}, "synthetic"),
    ],
)
def test_only_human_written_messages_are_eligible(
    fields: dict[str, object], reason: str | None
) -> None:
    assert message_reason(message("a", "assistant", "Hi.", **fields)) == reason


@pytest.mark.parametrize("key", ["synthetic", "model_name"])
def test_a_message_without_either_authorship_field_is_missing_metadata(key: str) -> None:
    raw = message("a", "assistant", "Hi.")
    del raw[key]
    assert message_reason(raw) == "missing_metadata"


def test_the_walk_skips_a_best_ranked_model_written_reply() -> None:
    written = {"model_name": "oasst-sft-0_3000"}
    trees = [
        # The rank-0 reply is model-written, so the walk takes the rank-1 human reply,
        # and a model-written prompter turn below it ends the walk there.
        tree(
            "next",
            message(
                "p1",
                "prompter",
                "Name a prime.",
                replies=[
                    message("a1", "assistant", "Seven.", rank=0, **written),
                    message(
                        "a2",
                        "assistant",
                        "Two.",
                        rank=1,
                        replies=[message("p2", "prompter", "Another?", **written)],
                    ),
                ],
            ),
        ),
        # The only reply is model-written: no eligible reply, so the tree is dropped.
        tree(
            "ends",
            message(
                "p3",
                "prompter",
                "Hi",
                replies=[message("a3", "assistant", "Hello.", rank=0, **written)],
            ),
        ),
        # A model-written root drops the whole tree.
        tree("root", message("p4", "prompter", "Hi", **written)),
    ]
    kept, counts = adapt_trees(trees, TOKENIZER)
    assert [(item.id, item.message_ids) for item in kept] == [("next", ("p1", "a2"))]
    assert [m.text for m in kept[0].messages] == ["Name a prime.", "Two."]
    assert counts["message:model_written"] == 4
    assert counts["tree:no_eligible_reply"] == 1
    assert counts["tree:root_model_written"] == 1
    assert counts["tree:kept"] == 1


def test_adapter_fails_closed_on_missing_metadata_and_duplicate_trees() -> None:
    bare = {"message_id": "p", "role": "prompter", "text": "Hi", "lang": "en"}
    _, counts = adapt_trees([tree("t", bare)], TOKENIZER)
    assert counts["tree:root_missing_metadata"] == 1
    listed = message("p", "prompter", "Hi", labels=[{"name": "pii", "value": 0.0}])
    _, counts = adapt_trees([tree("t", listed)], TOKENIZER)
    assert counts["tree:root_missing_metadata"] == 1
    with pytest.raises(ValueError, match="unique"):
        adapt_trees([tree("t", message("p", "prompter", "a"))] * 2, TOKENIZER)


@pytest.mark.parametrize(
    "labels",
    [
        {"name": ["pii"], "value": [1.0], "count": [3]},  # Columnar, not the README's form.
        {"pii": [1.0]},
        {"pii": {"value": "0"}},
        {"pii": {"value": False}},
        {"pii": {"count": 3}},
        {"spam": {"value": None}},
        [],
        "",
    ],
)
def test_labels_in_any_other_shape_fail_closed(labels: object) -> None:
    assert message_reason(message("p", "prompter", "Hi", labels=labels)) == "missing_metadata"


def test_pii_labels_drop_nonzero_values_and_unlabelled_messages_are_counted() -> None:
    assert message_reason(message("p", "prompter", "Hi")) is None
    assert message_reason(message("p", "prompter", "Hi", labels={"pii": {"value": 0}})) is None
    flagged = {"pii": {"value": 0.25, "count": 4}}
    assert message_reason(message("p", "prompter", "Hi", labels=flagged)) == "pii_label"
    no_label = [{"spam": {"value": 0.0, "count": 3}}, {}, None]
    for labels in no_label:
        assert message_reason(message("p", "prompter", "Hi", labels=labels)) is None
    bare = message("p", "prompter", "Hi")
    del bare["labels"]
    assert message_reason(bare) is None
    root = message(
        "p",
        "prompter",
        "Hello there?",
        labels=None,
        replies=[
            message("a", "assistant", "Hi.", rank=0, labels={}),
            message("b", "assistant", "Hey.", rank=1, labels=None),
        ],
    )
    kept, counts = adapt_trees([tree("t", root)], TOKENIZER)
    assert [item.message_ids for item in kept] == [("p", "a")]
    assert counts["message:eligible"] == 3
    assert counts["message:eligible_no_pii_label"] == 3  # Every eligible message, chosen or not.
    assert counts["kept_messages_no_pii_label"] == 2  # Only the chosen path's messages.
    assert counts["tree:kept_with_no_pii_label"] == 1


def choice(question: str) -> ChoiceTask:
    return ChoiceTask(
        id="arc/1",
        question=question,
        labels=("A", "B"),
        choices=("yes", "no"),
        answer="A",
        source="allenai/ai2_arc",
        source_revision="0" * 40,
        license="CC-BY-SA-4.0",
    )


def chat(identifier: str, *texts: str) -> Conversation:
    roles = ("user", "assistant")
    return Conversation(
        id=identifier,
        source="OpenAssistant/oasst2",
        messages=tuple(Message(roles[i % 2], text) for i, text in enumerate(texts)),
        message_ids=tuple(f"{identifier}-{i}" for i in range(len(texts))),
        tokens=1,
    )


def test_screens_drop_email_benchmark_arc_holdout_and_near_duplicate_text() -> None:
    benchmark = task(OCI, 1, prompt=" ".join(f"word{n}" for n in range(20)), split="test")
    holdout = task(OCI, 2, prompt=" ".join(str(n) for n in range(30)), split="dev")
    arc = choice("Which gas do plants take in from the air to make their food each day?")
    items = [
        chat("email", "Write to me", "Sure: person@example.org"),
        chat("bench", "Hi", "So " + " ".join(f"word{n}" for n in range(2, 16)) + " ok"),
        chat(
            "arc",
            "Which gas do plants take in from the air to make their food each day? A or B",
            "A",
        ),
        chat("dup", " ".join(str(n) for n in range(30)), "Count."),
        chat("clean", "Tell me a joke", "No."),
    ]
    kept, counts = screen_conversations(
        items, benchmarks=[benchmark], choices=[arc], holdouts=[holdout], min_alpha_tokens=3
    )
    assert [item.id for item in kept] == ["clean"]
    assert counts == {
        "contamination": 2,
        "email_address": 1,
        "kept": 1,
        "near_duplicate": 1,
        "seen": 5,
    }


def test_screens_keep_one_conversation_per_repeated_opening_prompt() -> None:
    opening = "How do I sort a list of numbers in Python without changing the original list?"
    items = [
        chat("first", opening, "Use sorted(numbers)."),
        chat("same", opening, "Call sorted on it."),
        # One extra word: 5-word-shingle Jaccard 11/12, above the 0.85 threshold.
        chat("near", opening.rstrip("?") + " please?", "sorted() returns a copy."),
        chat("spaced", "  " + opening.upper() + "  ", "Same words, other case."),
        chat("other", "Tell me a joke about cats", "No."),
        chat("rejected", "Write to me", "Sure: person@example.org"),
        chat("after", "Write to me", "Hello."),  # A dropped opening is not a seen prompt.
    ]
    kept, counts = screen_conversations(
        items, benchmarks=[], choices=[], holdouts=[], min_alpha_tokens=3
    )
    assert [item.id for item in kept] == ["first", "other", "after"]
    assert counts == {"email_address": 1, "kept": 3, "repeated_prompt": 3, "seen": 7}
    # The earliest in input order is kept, whatever its position.
    kept, _ = screen_conversations(
        list(reversed(items[:3])), benchmarks=[], choices=[], holdouts=[], min_alpha_tokens=3
    )
    assert [item.id for item in kept] == ["near"]


# ------------------------------------------------------------------ language spec

SPEC = ROOT / "configs" / "sft-language-oasst2.json"
TREES = "2023-11-05_oasst2_ready.trees.jsonl.gz"
DELETE = object()


def spec_file(tmp_path: Path, **changes: Any) -> Path:
    raw = json.loads(SPEC.read_text())
    raw.update(changes)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    return path


def terms_file(tmp_path: Path, section: str | None, key: str, value: Any) -> Path:
    """The committed spec with one terms-of-record field changed (or deleted)."""
    raw = json.loads(SPEC.read_text())
    target = raw["terms_of_record"] if section is None else raw["terms_of_record"][section]
    if value is DELETE:
        del target[key]
    else:
        target[key] = value
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    return path


def card_listing() -> dict[str, Any]:
    card = json.loads(SPEC.read_text())["terms_of_record"]["card"]
    return {"rfilename": card["file"], "size": card["size_bytes"], "blobId": card["git_blob_id"]}


class FakeHub(Hub):
    def __init__(self, files: Mapping[str, bytes], card: Mapping[str, Any] | None = None) -> None:
        super().__init__(None, sleep=lambda _: None)
        self.files = files
        self.card = card_listing() if card is None else card
        self.calls = 0
        self.downloads = 0

    def json(self, url: str) -> Any:
        self.calls += 1
        revision = url.split("/revision/")[1].split("?")[0]
        return {
            "sha": revision,
            "siblings": [
                dict(self.card),
                *(
                    {
                        "rfilename": name,
                        "size": len(data),
                        "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                    }
                    for name, data in self.files.items()
                ),
            ],
        }

    def download(self, url: str, destination: Path, size: int, sha256: str) -> int:
        self.downloads += 1
        data = self.files[url.rsplit("/", 1)[1]]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return len(data)


def gz_trees(trees: list[dict[str, Any]]) -> bytes:
    return gzip.compress("".join(json.dumps(item) + "\n" for item in trees).encode())


def test_committed_spec_records_the_ruling_and_the_terms_of_record() -> None:
    spec = sft_data.load_language_spec(SPEC)
    assert spec["schema_version"] == 2 and spec["dataset"] == "OpenAssistant/oasst2"
    assert spec["owner_ruling"] == "approved" and spec["owner_ruling_on"] == "2026-10-26"
    revision = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
    assert spec["revision"] == revision and spec["license_label"] == "Apache-2.0"
    terms = spec["terms_of_record"]
    assert terms["card"] == {
        "file": "README.md",
        "revision": revision,
        "size_bytes": 10606,
        "sha256": "52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31",
        "git_blob_id": "ce94775b146d09acf5c5ea31992a9f322e12de76",
        "license": "apache-2.0",
        "license_line": 2,
    }
    paper = terms["paper"]
    assert (paper["arxiv_id"], paper["version"], paper["date"]) == (
        "2304.07327",
        "v2",
        "2023-10-31",
    )
    assert paper["statement"] == "We release our code and data under a fully permissive licence."
    assert "human-written" in terms["scope"] and "model_name" in terms["scope"]
    finding = spec["terms_finding"]
    for fact in ("no LICENSE file", "only as a metadata label", "grant no license", "accepted"):
        assert fact in finding
    assert spec["files"] == [TREES] and spec["max_download_bytes"] == 54370156


def test_an_unruled_spec_blocks_the_download(tmp_path: Path) -> None:
    spec = sft_data.load_language_spec(spec_file(tmp_path, owner_ruling=None, owner_ruling_on=None))
    hub = FakeHub({})
    with pytest.raises(PermissionError, match="owner"):
        sft_data.fetch(spec, tmp_path / "intake", hub_factory=lambda _: hub)
    assert hub.calls == 0 and not (tmp_path / "intake").exists()
    with pytest.raises(PermissionError, match="owner"):
        sft_data.adapt_language(
            spec,
            tmp_path / "intake",
            encoder=TOKENIZER,
            benchmarks=[],
            choices=[],
            benchmark_identities={},
            holdout_files=[],
            min_alpha_tokens=3,
        )


def test_fetch_adapt_and_build_carry_the_provenance(tmp_path: Path) -> None:
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(
        spec_file(tmp_path, max_tokens=60, max_download_bytes=len(data))
    )
    directory = tmp_path / "intake"
    receipt = sft_data.fetch(spec, directory, hub_factory=lambda _: FakeHub({TREES: data}))
    assert receipt["files"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["files"][0]["size_bytes"] == len(data)
    assert file_sha256(directory / "raw" / TREES) == receipt["files"][0]["sha256"]
    assert receipt["owner_ruling"] == "approved" and receipt["revision"] == spec["revision"]
    assert receipt["terms_of_record"] == spec["terms_of_record"]
    assert receipt["card"]["git_blob_id"] == spec["terms_of_record"]["card"]["git_blob_id"]
    with pytest.raises(FileExistsError):
        sft_data.fetch(spec, directory, hub_factory=lambda _: FakeHub({TREES: data}))
    dev = tmp_path / "dev.jsonl"
    save_tasks(dev, [task(OCI, 2, prompt="Pick? an option", split="dev")])
    record = sft_data.adapt_language(
        spec,
        directory,
        encoder=TOKENIZER,
        benchmarks=[],
        choices=[],
        benchmark_identities={},
        holdout_files=[dev],
        min_alpha_tokens=3,
    )
    assert record["intake"][TREES]["tree:kept"] == 3
    assert record["screens"] == {"kept": 3, "seen": 3}
    path = directory / "language" / "conversations.jsonl"
    assert record["files"]["conversations.jsonl"]["sha256"] == file_sha256(path)
    assert [json.loads(line)["id"] for line in path.read_text().split("\n") if line] == [
        "t1",
        "t6",
        "t8",
    ]
    assert record["exchanges_per_conversation"] == {"1": 2, "2": 1}
    provenance = record["provenance"]
    assert provenance["revision"] == spec["revision"]
    assert provenance["owner_ruling"] == "approved"
    assert provenance["owner_ruling_on"] == spec["owner_ruling_on"]
    assert provenance["terms_of_record"] == spec["terms_of_record"]
    assert provenance["license_label"] == "Apache-2.0"
    assert provenance["spec_sha256"] == spec["sha256"]
    assert provenance["receipts_sha256"] == file_sha256(directory / "download-receipts.json")
    assert provenance["files"] == [
        {
            "filename": TREES,
            "url": receipt["files"][0]["url"],
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    ]
    with pytest.raises(FileExistsError):
        sft_data.adapt_language(
            spec,
            directory,
            encoder=TOKENIZER,
            benchmarks=[],
            choices=[],
            benchmark_identities={},
            holdout_files=[],
            min_alpha_tokens=3,
        )
    built = build(
        pool_train=pool(tmp_path),
        language=path,
        output=tmp_path / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(sequence_length=400, rl_candidates=5, language_dev=1),
    )
    assert built["inputs"]["language"]["provenance"] == provenance
    assert built["inputs"]["language"]["screened_against"] == record["screened_against"]
    assert built["inputs"]["language"]["record"]["sha256"] == file_sha256(
        directory / "language" / "record.json"
    )


def intake(tmp_path: Path, name: str = "intake") -> tuple[dict[str, Any], Path]:
    """The committed spec, sized to the authored trees, fetched into ``tmp_path/name``."""
    data = gz_trees(authored_trees())
    spec_path = spec_file(tmp_path, max_tokens=60, max_download_bytes=len(data))
    spec = sft_data.load_language_spec(spec_path)
    directory = tmp_path / name
    sft_data.fetch(spec, directory, hub_factory=lambda _: FakeHub({TREES: data}))
    return spec, directory


def test_the_language_record_names_the_benchmarks_and_holdout_files_it_screened(
    tmp_path: Path,
) -> None:
    spec, directory = intake(tmp_path)
    shutil.copytree(directory, tmp_path / "dev-only")
    dev, test = tmp_path / "dev.jsonl", tmp_path / "test.jsonl"
    # The dev task's prompt is t1's whole opening, so the n-gram screen drops t1.
    save_tasks(dev, [task(OCI, 2, prompt="What is two plus two?", split="dev")])
    save_tasks(test, [task(OCI, 3, split="test"), task(TACO, 4, split="test")])
    identities = {"humaneval": {"tasks": 1, "sha256": "c" * 64}, "arc-easy": {"suite": "ARC"}}
    benchmark = task(OCI, 1, prompt="Unrelated benchmark prompt.", split="test")
    arc = choice("Which gas do plants take in from the air to make their food each day?")

    def adapt(where: Path, files: list[Path]) -> dict[str, Any]:
        return sft_data.adapt_language(
            spec,
            where,
            encoder=TOKENIZER,
            benchmarks=[benchmark],
            choices=[arc],
            benchmark_identities=identities,
            holdout_files=files,
            min_alpha_tokens=3,
        )

    record = adapt(directory, [dev, test])
    assert record["screens"] == {"contamination": 1, "kept": 2, "seen": 3}
    assert record["screened_against"] == {
        "benchmarks": identities,
        "benchmark_tasks": 1,
        "choice_tasks": 1,
        "holdouts": [
            {"path": str(dev), "sha256": file_sha256(dev), "tasks": 1, "splits": {"dev": 1}},
            {"path": str(test), "sha256": file_sha256(test), "tasks": 2, "splits": {"test": 2}},
        ],
        "min_alpha_tokens": 3,
    }
    # A screen against dev alone is visible in its record, not only in its counts.
    alone = adapt(tmp_path / "dev-only", [dev])["screened_against"]["holdouts"]
    assert [item["splits"] for item in alone] == [{"dev": 1}]


def test_the_language_cli_screens_against_pool_dev_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import task_pool

    spec, directory = intake(tmp_path)
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    save_tasks(pool_dir / "dev.jsonl", [task(OCI, 2, split="dev")])
    save_tasks(pool_dir / "test.jsonl", [task(OCI, 3, split="test")])
    identities = {"humaneval": {"tasks": 0, "sha256": "c" * 64}}
    monkeypatch.setattr(task_pool, "load_spec", lambda _: {"protection": {"min_alpha_tokens": 3}})
    monkeypatch.setattr(task_pool, "protection", lambda _: ([], [], identities))
    monkeypatch.setattr(sft_data, "_encoder", lambda _: (TOKENIZER, IDENTITY))
    status = sft_data.main(
        [
            "language",
            "--language-spec",
            str(tmp_path / "spec.json"),
            "--directory",
            str(directory),
            "--pool-spec",
            str(tmp_path / "pool-spec.json"),
            "--pool-dir",
            str(pool_dir),
        ]
    )
    assert status == 0
    record = json.loads((directory / "language" / "record.json").read_text())
    screened = record["screened_against"]
    assert screened["benchmarks"] == identities and screened["min_alpha_tokens"] == 3
    assert [(item["path"], item["sha256"], item["splits"]) for item in screened["holdouts"]] == [
        (str(pool_dir / name), file_sha256(pool_dir / name), {split: 1})
        for name, split in (("dev.jsonl", "dev"), ("test.jsonl", "test"))
    ]


@pytest.mark.parametrize(
    "card",
    [
        {**card_listing(), "blobId": "0" * 40},
        {**card_listing(), "size": 10607},
        {"rfilename": "LICENSE", "size": 10606, "blobId": card_listing()["blobId"]},
    ],
)
def test_fetch_refuses_a_card_that_differs_from_the_terms_of_record(
    tmp_path: Path, card: dict[str, Any]
) -> None:
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(spec_file(tmp_path, max_download_bytes=len(data)))
    hub = FakeHub({TREES: data}, card=card)
    with pytest.raises(ValueError, match="terms of record"):
        sft_data.fetch(spec, tmp_path / "intake", hub_factory=lambda _: hub)
    assert hub.downloads == 0 and not (tmp_path / "intake").exists()


def test_fetch_refuses_oversized_plans_and_bad_specs(tmp_path: Path) -> None:
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(spec_file(tmp_path, max_download_bytes=len(data) - 1))
    with pytest.raises(RuntimeError, match="approved maximum"):
        sft_data.fetch(spec, tmp_path / "x", hub_factory=lambda _: FakeHub({TREES: data}))
    with pytest.raises(ValueError, match="allowlist"):
        sft_data.load_language_spec(spec_file(tmp_path, license_label="CC-BY-NC-4.0"))
    with pytest.raises(ValueError):
        sft_data.load_language_spec(spec_file(tmp_path, files=["data/train.parquet"]))
    with pytest.raises(ValueError):
        sft_data.load_language_spec(
            spec_file(tmp_path, owner_ruling="approved", owner_ruling_on="2999-01-01")
        )
    with pytest.raises(ValueError, match="schema"):
        sft_data.load_language_spec(spec_file(tmp_path, schema_version=1))
    raw = json.loads(SPEC.read_text())
    del raw["terms_of_record"]
    (tmp_path / "old.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly the keys"):
        sft_data.load_language_spec(tmp_path / "old.json")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "scope", " "),
        (None, "extra", "x"),
        (None, "card", DELETE),
        ("card", "extra", "x"),
        ("card", "sha256", DELETE),
        ("card", "revision", "0" * 40),  # Read at another revision than the pinned one.
        ("card", "sha256", "52BD" + "0" * 60),
        ("card", "git_blob_id", "ce94"),
        ("card", "size_bytes", 0),
        ("card", "license_line", "2"),
        ("card", "license", "mit"),  # Not the spec's license label.
        ("card", "file", ""),
        ("paper", "extra", "x"),
        ("paper", "statement", DELETE),
        ("paper", "statement", ""),
        ("paper", "arxiv_id", "arXiv:2304.07327"),
        ("paper", "arxiv_id", 2304.07327),
        ("paper", "version", "2"),
        ("paper", "version", "v0"),
        ("paper", "date", "2023-10-31T00:00"),
        ("paper", "date", "2999-01-01"),
        ("paper", "title", None),
    ],
)
def test_spec_validates_every_terms_of_record_field(
    tmp_path: Path, section: str | None, key: str, value: Any
) -> None:
    with pytest.raises(ValueError):
        sft_data.load_language_spec(terms_file(tmp_path, section, key, value))


@pytest.mark.parametrize(
    ("ruling", "on"),
    [
        (None, "2026-10-01"),  # A date alone is not a ruling.
        ("approved", None),
        ("yes", "2026-10-01"),
        ("Approved", "2026-10-01"),
        (True, "2026-10-01"),
    ],
)
def test_spec_rejects_an_incomplete_or_unknown_ruling(
    tmp_path: Path, ruling: object, on: object
) -> None:
    with pytest.raises(ValueError, match="owner_ruling"):
        sft_data.load_language_spec(spec_file(tmp_path, owner_ruling=ruling, owner_ruling_on=on))


def test_a_rejected_ruling_never_downloads(tmp_path: Path) -> None:
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(
        spec_file(tmp_path, owner_ruling="rejected", owner_ruling_on="2026-10-01")
    )
    hub = FakeHub({TREES: data})
    with pytest.raises(PermissionError, match="did not approve"):
        sft_data.fetch(spec, tmp_path / "intake", hub_factory=lambda _: hub)
    assert hub.calls == 0 and not (tmp_path / "intake").exists()
    assert asdict(BuildConfig())["language_dev"] == 300
