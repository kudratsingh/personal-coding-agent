"""SFT data builder, OpenAssistant adapter and language intake, on authored fixtures."""

import gzip
import hashlib
import json
from collections.abc import Mapping
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
from scratchcoder.data.sft import SFTRow, read_rows
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


def conversations(tmp_path: Path, count: int = 10) -> Path:
    path = tmp_path / "conversations.jsonl"
    with path.open("w") as stream:
        for index in range(count):
            item = Conversation(
                id=f"tree-{index}",
                source="OpenAssistant/oasst2",
                messages=(
                    Message("user", f"Question {index}?"),
                    Message("assistant", "An answer. " * (index + 3)),
                ),
                message_ids=(f"m{index}a", f"m{index}b"),
                tokens=0,
            )
            stream.write(json.dumps(conversation_json(item)) + "\n")
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


def test_language_dev_is_held_out_by_hash_and_train_is_capped_at_the_share(tmp_path: Path) -> None:
    record = run_build(tmp_path)
    output = tmp_path / "sft"
    dev = list(read_rows(output / "language_dev.jsonl"))
    train = [
        row for row in read_rows(output / "train.jsonl") if row.source == "OpenAssistant/oasst2"
    ]

    def key(name: str) -> bytes:
        return hashlib.sha256(f"language-dev-v1|{name}".encode()).digest()

    ordered = sorted((f"tree-{index}" for index in range(10)), key=key)
    assert [row.id for row in dev] == ordered[:3]
    assert {row.id for row in train} <= set(ordered[3:])
    tokens = record["tokens"]
    budget = tokens["code"]["tokens"] * 0.3 / 0.7
    assert tokens["language"]["tokens"] <= budget
    accepted = [name for name in ordered[3:] if name in {row.id for row in train}]
    assert accepted == ordered[3 : 3 + len(accepted)]  # A prefix of the hash order.
    language = record["language"]
    assert language["dev"] == 3 and language["train"] == len(train)
    assert language["train"] + language.get("over_share", 0) == 7
    shares = record["shares"]
    assert shares["language"] == pytest.approx(
        tokens["language"]["tokens"] / (tokens["language"]["tokens"] + tokens["code"]["tokens"])
    )
    assert language["over_share"] > 0 and shares["language_target_met"]
    first = next(row for row in train if row.id == accepted[0])
    index = int(accepted[0].split("-")[1])
    assert first == SFTRow.of(
        accepted[0],
        "OpenAssistant/oasst2",
        render_conversation(
            TOKENIZER,
            [
                Message("user", f"Question {index}?"),
                Message("assistant", "An answer. " * (index + 3)),
            ],
        ),
    )


def test_language_train_max_and_without_language(tmp_path: Path) -> None:
    record = run_build(tmp_path / "a", language_train_max=2, language_share=0.9)
    assert record["language"]["train"] == 2 and record["language"]["over_train_max"] == 5
    assert not record["shares"]["language_target_met"]
    alone = build(
        pool_train=pool(tmp_path / "b"),
        output=tmp_path / "b" / "sft",
        encoder=TOKENIZER,
        tokenizer_identity=IDENTITY,
        config=BuildConfig(sequence_length=400, rl_candidates=5),
    )
    assert alone["inputs"]["language"] is None and alone["tokens"]["language"]["rows"] == 0
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
    with pytest.raises(ValueError):
        BuildConfig(language_share=1.0)


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
        "message:failed_review": 1,
        "message:not_english": 1,
        "message:pii_label": 1,
        "message:synthetic": 1,
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


def spec_file(tmp_path: Path, **changes: Any) -> Path:
    raw = json.loads((ROOT / "configs" / "sft-language-oasst2.json").read_text())
    raw.update(changes)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    return path


class FakeHub(Hub):
    def __init__(self, files: Mapping[str, bytes]) -> None:
        super().__init__(None, sleep=lambda _: None)
        self.files = files
        self.calls = 0

    def json(self, url: str) -> Any:
        self.calls += 1
        revision = url.split("/revision/")[1].split("?")[0]
        return {
            "sha": revision,
            "siblings": [
                {
                    "rfilename": name,
                    "size": len(data),
                    "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                }
                for name, data in self.files.items()
            ],
        }

    def download(self, url: str, destination: Path, size: int, sha256: str) -> int:
        data = self.files[url.rsplit("/", 1)[1]]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return len(data)


def gz_trees(trees: list[dict[str, Any]]) -> bytes:
    return gzip.compress("".join(json.dumps(item) + "\n" for item in trees).encode())


def test_committed_spec_records_the_terms_finding_and_blocks_the_download(tmp_path: Path) -> None:
    spec = sft_data.load_language_spec(ROOT / "configs" / "sft-language-oasst2.json")
    assert spec["owner_ruling"] is None and spec["owner_ruling_on"] is None
    assert spec["dataset"] == "OpenAssistant/oasst2"
    assert "no LICENSE file" in spec["terms_finding"]
    hub = FakeHub({})
    with pytest.raises(PermissionError, match="owner"):
        sft_data.fetch(spec, tmp_path / "intake", hub_factory=lambda _: hub)
    assert hub.calls == 0 and not (tmp_path / "intake").exists()


def test_fetch_and_adapt_after_a_recorded_ruling(tmp_path: Path) -> None:
    name = "2023-11-05_oasst2_ready.trees.jsonl.gz"
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(
        spec_file(
            tmp_path,
            owner_ruling="approved",
            owner_ruling_on="2026-10-01",
            max_tokens=60,
            max_download_bytes=len(data),
        )
    )
    directory = tmp_path / "intake"
    receipt = sft_data.fetch(spec, directory, hub_factory=lambda _: FakeHub({name: data}))
    assert receipt["files"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert file_sha256(directory / "raw" / name) == receipt["files"][0]["sha256"]
    with pytest.raises(FileExistsError):
        sft_data.fetch(spec, directory, hub_factory=lambda _: FakeHub({name: data}))
    record = sft_data.adapt_language(
        spec,
        directory,
        encoder=TOKENIZER,
        benchmarks=[],
        choices=[],
        holdouts=[task(OCI, 2, prompt="Pick? an option", split="dev")],
        min_alpha_tokens=3,
    )
    assert record["intake"][name]["tree:kept"] == 3
    assert record["screens"] == {"kept": 3, "seen": 3}
    path = directory / "language" / "conversations.jsonl"
    assert record["files"]["conversations.jsonl"]["sha256"] == file_sha256(path)
    assert [json.loads(line)["id"] for line in path.read_text().split("\n") if line] == [
        "t1",
        "t6",
        "t8",
    ]
    assert record["exchanges_per_conversation"] == {"1": 2, "2": 1}
    with pytest.raises(FileExistsError):
        sft_data.adapt_language(
            spec,
            directory,
            encoder=TOKENIZER,
            benchmarks=[],
            choices=[],
            holdouts=[],
            min_alpha_tokens=3,
        )


def test_fetch_refuses_oversized_plans_and_bad_specs(tmp_path: Path) -> None:
    name = "2023-11-05_oasst2_ready.trees.jsonl.gz"
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(
        spec_file(
            tmp_path,
            owner_ruling="approved",
            owner_ruling_on="2026-10-01",
            max_download_bytes=len(data) - 1,
        )
    )
    with pytest.raises(RuntimeError, match="approved maximum"):
        sft_data.fetch(spec, tmp_path / "x", hub_factory=lambda _: FakeHub({name: data}))
    with pytest.raises(ValueError, match="allowlist"):
        sft_data.load_language_spec(spec_file(tmp_path, license_label="CC-BY-NC-4.0"))
    with pytest.raises(ValueError):
        sft_data.load_language_spec(spec_file(tmp_path, files=["data/train.parquet"]))
    with pytest.raises(ValueError):
        sft_data.load_language_spec(
            spec_file(tmp_path, owner_ruling="approved", owner_ruling_on="2999-01-01")
        )


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
    name = "2023-11-05_oasst2_ready.trees.jsonl.gz"
    data = gz_trees(authored_trees())
    spec = sft_data.load_language_spec(
        spec_file(tmp_path, owner_ruling="rejected", owner_ruling_on="2026-10-01")
    )
    hub = FakeHub({name: data})
    with pytest.raises(PermissionError, match="did not approve"):
        sft_data.fetch(spec, tmp_path / "intake", hub_factory=lambda _: hub)
    assert hub.calls == 0 and not (tmp_path / "intake").exists()
    assert asdict(BuildConfig())["language_dev"] == 300
