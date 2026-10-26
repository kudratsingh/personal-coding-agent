"""Hash-bound SFT data for the rehearsal: chat-v1 rows, RL candidates and dev sets.

Three stages, each refusing to overwrite a completed result:

``download``
    Fetch the language source's pinned files with size and SHA-256 receipts. It
    refuses until the spec records the owner's ruling on the source's terms.
``language``
    Adapt the downloaded OpenAssistant trees to chat-v1 conversations
    (``scratchcoder.data.oasst``) and screen them: e-mail addresses, benchmark and
    ARC-Easy n-gram matches, pool dev/test matches, and near-duplicate prompts.
``build``
    From the pool's train file and the language conversations:

    - hold out RL candidates by ID hash, stratified by source in train proportions;
    - render every other train task's verified reference in chat-v1;
    - hold out a language dev set by ID hash and cap the language rows at the target
      share of assistant-plus-context tokens;
    - write the rows, the candidate list, the dev sets and a record with counts,
      shares, drops and SHA-256 values.

AceCode is flagged and stays out unless ``--include-acecode`` is given. Nothing here
executes code or spends money.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import TextEncoder, render_code_task, render_conversation
from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.manifest import TRAINING_LICENSES
from scratchcoder.data.oasst import (
    Conversation,
    adapt_trees,
    conversation_from_json,
    conversation_json,
    has_email,
)
from scratchcoder.data.pool import SimilarityIndex
from scratchcoder.data.sft import SFTRow, fits, write_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task, load_tasks
from tools.rehearsal_build import Hub, _accepted, _exact_keys, _hex, _positive, _text

SCHEMA_VERSION = 1
CODE_SOURCES = ("nvidia/OpenCodeInstruct", "BAAI/TACO", "google-research-datasets/mbpp")
ACECODE = "TIGER-Lab/AceCode-87K"
"""Flagged under the owner's generator-terms review; used only by explicit option."""
RL_CANDIDATES = 1000
RL_CANDIDATES_WITH_ACECODE = 2000
RL_SALT = "rl-candidate-v1"
LANGUAGE_SALT = "language-dev-v1"


# ---------------------------------------------------------------------------- build


@dataclass(frozen=True)
class BuildConfig:
    sequence_length: int = 1024
    include_acecode: bool = False
    rl_candidates: int | None = None
    """None means 1,000 without AceCode and 2,000 with it."""
    language_dev: int = 300
    language_train_max: int = 3000
    language_share: float = 0.30
    """Target language share of assistant-plus-context tokens in the training rows."""

    def __post_init__(self) -> None:
        if type(self.include_acecode) is not bool:
            raise ValueError("include_acecode must be a boolean")
        for name in ("sequence_length", "language_train_max"):
            _positive(getattr(self, name), name)
        if type(self.language_dev) is not int or self.language_dev < 0:
            raise ValueError("language_dev must be a nonnegative integer")
        if self.rl_candidates is not None and (
            type(self.rl_candidates) is not int or self.rl_candidates < 0
        ):
            raise ValueError("rl_candidates must be a nonnegative integer or None")
        if not isinstance(self.language_share, float) or not 0 <= self.language_share < 1:
            raise ValueError("language_share must be a float in [0, 1)")

    @property
    def sources(self) -> tuple[str, ...]:
        return (*CODE_SOURCES, ACECODE) if self.include_acecode else CODE_SOURCES

    @property
    def candidates(self) -> int:
        if self.rl_candidates is not None:
            return self.rl_candidates
        return RL_CANDIDATES_WITH_ACECODE if self.include_acecode else RL_CANDIDATES


def hash_order(identifier: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}|{identifier}".encode()).digest()


def quotas(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Largest-remainder shares of ``total`` in proportion to ``sizes``.

    Floors first; the remaining units go to the largest fractional parts, ties to
    the source name in sorted order.
    """
    available = sum(sizes.values())
    if total > available:
        raise ValueError(f"{total} candidates requested from {available} tasks")
    if total == 0:
        return dict.fromkeys(sorted(sizes), 0)
    exact = {name: total * size / available for name, size in sizes.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    left = total - sum(result.values())
    for name in sorted(exact, key=lambda item: (-(exact[item] - result[item]), item))[:left]:
        result[name] += 1
    return dict(sorted(result.items()))


def stratified_holdout(
    tasks: Sequence[Task], total: int, *, salt: str = RL_SALT
) -> tuple[list[Task], dict[str, int]]:
    """``total`` tasks stratified by source, each source's lowest ``sha256(salt|id)``."""
    by_source: dict[str, list[Task]] = {}
    for task in tasks:
        by_source.setdefault(task.source, []).append(task)
    shares = quotas({name: len(items) for name, items in by_source.items()}, total)
    chosen: list[Task] = []
    for name in sorted(by_source):
        ordered = sorted(by_source[name], key=lambda task: hash_order(task.id, salt))
        chosen.extend(ordered[: shares[name]])
    return chosen, shares


def _domain(rows: Sequence[SFTRow]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "tokens": sum(len(row.ids) for row in rows),
        "assistant_tokens": sum(row.assistant_tokens for row in rows),
    }


def _file(path: Path, rows: int) -> dict[str, Any]:
    return {"sha256": file_sha256(path), "rows": rows, "bytes": path.stat().st_size}


def read_conversations(path: Path) -> list[Conversation]:
    conversations = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                conversation = conversation_from_json(json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid conversation on line {number}") from error
            if conversation.id in seen:
                raise ValueError(f"duplicate conversation id: {conversation.id}")
            seen.add(conversation.id)
            conversations.append(conversation)
    return conversations


def build(
    *,
    pool_train: Path,
    output: Path,
    encoder: TextEncoder,
    tokenizer_identity: str,
    config: BuildConfig,
    language: Path | None = None,
    pool_dev: Path | None = None,
) -> dict[str, Any]:
    """Write the SFT data set into the new directory ``output``; return its record."""
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    _hex(tokenizer_identity, 64, "tokenizer identity")
    tasks = list(load_tasks(pool_train))
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("the pool train file must hold only train-split tasks")
    seen_sources = Counter(task.source for task in tasks)
    allowed = [task for task in tasks if task.source in config.sources]
    excluded = {name: count for name, count in seen_sources.items() if name not in config.sources}
    candidates, shares = stratified_holdout(allowed, config.candidates)
    held = {task.id for task in candidates}

    code_rows: list[SFTRow] = []
    code_drops: Counter[str] = Counter()
    for task in sorted(allowed, key=lambda item: item.id):
        if task.id in held:
            continue
        if task.solution is None:
            code_drops["no_reference"] += 1
            continue
        row = SFTRow.of(task.id, task.source, render_code_task(encoder, task))
        if not fits(row, config.sequence_length):
            code_drops["over_length"] += 1
            continue
        code_rows.append(row)

    language_rows: list[SFTRow] = []
    dev_rows: list[SFTRow] = []
    language_counts: Counter[str] = Counter()
    conversations = read_conversations(language) if language is not None else []
    fitting: list[SFTRow] = []
    for conversation in conversations:
        language_counts["seen"] += 1
        rendered = render_conversation(encoder, conversation.messages)
        row = SFTRow.of(conversation.id, conversation.source, rendered)
        if not fits(row, config.sequence_length):
            language_counts["over_length"] += 1
            continue
        fitting.append(row)
    fitting.sort(key=lambda row: hash_order(row.id, LANGUAGE_SALT))
    if language is not None and len(fitting) < config.language_dev:
        raise ValueError(f"{len(fitting)} conversations cannot fill the language dev set")
    dev_rows = fitting[: config.language_dev]
    code_tokens = sum(len(row.ids) for row in code_rows)
    share = config.language_share
    budget = code_tokens * share / (1 - share)
    # The training conversations are a prefix of the hash order: the first one that
    # would pass the token budget or the count cap ends the selection.
    used, stop = 0, ""
    for row in fitting[config.language_dev :]:
        if not stop and len(language_rows) == config.language_train_max:
            stop = "over_train_max"
        elif not stop and used + len(row.ids) > budget:
            stop = "over_share"
        if stop:
            language_counts[stop] += 1
            continue
        language_rows.append(row)
        used += len(row.ids)
    language_counts["dev"] = len(dev_rows)
    language_counts["train"] = len(language_rows)

    code_dev_rows: list[SFTRow] = []
    code_dev_drops: Counter[str] = Counter()
    if pool_dev is not None:
        for task in sorted(load_tasks(pool_dev), key=lambda item: item.id):
            if task.split != "dev" or task.solution is None:
                raise ValueError("the pool dev file must hold dev tasks with references")
            row = SFTRow.of(task.id, task.source, render_code_task(encoder, task))
            if not fits(row, config.sequence_length):
                code_dev_drops["over_length"] += 1
                continue
            code_dev_rows.append(row)

    train = sorted(code_rows + language_rows, key=lambda row: row.id)
    if len({row.id for row in train}) != len(train):
        raise ValueError("code task and conversation IDs collide")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_rows(staging / "train.jsonl", train)
    write_rows(staging / "language_dev.jsonl", dev_rows)
    if pool_dev is not None:
        write_rows(staging / "code_dev.jsonl", code_dev_rows)
    with (staging / "rl_candidates.jsonl").open("x", encoding="utf-8") as stream:
        for task in candidates:
            stream.write(json.dumps({"id": task.id, "source": task.source}, sort_keys=True) + "\n")
    files = {
        "train.jsonl": _file(staging / "train.jsonl", len(train)),
        "language_dev.jsonl": _file(staging / "language_dev.jsonl", len(dev_rows)),
        "rl_candidates.jsonl": _file(staging / "rl_candidates.jsonl", len(candidates)),
    }
    if pool_dev is not None:
        files["code_dev.jsonl"] = _file(staging / "code_dev.jsonl", len(code_dev_rows))
    code, words = _domain(code_rows), _domain(language_rows)
    total = code["tokens"] + words["tokens"]
    total_assistant = code["assistant_tokens"] + words["assistant_tokens"]
    commit, dirty = git_state()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "protocol": "chat-v1",
        "tokenizer_identity": tokenizer_identity,
        "config": {**asdict(config), "sources": list(config.sources)},
        "inputs": {
            "pool_train": {"path": str(pool_train), "sha256": file_sha256(pool_train)},
            "language": None
            if language is None
            else {"path": str(language), "sha256": file_sha256(language)},
            "pool_dev": None
            if pool_dev is None
            else {"path": str(pool_dev), "sha256": file_sha256(pool_dev)},
        },
        "code": {
            "train_tasks_by_source": dict(sorted(seen_sources.items())),
            "excluded_sources": dict(sorted(excluded.items())),
            "rl_candidates": {"total": len(candidates), "by_source": shares, "salt": RL_SALT},
            "sft_rows_by_source": dict(sorted(Counter(r.source for r in code_rows).items())),
            "dropped": dict(sorted(code_drops.items())),
            "dev_rows": len(code_dev_rows),
            "dev_dropped": dict(sorted(code_dev_drops.items())),
        },
        "language": {**dict(sorted(language_counts.items())), "salt": LANGUAGE_SALT},
        "tokens": {"code": code, "language": words},
        "shares": {
            "basis": "assistant-plus-context tokens (every token of a row)",
            "target_language": share,
            "language": words["tokens"] / total if total else 0.0,
            "code": code["tokens"] / total if total else 0.0,
            "language_assistant_only": (
                words["assistant_tokens"] / total_assistant if total_assistant else 0.0
            ),
            "language_target_met": bool(language_counts["over_share"]),
        },
        "files": files,
        "paid_runs": 0,
    }
    atomic_json(staging / "record.json", record)
    staging.replace(output)
    return record


# ------------------------------------------------------------------------- language


def load_language_spec(path: Path) -> dict[str, Any]:
    """Parse the language-source spec; unknown or missing keys are errors."""
    spec = _exact_keys(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "dataset",
            "revision",
            "files",
            "license_label",
            "terms_finding",
            "terms_reviewed_on",
            "owner_ruling_on",
            "max_download_bytes",
            "max_tokens",
        },
        "language spec",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported language spec schema")
    _text(spec, ("dataset", "terms_finding"), "language spec")
    _hex(spec["revision"], 40, "language revision")
    _accepted(spec["terms_reviewed_on"], "language terms review")
    if spec["owner_ruling_on"] is not None:
        _accepted(spec["owner_ruling_on"], "owner ruling")
    if spec["license_label"] not in TRAINING_LICENSES:
        raise ValueError("the language license label is not on the D-05 allowlist")
    _positive(spec["max_download_bytes"], "max_download_bytes")
    _positive(spec["max_tokens"], "max_tokens")
    files = spec["files"]
    if (
        not isinstance(files, list)
        or not files
        or any(not isinstance(name, str) or not name.endswith(".trees.jsonl.gz") for name in files)
    ):
        raise ValueError("the language spec lists OpenAssistant .trees.jsonl.gz files")
    spec["sha256"] = file_sha256(path)
    return dict(spec)


def fetch(
    spec: Mapping[str, Any],
    directory: Path,
    *,
    hub_factory: Callable[[str | None], Hub] = Hub,
) -> dict[str, Any]:
    """Download the spec's files at the pinned revision; refuse without a ruling."""
    if spec["owner_ruling_on"] is None:
        raise PermissionError(
            "the source's terms await the owner's ruling (D-05); nothing was downloaded"
        )
    receipts_path = directory / "download-receipts.json"
    if receipts_path.exists():
        raise FileExistsError("downloads are already recorded for this directory")
    hub = hub_factory(None)
    info = hub.json(
        f"https://huggingface.co/api/datasets/{spec['dataset']}/revision/"
        f"{spec['revision']}?blobs=true"
    )
    if info.get("sha") != spec["revision"]:
        raise ValueError("the pinned revision resolved differently")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    planned = []
    for name in spec["files"]:
        item = siblings.get(name)
        if item is None or not item.get("lfs"):
            raise ValueError(f"{name} is not at the pinned revision")
        planned.append((name, int(item["size"]), str(item["lfs"]["sha256"])))
    listed = sum(size for _, size, _ in planned)
    if listed > spec["max_download_bytes"]:
        raise RuntimeError(f"the plan lists {listed:,} bytes, above the approved maximum")
    files = []
    for name, size, sha256 in planned:
        target = directory / "raw" / name
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        url = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{name}"
        moved = hub.download(url, target, size, sha256)
        if target.stat().st_size != size or file_sha256(target) != sha256:
            raise ValueError(f"{target}: bytes differ after publication")
        files.append(
            {"filename": name, "url": url, "size_bytes": size, "sha256": sha256, "moved": moved}
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "spec_sha256": spec["sha256"],
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "files": files,
        "listed_bytes": listed,
        "paid_runs": 0,
    }
    atomic_json(receipts_path, report)
    return report


def read_trees(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def screen_conversations(
    conversations: Iterable[Conversation],
    *,
    benchmarks: Sequence[Task],
    choices: Sequence[ChoiceTask],
    holdouts: Sequence[Task],
    min_alpha_tokens: int,
) -> tuple[list[Conversation], dict[str, int]]:
    """The pool's contamination screens, applied to every message of a conversation.

    In order: an e-mail address anywhere drops it; then n-gram matches against the
    benchmarks' prompts, solutions and tests, ARC-Easy questions and choices, and the
    pool dev/test tasks; then a first user turn that near-duplicates a benchmark or
    holdout prompt.
    """
    protected = [*benchmarks, *holdouts]
    texts = [
        part
        for task in protected
        for part in (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)
    ]
    texts.extend(text for task in choices for text in (task.question, task.public_prompt()))
    contamination = ContaminationIndex(texts, min_alpha_tokens=min_alpha_tokens)
    duplicates = SimilarityIndex()
    for task in protected:
        duplicates.add(task.prompt)
    counts: Counter[str] = Counter()
    kept = []
    for conversation in conversations:
        counts["seen"] += 1
        if has_email(conversation):
            counts["email_address"] += 1
        elif any(contamination.contains(message.text) for message in conversation.messages):
            counts["contamination"] += 1
        elif duplicates.contains(conversation.messages[0].text):
            counts["near_duplicate"] += 1
        else:
            kept.append(conversation)
            counts["kept"] += 1
    return kept, dict(sorted(counts.items()))


def adapt_language(
    spec: Mapping[str, Any],
    directory: Path,
    *,
    encoder: TextEncoder,
    benchmarks: Sequence[Task],
    choices: Sequence[ChoiceTask],
    holdouts: Sequence[Task],
    min_alpha_tokens: int,
) -> dict[str, Any]:
    """Adapt and screen the downloaded trees into ``directory/language``."""
    output = directory / "language"
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    receipts = json.loads((directory / "download-receipts.json").read_text())
    if receipts["status"] != "completed" or receipts["spec_sha256"] != spec["sha256"]:
        raise ValueError("download receipts do not match this spec")
    adapted: list[Conversation] = []
    intake: dict[str, dict[str, int]] = {}
    for item in receipts["files"]:
        path = directory / "raw" / item["filename"]
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"{path} differs from its receipt")
        kept, counts = adapt_trees(read_trees(path), encoder, max_tokens=spec["max_tokens"])
        adapted.extend(kept)
        intake[item["filename"]] = counts
    screened, screens = screen_conversations(
        adapted,
        benchmarks=benchmarks,
        choices=choices,
        holdouts=holdouts,
        min_alpha_tokens=min_alpha_tokens,
    )
    staging = directory / "language.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    path = staging / "conversations.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for conversation in sorted(screened, key=lambda item: item.id):
            stream.write(json.dumps(conversation_json(conversation), sort_keys=True) + "\n")
    turns = Counter(len(item.messages) // 2 for item in screened)
    commit, dirty = git_state()
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "spec_sha256": spec["sha256"],
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "max_tokens": spec["max_tokens"],
        "intake": intake,
        "screens": screens,
        "exchanges_per_conversation": {str(key): turns[key] for key in sorted(turns)},
        "tokens": sum(item.tokens for item in screened),
        "files": {"conversations.jsonl": _file(path, len(screened))},
        "paid_runs": 0,
    }
    atomic_json(staging / "record.json", record)
    staging.replace(output)
    return record


# ----------------------------------------------------------------------------- CLI


def _encoder(pool_spec: Mapping[str, Any]) -> tuple[TextEncoder, str]:
    from scratchcoder.tokenizer.bulk import FastEncoder
    from tools.pilot_data import _load_tokenizer

    tokenizer = pool_spec["tokenizer"]
    loaded = _load_tokenizer(
        Path(tokenizer["path"]),
        expected_identity=tokenizer["identity"],
        expected_file_sha256=tokenizer["file_sha256"],
        expected_vocab_size=tokenizer["vocab_size"],
        expected_bos_id=256,
    )
    return FastEncoder(loaded), loaded.sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "language", "build"))
    parser.add_argument("--pool-spec", type=Path, help="task-pool spec (tokenizer, protection)")
    parser.add_argument("--language-spec", type=Path, help="language source spec")
    parser.add_argument("--directory", type=Path, help="language intake directory")
    parser.add_argument("--pool-dir", type=Path, help="pool directory with train/dev/test")
    parser.add_argument("--language", type=Path, help="screened conversations.jsonl")
    parser.add_argument("--output", type=Path, help="new SFT data directory")
    parser.add_argument("--include-acecode", action="store_true", help="use the flagged source")
    parser.add_argument("--rl-candidates", type=int)
    args = parser.parse_args(argv)
    if args.stage == "download":
        if args.language_spec is None or args.directory is None:
            parser.error("download needs --language-spec and --directory")
        result = fetch(load_language_spec(args.language_spec), args.directory)
        print(json.dumps({"listed_bytes": result["listed_bytes"]}))
        return 0
    if args.pool_spec is None or args.pool_dir is None:
        parser.error(f"{args.stage} needs --pool-spec and --pool-dir")
    from tools.task_pool import load_spec, protection

    pool_spec = load_spec(args.pool_spec)
    encoder, identity = _encoder(pool_spec)
    if args.stage == "language":
        if args.language_spec is None or args.directory is None:
            parser.error("language needs --language-spec and --directory")
        benchmarks, choices, _ = protection(pool_spec)
        holdouts = [
            *load_tasks(args.pool_dir / "dev.jsonl"),
            *load_tasks(args.pool_dir / "test.jsonl"),
        ]
        record = adapt_language(
            load_language_spec(args.language_spec),
            args.directory,
            encoder=encoder,
            benchmarks=benchmarks,
            choices=choices,
            holdouts=holdouts,
            min_alpha_tokens=pool_spec["protection"]["min_alpha_tokens"],
        )
        print(json.dumps({"screens": record["screens"], "tokens": record["tokens"]}, indent=1))
        return 0
    if args.output is None:
        parser.error("build needs --output")
    record = build(
        pool_train=args.pool_dir / "train.jsonl",
        pool_dev=args.pool_dir / "dev.jsonl",
        language=args.language,
        output=args.output,
        encoder=encoder,
        tokenizer_identity=identity,
        config=BuildConfig(include_acecode=args.include_acecode, rl_candidates=args.rl_candidates),
    )
    print(json.dumps({key: record[key] for key in ("code", "language", "shares")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
