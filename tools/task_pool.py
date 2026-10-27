"""Build the capped rehearsal task pool: pinned sources, sandbox-verified references, holdouts.

Two stages share one pool directory; each refuses to overwrite a completed record.

``download``
    Resolve each Hugging Face source's pinned revision, record every spec file's size
    and LFS SHA-256, and fetch it once with a streaming hash check
    (``tools.rehearsal_build.Hub``). A spec ``local_copies`` entry is hard-linked
    instead, after the same size and hash check. MBPP is one pinned GitHub file.
``build``
    Adapt each source's rows to the shared task schema with the reviewed adapters. Then:

    - drop MBPP+ IDs, TACO components without known rights, tasks whose public prompt
      plus reference solution do not fit the context, and tasks too large for the
      sandbox;
    - drop ARC-Easy text matches, then let ``scratchcoder.data.pool.build_pool`` drop
      benchmark IDs, near-duplicates and benchmark n-gram matches;
    - admit a task only if its reference passes its public and its hidden tests in the
      disposable sandbox.

    Verification runs in parallel threads ahead of ``build_pool``, which reads the cached
    results in its own order. The dev and test holdouts are TACO test-split tasks
    divided evenly (topped up from MBPP if fewer than the spec's minimum), then
    model-written tasks by ID hash.

Generated and reference programs only ever run inside ``scratchcoder.sandbox.runner``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.adapters import (
    IO_CASE_LIMIT,
    adapt_row,
    adapt_taco_io_row,
    canonical_id,
    clean_completion,
)
from scratchcoder.data.arc import read_arc_easy
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.manifest import TRAINING_LICENSES, ManifestEntry, download
from scratchcoder.data.mbpp import mbpp_release, read_mbpp
from scratchcoder.data.pool import SimilarityIndex, build_pool, shingles, split_summary
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.sandbox.budget import io_seconds
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import (
    HOST_WATCHDOG_SECONDS,
    IMAGE,
    Limits,
    Result,
    Runtime,
    run,
)
from scratchcoder.tokenizer.bulk import FastEncoder
from tools.pilot_data import _load_tokenizer
from tools.real_samples import HUMANEVAL_SHA256
from tools.rehearsal_build import Hub, _accepted, _exact_keys, _hex, _positive, _text

SCHEMA_VERSION = 1
KINDS = ("opencode", "acecode", "taco", "mbpp")
DATASETS = {
    "opencode": "nvidia/OpenCodeInstruct",
    "acecode": "TIGER-Lab/AceCode-87K",
    "taco": "BAAI/TACO",
    "mbpp": "google-research-datasets/mbpp",
}
WRITTEN_BY = {"opencode": "model", "acecode": "model", "taco": "human", "mbpp": "human"}
COLUMNS = {
    "opencode": ["id", "input", "output", "unit_tests"],
    "acecode": ["id", "source", "question", "test_cases", "inferences"],
    "taco": [
        "question",
        "solutions",
        "starter_code",
        "input_output",
        "difficulty",
        "source",
        "url",
        "picture_num",
    ],
}
# A sandbox call that reports one of these never decides a task; it is retried and,
# if it persists, the build stops rather than guessing.
RETRY_STATUSES = frozenset({"infrastructure_error", "unavailable"})
RETRY_DELAY_SECONDS = 2.0
HOST_DEADLINE_SLACK = HOST_WATCHDOG_SECONDS
"""runner.run gives up this long after the candidate deadline: an infrastructure stall. A
fresh result says so itself (``Result.host_stalled``); this elapsed-time signature also
recognises stalls in cache rows written before that flag existed."""
LIMITS_FORMULA_VERSION = 1
"""Bump when limits_for changes; cached verdicts are bound to it."""
ACECODE_ATTEMPTS = 3
"""Recorded AceCode attempts with an upstream pass rate of 1.0 tried in our sandbox."""
VERDICT_STATUSES = frozenset({"passed", "failed", "timeout", "output_limit", "resource_limit"})


def load_spec(path: Path) -> dict[str, Any]:
    """Parse and validate a pool spec; unknown or missing keys are errors."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    spec = _exact_keys(
        raw,
        {
            "schema_version",
            "name",
            "tokenizer",
            "context_tokens",
            "context_basis",
            "sources",
            "protection",
            "splits",
            "verification",
            "local_copies",
            "max_download_bytes",
        },
        "spec",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported pool spec schema")
    _text(spec, ("name", "context_basis"), "spec")
    tokenizer = _exact_keys(
        spec["tokenizer"], {"path", "identity", "file_sha256", "vocab_size"}, "tokenizer"
    )
    _positive(tokenizer["vocab_size"], "tokenizer vocab_size")
    _hex(tokenizer["identity"], 64, "tokenizer identity")
    _hex(tokenizer["file_sha256"], 64, "tokenizer file_sha256")
    _positive(spec["context_tokens"], "context_tokens")
    _positive(spec["max_download_bytes"], "max_download_bytes")
    kinds = []
    for source in spec["sources"]:
        keys = {
            "kind",
            "dataset",
            "revision",
            "files",
            "rows",
            "license",
            "attribution",
            "terms",
            "terms_reviewed_on",
            "obligations",
        }
        extra = {
            "taco": {"difficulty", "excluded_sources", "truncate_cases"},
            "mbpp": {"url", "sha256", "size_bytes"},
        }
        _exact_keys(source, keys | extra.get(source.get("kind"), set()), "source")
        kind = source["kind"]
        if kind not in KINDS or source["dataset"] != DATASETS[kind]:
            raise ValueError("unknown source kind or dataset")
        _text(source, ("license", "attribution", "terms", "obligations"), kind)
        _hex(source["revision"], 40, f"{kind} revision")
        _accepted(source["terms_reviewed_on"], kind)
        if source["license"] not in TRAINING_LICENSES:
            raise ValueError(f"{kind}: dataset license is not on the D-05 allowlist")
        if source["rows"] is not None:
            _positive(source["rows"], f"{kind} rows")
        files = source["files"]
        if not isinstance(files, list) or (kind == "mbpp") != (not files):
            raise ValueError(f"{kind}: Hub sources list files; MBPP lists none")
        if any(not isinstance(name, str) or not name.endswith(".parquet") for name in files):
            raise ValueError(f"{kind}: files must be Parquet paths")
        if kind == "taco" and (
            source["difficulty"] != "EASY"
            or type(source["truncate_cases"]) is not bool
            or not isinstance(source["excluded_sources"], list)
            or not all(isinstance(item, str) for item in source["excluded_sources"])
        ):
            raise ValueError("TACO is EASY only, with an explicit excluded-source list")
        if kind == "mbpp":
            _hex(source["sha256"], 64, "MBPP sha256")
            _positive(source["size_bytes"], "MBPP size_bytes")
        kinds.append(kind)
    if sorted(kinds) != sorted(KINDS):
        raise ValueError("the spec needs exactly one source of each kind")
    protection_spec = _exact_keys(
        spec["protection"],
        {"humaneval", "evalplus_directory", "arc_directory", "min_alpha_tokens"},
        "protection",
    )
    _positive(protection_spec["min_alpha_tokens"], "protection min_alpha_tokens")
    splits = _exact_keys(
        spec["splits"],
        {"dev", "test", "human_holdout_minimum", "model_holdout_sources", "rl_prompt_tokens"},
        "splits",
    )
    for key in ("dev", "test", "human_holdout_minimum", "rl_prompt_tokens"):
        _positive(splits[key], f"splits {key}")
    sources = splits["model_holdout_sources"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(source not in KINDS or WRITTEN_BY[source] != "model" for source in sources)
    ):
        raise ValueError("model_holdout_sources must name model-written source kinds")
    verification = _exact_keys(
        spec["verification"],
        {"timeout_seconds", "io_seconds_per_case", "memory_mb", "workers"},
        "verification",
    )
    _positive(verification["workers"], "verification workers")
    if not 0 <= verification["io_seconds_per_case"] <= 1:
        raise ValueError("io_seconds_per_case must be between 0 and 1")
    Limits(timeout_seconds=verification["timeout_seconds"], memory_mb=verification["memory_mb"])
    if not isinstance(spec["local_copies"], dict):
        raise ValueError("local_copies maps dataset:file to a local path")
    spec["sha256"] = file_sha256(path)
    return dict(spec)


def _source(spec: Mapping[str, Any], kind: str) -> dict[str, Any]:
    found: dict[str, Any] = next(item for item in spec["sources"] if item["kind"] == kind)
    return found


def _verified(path: Path, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and file_sha256(path) == sha256


def fetch(
    spec: Mapping[str, Any],
    directory: Path,
    *,
    hub_factory: Callable[[str | None], Hub] = Hub,
    mbpp_download: Callable[[ManifestEntry, Path], Path] = download,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    receipts_path = directory / "download-receipts.json"
    if receipts_path.exists():
        prior = json.loads(receipts_path.read_text())
        if prior["status"] == "completed" and prior["spec_sha256"] == spec["sha256"]:
            raise FileExistsError("downloads are already complete for this spec")
    started = time.perf_counter()
    hub = hub_factory(None)
    planned: list[dict[str, Any]] = []
    for source in spec["sources"]:
        if source["kind"] == "mbpp":
            planned.append(
                {
                    "kind": "mbpp",
                    "dataset": source["dataset"],
                    "revision": source["revision"],
                    "filename": "mbpp.jsonl",
                    "size_bytes": source["size_bytes"],
                    "sha256": source["sha256"],
                    "url": source["url"],
                }
            )
            continue
        url = (
            f"https://huggingface.co/api/datasets/{source['dataset']}/revision/"
            f"{source['revision']}?blobs=true"
        )
        info = hub.json(url)
        if info.get("sha") != source["revision"]:
            raise ValueError(f"{source['dataset']}: the pinned revision resolved differently")
        siblings = {item["rfilename"]: item for item in info["siblings"]}
        for name in source["files"]:
            item = siblings.get(name)
            if item is None or not item.get("lfs"):
                raise ValueError(f"{source['dataset']}: {name} is not at the pinned revision")
            planned.append(
                {
                    "kind": source["kind"],
                    "dataset": source["dataset"],
                    "revision": source["revision"],
                    "filename": name,
                    "size_bytes": int(item["size"]),
                    "sha256": str(item["lfs"]["sha256"]),
                    "url": (
                        f"https://huggingface.co/datasets/{source['dataset']}/resolve/"
                        f"{source['revision']}/{name}"
                    ),
                }
            )
    listed = sum(item["size_bytes"] for item in planned)
    if listed > spec["max_download_bytes"]:
        raise RuntimeError(f"the plan lists {listed:,} bytes, above the approved maximum")
    files = []
    for item in planned:
        target = directory / "raw" / item["kind"] / item["filename"]
        began = time.perf_counter()
        local = spec["local_copies"].get(f"{item['dataset']}:{item['filename']}")
        if target.exists():
            if not _verified(target, item["size_bytes"], item["sha256"]):
                raise ValueError(f"{target} exists with different bytes")
            method, moved = "present", 0
        elif local is not None and _verified(Path(local), item["size_bytes"], item["sha256"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(local, target)
            method, moved = "hard-linked", 0
        elif item["kind"] == "mbpp":
            entry = replace(mbpp_release().manifest, url=item["url"])
            mbpp_download(entry, target)
            method, moved = "downloaded", item["size_bytes"]
        else:
            moved = hub.download(item["url"], target, item["size_bytes"], item["sha256"])
            method = "downloaded"
        if not _verified(target, item["size_bytes"], item["sha256"]):
            raise ValueError(f"{target}: bytes differ after publication")
        files.append(
            {
                **item,
                "path": str(target),
                "method": method,
                "bytes_transferred": moved,
                "verified": True,
                "elapsed_seconds": time.perf_counter() - began,
            }
        )
        progress(f"{method} {item['kind']} {item['filename']} ({item['size_bytes']:,} bytes)")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "spec_sha256": spec["sha256"],
        "files": files,
        "listed_bytes": listed,
        "bytes_transferred": sum(item["bytes_transferred"] for item in files),
        "elapsed_seconds": time.perf_counter() - started,
        "paid_runs": 0,
    }
    atomic_json(receipts_path, report)
    return report


def _provenance(source: Mapping[str, Any], receipt: Mapping[str, Any]) -> ManifestEntry:
    return ManifestEntry(
        source=source["dataset"],
        revision=source["revision"],
        url=receipt["url"],
        license=source["license"],
        attribution=source["attribution"],
        sha256=receipt["sha256"],
        size_bytes=receipt["size_bytes"],
        purpose="train",
        license_verified=True,
    )


def _parquet_rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    parquet = importlib.import_module("pyarrow.parquet")
    with parquet.ParquetFile(path, arrow_extensions_enabled=False) as handle:
        for batch in handle.iter_batches(batch_size=512, columns=list(columns)):
            yield from batch.to_pylist()


def candidates(
    spec: Mapping[str, Any], receipts: Mapping[str, Any], benchmarks: Sequence[Task]
) -> tuple[list[Task], dict[str, Counter[str]], dict[str, Any], dict[str, list[str]]]:
    """Adapt every selected row; count each row's intake outcome per source.

    Also returns, per AceCode task, its first ``ACECODE_ATTEMPTS`` distinct recorded
    completions with an upstream pass rate of 1.0, in upstream order.
    """
    alternatives: dict[str, list[str]] = {}
    counts: dict[str, Counter[str]] = {kind: Counter() for kind in KINDS}
    tasks: list[Task] = []
    details: dict[str, Any] = {"taco_sources": Counter(), "acecode_seed_sources": Counter()}
    protected = {canonical_id(task.id) for task in benchmarks}
    by_kind: dict[str, list[Mapping[str, Any]]] = {kind: [] for kind in KINDS}
    for item in receipts["files"]:
        by_kind[item["kind"]].append(item)

    def admit(kind: str, make: Callable[..., Task], *args: Any, **kwargs: Any) -> None:
        try:
            tasks.append(make(*args, **kwargs))
            counts[kind]["adapted"] += 1
        except (ValueError, SyntaxError, KeyError, TypeError, json.JSONDecodeError):
            counts[kind]["unsupported_schema"] += 1

    for kind in ("opencode", "acecode"):
        source = _source(spec, kind)
        taken = 0
        for receipt in by_kind[kind]:
            provenance = _provenance(source, receipt)
            for row in _parquet_rows(Path(receipt["path"]), COLUMNS[kind]):
                if source["rows"] is not None and taken >= source["rows"]:
                    break
                taken += 1
                counts[kind]["selected_rows"] += 1
                if kind == "acecode":
                    details["acecode_seed_sources"][str(row.get("source"))] += 1
                before = len(tasks)
                admit(kind, adapt_row, DATASETS[kind], row, provenance)
                if kind == "acecode" and len(tasks) > before:
                    options = dict.fromkeys(
                        clean_completion(item["completion"])
                        for item in row["inferences"]
                        if item["pass_rate"] == 1.0
                    )
                    alternatives[tasks[-1].id] = list(options)[:ACECODE_ATTEMPTS]
        if source["rows"] is not None and taken < source["rows"]:
            raise ValueError(f"{kind}: files hold {taken} rows, fewer than {source['rows']}")
    taco = _source(spec, "taco")
    excluded = {item.casefold() for item in taco["excluded_sources"]}
    for receipt in by_kind["taco"]:
        provenance = _provenance(taco, receipt)
        upstream = "test" if "/test-" in receipt["filename"] else "train"
        for row in _parquet_rows(Path(receipt["path"]), COLUMNS["taco"]):
            counts["taco"][f"upstream_{upstream}_rows"] += 1
            if row["difficulty"] != taco["difficulty"]:
                counts["taco"]["not_easy"] += 1
                continue
            counts["taco"]["selected_rows"] += 1
            component = str(row.get("source") or "").casefold()
            details["taco_sources"][f"{upstream}:{component}"] += 1
            if component in excluded:
                counts["taco"]["excluded_component"] += 1
                continue
            # Upstream test tasks are holdouts only; "test" is a placeholder until split.
            split = "test" if upstream == "test" else "train"
            before = counts["taco"]["adapted"]
            admit(
                "taco",
                adapt_taco_io_row,
                row,
                provenance,
                upstream_split=upstream,
                split=split,
                truncate_cases=taco["truncate_cases"],
            )
            if counts["taco"]["adapted"] > before and _case_count(row) > IO_CASE_LIMIT:
                counts["taco"][f"{upstream}_cases_truncated_to_{IO_CASE_LIMIT}"] += 1
    mbpp = _source(spec, "mbpp")
    (receipt,) = by_kind["mbpp"]
    provenance = _provenance(mbpp, receipt)
    for row in read_mbpp(Path(receipt["path"]), mbpp_release()):
        counts["mbpp"]["selected_rows"] += 1
        if f"mbpp/{row['task_id']}" in protected:
            counts["mbpp"]["mbpp_plus_id_excluded"] += 1
            continue
        admit("mbpp", adapt_row, DATASETS["mbpp"], row, provenance)
    details["taco_sources"] = dict(sorted(details["taco_sources"].items()))
    details["acecode_seed_sources"] = dict(details["acecode_seed_sources"])
    identifiers = Counter(task.id for task in tasks)
    repeated = sorted(name for name, number in identifiers.items() if number > 1)
    if repeated:
        raise ValueError(f"task IDs repeat before verification: {repeated[:5]}")
    return tasks, counts, details, alternatives


def _case_count(row: Mapping[str, Any]) -> int:
    """Distinct inputs, which is what the adapter's truncation counts."""
    cases = json.loads(row["input_output"])
    inputs = cases.get("inputs") or [] if isinstance(cases, dict) else []
    return len(dict.fromkeys(inputs))


def _kind(task: Task) -> str:
    return next(kind for kind, dataset in DATASETS.items() if task.source == dataset)


def limits_for(verification: Mapping[str, Any], tests: str, mode: str) -> Limits:
    """The sandbox deadline covers every I/O case in one call, so it grows with the cases.

    The I/O rule is :func:`scratchcoder.sandbox.budget.io_seconds`, which the reward channel
    also uses, so both give a task the same deadline."""
    timeout = verification["timeout_seconds"]
    if mode == "io":
        timeout = io_seconds(
            len(json.loads(tests)),
            base_seconds=timeout,
            seconds_per_case=verification["io_seconds_per_case"],
        )
    return Limits(timeout_seconds=min(60.0, timeout), memory_mb=verification["memory_mb"])


def policy_hash(verification: Mapping[str, Any]) -> str:
    """Names the verdict cache: the limits settings, without the worker count."""
    policy = {key: value for key, value in verification.items() if key != "workers"}
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]


def runtime_policy(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Everything else a cached verdict depends on: image, sandbox code and limits."""
    sandbox = Path(__file__).resolve().parents[1] / "src" / "scratchcoder" / "sandbox"
    return {
        "runtime": dict(runtime),
        "runner_sha256": file_sha256(sandbox / "runner.py"),
        "program_sha256": file_sha256(sandbox / "program.py"),
        "limits_defaults": asdict(Limits()),
        "limits_formula_version": LIMITS_FORMULA_VERSION,
        "io_case_limit": IO_CASE_LIMIT,
    }


def call_key(code: str, tests: str, mode: str) -> str:
    return hashlib.sha256(json.dumps([code, tests, mode]).encode()).hexdigest()


def _read_cache(cache: Path) -> list[dict[str, Any]]:
    """Complete rows only; a torn final line (a crash mid-write) is cut off and redone."""
    if not cache.exists():
        return []
    raw = cache.read_bytes()
    cut = raw.rfind(b"\n") + 1
    if cut != len(raw):
        with cache.open("r+b") as stream:
            stream.truncate(cut)
    rows = []
    for line in raw[:cut].splitlines():
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("key"), str)
            or len(row["key"]) != 64
            or row.get("status") not in VERDICT_STATUSES
        ):
            raise ValueError(f"{cache}: malformed verdict row")
        rows.append(row)
    return rows


def verify(
    calls: Sequence[tuple[str, str, str]],
    cache: Path,
    *,
    execute: Callable[[str, str, str], Result],
    deadline: Callable[[str, str], float],
    workers: int,
    attempts: int = 3,
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, Result], dict[str, int]]:
    """Run every distinct (code, tests, mode) once, persisting verdicts for resume.

    ``deadline(tests, mode)`` is the candidate deadline of that call. A result the runner
    marks ``host_stalled``, or a ``timeout`` reported ``HOST_DEADLINE_SLACK`` or more past the
    deadline, is the runner giving up on a stalled container, not a verdict, and is retried
    like an unavailable sandbox. A ``resource_limit`` is accepted only when a second attempt
    repeats it. Cached rows with those signatures from before this rule are re-run.
    """
    results: dict[str, Result] = {}
    counters: Counter[str] = Counter()

    def stalled(result: Result, tests: str, mode: str) -> bool:
        return result.host_stalled or (
            result.status == "timeout"
            and result.elapsed_seconds >= deadline(tests, mode) + HOST_DEADLINE_SLACK
        )

    calls_by_key = {call_key(*call): call for call in calls}
    for row in _read_cache(cache):
        call = calls_by_key.get(row["key"])
        result = Result(row["status"], row["stdout"], row["exit_code"], row["elapsed_seconds"])
        if call is not None and stalled(result, call[1], call[2]):
            counters["cached_host_timeouts_redone"] += 1
            continue
        if result.status == "resource_limit" and row.get("attempts", 1) < 2:
            counters["cached_resource_limits_redone"] += 1
            continue
        results[row["key"]] = result
    pending = {key: call for key, call in calls_by_key.items() if key not in results}
    counters["cached"] = len(calls_by_key) - len(pending)
    progress(f"verifying {len(pending)} sandbox calls ({counters['cached']} cached)")
    lock = threading.Lock()

    def one(key: str, call: tuple[str, str, str]) -> None:
        used = 0
        while True:
            used += 1
            result = execute(*call)
            retry = result.status in RETRY_STATUSES or stalled(result, call[1], call[2])
            if result.status == "resource_limit" and used < 2:
                retry = True
            if not retry or used >= attempts:
                break
            with lock:
                counters["retried_calls"] += 1
            time.sleep(RETRY_DELAY_SECONDS * used)
        if result.status not in VERDICT_STATUSES or stalled(result, call[1], call[2]):
            raise RuntimeError(f"sandbox unavailable or failing: {result.status}")
        # The output is kept only to classify failures; it is never shown to a model.
        kept = Result(result.status, result.stdout[:4096], result.exit_code, result.elapsed_seconds)
        row = {
            "key": key,
            **asdict(kept),
            "category": failure_category(result),
            "attempts": used,
        }
        with lock:
            results[key] = kept
            stream.write(json.dumps(row) + "\n")
            stream.flush()

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, key, call) for key, call in pending.items()]
            try:
                for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    future.result()
                    if number % 2000 == 0 or number == len(futures):
                        progress(f"verified {number}/{len(futures)} calls")
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    counters["executed"] = len(pending)
    return results, dict(counters)


def protection(spec: Mapping[str, Any]) -> tuple[list[Task], list[Any], dict[str, Any]]:
    """HumanEval, HumanEval+ and MBPP+ tasks, plus ARC-Easy test questions."""
    paths = spec["protection"]
    raw = Path(paths["humaneval"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != HUMANEVAL_SHA256:
        raise ValueError("original HumanEval bytes differ")
    humaneval = ManifestEntry(
        source="openai/human-eval",
        revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
        url="https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
        license="MIT",
        attribution="OpenAI HumanEval",
        sha256=HUMANEVAL_SHA256,
        size_bytes=len(raw),
        purpose="eval",
        license_verified=True,
    )
    benchmarks = [
        adapt_row("openai/human-eval", json.loads(line), humaneval, split="test")
        for line in gzip.decompress(raw).splitlines()
    ]
    identities: dict[str, Any] = {
        "humaneval": {"tasks": len(benchmarks), "sha256": HUMANEVAL_SHA256}
    }
    for name in ("humaneval-plus", "mbpp-plus"):
        specification = release(name)
        tasks = read_release(Path(paths["evalplus_directory"]) / f"{name}.jsonl.gz", specification)
        benchmarks.extend(tasks)
        identities[name] = {"tasks": len(tasks), "sha256": specification.manifest.sha256}
    choices, metadata = read_arc_easy(Path(paths["arc_directory"]))
    identities["arc-easy"] = metadata
    return benchmarks, choices, identities


def assign(kept: Sequence[Task], splits: Mapping[str, Any]) -> tuple[list[Task], dict[str, Any]]:
    """TACO test tasks split evenly, MBPP top-up to the human minimum, then model tasks.

    Model-written holdouts come only from ``splits["model_holdout_sources"]``.
    """

    def order(tasks: Sequence[Task]) -> list[Task]:
        return sorted(tasks, key=lambda task: hashlib.sha256(task.id.encode()).digest())

    taco_test = order([t for t in kept if t.source == DATASETS["taco"] and t.split != "train"])
    human = list(taco_test)
    topped = 0
    if len(human) < splits["human_holdout_minimum"]:
        mbpp = order([t for t in kept if t.source == DATASETS["mbpp"]])
        extra = mbpp[: splits["human_holdout_minimum"] - len(human)]
        topped = len(extra)
        human.extend(extra)
    assigned: dict[str, str] = {
        task.id: ("dev" if index % 2 == 0 else "test") for index, task in enumerate(human)
    }
    room = {
        name: splits[name] - sum(value == name for value in assigned.values())
        for name in ("dev", "test")
    }
    if min(room.values()) < 0:
        raise ValueError("human holdouts exceed the requested split sizes")
    allowed = {DATASETS[kind] for kind in splits["model_holdout_sources"]}
    model = order([t for t in kept if t.written_by == "model" and t.source in allowed])
    if len(model) < room["dev"] + room["test"]:
        raise ValueError("not enough verified model-written tasks for the holdouts")
    for index, task in enumerate(model[: room["dev"] + room["test"]]):
        assigned[task.id] = "dev" if index < room["dev"] else "test"
    result = [replace(task, split=assigned.get(task.id, "train")) for task in kept]
    held = {task.id for task in taco_test}
    if any(task.split == "train" and task.id in held for task in result):
        raise RuntimeError("an upstream TACO test task reached the training split")
    return result, {
        "taco_test_holdouts": len(taco_test),
        "mbpp_holdouts_added": topped,
        "human_holdout_minimum": splits["human_holdout_minimum"],
        "rule": (
            "TACO test-split tasks alternate dev/test by ID hash; if fewer than the minimum, "
            "MBPP tasks (MBPP+ IDs already removed) are added by ID hash; model-written tasks "
            "from the named sources, by ID hash, fill the rest of dev, then test"
        ),
        "model_holdout_sources": list(splits["model_holdout_sources"]),
    }


PARAPHRASE_THRESHOLD = 0.5
PARAPHRASE_WIDTH = 3
NAMED_PARAPHRASE_THRESHOLD = 0.25
"""A training or RL-only prompt is dropped as a holdout paraphrase when its 3-word
shingles overlap a dev or test prompt's by half, or by a quarter when both define the
same function name."""
DEFINED = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def defined_names(task: Task) -> frozenset[str]:
    return frozenset(DEFINED.findall(task.solution or "")) - {"main", "solve", "solution"}


class HoldoutParaphrases:
    """Loose paraphrase test against every dev and test prompt."""

    def __init__(self, holdouts: Sequence[Task]) -> None:
        self.index = SimilarityIndex(PARAPHRASE_THRESHOLD, width=PARAPHRASE_WIDTH)
        self.by_name: dict[str, list[frozenset[tuple[str, ...]]]] = {}
        for task in holdouts:
            self.index.add(task.prompt)
            for name in defined_names(task):
                self.by_name.setdefault(name, []).append(shingles(task.prompt, PARAPHRASE_WIDTH))

    def matches(self, task: Task) -> bool:
        if self.index.contains(task.prompt):
            return True
        value = shingles(task.prompt, PARAPHRASE_WIDTH)
        return any(
            len(value & other) / max(len(value | other), 1) >= NAMED_PARAPHRASE_THRESHOLD
            for name in defined_names(task)
            for other in self.by_name.get(name, ())
        )


def drop_holdout_paraphrases(tasks: Sequence[Task]) -> tuple[list[Task], Counter[str]]:
    """Remove training tasks whose prompt loosely paraphrases any dev or test prompt."""
    paraphrases = HoldoutParaphrases([t for t in tasks if t.split in {"dev", "test"}])
    dropped: Counter[str] = Counter()
    kept = []
    for task in tasks:
        if task.split == "train" and paraphrases.matches(task):
            dropped[_kind(task)] += 1
            continue
        kept.append(task)
    return kept, dropped


def rl_only_pool(
    candidates: Sequence[Task],
    pool: Sequence[Task],
    benchmarks: Sequence[Task],
    *,
    min_alpha_tokens: int,
    prompt_tokens: int,
    encoder: FastEncoder,
) -> tuple[list[Task], Counter[str]]:
    """AceCode tasks with no attempt that passed in our sandbox, kept for RL rewards only.

    They are never holdouts. A task is dropped if its prompt is over ``prompt_tokens``
    tokens; if it matches benchmark or holdout n-grams; or if it near-duplicates a
    benchmark, a pool task or an earlier RL-only task. A loose paraphrase of a holdout
    also drops it.
    """
    holdouts = [t for t in pool if t.split in {"dev", "test"}]
    texts = [
        part
        for task in [*benchmarks, *holdouts]
        for part in (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)
    ]
    contamination = ContaminationIndex(texts, min_alpha_tokens=min_alpha_tokens)
    duplicates = SimilarityIndex()
    for task in [*benchmarks, *pool]:
        duplicates.add(task.prompt)
    paraphrases = HoldoutParaphrases(holdouts)
    benchmark_ids = {canonical_id(task.id) for task in benchmarks}
    counts: Counter[str] = Counter()
    kept = []
    for task in sorted(candidates, key=lambda item: item.id):
        counts["seen"] += 1
        if len(encoder.encode(task.public_prompt())) > prompt_tokens:
            counts["over_prompt_tokens"] += 1
        elif canonical_id(task.id) in benchmark_ids or any(
            contamination.contains(part)
            for part in (task.prompt, task.public_tests, task.hidden_tests)
        ):
            counts["contamination"] += 1
        elif duplicates.contains(task.prompt):
            counts["near_duplicate"] += 1
        elif paraphrases.matches(task):
            counts["holdout_paraphrase"] += 1
        else:
            duplicates.add(task.prompt)
            kept.append(replace(task, split="train", solution=None))
            counts["kept"] += 1
    return kept, counts


def _task_hash(tasks: Sequence[Task]) -> str:
    return hashlib.sha256(
        json.dumps([asdict(t) for t in sorted(tasks, key=lambda t: t.id)], sort_keys=True).encode()
    ).hexdigest()


def _screen(
    task: Task, solution: str, encoder: FastEncoder, context: int, limits: Limits
) -> str | None:
    """Why a (task, solution) pair cannot be verified, or None if it can."""
    if len(encoder.encode(task.public_prompt() + "\n" + solution)) > context:
        return "over_context"
    if max(len((solution + t).encode()) for t in (task.public_tests, task.hidden_tests)) > (
        limits.input_bytes
    ):
        return "oversized_for_sandbox"
    return None


def build(
    spec: Mapping[str, Any],
    directory: Path,
    *,
    execute: Callable[[str, str, str], Result],
    runtime: Mapping[str, Any],
    adopt_cache_policy: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Screen, verify and split the pool; one build per directory at a time.

    ``adopt_cache_policy`` records, for a verdict cache written before policy records
    existed, that it was produced by the current sandbox runtime.
    """
    output = directory / "pool"
    if output.exists():
        raise FileExistsError("the pool already exists")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "build.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another build holds {lock}") from error
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        return _build(spec, directory, execute, runtime, adopt_cache_policy, progress)
    finally:
        lock.unlink(missing_ok=True)


def _build(
    spec: Mapping[str, Any],
    directory: Path,
    execute: Callable[[str, str, str], Result],
    runtime: Mapping[str, Any],
    adopt_cache_policy: bool,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    output = directory / "pool"
    started = time.perf_counter()
    receipts = json.loads((directory / "download-receipts.json").read_text())
    if receipts.get("status") != "completed" or receipts.get("spec_sha256") != spec["sha256"]:
        raise ValueError("completed download receipts for this exact spec are required")
    for item in receipts["files"]:
        if not _verified(Path(item["path"]), item["size_bytes"], item["sha256"]):
            raise ValueError(f"{item['filename']}: bytes differ from the receipt")
    tokenizer = _load_tokenizer(
        Path(spec["tokenizer"]["path"]),
        expected_identity=spec["tokenizer"]["identity"],
        expected_file_sha256=spec["tokenizer"]["file_sha256"],
        expected_vocab_size=spec["tokenizer"]["vocab_size"],
        expected_bos_id=256,
    )
    encoder = FastEncoder(tokenizer)
    verification = spec["verification"]
    cache = directory / "verification" / f"results-{policy_hash(verification)}.jsonl"
    policy_path = cache.with_suffix(".policy.json")
    policy = runtime_policy(runtime)
    adopted = False
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("cached verdicts were produced by a different sandbox runtime")
    elif cache.exists() and not adopt_cache_policy:
        raise ValueError("a verdict cache without a policy record needs --adopt-cache-policy")
    else:
        adopted = cache.exists()
        atomic_json(policy_path, policy)
    min_alpha = spec["protection"]["min_alpha_tokens"]
    benchmarks, choices, coverage = protection(spec)
    adapted, counts, details, alternatives = candidates(spec, receipts, benchmarks)
    limits = Limits(memory_mb=verification["memory_mb"])
    choice_index = ContaminationIndex(
        (text for task in choices for text in (task.question, task.public_prompt())),
        min_alpha_tokens=min_alpha,
    )
    lengths: dict[str, list[int]] = {kind: [] for kind in KINDS}
    screened: list[Task] = []
    for task in adapted:
        kind = _kind(task)
        if task.solution is not None:
            reason = _screen(task, task.solution, encoder, spec["context_tokens"], limits)
            if reason is not None and kind != "acecode":
                counts[kind][reason] += 1
                continue
            if reason is not None:
                # Another recorded AceCode attempt may still fit; failing that, RL-only.
                counts[kind][f"first_attempt_{reason}"] += 1
        if any(
            choice_index.contains(text)
            for text in (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)
        ):
            counts[kind]["choice_contamination"] += 1
            continue
        if task.solution is not None:
            lengths[kind].append(len(encoder.encode(task.public_prompt() + "\n" + task.solution)))
        screened.append(task)
        counts[kind]["screened"] += 1
    timings = {"intake": time.perf_counter() - started}

    def deadline(tests: str, mode: str) -> float:
        return limits_for(verification, tests, mode).timeout_seconds

    def run_calls(pairs: Sequence[tuple[Task, str]]) -> dict[str, Result]:
        calls = [
            (solution, tests, task.mode)
            for task, solution in pairs
            for tests in (task.public_tests, task.hidden_tests)
        ]
        found, counters = verify(
            calls,
            cache,
            execute=execute,
            deadline=deadline,
            workers=verification["workers"],
            progress=progress,
        )
        for name, value in counters.items():
            call_counts[name] += value
        return found

    def passes(results: Mapping[str, Result], task: Task, solution: str) -> bool:
        return all(
            results[call_key(solution, tests, task.mode)].passed
            for tests in (task.public_tests, task.hidden_tests)
        )

    began = time.perf_counter()
    call_counts: Counter[str] = Counter()
    fits = {
        task.id: task.solution is not None
        and _screen(task, task.solution, encoder, spec["context_tokens"], limits) is None
        for task in screened
    }
    first = [
        (task, task.solution) for task in screened if task.solution is not None and fits[task.id]
    ]
    results = dict(run_calls([(t, sol) for t, sol in first if sol is not None]))
    # AceCode: try further recorded attempts until one passes in our sandbox.
    chosen: dict[str, int | None] = {}
    for task in screened:
        if _kind(task) == "acecode":
            chosen[task.id] = (
                0 if fits[task.id] and passes(results, task, task.solution or "") else None
            )
    for attempt in range(1, ACECODE_ATTEMPTS):
        pairs = [
            (task, alternatives[task.id][attempt])
            for task in screened
            if _kind(task) == "acecode"
            and chosen[task.id] is None
            and len(alternatives.get(task.id, [])) > attempt
            and _screen(
                task, alternatives[task.id][attempt], encoder, spec["context_tokens"], limits
            )
            is None
        ]
        if pairs:
            results.update(run_calls(pairs))
        for task, solution in pairs:
            if passes(results, task, solution):
                chosen[task.id] = attempt
    timings["verification"] = time.perf_counter() - began
    updated: list[Task] = []
    for task in screened:
        if _kind(task) == "acecode":
            index = chosen[task.id]
            task = replace(task, solution=None if index is None else alternatives[task.id][index])
        elif task.solution is not None and not fits[task.id]:
            task = replace(task, solution=None)
        updated.append(task)
    attempt_use = Counter(
        "none" if index is None else f"attempt_{index + 1}" for index in chosen.values()
    )
    # Every screened reference's public and hidden verdicts, by source.
    outcomes: dict[str, Counter[str]] = {kind: Counter() for kind in KINDS}
    for task in updated:
        if task.solution is not None:
            for part, tests in (("public", task.public_tests), ("hidden", task.hidden_tests)):
                result = results[call_key(task.solution, tests, task.mode)]
                outcomes[_kind(task)][f"{part}:{failure_category(result)}"] += 1
    kept_all, pool = build_pool(
        updated,
        benchmarks,
        execute=lambda code, tests, mode: results[call_key(code, tests, mode)],
        min_alpha_tokens=min_alpha,
    )
    for kind in KINDS:
        dataset = DATASETS[kind]
        for reason in (
            "contamination_or_duplicate",
            "missing_solution",
            "invalid_solution",
            "kept",
        ):
            counts[kind][reason] = pool.get(f"source:{dataset}:{reason}", 0)
    assigned, holdout_rule = assign(kept_all, spec["splits"])
    tasks, paraphrased = drop_holdout_paraphrases(assigned)
    for kind, number in paraphrased.items():
        counts[kind]["holdout_paraphrase"] = number
    kept_ids = {task.id for task in kept_all}
    rl_candidates = [
        task
        for task in updated
        if _kind(task) == "acecode" and task.solution is None and task.id not in kept_ids
    ]
    rl_only, rl_counts = rl_only_pool(
        rl_candidates,
        tasks,
        benchmarks,
        min_alpha_tokens=min_alpha,
        prompt_tokens=spec["splits"]["rl_prompt_tokens"],
        encoder=encoder,
    )
    staging = directory / "pool.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    hashes = {}
    for name, selected in (
        ("train", [t for t in tasks if t.split == "train"]),
        ("dev", [t for t in tasks if t.split == "dev"]),
        ("test", [t for t in tasks if t.split == "test"]),
        ("holdout", [t for t in tasks if t.split in {"dev", "test"}]),
        ("rl_only", rl_only),
    ):
        save_tasks(staging / f"{name}.jsonl", selected)
        hashes[name] = {
            "file_sha256": file_sha256(staging / f"{name}.jsonl"),
            "task_hash": _task_hash(selected),
            "tasks": len(selected),
        }

    def stats(values: list[int]) -> dict[str, int]:
        if not values:
            return {}
        ordered_values = sorted(values)
        return {
            "with_solution": len(values),
            "median": ordered_values[len(values) // 2],
            "p90": ordered_values[int(len(values) * 0.9)],
            "max": ordered_values[-1],
        }

    commit, dirty = git_state()
    splits = split_summary(tasks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "name": spec["name"],
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "spec_sha256": spec["sha256"],
        "download_receipts_sha256": file_sha256(directory / "download-receipts.json"),
        "sources": [
            {key: value for key, value in source.items() if key != "files"}
            for source in spec["sources"]
        ],
        "selection": {
            "opencode": "first rows of the listed files in file order",
            "acecode": "first rows of the listed files in file order",
            "taco": "every EASY row of the listed train and test files, HackerRank excluded",
            "mbpp": "all 974 tasks minus MBPP+ IDs",
        },
        "intake": {kind: dict(counter) for kind, counter in counts.items()},
        "details": details,
        "acecode_attempts": {
            "tried_per_task": ACECODE_ATTEMPTS,
            "solution_from": dict(attempt_use),
        },
        "verification_outcomes": {kind: dict(counter) for kind, counter in outcomes.items()},
        "pool_counts": pool,
        "context": {
            "limit_tokens": spec["context_tokens"],
            "basis": spec["context_basis"],
            "measured": "frozen-tokenizer tokens of the public prompt plus reference solution",
            "lengths": {kind: stats(values) for kind, values in lengths.items()},
        },
        "holdouts": holdout_rule,
        "public_tests_per_task": (
            "1 public, the rest hidden, for every source (data.md's starting point of up to 3 "
            "public is not applied in this rehearsal pool)"
        ),
        "paraphrase_rule": (
            f"training and RL-only prompts sharing {PARAPHRASE_THRESHOLD:.0%} of "
            f"{PARAPHRASE_WIDTH}-word shingles with any dev or test prompt, or "
            f"{NAMED_PARAPHRASE_THRESHOLD:.0%} when both define the same function name, are dropped"
        ),
        "splits": splits,
        "kept_by_source": {kind: sum(t.source == DATASETS[kind] for t in tasks) for kind in KINDS},
        "authorship": {
            split: {
                who: splits.get(f"split:{split}:authorship:{who}", 0) for who in ("human", "model")
            }
            for split in ("train", "dev", "test")
        },
        "rl_only": {
            **dict(rl_counts),
            "source": "AceCode tasks with no recorded attempt that passed in our sandbox",
            "use": "RL rewards only; never SFT, never holdouts",
            "prompt_tokens_limit": spec["splits"]["rl_prompt_tokens"],
        },
        "files": hashes,
        "protection": {**coverage, "min_alpha_tokens": min_alpha},
        "sandbox": {
            "limits": {
                "memory_mb": verification["memory_mb"],
                "function_timeout_seconds": verification["timeout_seconds"],
                "io_timeout_seconds": "timeout_seconds + io_seconds_per_case x cases, at most 60",
                "io_seconds_per_case": verification["io_seconds_per_case"],
                "input_bytes": limits.input_bytes,
                "policy_hash": policy_hash(verification),
            },
            "policy": policy,
            "cache_policy_adopted": adopted,
            "calls": dict(call_counts),
        },
        "timings_seconds": {**timings, "total": time.perf_counter() - started},
        "paid_runs": 0,
        "limitations": [
            "dataset licenses as declared on each card; seed/component provenance of the "
            "model-written sources is not independently cleared",
            "a reference that times out under the sandbox limits is treated as invalid",
            "near-duplicates are 5-word-shingle Jaccard >= 0.85 and paraphrases of holdouts "
            "3-word-shingle Jaccard >= 0.5; looser rewordings remain",
            "RL-only tasks have model-synthesized tests that no reference has passed here",
        ],
    }
    atomic_json(staging / "report.json", report)
    staging.replace(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "build"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--adopt-cache-policy",
        action="store_true",
        help="record that an existing verdict cache was produced by the current sandbox",
    )
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    if args.stage == "download":
        result = fetch(spec, args.directory)
        print(json.dumps({k: result[k] for k in ("listed_bytes", "bytes_transferred")}))
        return 0
    runtime = Runtime()
    verification = spec["verification"]
    report = build(
        spec,
        args.directory,
        execute=lambda code, tests, mode: run(
            code, tests, limits_for(verification, tests, mode), mode=mode, runtime=runtime
        ),
        runtime={"image": IMAGE, **asdict(runtime)},
        adopt_cache_policy=args.adopt_cache_policy,
    )
    print(json.dumps({k: report[k] for k in ("kept_by_source", "authorship", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
