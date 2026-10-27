"""Bind a post-training run's inputs: each object's R2 key, size and SHA-256; stage, fetch
and verify them. Modelled on ``tools.rehearsal_binding``; nothing here contacts Runpod.

Run 1 reads three parent checkpoints already in R2 (run B's ``mixed-final`` and run A's two
stage finals), the tokenizer already in R2, and files the Mac stages at the go: the SFT
rows and dev sets, the hidden-test-free Pod exports of the pool dev and test splits and
the RL candidates (``scratchcoder pod-task-export``), the RL candidate list, the pinned
public benchmark releases and the ARC-Easy test questions.

``build`` (Mac)
    From a clean tree, read the specification's inputs and the local files given with
    ``--input NAME=PATH``, hash each, check every size and SHA-256 the specification already
    pins, and write the binding: for each input its kind, R2 key, Pod path, size and SHA-256.
    An input with a key in the specification is ``existing`` (already in R2, never
    uploaded); every other input is ``staged`` under ``posttrain-inputs/<sha256>/<file>``.
``stage`` (Mac, at the go)
    Upload every staged input with the Mac's key under the storage cap, and HEAD every
    existing input: its stored SHA-256 and size must be the binding's.
``fetch`` (Pod; also the Mac's copy of the parent checkpoints)
    Download inputs pinned to their size and SHA-256 into a root; ``--only`` selects some.
``verify`` (Pod or Mac)
    Re-hash every input under a root against the binding.

R2 credentials come only from the environment and are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.shards import file_sha256
from tools.r2_store import Store, upload
from tools.rehearsal_binding import (
    _hex,
    _pinned,
    _plain_file,
    _walk,
    _write_new,
    r2_credentials,
)
from tools.rehearsal_ops import head_sha256

SCHEMA_VERSION = 1
KIND = "posttrain-input-binding"
STAGED_ROOT = "posttrain-inputs"
INPUT_KINDS = (
    "checkpoint",
    "tokenizer",
    "sft-rows",
    "pool-export",
    "candidates-export",
    "candidate-list",
    "benchmark",
    "arc-export",
)
BINDING_KEYS = {"schema_version", "kind", "run", "created_at", "binder", "specification", "inputs"}
INPUT_KEYS = {"kind", "key", "path", "size_bytes", "sha256", "origin", "source"}
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _serialized(binding: Mapping[str, Any]) -> bytes:
    return (json.dumps(binding, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def binding_sha256(binding: Mapping[str, Any]) -> str:
    return hashlib.sha256(_serialized(binding)).hexdigest()


def staged_key(sha256: str, file: str) -> str:
    """Where a staged input lives in R2: named by its SHA-256, so no upload replaces another."""
    _hex(sha256, "the staged input's SHA-256")
    if not FILE.fullmatch(file):
        raise ValueError(f"{file!r} is not a plain file name")
    return f"{STAGED_ROOT}/{sha256}/{file}"


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    return value


def check_input(name: str, entry: Any) -> dict[str, Any]:
    """One bound input: its kind, plain Pod path ``<name>/<file>``, key, size and SHA-256."""
    row = _exact(entry, INPUT_KEYS, f"input {name}")
    if not NAME.fullmatch(name) or row["kind"] not in INPUT_KINDS:
        raise ValueError(f"input {name!r} has an unsafe name or unknown kind {row['kind']!r}")
    path = row["path"]
    if not isinstance(path, str) or path.count("/") != 1 or not path.startswith(name + "/"):
        raise ValueError(f"input {name}: the Pod path must be {name}/<file>")
    if not FILE.fullmatch(path.split("/")[1]):
        raise ValueError(f"input {name}: the Pod path's file name is not plain")
    _hex(row["sha256"], f"input {name} sha256")
    if type(row["size_bytes"]) is not int or row["size_bytes"] < 1:
        raise ValueError(f"input {name}: size_bytes must be a positive integer")
    key = row["key"]
    if row["origin"] == "staged":
        if key != staged_key(row["sha256"], path.split("/")[1]):
            raise ValueError(f"input {name}: a staged key is posttrain-inputs/<sha256>/<file>")
    elif row["origin"] == "existing":
        if (
            not isinstance(key, str)
            or key.startswith(STAGED_ROOT + "/")
            or not all(FILE.fullmatch(part) for part in key.split("/"))
        ):
            raise ValueError(f"input {name}: an existing key must be plain path components")
    else:
        raise ValueError(f"input {name}: origin must be existing or staged")
    if row["source"] is not None and not isinstance(row["source"], str):
        raise ValueError(f"input {name}: source must be a path string or null")
    return row


def check_binding(binding: Any) -> dict[str, Any]:
    """Validate a binding's structure without touching any file."""
    top = _exact(binding, BINDING_KEYS, "the binding")
    if top["schema_version"] != SCHEMA_VERSION or top["kind"] != KIND:
        raise ValueError("not a post-training input binding of this schema")
    binder = _exact(top["binder"], {"git_commit", "dirty"}, "binder")
    if binder["dirty"] is not False or not isinstance(binder["git_commit"], str):
        raise ValueError("the binding was not written from a clean tree")
    if not GIT_COMMIT.fullmatch(binder["git_commit"]):
        raise ValueError("the binder's git_commit must be 40 lower-case hex digits")
    specification = _exact(top["specification"], {"path", "sha256"}, "specification")
    _hex(specification["sha256"], "the specification's SHA-256")
    inputs = top["inputs"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("the binding must list its inputs")
    for name, entry in inputs.items():
        check_input(name, entry)
    keys = [entry["key"] for entry in inputs.values()]
    if len(set(keys)) != len(keys):
        raise ValueError("two inputs share an R2 key")
    return top


def load_binding(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Read and check a binding whose bytes hash to ``expected_sha256``."""
    _hex(expected_sha256, "the expected binding SHA-256")
    data = _plain_file(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("the binding's SHA-256 differs from the expected value")
    binding = check_binding(json.loads(data))
    if _serialized(binding) != data:
        raise ValueError("the binding is not in its canonical serialization")
    return binding


def require_specification(binding: Mapping[str, Any], specification: Mapping[str, Any]) -> None:
    """The binding names exactly the specification's inputs, with every pinned value."""
    wanted = specification.get("inputs")
    if not isinstance(wanted, dict) or set(wanted) != set(binding["inputs"]):
        raise ValueError("the binding's inputs differ from the specification's")
    for name, pin in wanted.items():
        bound = binding["inputs"][name]
        if bound["kind"] != pin.get("kind"):
            raise ValueError(f"input {name}: the binding's kind differs from the specification")
        for field in ("key", "size_bytes", "sha256"):
            if pin.get(field) is not None and bound[field] != pin[field]:
                raise ValueError(f"input {name}: the bound {field} differs from the specification")
        if (pin.get("key") is None) != (bound["origin"] == "staged"):
            raise ValueError(f"input {name}: only inputs without a key are staged")


def build_binding(
    specification_path: Path,
    local: Mapping[str, Path],
    output: Path,
    *,
    run: str,
) -> dict[str, Any]:
    """Write the binding of ``local`` files against the specification; return it.

    ``local`` gives a file for every staged input and, optionally, the Mac's copy of an
    existing one (its size and SHA-256 are then checked too).
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists; a binding is never rewritten")
    commit, dirty = git_state()
    if dirty or commit is None or not GIT_COMMIT.fullmatch(commit):
        raise ValueError("a binding is written only from a clean tree at a known commit")
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict) or specification.get("run") != run:
        raise ValueError(f"the specification is not run {run}'s")
    wanted = specification.get("inputs")
    if not isinstance(wanted, dict):
        raise ValueError("the specification lists no inputs")
    unknown = sorted(set(local) - set(wanted))
    if unknown:
        raise ValueError(f"--input names {unknown} are not inputs of the specification")
    inputs: dict[str, dict[str, Any]] = {}
    for name, pin in sorted(wanted.items()):
        if not isinstance(pin, dict):
            raise ValueError(f"input {name} must be an object")
        path = local.get(name)
        if pin.get("key") is None:
            if path is None:
                raise ValueError(f"input {name} is staged: give it with --input {name}=PATH")
            source = _plain_file(path)
            digest, size = file_sha256(source), source.stat().st_size
            entry = {
                "kind": pin.get("kind"),
                "key": staged_key(digest, source.name),
                "path": f"{name}/{source.name}",
                "size_bytes": size,
                "sha256": digest,
                "origin": "staged",
                "source": str(source.absolute()),
            }
        else:
            if pin.get("size_bytes") is None or pin.get("sha256") is None:
                raise ValueError(f"input {name}: an existing input needs its size and SHA-256")
            if path is not None:
                copy = _plain_file(path)
                if (copy.stat().st_size, file_sha256(copy)) != (pin["size_bytes"], pin["sha256"]):
                    raise ValueError(f"input {name}: the Mac copy differs from the specification")
            entry = {
                "kind": pin.get("kind"),
                "key": pin["key"],
                "path": f"{name}/{pin['key'].rsplit('/', 1)[-1]}",
                "size_bytes": pin["size_bytes"],
                "sha256": pin["sha256"],
                "origin": "existing",
                "source": None if path is None else str(path.absolute()),
            }
        for field in ("size_bytes", "sha256"):
            if pin.get(field) is not None and entry[field] != pin[field]:
                raise ValueError(f"input {name}: its {field} differs from the specification")
        inputs[name] = check_input(name, entry)
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run": run,
        "created_at": utc_now(),
        "binder": {"git_commit": commit, "dirty": False},
        "specification": {
            "path": str(specification_path),
            "sha256": file_sha256(specification_path),
        },
        "inputs": inputs,
    }
    check_binding(binding)
    require_specification(binding, specification)
    _write_new(output, _serialized(binding))
    return binding


def stage(
    binding: Mapping[str, Any], store: Store, *, cap_usd: float, receipt: Path
) -> dict[str, Any]:
    """Upload every staged input (re-hashed locally first) and HEAD every existing one."""
    checked = check_binding(binding)
    pairs = []
    for name, entry in sorted(checked["inputs"].items()):
        if entry["origin"] != "staged":
            continue
        source = _plain_file(Path(entry["source"]))
        if (source.stat().st_size, file_sha256(source)) != (entry["size_bytes"], entry["sha256"]):
            raise ValueError(f"input {name}: the local file differs from the binding")
        pairs.append((entry["key"], source))
    existing = []
    for name, entry in sorted(checked["inputs"].items()):
        if entry["origin"] == "existing":
            row = head_sha256(store, entry["key"], expected=entry["sha256"])
            if row["bytes"] != entry["size_bytes"]:
                raise ValueError(f"input {name}: the size in R2 differs from the binding")
            existing.append({"name": name, **row})
    record = upload(store, pairs, cap_usd=cap_usd, receipt=receipt) if pairs else {"files": []}
    record = {**record, "existing_verified": existing, "binding_sha256": binding_sha256(checked)}
    atomic_json(receipt, record)
    return record


def fetch(
    binding: Mapping[str, Any],
    root: Path,
    store: Store,
    *,
    receipt: Path,
    only: Sequence[str] = (),
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Download the inputs (``only`` some) into ``root``, each pinned; write a receipt."""
    checked = check_binding(binding)
    names = list(only) or sorted(checked["inputs"])
    unknown = sorted(set(names) - set(checked["inputs"]))
    if unknown:
        raise ValueError(f"--only names {unknown} are not bound inputs")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for name in names:
        entry = checked["inputs"][name]
        target, info = _walk(root, entry["path"], create=True)
        if info is not None:
            if info.st_size != entry["size_bytes"] or file_sha256(target) != entry["sha256"]:
                raise ValueError(
                    f"{entry['path']} is present with other bytes; remove it to refetch"
                )
            row = {"name": name, "method": "present", "bytes_transferred": 0}
        else:
            got = store.download(
                entry["key"], target, sha256=entry["sha256"], size=entry["size_bytes"]
            )
            row = {
                "name": name,
                "method": "downloaded",
                "bytes_transferred": got["bytes"],
                "attempts": got["attempts"],
                "seconds": got["seconds"],
            }
        rows.append({**row, **{k: entry[k] for k in ("key", "path", "size_bytes", "sha256")}})
        progress(f"{row['method']} {entry['path']} ({entry['size_bytes']:,} bytes)")
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "posttrain-input-fetch",
        "created_at": utc_now(),
        "run": checked["run"],
        "binding_sha256": binding_sha256(checked),
        "bucket": store.credentials.bucket,
        "root": str(root),
        "inputs": rows,
        "bytes_transferred": sum(row["bytes_transferred"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(receipt, record)
    return record


def verify(binding: Mapping[str, Any], root: Path, *, only: Sequence[str] = ()) -> dict[str, Any]:
    """Re-hash every input (``only`` some) under ``root``; return the verified paths."""
    checked = check_binding(binding)
    names = list(only) or sorted(checked["inputs"])
    paths = {}
    for name in names:
        entry = checked["inputs"][name]
        paths[name] = _pinned(root, entry["path"], entry["size_bytes"], entry["sha256"])
    return {
        "binding_sha256": binding_sha256(checked),
        "run": checked["run"],
        "inputs": {name: checked["inputs"][name]["sha256"] for name in names},
        "paths": {name: str(path) for name, path in paths.items()},
    }


def check_fetch_receipt(
    record: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    names: Sequence[str] | None = None,
) -> None:
    """A fetch receipt covers exactly ``names`` (default: every input) at the bound pins."""
    if record.get("binding_sha256") != binding_sha256(binding):
        raise ValueError("the fetch receipt is for another binding")
    wanted = sorted(binding["inputs"]) if names is None else sorted(names)
    rows = record.get("inputs")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("the fetch receipt lists no inputs")
    found = {row.get("name"): row for row in rows}
    if sorted(found) != wanted or len(found) != len(rows):
        raise ValueError(f"the fetch receipt must cover exactly {wanted}")
    for name, row in found.items():
        entry = binding["inputs"][name]
        if row.get("method") not in {"downloaded", "present"} or any(
            row.get(field) != entry[field] for field in ("key", "path", "size_bytes", "sha256")
        ):
            raise ValueError(f"the fetch receipt's {name} differs from the binding")


BENCHMARK_FILES = {
    "humaneval": "HumanEval.jsonl.gz",
    "humaneval-plus": "HumanEvalPlus-OriginFmt.jsonl.gz",
    "mbpp-plus": "MbppPlus-OriginFmt.jsonl.gz",
}


def benchmark_releases(
    output: Path, *, download: Callable[[Any, Path], Any] | None = None
) -> dict[str, Path]:
    """The pinned HumanEval and EvalPlus releases under ``output``, each size- and
    SHA-256-checked by ``scratchcoder.data.manifest.download``; a present file is checked."""
    from scratchcoder.data import evalplus, humaneval
    from scratchcoder.data.manifest import download as fetch_manifest

    get = fetch_manifest if download is None else download
    manifests = {
        "humaneval": humaneval.manifest(),
        "humaneval-plus": evalplus.release("humaneval-plus").manifest,
        "mbpp-plus": evalplus.release("mbpp-plus").manifest,
    }
    paths = {}
    for suite, manifest in manifests.items():
        path = output / BENCHMARK_FILES[suite]
        if not path.exists():
            get(manifest, path)
        if (path.stat().st_size, file_sha256(path)) != (manifest.size_bytes, manifest.sha256):
            raise ValueError(f"{path} is not the pinned {suite} release")
        paths[suite] = path
    return paths


def _local(values: Sequence[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for item in values:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in found:
            raise ValueError("--input takes NAME=PATH, each name once")
        found[name] = Path(path)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write the input binding (Mac)")
    build.add_argument("--specification", type=Path, required=True)
    build.add_argument("--run", required=True)
    build.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    build.add_argument("--output", type=Path, required=True)
    releases = commands.add_parser(
        "benchmarks", help="download and check the pinned benchmark releases (Mac, at the go)"
    )
    releases.add_argument("--output", type=Path, required=True)
    stage_command = commands.add_parser("stage", help="upload the staged inputs (Mac)")
    fetch_command = commands.add_parser("fetch", help="download inputs pinned (Pod or Mac)")
    verify_command = commands.add_parser("verify", help="re-hash inputs under a root")
    for command in (stage_command, fetch_command, verify_command):
        command.add_argument("--binding", type=Path, required=True)
        command.add_argument("--expected-sha256", required=True)
    stage_command.add_argument("--receipt", type=Path, required=True)
    stage_command.add_argument("--cap-usd", type=float, default=5.0)
    for command in (fetch_command, verify_command):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--only", action="append", default=[], help="one input name")
    fetch_command.add_argument("--receipt", type=Path, required=True)
    fetch_command.add_argument("--key-prefix", default="R2_", help="e.g. R2_POD_ on the Pod")
    args = parser.parse_args(argv)
    if args.command == "build":
        binding = build_binding(args.specification, _local(args.input), args.output, run=args.run)
        print(json.dumps({"binding": str(args.output), "sha256": binding_sha256(binding)}))
        return 0
    if args.command == "benchmarks":
        paths = benchmark_releases(args.output)
        print(json.dumps({suite: str(path) for suite, path in paths.items()}, indent=1))
        return 0
    binding = load_binding(args.binding, expected_sha256=args.expected_sha256)
    if args.command == "stage":
        record = stage(binding, Store(r2_credentials()), cap_usd=args.cap_usd, receipt=args.receipt)
        print(
            json.dumps(
                {"staged": len(record["files"]), "existing": len(record["existing_verified"])}
            )
        )
    elif args.command == "fetch":
        store = Store(r2_credentials(key_prefix=args.key_prefix))
        record = fetch(binding, args.root, store, receipt=args.receipt, only=args.only)
        result = verify(binding, args.root, only=args.only)
        print(json.dumps({"bytes_transferred": record["bytes_transferred"], **result}, indent=1))
    else:
        print(json.dumps(verify(binding, args.root, only=args.only), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
