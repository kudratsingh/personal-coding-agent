"""Pinned, CPU-only reference check using the project's isolated evaluation harness."""

import argparse
import ast
import base64
import gzip
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

import torch
from torch import Tensor

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.harness import EvaluationConfig, evaluate, save_report
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.diagnostics import failure_category as failure_category
from scratchcoder.sandbox.runner import Result, run

MODEL = "bigcode/tiny_starcoder_py"
MODEL_REVISION = "8547527bef0bc927268c1653cce6948c5c242dd1"
DATA_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
DATA_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
STOPS = ("\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n#")


def normalize(rows: list[dict[str, Any]]) -> list[Task]:
    tasks = []
    for row in rows:
        entry = row["entry_point"]
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError("invalid benchmark entry point")
        tasks.append(
            Task(
                id=row["task_id"],
                prompt=row["prompt"],
                public_tests=f"assert callable({entry})",
                hidden_tests=row["test"] + f"\ncheck({entry})",
                solution=row["prompt"] + row["canonical_solution"],
                entry_point=entry,
                source="openai/human-eval",
                license="MIT",
                written_by="human",
                split="eval",
            )
        )
    return tasks


def truncate_completion(text: str, *, stops: Sequence[str] = STOPS) -> str:
    offsets = [text.find(stop) for stop in stops if stop in text]
    return text[: min(offsets)] if offsets else text


def completion_diagnostics(
    prompt: str,
    completion: str,
    generated_tokens: Sequence[int],
    *,
    budget: int,
    eos_token_id: int | None,
    stops: Sequence[str] = STOPS,
) -> dict[str, Any]:
    """Observe the existing candidate without executing or returning its source text."""
    if (
        type(budget) is not int
        or budget < 1
        or len(generated_tokens) > budget
        or any(type(token) is not int or token < 0 for token in generated_tokens)
        or eos_token_id is not None
        and (type(eos_token_id) is not int or eos_token_id < 0)
    ):
        raise ValueError("valid generation counts and token IDs required")
    kept = truncate_completion(completion, stops=stops)
    matches = [
        (completion.index(stop), index) for index, stop in enumerate(stops) if stop in completion
    ]
    stop_index = min(matches)[1] if matches else None
    try:
        ast.parse(prompt + kept)
        syntax_valid = True
    except (SyntaxError, ValueError, RecursionError):
        syntax_valid = False
    eos_seen = eos_token_id is not None and eos_token_id in generated_tokens
    return {
        "generated_tokens": len(generated_tokens),
        "budget_reached": len(generated_tokens) == budget,
        "budget_without_eos": len(generated_tokens) == budget and not eos_seen,
        "eos_seen": eos_seen,
        "stop_index": stop_index,
        "decoded_characters": len(completion),
        "retained_characters": len(kept),
        "empty_retained_completion": not kept.strip(),
        "candidate_syntax_valid": syntax_valid,
    }


def encrypted_failures(rows: Sequence[dict[str, str]], recipient: Path) -> str:
    """Return ciphertext only; the recipient's private key never leaves the owner."""
    result = subprocess.run(
        [
            "openssl",
            "cms",
            "-encrypt",
            "-binary",
            "-aes-256-cbc",
            "-outform",
            "DER",
            str(recipient),
        ],
        input=json.dumps({"failures": list(rows)}, ensure_ascii=False).encode(),
        capture_output=True,
        check=True,
    )
    return base64.b64encode(result.stdout).decode("ascii")


def validate_references(
    tasks: Sequence[Task],
    *,
    execute: Callable[[str, str, str], Result],
) -> dict[str, Any]:
    """Check known answers through isolation before interpreting model performance."""
    if any(not task.solution for task in tasks):
        raise ValueError("canonical solution is missing")
    return evaluate(
        tasks,
        lambda task, _: cast(str, task.solution),
        config=EvaluationConfig(),
        checkpoint_hash="canonical-solutions:" + DATA_SHA256,
        tokenizer_hash="not-applicable:canonical-source",
        execute=execute,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=164)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--disable-stops", action="store_true")
    parser.add_argument("--failure-recipient", type=Path, help="public CMS recipient certificate")
    args = parser.parse_args()
    if not 1 <= args.limit <= 164 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error("invalid limit, thread count, or token budget")
    args.output.mkdir(parents=True, exist_ok=False)
    stops = () if args.disable_stops else STOPS
    url = f"https://raw.githubusercontent.com/openai/human-eval/{DATA_REVISION}/data/HumanEval.jsonl.gz"
    with urlopen(url, timeout=30) as response:
        raw = response.read(100000)
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("benchmark content hash differs from pinned bytes")
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    if len(rows) != 164:
        raise ValueError("expected complete 164-task benchmark")
    tasks = normalize(rows)[: args.limit]
    metadata = {
        "source": url,
        "revision": DATA_REVISION,
        "sha256": DATA_SHA256,
        "purpose": "evaluation-only",
        "license": "MIT",
        "attribution": "OpenAI HumanEval",
        "license_url": f"https://github.com/openai/human-eval/blob/{DATA_REVISION}/LICENSE",
    }
    atomic_json(args.output / "data-manifest.json", metadata)
    canonical = validate_references(
        tasks, execute=lambda code, tests, mode: run(code, tests, mode=mode)
    )
    save_report(
        canonical,
        args.output / "canonical-evaluation",
        scorecard=args.output / "canonical-scorecard.jsonl",
    )
    print("CANONICAL_REPORT=" + json.dumps(canonical, sort_keys=True), flush=True)
    if canonical["metrics"]["greedy_pass_at_1"] != 1.0:
        raise RuntimeError("canonical validation failed; model benchmark is not trustworthy")
    if args.canonical_only:
        return 0
    torch.set_num_threads(args.threads)
    transformers = importlib.import_module("transformers")
    hub = importlib.import_module("huggingface_hub")
    cache = str(args.output / "model-cache")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=False, cache_dir=cache
    )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            use_safetensors=True,
            cache_dir=cache,
        )
        .to("cpu")
        .eval()
    )
    weights_path = Path(
        hub.hf_hub_download(MODEL, "model.safetensors", revision=MODEL_REVISION, cache_dir=cache)
    )
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    counter = 0
    generation_rows: dict[str, dict[str, Any]] = {}
    current: tuple[str, str] = ("", "")
    failures: list[dict[str, str]] = []
    task_categories: dict[str, str] = {}

    def candidate(task: Task, seed: int) -> str:
        nonlocal counter, current
        torch.manual_seed(seed)
        encoded = cast(dict[str, Tensor], tokenizer(task.prompt, return_tensors="pt"))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = str(
            tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
        )
        generation_rows[task.id] = completion_diagnostics(
            task.prompt,
            completion,
            cast(list[int], generated[0, encoded["input_ids"].shape[1] :].tolist()),
            budget=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            stops=stops,
        )
        current = (task.id, completion)
        counter += 1
        print(f"Generated reference candidate {counter}/{len(tasks)}", file=sys.stderr, flush=True)
        return task.prompt + truncate_completion(completion, stops=stops)

    categories: Counter[str] = Counter()

    def diagnostic_execution(code: str, tests: str, mode: str) -> Result:
        result = run(code, tests, mode=mode)
        category = failure_category(result)
        categories[category] += 1
        task_categories[current[0]] = category
        if args.failure_recipient is not None and not result.passed and len(failures) < 5:
            failures.append({"id": current[0], "category": category, "completion": current[1]})
        return result

    report = evaluate(
        tasks,
        candidate,
        config=EvaluationConfig(max_new_tokens=args.max_new_tokens),
        checkpoint_hash=file_sha256(weights_path),
        tokenizer_hash=tokenizer_hash,
        execute=diagnostic_execution,
    )
    score = report["metrics"]["greedy_pass_at_1"]
    report["reference"] = {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "stops": stops,
        "published_pass_at_1": 0.0784,
        "published_marked_verified": False,
        "published_decoding_protocol": "not specified in model card",
        "difference_equivalent_tasks": abs(score - 0.0784) * 164,
        "within_two_tasks": len(tasks) == 164 and abs(score - 0.0784) * 164 <= 2,
        "model_license": "BigCode OpenRAIL-M v1",
        "device": "cpu",
        "threads": args.threads,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": str(transformers.__version__),
        "model_dtype": str(model.dtype),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "canonical_passed": canonical["task_count"],
        "outcome_categories": dict(categories),
        "per_task_categories": task_categories,
        "generation_diagnostics": {
            "policy": "effective-recorded-stop-policy",
            "task_count": len(generation_rows),
            "budget_reached": sum(row["budget_reached"] for row in generation_rows.values()),
            "budget_without_eos": sum(
                row["budget_without_eos"] for row in generation_rows.values()
            ),
            "eos_seen": sum(row["eos_seen"] for row in generation_rows.values()),
            "empty_retained_completions": sum(
                row["empty_retained_completion"] for row in generation_rows.values()
            ),
            "syntactically_valid_candidates": sum(
                row["candidate_syntax_valid"] for row in generation_rows.values()
            ),
            "stop_counts": dict(
                Counter(str(row["stop_index"]) for row in generation_rows.values())
            ),
            "per_task": generation_rows,
        },
    }
    save_report(report, args.output / "evaluation", scorecard=args.output / "scorecard.jsonl")
    # These records contain only public benchmark settings, hashes, and outcome statuses.
    # Console output preserves the run evidence without uploading caches, code, or weights.
    print("REFERENCE_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    if args.failure_recipient is not None:
        print(
            "ENCRYPTED_FAILURES=" + encrypted_failures(failures, args.failure_recipient), flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
