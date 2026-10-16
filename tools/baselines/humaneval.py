"""Pinned, CPU-only reference check using the project's isolated evaluation harness."""

import argparse
import gzip
import hashlib
import importlib
import json
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


def truncate_completion(text: str) -> str:
    offsets = [text.find(stop) for stop in STOPS if stop in text]
    return text[: min(offsets)] if offsets else text


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
    args = parser.parse_args()
    if not 1 <= args.limit <= 164 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error("invalid limit, thread count, or token budget")
    args.output.mkdir(parents=True, exist_ok=False)
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

    def candidate(task: Task, seed: int) -> str:
        nonlocal counter
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
        counter += 1
        print(f"Generated reference candidate {counter}/{len(tasks)}", file=sys.stderr, flush=True)
        return task.prompt + truncate_completion(completion)

    categories: Counter[str] = Counter()

    def diagnostic_execution(code: str, tests: str, mode: str) -> Result:
        result = run(code, tests, mode=mode)
        categories[failure_category(result)] += 1
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
        "stops": STOPS,
        "published_pass_at_1": 0.0784,
        "published_marked_verified": False,
        "published_decoding_protocol": "not specified in model card",
        "difference_equivalent_tasks": abs(score - 0.0784) * 164,
        "within_two_tasks": len(tasks) == 164 and abs(score - 0.0784) * 164 <= 2,
        "model_license": "BigCode OpenRAIL-M v1",
        "device": "cpu",
        "threads": args.threads,
        "canonical_passed": canonical["task_count"],
        "outcome_categories": dict(categories),
    }
    save_report(report, args.output / "evaluation", scorecard=args.output / "scorecard.jsonl")
    # These records contain only public benchmark settings, hashes, and outcome statuses.
    # Console output preserves the run evidence without uploading caches, code, or weights.
    print("REFERENCE_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
