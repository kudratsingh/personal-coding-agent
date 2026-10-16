"""Conservative reviewed-schema adapters requiring declared source provenance."""

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.tasks import Task


def adapt_taco_io_row(
    row: Mapping[str, Any],
    provenance: ManifestEntry,
    *,
    upstream_split: str,
    split: str,
    public_count: int = 1,
) -> Task:
    """Conservative EASY stdin/stdout subset; no execution or metadata eval."""
    provenance.validate()
    if provenance.source != "BAAI/TACO" or upstream_split not in {"train", "test"}:
        raise ValueError("TACO provenance and explicit original split are required")
    if provenance.purpose == "eval" and split == "train":
        raise ValueError("evaluation-only provenance cannot become training data")
    if upstream_split == "test" and split == "train":
        raise ValueError("upstream TACO test tasks cannot become training data")
    if (
        row["difficulty"] != "EASY"
        or row["starter_code"]
        or str(row.get("picture_num", "0")) != "0"
    ):
        raise ValueError("only EASY standalone text I/O tasks are supported")
    cases = json.loads(row["input_output"])
    solutions = json.loads(row["solutions"])
    if (
        not isinstance(cases, dict)
        or cases.get("fn_name")
        or set(cases) - {"inputs", "outputs", "fn_name"}
        or not isinstance(cases.get("inputs"), list)
        or not isinstance(cases.get("outputs"), list)
        or len(cases["inputs"]) != len(cases["outputs"])
        or not 2 <= len(cases["inputs"]) <= 100
        or not all(isinstance(value, str) for value in [*cases["inputs"], *cases["outputs"]])
    ):
        raise ValueError("unsupported TACO I/O tests")
    if (
        not isinstance(solutions, list)
        or not solutions
        or any(not isinstance(code, str) or not code.strip() for code in solutions)
    ):
        raise ValueError("TACO needs nonempty reference solution strings")
    unique: dict[str, str] = {}
    for given, expected in zip(cases["inputs"], cases["outputs"], strict=True):
        if given in unique and unique[given].strip() != expected.strip():
            raise ValueError("conflicting outputs for identical TACO input")
        unique[given] = expected
    items = [{"input": given, "output": expected} for given, expected in unique.items()]
    if type(public_count) is not int or not 1 <= public_count < len(items):
        raise ValueError("distinct public and hidden I/O cases required")
    question = row["question"]
    if not isinstance(question, str) or not isinstance(row["url"], str):
        raise ValueError("TACO question and stable source URL must be strings")
    identity = hashlib.sha256((row["url"] + "\n" + question).encode()).hexdigest()
    return Task(
        id="BAAI/TACO/" + identity,
        prompt=question,
        public_tests=json.dumps(items[:public_count]),
        hidden_tests=json.dumps(items[public_count:]),
        source=provenance.source,
        source_revision=provenance.revision,
        license=provenance.license,
        written_by="human",
        split=split,
        mode="io",
        solution=solutions[0],
    )


def canonical_id(identifier: str) -> str:
    family, separator, rest = identifier.partition("/")
    return (
        family.casefold() + "/" + rest
        if separator and family.casefold() in {"mbpp", "humaneval"}
        else identifier
    )


def assertion_partitions(
    value: object, *, setup: str = "", public_count: int = 1
) -> tuple[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("expected a JSON assertion list") from error
    if not isinstance(value, list) or not all(isinstance(test, str) for test in value):
        raise ValueError("tests must be strings in an assertion list")
    assertions: list[str] = []
    imports: list[str] = []
    try:
        setup_nodes = ast.parse(setup).body
    except SyntaxError as error:
        raise ValueError("unsupported test setup") from error
    if any(not isinstance(node, (ast.Import, ast.ImportFrom)) for node in setup_nodes):
        raise ValueError("shared setup must contain only imports")
    for text in value:
        try:
            nodes = ast.parse(text).body
        except SyntaxError as error:
            raise ValueError("unsupported test syntax") from error
        for node in nodes:
            rendered = ast.unparse(node)
            if isinstance(node, ast.Assert):
                if rendered not in assertions:
                    assertions.append(rendered)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if rendered not in imports:
                    imports.append(rendered)
            else:
                raise ValueError(
                    "only imports and explicit assertions are supported by this adapter"
                )
    if not 1 <= public_count < len(assertions):
        raise ValueError("need distinct public and hidden assertions")
    prefix = "\n".join([setup, *imports]) + "\n"
    return prefix + "\n".join(assertions[:public_count]), prefix + "\n".join(
        assertions[public_count:]
    )


def clean_completion(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text


def adapt_row(
    source: str,
    row: Mapping[str, Any],
    provenance: ManifestEntry,
    *,
    split: str = "train",
    public_count: int = 1,
) -> Task:
    provenance.validate()
    if provenance.source != source:
        raise ValueError("source and provenance differ")
    if provenance.purpose == "eval" and split == "train":
        raise ValueError("evaluation-only provenance cannot become training data")
    setup = ""
    solution: str | None
    if source == "nvidia/OpenCodeInstruct":
        identifier, prompt, solution = str(row["id"]), row["input"], row["output"]
        tests, author = row["unit_tests"], "model"
    elif source == "TIGER-Lab/AceCode-87K":
        identifier, prompt = str(row["id"]), row["question"]
        tests, author = row["test_cases"], "model"
        solution = next(
            (
                clean_completion(item["completion"])
                for item in row["inferences"]
                if item["pass_rate"] == 1.0
            ),
            None,
        )
    elif source == "google-research-datasets/mbpp":
        identifier, prompt, solution = "mbpp/" + str(row["task_id"]), row["text"], row["code"]
        tests, author, setup = row["test_list"], "human", row.get("test_setup_code", "")
        challenge = row.get("challenge_test_list", [])
        if not isinstance(tests, list) or not isinstance(challenge, list):
            raise ValueError("MBPP primary/challenge tests must be explicit assertion arrays")
        tests = [*tests, *challenge]
    elif source == "openai/human-eval":
        if split not in {"eval", "test"} or provenance.purpose != "eval":
            raise ValueError("HumanEval is evaluation-only")
        entry = row["entry_point"]
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError("invalid benchmark entry point")
        return Task(
            id=canonical_id(row["task_id"]),
            prompt=row["prompt"],
            public_tests=f"assert callable({entry})",
            hidden_tests=row["test"] + f"\ncheck({entry})",
            source=source,
            license=provenance.license,
            written_by="human",
            split=split,
            solution=row["prompt"] + row["canonical_solution"],
            entry_point=entry,
            source_revision=provenance.revision,
        )
    else:
        raise ValueError("source schema has not been reviewed")
    public, hidden = assertion_partitions(tests, setup=setup, public_count=public_count)
    task_id = identifier if source.endswith("/mbpp") else source + "/" + identifier
    return Task(
        id=task_id,
        prompt=prompt,
        public_tests=public,
        hidden_tests=hidden,
        source=source,
        license=provenance.license,
        written_by=author,
        split=split,
        solution=None if solution is None else clean_completion(solution),
        source_revision=provenance.revision,
    )
