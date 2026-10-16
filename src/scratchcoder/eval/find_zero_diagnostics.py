"""Diagnostic instrumentation, explicitly separate from benchmark scoring."""

import ast
import hashlib
import json
from collections.abc import Callable
from typing import Any

from scratchcoder.data.evalplus import EvalPlusRelease
from scratchcoder.data.evalplus_corrections import CORRECTED_ASSERTION, correct_find_zero
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Result

MARKER = "FIND_ZERO_DIAGNOSTIC="
DRIVER = """
import json as _diagnostic_json
_diagnostic_record = {"case_count": 0, "passed": 0, "residual_failures": 0,
                      "exceptions": {}, "failure_examples": []}
def _diagnose_case(candidate, inp, exp, index):
    record = _diagnostic_record
    record["case_count"] += 1
    example = {"index": index, "coefficient_count": len(inp[0]),
               "highest_coefficient_nonzero": inp[0][-1] != 0,
               "linear_coefficient_nonzero": len(inp[0]) > 1 and inp[0][1] != 0}
    try:
        root = candidate(*inp)
        residual = _poly(*inp, root)
        if abs(residual) <= 0.0001:
            record["passed"] += 1
            return
        record["residual_failures"] += 1
        try:
            expected_residual = repr(_poly(*inp, exp))[:80]
        except Exception:
            expected_residual = "unavailable"
        example.update(status="residual_failure", candidate_root=repr(root)[:80],
                       candidate_residual=repr(residual)[:80],
                       release_expected_root_residual=expected_residual)
    except Exception as error:
        category = type(error).__name__
        if category not in {"ZeroDivisionError", "OverflowError", "ValueError", "TypeError"}:
            category = "OtherException"
        record["exceptions"][category] = record["exceptions"].get(category, 0) + 1
        example.update(status="exception", category=category)
    if len(record["failure_examples"]) < 8:
        record["failure_examples"].append(example)
"""


def diagnose_find_zero(
    task: Task, specification: EvalPlusRelease, *, execute: Callable[[str, str, str], Result]
) -> dict[str, Any]:
    corrected, correction = correct_find_zero([task], specification)
    tree = ast.parse(task.hidden_tests)
    check = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check"
    )
    assignments = {
        node.targets[0].id: node
        for node in check.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    inputs_node = assignments["inputs"]
    inputs = ast.literal_eval(inputs_node.value)
    results = ast.literal_eval(assignments["results"].value)
    if not 0 < len(inputs) <= 4096 or len(inputs) != len(results):
        raise ValueError("diagnostic input/result counts differ")
    literal = ast.get_source_segment(task.hidden_tests, inputs_node)
    assert literal is not None
    script = (
        DRIVER
        + corrected[0].hidden_tests.replace(
            CORRECTED_ASSERTION, "_diagnose_case(candidate, inp, exp, i)", 1
        )
        + '\nprint("FIND_ZERO_DIAGNOSTIC=" + _diagnostic_json.dumps(_diagnostic_record))\n'
    )
    metadata: dict[str, Any] = {
        "diagnostic_only": True,
        "benchmark_acceptance": False,
        "oracle_correction": correction,
        "canonical_source_sha256": hashlib.sha256((task.solution or "").encode()).hexdigest(),
        "diagnostic_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "input_assignment_sha256": hashlib.sha256(literal.encode()).hexdigest(),
        "input_count": len(inputs),
        "input_domain": {
            "odd_coefficient_counts": sum(len(inp[0]) % 2 == 1 for inp in inputs),
            "zero_highest_coefficients": sum(inp[0][-1] == 0 for inp in inputs),
            "zero_linear_coefficients": sum(len(inp[0]) > 1 and inp[0][1] == 0 for inp in inputs),
        },
    }
    result = execute(task.solution or "", script, task.mode)
    metadata.update(
        status="completed" if result.passed else "failed",
        sandbox_status=result.status,
        failure_category=failure_category(result),
    )
    if not result.passed:
        return metadata
    lines = [
        line.removeprefix(MARKER) for line in result.stdout.splitlines() if line.startswith(MARKER)
    ]
    if len(lines) != 1:
        raise ValueError("one complete diagnostic marker required")
    observed = json.loads(lines[0])
    if (
        not isinstance(observed, dict)
        or set(observed)
        != {"case_count", "passed", "residual_failures", "exceptions", "failure_examples"}
        or any(
            type(observed[key]) is not int or observed[key] < 0
            for key in ("case_count", "passed", "residual_failures")
        )
        or not isinstance(observed["exceptions"], dict)
        or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in observed["exceptions"].items()
        )
        or not isinstance(observed["failure_examples"], list)
        or len(observed["failure_examples"]) > 8
        or observed["case_count"] != len(inputs)
        or observed["passed"] + observed["residual_failures"] + sum(observed["exceptions"].values())
        != len(inputs)
    ):
        raise ValueError("complete diagnostic case accounting required")
    allowed_categories = {
        "ZeroDivisionError",
        "OverflowError",
        "ValueError",
        "TypeError",
        "OtherException",
    }
    if not set(observed["exceptions"]) <= allowed_categories:
        raise ValueError("unsupported diagnostic exception category")
    seen = set()
    base_fields = {
        "index",
        "coefficient_count",
        "highest_coefficient_nonzero",
        "linear_coefficient_nonzero",
        "status",
    }
    numeric_fields = {"candidate_root", "candidate_residual", "release_expected_root_residual"}
    for example in observed["failure_examples"]:
        if not isinstance(example, dict):
            raise ValueError("invalid diagnostic example")
        fields = base_fields | (
            {"category"} if example.get("status") == "exception" else numeric_fields
        )
        index = example.get("index")
        if (
            set(example) != fields
            or type(index) is not int
            or not 0 <= index < len(inputs)
            or index in seen
            or type(example["coefficient_count"]) is not int
            or example["coefficient_count"] != len(inputs[index][0])
            or type(example["highest_coefficient_nonzero"]) is not bool
            or example["highest_coefficient_nonzero"] != (inputs[index][0][-1] != 0)
            or type(example["linear_coefficient_nonzero"]) is not bool
            or example["linear_coefficient_nonzero"]
            != (len(inputs[index][0]) > 1 and inputs[index][0][1] != 0)
            or example["status"] not in {"exception", "residual_failure"}
            or example.get("category", "OtherException") not in allowed_categories
        ):
            raise ValueError("invalid diagnostic example fields")
        for field in numeric_fields & set(example):
            value = example[field]
            if not isinstance(value, str) or len(value) > 80:
                raise ValueError("bounded numeric diagnostics required")
            if field == "release_expected_root_residual" and value == "unavailable":
                continue
            try:
                float(value)
            except ValueError:
                raise ValueError("numeric diagnostic values required") from None
        seen.add(index)
    metadata["observations"] = observed
    return metadata
