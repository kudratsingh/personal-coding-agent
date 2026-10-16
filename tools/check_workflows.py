"""Offline runner-declaration policy; not a billing or arbitrary-command safety check."""

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RUNNERS = frozenset({"ubuntu-latest", "macos-latest"})
MATRIX_RUNNER = re.compile(r"\$\{\{\s*matrix\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def validate_workflow(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), dict) or not value["jobs"]:
        raise ValueError("workflow requires a nonempty jobs mapping")
    for name, job in value["jobs"].items():
        if not isinstance(job, dict) or "uses" in job:
            raise ValueError(f"job {name!r}: reusable/invalid jobs require runner-policy review")
        runner = job.get("runs-on")
        if isinstance(runner, str) and runner in ALLOWED_RUNNERS:
            continue
        match = MATRIX_RUNNER.fullmatch(runner) if isinstance(runner, str) else None
        if match is None:
            raise ValueError(f"job {name!r}: runner declaration requires owner review")
        strategy = job.get("strategy")
        matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
        if not isinstance(matrix, dict) or "include" in matrix or "exclude" in matrix:
            raise ValueError(f"job {name!r}: only simple literal runner matrices are reviewed")
        runners = matrix.get(match[1])
        if (
            not isinstance(runners, list)
            or not runners
            or any(not isinstance(label, str) or label not in ALLOWED_RUNNERS for label in runners)
        ):
            raise ValueError(f"job {name!r}: matrix runners require owner review")


def main() -> int:
    directory = ROOT / ".github" / "workflows"
    if directory.is_symlink() or directory.parent.is_symlink():
        print("Workflow check failed: symlink workflow directory is not read.")
        return 1
    files = sorted({*directory.glob("*.yml"), *directory.glob("*.yaml")})
    if not files:
        print("Workflow check failed: no workflow files found.")
        return 1
    errors = []
    for path in files:
        try:
            if path.is_symlink():
                raise ValueError("symlink workflow is not read")
            validate_workflow(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
            # YAML parser errors can contain source excerpts; never print their text.
            detail = str(error) if type(error) is ValueError else type(error).__name__
            errors.append(f"{path.name!r}: {detail}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked runner declarations in {len(files)} workflows; billing/commands not audited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
