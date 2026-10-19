import json
from pathlib import Path

import pytest

from scratchcoder.data.filters import ContaminationIndex
from tools.real_samples import declared_license, process_sample, sample_rows


def test_sample_whole_document_boundary_and_insufficient_bytes() -> None:
    rows = [{"text": "éé"}, {"text": "abc"}, {"text": "not consumed"}]
    assert list(sample_rows(rows, "web", 5)) == rows[:2]
    assert list(sample_rows(rows, "web", 4)) == rows[:1]
    with pytest.raises(ValueError, match="before target"):
        list(sample_rows(rows[:1], "web", 5))


def test_original_file_license_required_not_dataset_label() -> None:
    assert declared_license("# SPDX-License-Identifier: MIT\npass") == "MIT"
    assert (
        declared_license("# Licensed under the Apache License, Version 2.0\npass") == "Apache-2.0"
    )
    for text in (
        "# dataset is permissively licensed\npass",
        "# SPDX-License-Identifier: MIT OR GPL-3.0\npass",
        "# SPDX-License-Identifier: GPL-3.0-only\npass",
        "# SPDX-License-Identifier: MIT\n# SPDX-License-Identifier: Apache-2.0\npass",
    ):
        assert declared_license(text) is None


def test_license_filter_contamination_counts_split_and_provenance(tmp_path: Path) -> None:
    prefix = "# SPDX-License-Identifier: MIT\n"
    benchmark = "This authored benchmark marker must never enter our tokenizer training data"
    clean = prefix + "def fixture(value):\n    return value + 123\n"
    texts = [
        clean,
        clean,
        "def unlicensed(value): return value\n",
        prefix + "# auto-generated\n" + "pass\n" * 10,
        prefix + benchmark,
    ]
    rows = [
        dict(
            id=str(i),
            content=text,
            max_stars_repo_name="fixture/repo",
            max_stars_repo_path=f"{i}.py",
        )
        for i, text in enumerate(texts)
    ]
    report = process_sample(
        rows,
        domain="python",
        source="fixture",
        target=sum(len(t.encode()) for t in texts),
        contamination=ContaminationIndex([benchmark]),
        output=tmp_path / "review",
    )
    assert report["sample_documents"] == 5 and report["license_admitted_documents"] == 4
    assert report["missing_or_unapproved_file_license"] == 1
    assert report["filters"]["duplicate"] == 1
    assert report["filters"]["generated"] == 1
    assert report["filters"]["contamination"] == 1
    assert report["kept_documents"] == 1
    assert report["tokens"] == report["kept_text_bytes"] == len(clean.encode())
    document = json.loads((tmp_path / "review/documents.jsonl").read_text())
    assert document["text"] == clean and document["provenance"]["repository"] == "fixture/repo"
    assert document["provenance"]["path"] == "0.py"
    assert document["provenance"]["license"] == "MIT"
    assert sum(value["documents"] for value in report["splits"].values()) == 1
    assert "pre-merge" in report["token_count_protocol"]
