import json
from pathlib import Path

import pytest

from scratchcoder.data.filters import ContaminationIndex
from tools.header_licenses import PERMISSIVE, classify_header
from tools.real_samples import process_sample

BSD = """# Copyright fixture. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice.
# 2. Redistributions in binary form must reproduce the above copyright notice.
"""


@pytest.mark.parametrize("name", sorted(PERMISSIVE))
def test_recognized_permissive_spdx(name: str) -> None:
    result = classify_header(f"# SPDX-License-Identifier: {name}\npass\n")
    assert result.category == "permissive" and result.license_id == name


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("# Licensed under the Apache License, Version 2.0", "Apache-2.0"),
        ("# Released under the MIT license", "MIT"),
        (
            "# Permission is hereby granted, free of charge, "
            "to deal in the Software without restriction.",
            "MIT",
        ),
        (BSD, "BSD-2-Clause"),
        (
            BSD + "# Neither the name nor contributors may endorse or promote derived products.\n",
            "BSD-3-Clause",
        ),
        (
            "# Permission to use, copy, modify, and/or distribute this software "
            "for any purpose with or without fee is hereby granted.",
            "ISC",
        ),
        (
            "# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2\n"
            "# PSF hereby grants Licensee a nonexclusive royalty-free license.",
            "PSF-2.0",
        ),
        (
            "# This is free and unencumbered software released into the public domain.\n"
            "# Anyone is free to copy, modify, publish, use, compile, sell, or distribute "
            "this software for any purpose, commercial or non-commercial.",
            "Unlicense",
        ),
        ("# To the extent possible, rights are waived under CC0 1.0 Universal.", "CC0-1.0"),
    ],
)
def test_short_synthetic_permissive_grants(header: str, name: str) -> None:
    result = classify_header(header + "\npass\n")
    assert result.category == "permissive" and result.license_id == name


@pytest.mark.parametrize("name", ["GPL-2.0-only", "LGPL-2.1-only", "AGPL-3.0-only", "MPL-2.0"])
def test_copyleft_dropped_even_alongside_a_permissive_grant(name: str) -> None:
    result = classify_header(f"# SPDX-License-Identifier: MIT OR {name}\npass\n")
    assert result.category == "copyleft" and result.license_id is None


@pytest.mark.parametrize(
    "header",
    ["# Non-commercial use only", "# Proprietary software", "# Commercial use is prohibited"],
)
def test_restricted_header_is_not_admitted(header: str) -> None:
    assert classify_header(header + "\npass").category == "restricted"


@pytest.mark.parametrize(
    "header",
    [
        "# All rights reserved",
        "# SPDX-License-Identifier: Mystery-1.0",
        "# See LICENSE",
        "import os\n# SPDX-License-Identifier: MIT",
        "# SPDX-License-Identifier: CC-BY-4.0",
    ],
)
def test_unrecognized_or_absent_grant_is_separate_from_copyleft(header: str) -> None:
    result = classify_header(header + "\npass\n")
    assert result.category == "unrecognized" and result.license_id is None


def test_module_docstring_header_is_read_without_execution() -> None:
    assert (
        classify_header('"""Licensed under the MIT license."""\nraise RuntimeError').license_id
        == "MIT"
    )


def test_sample_class_attribution_examples_and_per_file_credit(tmp_path: Path) -> None:
    texts = [
        "# SPDX-License-Identifier: ISC\nanswer = 123456\n",
        "# SPDX-License-Identifier: MPL-2.0\nanswer = 2\n",
        "# Proprietary software\nanswer = 3\n",
    ]
    texts += ["# Fixture without a grant\n" + "pass\n" * 8 for _ in range(12)]
    texts += ["# SPDX-License-Identifier: MIT\n# auto-generated\nanswer = 4\n"]
    report = process_sample(
        [dict(id=str(i), content=text) for i, text in enumerate(texts)],
        domain="python",
        source="fixture",
        target=sum(len(t.encode()) for t in texts),
        contamination=ContaminationIndex([]),
        output=tmp_path / "sample",
    )
    decisions = report["primary_decisions"]
    assert decisions["license_copyleft"]["files"] == 1
    assert decisions["license_restricted"]["files"] == 1
    assert decisions["license_unrecognized"]["files"] == 12
    assert decisions["generated"]["files"] == 1
    assert decisions["kept"]["files"] == 1
    assert sum(item["text_bytes"] for item in decisions.values()) == sum(
        len(t.encode()) for t in texts
    )
    examples = json.loads((tmp_path / "sample/unrecognized-examples.json").read_text())["examples"]
    assert len(examples) == 10
    assert all(item["first_five_lines"] == texts[3].splitlines()[:5] for item in examples)
    row = json.loads((tmp_path / "sample/documents.jsonl").read_text())
    assert row["provenance"]["license"] == "ISC"
    assert row["provenance"]["license_class"] == "permissive"
