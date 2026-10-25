import json
from pathlib import Path

import pytest

from scratchcoder.data.filters import ContaminationIndex
from tools.header_licenses import (
    PERMISSIVE,
    classify_file,
    classify_header,
    repository_license_ids,
)
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


def test_upstream_metadata_does_not_hide_an_explicit_license_header() -> None:
    text = "<gh_stars>1-10\n<filename>fixture.py\n# License: MIT license\nanswer = 123\n"
    assert classify_header(text).license_id == "MIT"


@pytest.mark.parametrize("prefix", ["", "# helper utility\n", '"""Helper utility."""\n'])
def test_no_license_header_uses_allowlisted_repository_metadata(prefix: str) -> None:
    text = prefix + "import os\n# SPDX-License-Identifier: MIT\n"
    assert classify_header(text).category == "no_header"
    result = classify_file(text, "Apache-2.0")
    assert result.category == "permissive" and result.license_id == "Apache-2.0"
    assert "repository" in result.evidence


@pytest.mark.parametrize("name", ["GPL-3.0-only", "CC-BY-NC-4.0", "Mystery-1.0"])
def test_repository_license_outside_allowlist_is_attributed(name: str) -> None:
    result = classify_file("answer = 123456789\n", name)
    assert result.category == "repository_disallowed" and result.license_id == name


def test_missing_repository_license_is_not_unrecognized_header() -> None:
    assert classify_file("import os\n", None).category == "missing_repository_license"
    assert classify_file("# License: Mystery-1.0\npass\n", "MIT").category == "unrecognized"
    assert (
        classify_file("# SPDX-License-Identifier: GPL-3.0-only\npass\n", "MIT").category
        == "copyleft"
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


def test_repository_license_lists_need_every_id_on_the_allowlist() -> None:
    text = "answer = 123456789\n"
    both = classify_file(text, ["MIT", "Apache-2.0", "MIT"])
    assert both.category == "permissive" and both.license_id == "Apache-2.0 AND MIT"
    assert both.evidence == "upstream repository license metadata"
    mixed = classify_file(text, ("MIT", "Zlib", "BSL-1.0"))
    assert mixed.category == "repository_disallowed"
    assert mixed.license_id == "BSL-1.0 AND MIT AND Zlib"
    assert mixed.evidence.endswith("BSL-1.0, Zlib")  # Only the blocking IDs are named.
    # D-05 correction (2026-10-24): ISC and Unlicense were always allowed.
    corrected = classify_file(text, ["ISC", "Unlicense", "MIT"])
    assert corrected.category == "permissive"
    assert corrected.license_id == "ISC AND MIT AND Unlicense"
    # An empty upstream list is no verified repository license, not an approval.
    assert classify_file(text, []).category == "missing_repository_license"
    assert classify_file(text, ["", "  "]).category == "missing_repository_license"


def test_explicit_header_still_wins_over_a_repository_list() -> None:
    copyleft = "# SPDX-License-Identifier: GPL-3.0-only\npass\n"
    assert classify_file(copyleft, ["MIT", "Apache-2.0"]).category == "copyleft"
    inline = "# SPDX-License-Identifier: BSD-3-Clause\npass\n"
    result = classify_file(inline, ["MIT"])
    assert result.license_id == "BSD-3-Clause" and "repository" not in result.evidence


def test_repository_license_ids_normalize_and_reject_non_text() -> None:
    assert repository_license_ids(None) == ()
    assert repository_license_ids("MIT") == ("MIT",)
    assert repository_license_ids([" MIT ", "Apache-2.0", "MIT"]) == ("Apache-2.0", "MIT")
    with pytest.raises(ValueError, match="strings"):
        repository_license_ids(["MIT", 3])  # type: ignore[list-item]


def test_a_byte_order_mark_never_hides_an_explicit_header() -> None:
    agpl = "\ufeff# License AGPL-3.0 or later (https://www.gnu.org/licenses/agpl).\n"
    agpl += "from odoo import models\n"
    assert classify_file(agpl, ["MIT"]).category == "copyleft"
    mit = "\ufeff# SPDX-License-Identifier: MIT\nanswer = 1\n"
    result = classify_file(mit, ["Apache-2.0"])
    assert result.category == "permissive" and result.license_id == "MIT"
    assert "repository" not in result.evidence


def test_an_unreadable_leading_header_is_unrecognized_not_absent() -> None:
    truncated = '"""' + "x" * 20000 + '\n"""\nanswer = 1\n'  # Cut by the 16,384-character window.
    result = classify_file(truncated, ["MIT"])
    assert result.category == "unrecognized" and result.evidence == "unreadable leading header"
    assert (
        classify_file("import os\n", ["MIT"]).category == "permissive"
    )  # Ordinary code still falls back.
    assert classify_file("", None).category == "missing_repository_license"


def test_license_blocks_after_future_imports_are_read() -> None:
    text = (
        "#!/usr/bin/python\nfrom __future__ import (absolute_import, division)\n"
        "# SPDX-License-Identifier: GPL-3.0-only\nimport os\n"
    )
    assert classify_file(text, ["MIT"]).category == "copyleft"
    plain = "from __future__ import annotations\nimport os\n"
    assert classify_file(plain, ["MIT"]).category == "permissive"


@pytest.mark.parametrize(
    ("value", "category"),
    [
        ("GPL", "copyleft"),
        ("GPLv3", "copyleft"),
        ("Proprietary", "restricted"),
        ("Mystery", "unrecognized"),
    ],
)
def test_module_license_dunder_is_an_explicit_declaration(value: str, category: str) -> None:
    text = f'#encoding: utf8\n__author__ = "someone"\n__license__ = "{value}"\nimport os\n'
    result = classify_file(text, ["MIT"])
    assert result.category == category
    assert "repository" not in result.evidence


def test_permissive_license_dunder_keeps_its_own_id() -> None:
    result = classify_file('__license__ = "MIT"\nimport os\n', ["Apache-2.0"])
    assert (result.category, result.license_id, result.evidence) == (
        "permissive",
        "MIT",
        "__license__ declaration",
    )


@pytest.mark.parametrize(
    "header", ["# License: GPLv3", "# Licensed under the AGPLv3", "# LGPL-2.1"]
)
def test_versioned_copyleft_spellings_are_copyleft(header: str) -> None:
    assert classify_header(header + "\nimport os\n").category == "copyleft"


def test_d05_correction_admits_0bsd_and_wtfpl_repository_licenses() -> None:
    # Owner correction 2026-10-24 (not a new decision): both were always allowed.
    text = "answer = 123456789\n"
    for listed, joined in ((["0BSD"], "0BSD"), (["WTFPL", "MIT"], "MIT AND WTFPL")):
        result = classify_file(text, listed)
        assert result.category == "permissive" and result.license_id == joined
    # The other off-allowlist IDs stay out.
    assert classify_file(text, ["0BSD", "ECL-2.0"]).category == "repository_disallowed"
    # The inline header classifier is unchanged by this correction.
    assert classify_header("# SPDX-License-Identifier: 0BSD\nanswer = 1\n").category != "permissive"
