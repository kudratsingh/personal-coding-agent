"""Conservative, non-executing classification of Python leading license headers."""

import io
import re
import tokenize
from dataclasses import dataclass

PERMISSIVE = frozenset(
    {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "PSF-2.0", "Unlicense", "CC0-1.0"}
)
HEADER_CHARACTERS = 16384


@dataclass(frozen=True)
class HeaderLicense:
    category: str
    license_id: str | None
    evidence: str


def leading_header(text: str) -> str:
    """Read comments and leading string literals only; never execute or import code."""
    parts = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text[:HEADER_CHARACTERS]).readline):
            if token.type in {tokenize.COMMENT, tokenize.STRING}:
                parts.append(token.string)
            elif token.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}:
                break
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return "\n".join(parts)


def classify_header(text: str) -> HeaderLicense:
    header = leading_header(text)
    normalized = " ".join(re.sub(r"(?m)^\s*#\s?", "", header).casefold().split())
    # A public-domain grant explicitly permits both purposes; it is not an NC restriction.
    restriction_text = re.sub(
        r"(?:for any purpose,?\s*)?(?:both\s+)?commercial (?:or|and) non-commercial",
        "commercial use permitted",
        normalized,
    )
    if re.search(
        r"\b(?:agpl|lgpl|gpl|mpl)(?:[- ]\d|\b)|mozilla public license|"
        r"gnu (?:general|lesser|library|affero) public license",
        restriction_text,
    ):
        return HeaderLicense("copyleft", None, "explicit copyleft marker")
    if re.search(
        r"non[- ]?commercial|cc-by-nc|proprietary|confidential|"
        r"not (?:licensed |permitted )?for commercial|commercial use (?:is )?(?:prohibited|not permitted)",
        restriction_text,
    ):
        return HeaderLicense("restricted", None, "non-commercial or proprietary restriction")
    identifiers = re.findall(r"SPDX-License-Identifier:\s*([^\r\n]+)", header, re.I)
    if identifiers:
        names = {value.strip().removesuffix("*/").strip() for value in identifiers}
        if len(names) == 1 and names <= PERMISSIVE:
            return HeaderLicense("permissive", next(iter(names)), "single recognized SPDX id")
        return HeaderLicense("unrecognized", None, "unsupported or compound SPDX declaration")

    # Explicit named-license grants, not a bare name appearing in unrelated code.
    for name, pattern in (
        ("Apache-2.0", r"apache (?:software )?license,? (?:version |v)?2\.0"),
        ("MIT", r"mit(?: license)?"),
        ("BSD-2-Clause", r"(?:bsd[- ]2[- ]clause|2[- ]clause bsd)(?: license)?"),
        ("BSD-3-Clause", r"(?:bsd[- ]3[- ]clause|3[- ]clause bsd)(?: license)?"),
        ("ISC", r"isc(?: license)?"),
        (
            "PSF-2.0",
            r"(?:psf(?: license)?|python software foundation license)(?: version)? 2(?:\.0)?",
        ),
        ("Unlicense", r"(?:the )?unlicense"),
        ("CC0-1.0", r"cc0(?:[- ]1\.0)?(?: universal)?"),
    ):
        if re.search(
            r"\b(?:licensed|released|distributed) under (?:the )?" + pattern + r"\b", normalized
        ):
            return HeaderLicense("permissive", name, "explicit named-license grant")

    if (
        "permission is hereby granted, free of charge" in normalized
        and "software" in normalized
        and (
            "without restriction" in normalized
            or (
                "the above copyright notice" in normalized
                and "the software is provided" in normalized
            )
        )
    ):
        return HeaderLicense("permissive", "MIT", "MIT permission grant")
    if (
        "redistribution and use in source and binary forms" in normalized
        and "are permitted" in normalized
        and "redistributions of source code" in normalized
        and "redistributions in binary form" in normalized
        and "copyright notice" in normalized
        and "advertising materials" not in normalized
    ):
        name = (
            "BSD-3-Clause"
            if re.search(r"neither the name|names?.*endorse or promote", normalized)
            else "BSD-2-Clause"
        )
        return HeaderLicense("permissive", name, "BSD redistribution grant and conditions")
    if (
        re.search(r"permission to use, copy, modify,? and(?:/or| or)? distribute", normalized)
        and "for any purpose with or without fee" in normalized
        and "is hereby granted" in normalized
    ):
        return HeaderLicense("permissive", "ISC", "ISC permission grant")
    if (
        "python software foundation license version 2" in normalized
        and re.search(r"psf.{0,20}(?:hereby )?grants licensee", normalized)
        and "royalty-free" in normalized
    ):
        return HeaderLicense("permissive", "PSF-2.0", "PSF version-2 grant")
    if (
        "free and unencumbered software released into the public domain" in normalized
        and "anyone is free to copy, modify, publish, use, compile, sell, or distribute"
        in normalized
    ):
        return HeaderLicense("permissive", "Unlicense", "Unlicense public-domain grant")
    if re.search(
        r"(?:creativecommons.org/publicdomain/zero/1\.0|cc0 1\.0 universal)", normalized
    ) and re.search(r"waiv|dedicat|released under", normalized):
        return HeaderLicense("permissive", "CC0-1.0", "CC0 public-domain dedication")
    return HeaderLicense("unrecognized", None, "no recognized permissive grant in leading header")
