# ADR 0011: classify license headers separately from quality

Status: accepted for sample review. Date: 2026-10-20.

## Context

The real Python sample first required a recognized original-file license header.
That gate discarded files before the quality filters, conflating missing inline
evidence with bad content. A permissive repository can contain files without a
license header. Conversely, a dataset's license tag does not prove downstream
rights for every file.

## Decision

Replace the original blanket check with a header classifier. Recognize Apache-2.0,
MIT, BSD-2-Clause, BSD-3-Clause, ISC, PSF-2.0, Unlicense and CC0-1.0. Preserve the
license ID and recognition evidence per retained file. GPL/LGPL/AGPL/MPL markers
are a separate copyleft rejection; non-commercial/proprietary restrictions have
their own rejection class. An unsupported inline grant is unrecognized, never
silently approved. "All rights reserved" alone is not a restrictive-license
trigger: a recognized BSD grant that follows can be retained.

Inspect leading comments/module strings within the existing 16,384-character
window. Skip only known leading StarCoderData metadata lines (`<reponame>`,
`<filename>`, `<gh_stars>`) for classification, without changing stored text.
Recognize single supported SPDX IDs, explicit named declarations/grants and
specific grant patterns. Unsupported/compound declarations remain unrecognized;
restrictive markers take precedence. This conservative heuristic is not a general
license parser or legal warranty. The Unlicense's affirmative permission for
commercial or non-commercial purposes is not an NC restriction.

The earlier quality-only bypass was diagnostic only and is removed from the
current tool. Keep unrecognized examples (first five raw lines of ten files)
in ignored run storage, not Git. No source or full build is approved by a review.

Attribute each file and its complete UTF-8 bytes to one primary decision in
filter order. The pre-quality permissive class includes files later rejected by
quality rules and must not be added to final-kept counts. Do not change thresholds
to meet a desired retention percentage. Historical sample and frozen-tokenizer
artifacts remain immutable.

## Consequences and verification

Capacity reports must label which policy their retention describes. The historical
93.4% quality-only result is not the classifier's admitted-byte forecast. Authored
tests cover each class, per-file IDs, byte/file conservation, private examples,
BSD rights-reserved wording, metadata prefixes and no source execution. Real
corpus rights, final holdouts and build authorization remain separate conditions.
The [sample evidence](../real-samples.md) records the actual offline runs.

## Owner amendment: repository fallback (2026-10-20)

An absent inline license is **not** an unrecognized header. Ordinary module
comments/docstrings without licensing signals do not require an inline grant.
Use the explicit upstream `repository_license` metadata field, when available,
only in this case: D-05 allows MIT, Apache-2.0, BSD-2/3-Clause, CC0-1.0,
CC-BY-4.0, ODC-By-1.0 and CDLA-Permissive-2.0. Preserve that ID and evidence
per file. Count off-allowlist repository IDs separately. An unreadable explicit
header cannot be overridden by a permissive repository declaration.

Missing repository metadata is its own unresolved rejection, not a header
classification and not proof that a repository is unlicensed. The pinned local
Python Parquet has **no repository-license column**: its five fields are
`max_stars_repo_path`, `max_stars_repo_name`, `max_stars_count`, `id`, `content`.
Arrow/Hugging Face schema metadata adds no license mapping. The earlier text
prefix parsing fix did not supply licenses. Do not infer original rights from
the dataset tag or a present-day repository lookup. A provenance-backed local
mapping is needed to admit the currently unresolved no-header files.

## Owner amendment: upstream repository licenses (2026-10-24)

The owner directed recovering the missing repository metadata from upstream and
confirmed gated access to `bigcode/the-stack-dedup`, which carries the same Terms
of Use for The Stack as StarCoderData. Its v1.2 `data/python` files (revision
`17cad72c886a2858e08d4c349a00d6466f54df63`, byte-identical to the v1.2 release
candidate) record `max_stars_repo_licenses` for each file. `tools/stack_licenses.py`
reads only `max_stars_repo_name`, `max_stars_repo_path` and that license column
with HTTPS range requests. A request that would overlap a `content` column chunk
is refused before it is sent. Every column-projected read must also fetch the
Parquet footer, which stores per-row-group min/max statistics for every column;
for `content` those values can be short complete files. The 2026-10-24 run held
footers in memory only and stored none of their text; one measured footer is
169,400 of 545,052 bytes such text, about 24 MB across the 144 footers. Later
runs report the exact content-statistics bytes per footer.

The join key is the exact (repository, path) pair from StarCoderData's own
provenance. Duplicate upstream keys are admitted only when their license lists
agree. A file whose path is absent upstream but whose repository is present is a
diagnostic, never an admission, and neither is any present-day repository lookup.
The raw detector candidates in `bigcode/the-stack-metadata` are not used: one
license file yields several look-alike IDs there, and choosing among them would be
a new attribution rule.

The D-05 allowlist applies to every listed ID: a repository list is admitted only
when all of its IDs are allowed, because the repository may hold code under each
one. The admitted ID is the IDs joined with " AND ". Off-allowlist IDs are counted
per ID. Explicit headers keep precedence: copyleft, restricted and unrecognized
headers are still rejected. A leading byte-order mark is ignored, a license block
after `from __future__` imports is read, a module-level `__license__` string is an
explicit declaration, and a leading header that cannot be tokenized is
unrecognized rather than absent, so repository metadata never overrides it.

**Correction, not a new decision (2026-10-24):** ISC and Unlicense were always
allowed under D-05. The allowlist written in the 2026-10-20 amendment above, in
the private data rules and in the code omitted them. The repository allowlist and
the manifest's training-license list now read MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause, ISC, Unlicense, CC0-1.0, CC-BY-4.0, ODC-By-1.0 and
CDLA-Permissive-2.0. PSF-2.0 remains an inline-header grant only.
