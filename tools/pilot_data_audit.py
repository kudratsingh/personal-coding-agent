"""Audit a verified pilot data binding for exact token-level validation/training overlap.

The audit re-runs ``tools.pilot_data.verify_binding`` and then checks each validation
document in two ways: whether its full token content is an exact repeat of a training
document (under any document ID), and whether its validation window (the first
``sequence_length + 1`` tokens, as ``scratchcoder.train.validation.shard_windows`` takes
them, minus the leading BOS) appears verbatim as a contiguous token run anywhere in the
training stream, document boundaries ignored. Exact training-document repeats are counted
too.

This is an exact-containment check, not a near-duplicate detector: it reports nothing about
paraphrases, partially overlapping n-grams, text that tokenizes differently in another
context, or validation content that begins after the audited window. The report is written
atomically to ``--output`` whether or not a leak is found; a leak or a verification failure
exits with status 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools.pilot_data import LeakageError, verify_binding


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binding", type=Path, required=True, help="path to binding.json")
    parser.add_argument("--output", type=Path, required=True, help="audit report JSON to write")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--validation-documents",
        type=int,
        default=None,
        help="validation documents to audit (default: all bound validation documents)",
    )
    args = parser.parse_args(argv)
    try:
        binding_sha256 = file_sha256(args.binding)
        try:
            verified = verify_binding(
                args.binding,
                leakage_audit=True,
                sequence_length=args.sequence_length,
                validation_documents=args.validation_documents,
            )
        except LeakageError as exc:
            atomic_json(args.output, {"binding_sha256": binding_sha256, **exc.audit})
            print(f"pilot-data-audit: {exc}; report written to {args.output}", file=sys.stderr)
            return 1
        audit = verified["leakage_audit"]
        atomic_json(args.output, {"binding_sha256": binding_sha256, **audit})
    except (OSError, ValueError) as exc:
        print(f"pilot-data-audit: verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"pilot-data-audit: no leak; {audit['validation_documents_audited']} validation "
        f"documents audited against {audit['train_tokens']} training tokens in "
        f"{audit['train_documents']} documents; duplicate training documents "
        f"{audit['duplicate_training_documents']}; report written to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
