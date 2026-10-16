# ADR 0006: explicit dependency runtime for standalone evaluation tests

**Status:** Accepted for disposable development workers, subject to CI validation
**Date:** 2026-10-16

## Context and decision

Official standalone benchmark tests require NumPy and larger input scripts than the stdlib sandbox's default. Preserve the default runtime, input cap, and `-S` child behavior. Add an explicit immutable local image-ID profile that can enable system site packages under Python `-I`. Construct this profile only from an owner-controlled build, never from a dataset row or model output.

The separate Docker build context contains only its Dockerfile and hash-locked requirement, based on a pinned Linux/amd64 Python 3.12.13 image. NumPy uses the same version and CPython 3.12 wheel hash as the lockfile, binary-only without dependency resolution. Record the actual built image ID; a mutable tag is rejected at execution. Set BLAS/OMP threads to one in the clean candidate environment.

## Safety and limitations

No candidate network, host mounts, inherited credentials, elevated user/capabilities, or unrestricted execution are introduced. Memory/process/time/output limits remain mandatory. An explicit bounded input cap can be raised for larger tests; the file limit accounts for that bound. Build-time dependency network access does not grant candidate network access. The dedicated build context excludes unrelated workspace contents.

This remains a disposable development boundary with the limitations of [ADR 0004](0004-development-sandbox.md), not production signoff or kernel-proof isolation. Loading system packages enlarges the software surface. A compatible runtime does not prove equivalence to upstream timing/memory policy or validate a benchmark score. Dedicated CI tests must exercise functional, failure, timeout, environment, filesystem and network cases before use.
