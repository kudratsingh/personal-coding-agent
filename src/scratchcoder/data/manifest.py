"""Fail-closed provenance checks and content-addressed downloads."""

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from http.client import HTTPMessage
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

# D-05 training-data allowlist. ISC, Unlicense, 0BSD and WTFPL were always allowed;
# the written list omitted them until the owner's 2026-10-24 corrections (not new
# decisions).
TRAINING_LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "0BSD",
        "WTFPL",
        "CC0-1.0",
        "CC-BY-4.0",
        "ODC-By-1.0",
        "CDLA-Permissive-2.0",
    }
)


def _require_https_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not any(character.isspace() for character in url)
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("download URL requires HTTPS, a host, and no inline credentials")


class HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    """Keep the manifest's transport restriction across every redirect hop."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        try:
            _require_https_url(newurl)
        except ValueError:
            fp.close()
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    revision: str
    url: str
    license: str
    attribution: str
    sha256: str
    size_bytes: int
    purpose: str = "train"
    license_verified: bool = False
    terms_required: bool = False
    terms_accepted_on: str | None = None

    def validate(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.source,
                self.revision,
                self.url,
                self.license,
                self.attribution,
                self.sha256,
                self.purpose,
            )
        ):
            raise ValueError("manifest provenance fields must be nonempty strings")
        if type(self.license_verified) is not bool or type(self.terms_required) is not bool:
            raise ValueError("manifest review and terms flags must be booleans")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("expected byte size must be a nonnegative integer")
        if not self.source or not re.fullmatch(r"[0-9a-f]{40,64}", self.revision):
            raise ValueError("source and immutable revision are required")
        _require_https_url(self.url)
        if not self.license_verified or not self.license or not self.attribution:
            raise ValueError("license verification and attribution are required")
        if self.purpose not in {"train", "eval"}:
            raise ValueError("purpose must be train or eval")
        if self.purpose == "train" and self.license not in TRAINING_LICENSES:
            raise ValueError("training license is not allowlisted")
        if self.terms_required and self.terms_accepted_on is None:
            raise ValueError("owner must accept source terms before download")
        if self.terms_accepted_on is not None:
            if not isinstance(self.terms_accepted_on, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.terms_accepted_on
            ):
                raise ValueError("terms acceptance must be a YYYY-MM-DD date")
            if date.fromisoformat(self.terms_accepted_on) > date.today():
                raise ValueError("terms acceptance cannot be in the future")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.size_bytes < 0:
            raise ValueError("expected SHA-256 and byte size are required")


def _network_chunks(url: str) -> Iterator[bytes]:
    _require_https_url(url)
    opener = build_opener(HTTPSOnlyRedirectHandler())
    try:
        response = opener.open(url, timeout=30)
    except HTTPError as error:
        error.close()
        raise
    with response:
        while chunk := response.read(1024 * 1024):
            yield chunk


def download(
    entry: ManifestEntry, destination: Path, *, chunks: Iterable[bytes] | None = None
) -> Path:
    """Validate before opening the network; publish only verified complete bytes."""
    entry.validate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for chunk in _network_chunks(entry.url) if chunks is None else chunks:
                count += len(chunk)
                if count > entry.size_bytes:
                    raise ValueError("download exceeds manifested size")
                digest.update(chunk)
                stream.write(chunk)
            if count != entry.size_bytes or digest.hexdigest() != entry.sha256:
                raise ValueError("download size or hash mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return destination
