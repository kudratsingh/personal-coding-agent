"""Check relative Markdown file links in the public starter documentation."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.publication import public_documents

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    try:
        files = public_documents(ROOT)
    except (OSError, subprocess.CalledProcessError):
        print("Documentation check failed: Git inventory unavailable.")
        return 1
    errors: list[str] = []
    for document in files:
        parts = document.relative_to(ROOT).parts
        if any((ROOT.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)):
            errors.append(f"{document.relative_to(ROOT)}: symlink document is not read")
            continue
        for raw_link in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_link.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (document.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT):
                errors.append(f"{document.relative_to(ROOT)}: link escapes repository: {target}")
            elif resolved.relative_to(ROOT).parts[:1] in {("private",), ("plans",)} or (
                resolved.is_relative_to(ROOT / "docs" / "internal")
            ):
                errors.append(f"{document.relative_to(ROOT)}: links to private material: {target}")
            elif not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing link target: {target}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked relative file links in {len(files)} public Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
