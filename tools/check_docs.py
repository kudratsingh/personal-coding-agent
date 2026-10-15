"""Check relative Markdown file links in the public starter documentation."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    files = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    errors: list[str] = []
    for document in files:
        for raw_link in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_link.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (document.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT):
                errors.append(f"{document.relative_to(ROOT)}: link escapes repository: {target}")
            elif resolved.relative_to(ROOT).parts[0] in {"private", "plans"} or (
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
