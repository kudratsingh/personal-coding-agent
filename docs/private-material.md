# Keeping private material private

This project uses a public repository with private working material stored outside version control. The following paths are ignored by default:

- `private/` for owner-only source documents and notes
- `plans/` for detailed work plans
- `docs/internal/` for internal operating documentation

Credentials belong in `.env` or a dedicated secret manager; datasets, checkpoints, and experiment artifacts belong in the ignored paths listed in `.gitignore`.

Before committing, inspect the staged diff for secrets, personal data, supplier terms, unreleased results, or detailed operational plans.

## Read-only publication checks

Run `make privacy-check` before publishing (also included in `make check`). It inventories the Git index and rejects paths matching the checkout's `.gitignore` files, including files added with `git add --force`. Negative rules such as `.env.example` remain allowed. Machine-global ignore files and `.git/info/exclude` do not change this policy. The check reads paths, not file contents, and never removes or unstages anything.

`make docs-check` inventories tracked and nonignored untracked root/`docs/` Markdown, so new public docs are checked before staging. Ignored local notes are excluded even if force-added; the separate publication check rejects those index entries. Symlink documents or symlink ancestors are reported without reading the target. Public links to private planning directories remain errors. Invoke the utilities directly with `python -m tools.check_public_tree` and `python -m tools.check_docs` from the repository root.

These checks require a Git checkout. They are not secret detection, encryption, or a guarantee that public files contain no private details. Review staged content yourself. CI is a second line of defense: it cannot undo disclosure after a push. If a path is rejected, review its intended classification and unstage it yourself if appropriate; do not weaken ignore rules simply to silence the check.
