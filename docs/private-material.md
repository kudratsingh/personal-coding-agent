# Keeping private material private

This project uses a public repository with private working material stored outside version control. The following paths are ignored by default:

- `private/` for owner-only source documents and notes
- `plans/` for detailed work plans
- `docs/internal/` for internal operating documentation

Credentials belong in `.env` or a dedicated secret manager; datasets, checkpoints, and experiment artifacts belong in the ignored paths listed in `.gitignore`.

Before committing, inspect the staged diff for secrets, personal data, supplier terms, unreleased results, or detailed operational plans.
