# Getting started

## Public contributors

1. Clone `https://github.com/kudratsingh/personal-coding-agent.git`.
2. Read the root `AGENTS.md`, [status](status.md), and [roadmap](roadmap.md).
3. Install uv, then run `make setup && make check` from the repository root.
4. Discuss a focused implementation task in an issue before a substantial change.

Run `make smoke` for the offline synthetic pipeline. It builds a tokenizer and shards, runs two tiny training stages, saves checkpoints, samples text, and validates held-out token loss. It does not download external data or execute generated programs.

## Owner checkout

If `private/START_HERE.md` exists locally, follow it for the detailed work-order sequence. Keep its contents local and back it up in private storage separately from the public repository.
