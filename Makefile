UV ?= uv

.PHONY: setup lint typecheck test check docs-check smoke tok-train data-build train eval serve
.PHONY: paid-up paid-sync paid-train paid-eval paid-status paid-down

setup:
	$(UV) sync --locked

lint:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

typecheck:
	$(UV) run --locked mypy

test:
	$(UV) run --locked pytest

check: lint typecheck test docs-check

docs-check:
	$(UV) run --locked python tools/check_docs.py

smoke:
	$(UV) run --locked scratchcoder smoke

tok-train data-build train eval serve:
	@echo "$@ is not implemented yet. See docs/status.md." >&2
	@exit 2

paid-up paid-sync paid-train paid-eval paid-status paid-down:
	@echo "$@ is not implemented. Paid operations require approval for the specific run." >&2
	@exit 2
