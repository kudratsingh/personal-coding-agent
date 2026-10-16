UV ?= uv

.PHONY: setup lint typecheck test check docs-check privacy-check smoke tok-train data-build train eval serve
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

check: privacy-check lint typecheck test docs-check

privacy-check:
	$(UV) run --locked python -m tools.check_public_tree

docs-check:
	$(UV) run --locked python -m tools.check_docs

smoke:
	$(UV) run --locked scratchcoder smoke

tok-train:
	$(UV) run --locked scratchcoder tok-train $(ARGS)

data-build:
	$(UV) run --locked scratchcoder data-build $(ARGS)

train:
	$(UV) run --locked scratchcoder train $(ARGS)

eval:
	$(UV) run --locked scratchcoder eval $(ARGS)

serve:
	@echo "$@ is not implemented yet. See docs/status.md." >&2
	@exit 2

paid-up paid-sync paid-train paid-eval paid-status paid-down:
	@echo "$@ is not implemented. Paid operations require approval for the specific run." >&2
	@exit 2
