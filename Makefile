UV := $(HOME)/.local/bin/uv
PYTHON := $(UV) run python
SGOINFRE := $(shell [ -d /sgoinfre/$(USER) ] && echo /sgoinfre/$(USER) || echo $(HOME))
UV_ENV := UV_CACHE_DIR=$(SGOINFRE)/.cache/uv UV_PROJECT_ENVIRONMENT=$(SGOINFRE)/.venv_rag HF_HOME=$(SGOINFRE)/.cache/huggingface

.PHONY: help install run debug clean lint test

help:
	@echo "Available targets:"
	@echo "  make install      - Install project dependencies"
	@echo "  make run          - Run the project"
	@echo "  make debug        - Run the project with pdb"
	@echo "  make test         - Run unit tests"
	@echo "  make clean        - Remove caches and generated files"
	@echo "  make lint         - Run flake8 and required mypy checks"

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	mkdir -p $(SGOINFRE)/.cache/uv
	mkdir -p $(SGOINFRE)/.cache/huggingface
	@grep -qxF 'export UV_CACHE_DIR=$(SGOINFRE)/.cache/uv' $(HOME)/.zshrc \
		|| echo 'export UV_CACHE_DIR=$(SGOINFRE)/.cache/uv' >> $(HOME)/.zshrc
	@grep -qxF 'export UV_PROJECT_ENVIRONMENT=$(SGOINFRE)/.venv_rag' $(HOME)/.zshrc \
		|| echo 'export UV_PROJECT_ENVIRONMENT=$(SGOINFRE)/.venv_rag' >> $(HOME)/.zshrc
	@grep -qxF 'export HF_HOME=$(SGOINFRE)/.cache/huggingface' $(HOME)/.zshrc \
		|| echo 'export HF_HOME=$(SGOINFRE)/.cache/huggingface' >> $(HOME)/.zshrc
	@echo ""
	@echo "Installation Finished!"
	@echo "Run: source ~/.zshrc"
	$(UV_ENV) $(UV) sync

run:
	$(UV_ENV) $(PYTHON) -m src

debug:
	$(UV_ENV) $(PYTHON) -m pdb -m src

test:
	$(UV_ENV) $(UV) run pytest tests/ -v

clean:
	rm -rf .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(UV_ENV) $(UV) run flake8 . --exclude=.venv,data
	$(UV_ENV) $(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude .venv \
		--exclude data

