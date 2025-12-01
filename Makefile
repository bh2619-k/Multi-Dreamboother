# ===============================
# Makefile for DreamBooth
# ===============================

SHELL := /bin/bash

# ---------
# Variables
# ---------
PYTHON        := python3
PIP           := pip
TORCH_CUDA_URL:= https://download.pytorch.org/whl/cu128
CODE_DIRS     := src tests scripts
EXCLUDES      := (\.venv|\.git|\.mypy_cache|\.pytest_cache|\.ruff_cache|\.eggs|build|dist|.*egg-info|checkpoints|data|outputs|logs)

.DEFAULT_GOAL := help

# Ensure tools fail fast if missing
define require_cmd
	@command -v $(1) >/dev/null 2>&1 || { echo "Error: '$(1)' is not installed or not on PATH."; exit 1; }
endef

# -------------
# Setup / Deps
# -------------

venv: ## Create a virtual environment (.venv)
	$(PYTHON) -m venv .venv
	@echo "To activate: source .venv/bin/activate"
	@. .venv/bin/activate && \
		pip install --upgrade pip setuptools wheel

install-gpu: ## Install GPU PyTorch (CUDA 12.8) + project
	@echo "Installing PyTorch (CUDA 12.8 wheels)..."
	$(PIP) install --index-url $(TORCH_CUDA_URL) torch torchvision
	@echo "Installing project in editable mode..."
	$(PIP) install -e .

install-cpu: ## Install CPU-only PyTorch + project
	@echo "Installing CPU-only PyTorch..."
	$(PIP) install torch torchvision
	@echo "Installing project in editable mode..."
	$(PIP) install -e .

install-dev: ## Install project with dev dependencies
	$(PIP) install -e ".[dev]"

freeze: ## Export current environment to requirements.txt
	$(PIP) freeze | grep -vE "^-e git\+|^-e \.$$" > requirements.txt
	@echo "Wrote requirements.txt"

# ------------------
# Code Quality / QA
# ------------------
format: ## Auto-format code (black + isort) within CODE_DIRS
	$(call require_cmd,black)
	$(call require_cmd,isort)
	@echo "Formatting: $(CODE_DIRS)"
	black $(CODE_DIRS) --extend-exclude "$(EXCLUDES)"
	isort $(CODE_DIRS) --extend-skip-glob "$(EXCLUDES)"

lint: ## Lint code (ruff)
	$(call require_cmd,ruff)
	@echo "Linting: $(CODE_DIRS)"
	ruff check $(CODE_DIRS)

lint-fix: ## Auto-fix lint/style (ruff + isort + black)
	$(call require_cmd,ruff)
	$(call require_cmd,isort)
	$(call require_cmd,black)
	@echo "Auto-fixing lint/style in: $(CODE_DIRS)"
	ruff check $(CODE_DIRS) --fix
	isort $(CODE_DIRS) --extend-skip-glob "$(EXCLUDES)"
	black $(CODE_DIRS) --extend-exclude "$(EXCLUDES)"

# -------------
# Testing
# -------------
test: ## Run tests
	$(call require_cmd,pytest)
	pytest -v --maxfail=1 --disable-warnings

test-fast: ## Run tests without coverage, stop early
	$(call require_cmd,pytest)
	pytest -q -x

# -------------
# GPU & Torch
# -------------
check-gpu: ## Show PyTorch/CUDA/GPU info
	$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# -------------
# Utilities
# -------------
clean: ## Clean caches/build artifacts
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache .ruff_cache .venv build dist *.egg-info
	find . -type f -name '*.pyc' -delete
	@echo "Cleaned."

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: venv install-gpu install-cpu install-dev freeze format lint lint-fix test test-fast check-gpu clean help
