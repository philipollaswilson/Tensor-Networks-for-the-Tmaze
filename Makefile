# Makefile for installing requirements and UnsupGenModbyMPS

# Default Python interpreter
PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip

# Repository URL
REPO_URL := git@github.com:congzlwag/UnsupGenModbyMPS.git

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Create virtual environment, install requirements and UnsupGenModbyMPS"
	@echo "  venv        - Create virtual environment (.venv)"
	@echo "  requirements- Install only requirements from requirements.txt"
	@echo "  repo        - Install only UnsupGenModbyMPS from GitHub"
	@echo "  clean       - Clean up temporary files and virtual environment"
	@echo "  help        - Show this help message"

# Main install target
.PHONY: install
install: venv requirements repo
	@echo "Installation complete!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Create virtual environment
.PHONY: venv
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Upgrading pip in virtual environment..."; \
		$(VENV_PIP) install --upgrade pip; \
	else \
		echo "Virtual environment $(VENV_DIR) already exists"; \
	fi

# Install requirements from requirements.txt if it exists
.PHONY: requirements
requirements: venv
	@if [ -f requirements.txt ]; then \
		echo "Installing requirements from requirements.txt..."; \
		$(VENV_PIP) install -r requirements.txt; \
	else \
		echo "No requirements.txt found, skipping requirements installation"; \
	fi

# Install the GitHub repository
.PHONY: repo
repo: venv
	@echo "Installing UnsupGenModbyMPS from GitHub..."
	git clone $(REPO_URL) > lib/

# Clean up temporary files and virtual environment
.PHONY: clean
clean:
	@echo "Cleaning up temporary files and virtual environment..."
	rm -rf $(VENV_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
