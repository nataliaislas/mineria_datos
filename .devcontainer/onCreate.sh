#!/bin/bash

set -e

echo "Setting up development environment..."

# Install Just command runner (one-time installation)
if ! command -v just &> /dev/null; then
    echo "Installing Just..."
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
fi

# Sync Python dependencies
echo "Installing Python dependencies..."
uv sync --all-extras --quiet --dev --frozen

# Register Jupyter kernel
echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name mineria-datos --display-name "Python 3.13 (mineria-datos)"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install
