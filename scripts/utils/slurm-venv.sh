#!/usr/bin/env bash
# usage: source setup-erisk-venv.sh [venv_dir]

# Set project root
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(realpath "$(dirname "$(dirname "$SCRIPT_DIR")")")"  # assuming script is in scripts/utils

# Default venv directory: ~/scratch/erisk-venv (or pass custom path as $1)
VENV_DIR="${1:-$HOME/scratch/erisk-venv}"
mkdir -p "$VENV_DIR"
cd "$VENV_DIR" || exit

# Load Python module if on HPC
module load python/3.10

# Install pip and uv if missing
if ! command -v uv &> /dev/null; then
    python -m ensurepip --upgrade
    pip install --upgrade pip uv
fi

# Create virtual environment if missing
if [[ ! -d venv ]]; then
    echo "Creating virtual environment in ${VENV_DIR}/venv ..."
    uv venv venv
fi

# Activate environment
source venv/bin/activate

# Install required packages
uv pip install -r "$PROJECT_ROOT/requirements.txt"

# Confirm
echo "✓ Virtual environment is ready."
echo "Python: $(which python)"
echo "Version: $(python --version)"
