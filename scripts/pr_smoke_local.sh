#!/usr/bin/env bash
set -euo pipefail

# Local smoke test to mirror CI without building C++ from source.
# Requires: Python 3.10+ and internet access to install the open_spiel wheel.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}
VENV_DIR="${ROOT_DIR}/.venv-pr-smoke"

echo "[1/5] Creating virtual environment at ${VENV_DIR}"
${PYTHON} -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[2/5] Upgrading pip and installing deps"
pip install --upgrade pip
pip install --upgrade open_spiel numpy absl-py

echo "[3/5] Environment info"
python --version
python -c "import pyspiel,sys;print('pyspiel:', pyspiel.__file__);print('sys.path[0:3]:', sys.path[0:3])"

echo "[4/5] Syntax checks"
python -m py_compile \
  "${ROOT_DIR}/open_spiel/python/examples/example.py" \
  "${ROOT_DIR}/open_spiel/python/examples/game_stats.py"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/build/python:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

echo "[5/5] Run example scripts"
python "${ROOT_DIR}/open_spiel/python/examples/example.py" --game_string=tic_tac_toe
python "${ROOT_DIR}/open_spiel/python/examples/game_stats.py" --game_name=tic_tac_toe --num_samples=50 --output_json=stats.json

ls -l stats.json || true

echo "✓ Smoke tests passed"
