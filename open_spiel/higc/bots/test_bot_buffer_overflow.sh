#!/usr/bin/bash

BASE=$(dirname "$0")
source "$BASE/../../../venv/bin/activate"

echo "ready"
echo "start"

# Test for some fun overflows and killing child processes.
python -c 'print("x"*100000)'
